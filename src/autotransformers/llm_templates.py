from typing import Any, Optional
from transformers import (
    TrainerCallback,
    PreTrainedModel,
    PreTrainedTokenizer,
    BitsAndBytesConfig,
    Trainer,
    PreTrainedModel,
)
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from peft import prepare_model_for_kbit_training, get_peft_model, PeftModel
import torch
from peft.tuners.lora import LoraLayer
import os
from functools import wraps
import random

CHAT_NATURAL_TEMPLATE = """{% for message in messages %}
    {% if message['role'] == 'user' %}
        {{'<user> ' + message['content'].strip() + ' </user>' }}
    {% elif message['role'] == 'system' %}
        {{'<system>\\n' + message['content'].strip() + '\\n</system>\\n\\n' }}
    {% elif message['role'] == 'assistant' %}
        {{ message['content'].strip() + ' </assistant>' + eos_token }}
    {% elif message['role'] == 'input' %}
        {{'<input> ' + message['content'] + ' </input>' }}
    {% endif %}
{% endfor %}"""

def instructions_to_chat(
    sample: dict,
    input_field: str,
    output_field: str,
    context_field: str = None,
    nested_field: str = None,
    system_message: Optional[str|list] = None,
) -> dict:
    """
    Processes a single sample from any dataset to structure it for chatbot training or instruction-based tasks,
    supporting nested structures and optional fields with a more Pythonic approach.

    Parameters
    ----------
    sample : dict
        A dictionary representing a single sample from the dataset.
    input_field : str
        The key for the user's input.
    output_field : str
        The key for the assistant's output.
    context_field : str, optional
        The key for additional context, if any.
    nested_field : str, optional
        The key for a nested field within the sample, if the data structure is nested.
    system_message : str or list, optional
        A system message to initialize the conversation. If None, a default message is used. If a list is provided, a random system message is selected from the list for each element.

    Returns
    -------
    dict
        A modified dictionary with a 'messages' key containing ordered messages,
        each annotated with its role in the conversation.
    """
    if not isinstance(system_message, list):
        system_msg = (
            system_message
            or "You are an assistant that solves user's instructions. Use additional context if provided to complete the instruction."
        )
    else:
        system_msg = random.choice(system_message)
    chat = [{"role": "system", "content": system_msg}]

    def extract_data(field, nested=None):
        return (sample.get(nested, {}) if nested else sample).get(field, "")

    if context := extract_data(context_field, nested_field):
        chat.append({"role": "context", "content": context})

    input_content = extract_data(input_field, nested_field)
    output_content = extract_data(output_field, nested_field)

    if input_content:
        chat.append({"role": "input", "content": input_content})

    chat.extend(
        [
            {
                "role": "user",
                "content": input_content,
            },
            {"role": "assistant", "content": output_content},
        ]
    )

    sample["messages"] = chat
    return sample


def neftune_forward(self, input: torch.Tensor):
    """
    Implement the NEFTune forward pass for the model. Note this works only for torch.nn.Embedding layers.

    This method is slightly adapted from the original source code that can be found here: https://github.com/neelsjain/NEFTune

    Parameters
    ----------
    input (`torch.Tensor`):
        The input tensor to the model.
    noise_alpha (`float`):
        The noise alpha value to use for the NEFTune forward pass.

    Returns
    -------
    embeddings: torch.Tensor
        Embeddings with random noise added.
    """
    embeddings = torch.nn.functional.embedding(
        input,
        self.weight,
        self.padding_idx,
        self.max_norm,
        self.norm_type,
        self.scale_grad_by_freq,
        self.sparse,
    )

    if self.training:
        dims = torch.tensor(embeddings.size(1) * embeddings.size(2))
        mag_norm = self.neftune_noise_alpha / torch.sqrt(dims)
        embeddings = embeddings + torch.zeros_like(embeddings).uniform_(
            -mag_norm, mag_norm
        )

    return embeddings


def activate_neftune(model, neftune_noise_alpha=5):
    """
    Activates Neftune noise injection in the input embeddings of a given model by replacing its original forward method with a custom forward method that includes Neftune noise.

    This function supports models of type PreTrainedModel and PeftModel. It modifies the model in-place by adding Neftune noise with a specified alpha value to the input embeddings and keeps a reference to the original forward method.

    Parameters
    ----------
    model : Union[PreTrainedModel, PeftModel]
        The model to modify. It should be an instance of either PreTrainedModel or PeftModel.
    neftune_noise_alpha : float, optional
        The alpha value for the Neftune noise to be applied. It controls the intensity of the noise injected into the input embeddings. Default value is 5.

    Returns
    -------
    model : Union[PreTrainedModel, PeftModel]
        The modified model with Neftune noise injection activated in its input embeddings.

    Notes
    -----
    - The activation of Neftune noise involves modifying the forward pass of the model's input embeddings to include noise injection.
    - This function retains a reference to the original forward method of the embeddings, allowing for potential restoration if needed.
    - The technique used to replace the forward method is based on a discussion from PyTorch's forums, acknowledging the complexity and hacky nature of this operation.

    References
    ----------
    - PyTorch Forum Discussion: https://discuss.pytorch.org/t/how-can-i-replace-the-forward-method-of-a-predefined-torchvision-model-with-my-customized-forward-function/54224/11
    """
    if isinstance(model, PreTrainedModel):
        embeddings = model.get_input_embeddings()
    elif isinstance(model, PeftModel):
        embeddings = model.base_model.get_input_embeddings()

    embeddings.neftune_noise_alpha = neftune_noise_alpha
    old_forward = embeddings.forward

    # This hack seems to be needed to properly use a custom forward pass
    # all credits to: https://discuss.pytorch.org/t/how-can-i-replace-the-forward-method-of-a-predefined-torchvision-model-with-my-customized-forward-function/54224/11
    bound_method = neftune_forward.__get__(embeddings, embeddings.__class__)
    setattr(embeddings, "forward", bound_method)

    embeddings._trl_old_forward = old_forward
    return model


class NEFTuneTrainer(Trainer):
    """
    A custom trainer class for integrating Neftune noise into the training process of models derived from PreTrainedModel or PeftModel.

    This trainer extends the functionality of the `Trainer` class by allowing the injection of Neftune noise into the input embeddings
    during training, and ensures the original forward method is restored after training completes.

    Parameters
    ----------
    neftune_noise_alpha : float, optional
        The alpha value for the Neftune noise to be applied. It controls the intensity of the noise injected into the input embeddings.
        If None, Neftune noise injection is not activated. Default is None.
    *args : variable length argument list
        Arguments passed to the `Trainer` class initializer.
    **kwargs : arbitrary keyword arguments
        Keyword arguments passed to the `Trainer` class initializer.

    Methods
    -------
    train(*args, **kwargs)
        Extends the `Trainer.train` method by injecting Neftune noise into the model's input embeddings before training and restoring
        the original forward pass method of the embeddings after training.

    Notes
    -----
    - It is important to ensure that the `neftune_noise_alpha` is set appropriately to avoid excessively distorting the input embeddings.
    - The restoration of the original forward method after training is crucial for maintaining the expected behavior of the model outside of training.
    """

    def __init__(self, neftune_noise_alpha=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.neftune_noise_alpha = neftune_noise_alpha

    @wraps(Trainer.train)
    def train(self, *args, **kwargs):
        """
        Start the training loop, with an optional integration of Neftune noise into the model's input embeddings. After training, it ensures that the model's input embeddings are restored to their original forward pass method, effectively removing the Neftune noise injection.

        This method extends the `train` method of the `Trainer` class, incorporating a step to modify the model for Neftune noise injection before the training begins, if `neftune_noise_alpha` is not None. After the training process, it restores the original forward method of the model's input embeddings to ensure the model can be used normally post-training.

        Parameters
        ----------
        *args : variable length argument list
            Arguments to be passed to the `Trainer.train` method.
        **kwargs : arbitrary keyword arguments
            Keyword arguments to be passed to the `Trainer.train` method.

        Returns
        -------
        output : torch.utils.data.DataLoader
            The output from the `Trainer.train` method, typically including training statistics and results.

        Notes
        -----
        - The method checks if `neftune_noise_alpha` is set and applies Neftune noise injection accordingly.
        - Neftune noise is applied only during training. This method ensures that any modifications are reversed post-training, restoring the original forward pass of the input embeddings.
        - This design allows the temporary integration of noise for experimental or augmentation purposes without permanently altering the model's behavior.
        """
        output = super().train(*args, **kwargs)
        # After training we make sure to retrieve back the original forward pass method
        # for the embedding layer
        if self.neftune_noise_alpha is not None:

            if isinstance(self.model, PreTrainedModel):
                embeddings = self.model.get_input_embeddings()
            elif isinstance(self.model, PeftModel):
                embeddings = self.model.base_model.get_input_embeddings()

            if hasattr(embeddings, "_trl_old_forward"):
                embeddings.forward = embeddings._trl_old_forward
                del embeddings._trl_old_forward
                del embeddings.neftune_noise_alpha
        return output


class QLoraWrapperModelInit:
    """
    A wrapper class for initializing transformer-based models with QLoRa and gradient checkpointing.

    This class serves as a wrapper for the `model_init` function, which initializes the model.
    It activates gradient checkpointing when possible and applies QLoRa to the model.

    Parameters
    ----------
    model_init : callable
        A function that initializes the transformer-based model for training.
    model_config : Any
        The configuration for the model.
    tokenizer : Any
        The tokenizer used for tokenization.

    Returns
    -------
    Pre-trained model with QLoRa and gradient checkpointing, if enabled.
    """

    def __init__(self, model_init: Any, model_config: Any, tokenizer: Any) -> None:
        self.model_init = model_init
        self.model_config = model_config
        self.tokenizer = tokenizer

    def __call__(self) -> PreTrainedModel:
        """
        Initialize the model and apply QLoRa and gradient checkpointing when configured.

        Returns
        -------
        Pre-trained model with QLoRa and gradient checkpointing, if enabled.
        """
        model = self.model_init()
        has_gradient_checkpointing = False
        if not model.__class__.__name__ in [
            "MPTForCausalLM",
            "MixFormerSequentialForCausalLM",
        ]:
            try:
                model.resize_token_embeddings(len(self.tokenizer))
            except Exception as e:
                print(
                    f"Could not resize token embeddings due to {e}, but will continue anyway..."
                )
            try:
                model.gradient_checkpointing_enable()
                has_gradient_checkpointing = True
            except Exception as e:
                print(f"Model checkpointing did not work: {e}")
        if model.__class__.__name__ == "LlamaForCausalLM":
            model.config.pretraining_tp = 1
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=has_gradient_checkpointing
        )
        model = get_peft_model(model, self.model_config.peft_config)
        model.config.use_cache = False
        if self.model_config.neftune_noise_alpha is not None:
            model = activate_neftune(model, self.model_config.neftune_noise_alpha)
        model = self.change_layer_types_for_stability(model)
        return model

    def change_layer_types_for_stability(
        self, model: PreTrainedModel
    ) -> PreTrainedModel:
        """
        Change layer types of the model for stability.

        Parameters
        ----------
        model : PreTrainedModel
            The pre-trained model.

        Returns
        -------
        Pre-trained model with modified layer types for stability.
        """
        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                module = module.to(torch.bfloat16)
            if "norm" in name:
                module = module.to(torch.float32)
            if "lm_head" in name or "embed_tokens" in name:
                if hasattr(module, "weight"):
                    module = module.to(torch.bfloat16)
        return model


def modify_tokenizer(
    tokenizer: PreTrainedTokenizer,
    pad_token_id: int = None,
    padding_side: str = None,
    new_model_seq_length: int = None,
    add_special_tokens: dict = None,
    add_tokens: list = None,
    chat_template: str = None,
) -> PreTrainedTokenizer:
    """
    Modify properties of a pre-trained tokenizer.

    This function allows you to modify various properties of a pre-trained tokenizer,
    such as the pad_token_id, padding_side, model_max_length, special tokens, and additional tokens.

    Parameters
    ----------
    tokenizer : PreTrainedTokenizer
        The pre-trained tokenizer to be modified.
    pad_token_id : int, optional (default=None)
        The ID of the padding token to set for the tokenizer.
    padding_side : str, optional (default=None)
        The side (either 'left' or 'right') to apply padding.
    new_model_seq_length : int, optional (default=None)
        The new maximum sequence length allowed by the tokenizer.
    add_special_tokens : dict, optional (default=None)
        A dictionary specifying special tokens to be added to the tokenizer.
    add_tokens : list, optional (default=None)
        A list of additional tokens to be added to the tokenizer's vocabulary.
    chat_template : str, optional (default=None)
        The chat template to use for the tokenizer.

    Returns
    -------
    PreTrainedTokenizer
        The modified pre-trained tokenizer.
    """
    if pad_token_id:
        tokenizer.pad_token_id = pad_token_id
    if padding_side:
        tokenizer.padding_side = padding_side
    if new_model_seq_length:
        tokenizer.model_max_length = new_model_seq_length
    if add_special_tokens:
        tokenizer.add_special_tokens(add_special_tokens)
    if add_tokens:
        tokenizer.add_tokens(add_tokens)
    if chat_template:
        tokenizer.chat_template = chat_template
    return tokenizer


qlora_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)


class SavePeftModelCallback(TrainerCallback):
    """
    Callback for saving only adapter layers from PEFT (Parameter Efficient Fine-Tuning) checkpoints.

    This callback is designed to be applied to the `transformers.Trainer` to save adapter layers
    from PEFT checkpoints instead of saving full model weights.

    Methods
    -------
    save_model(args, state, kwargs):
        Save the adapter model from the PEFT checkpoint.
    on_save(args, state, control, **kwargs):
        Triggered when saving the model during training. Calls `save_model` and returns control.
    on_train_end(args, state, control, **kwargs):
        Triggered at the end of training. Creates a 'completed' file and calls `save_model`.
    """

    def save_model(self, args, state, kwargs):
        """
        Save the adapter model from the PEFT checkpoint.

        Parameters
        ----------
        args : TrainerArguments
            Arguments for the trainer.
        state : TrainerState
            Current trainer state.
        kwargs : dict
            Additional keyword arguments, including the model.

        Returns
        -------
        None
        """
        print("Saving PEFT checkpoint...")
        if state.best_model_checkpoint is not None:
            checkpoint_folder = os.path.join(
                state.best_model_checkpoint, "adapter_model"
            )
        else:
            checkpoint_folder = os.path.join(
                args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}"
            )

        peft_model_path = os.path.join(checkpoint_folder, "adapter_model")
        kwargs["model"].save_pretrained(peft_model_path)

        pytorch_model_path = os.path.join(checkpoint_folder, "pytorch_model.bin")
        if os.path.exists(pytorch_model_path):
            os.remove(pytorch_model_path)

    def on_save(self, args, state, control, **kwargs):
        """
        Triggered when saving the model during training. Calls `save_model` and returns control.

        Parameters
        ----------
        args : TrainerArguments
            Arguments for the trainer.
        state : TrainerState
            Current trainer state.
        control : str
            Control string for trainer callback.
        kwargs : dict
            Additional keyword arguments.

        Returns
        -------
        str
            Control string.
        """
        self.save_model(args, state, kwargs)
        return control

    def on_train_end(self, args, state, control, **kwargs):
        """
        Triggered at the end of training. Creates a 'completed' file and calls `save_model`.

        Parameters
        ----------
        args : TrainerArguments
            Arguments for the trainer.
        state : TrainerState
            Current trainer state.
        control : str
            Control string for trainer callback.
        kwargs : dict
            Additional keyword arguments.

        Returns
        -------
        None
        """

        def touch(fname, times=None):
            with open(fname, "a"):
                os.utime(fname, times)

        touch(os.path.join(args.output_dir, "completed"))
        self.save_model(args, state, kwargs)
