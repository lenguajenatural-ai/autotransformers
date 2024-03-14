from dataclasses import dataclass, field, asdict
from typing import List, Any, Dict
from transformers import PretrainedConfig, PreTrainedModel, BitsAndBytesConfig


@dataclass
class ModelConfig:
    """
    Configure a model to use inside the AutoTrainer class.

    With this we determine every choice related to the model
    such as the original name, the name to save the model with, the hyperparameter space, and a long etc.

    Parameters
    ----------
    name: str
        Name of the model, either in the HF hub or a path to the local directory where it is stored.
    save_name: str
        Alias for the model, used for saving it.
    hp_space
        The hyperparameter space for hyperparameter search with optuna. Must be a function receiving a trial and returning a dictionary with the corresponding suggest_categorical and float fields.
    dropout_vals: List
        Dropout values to try.
    custom_config_class: transformers.PretrainedConfig
        Custom configuration for a model. Useful for training ensembles of transformers.
    custom_model_class: transformers.PreTrainedModel
        Custom model. None by default. Only used for ensemble models and other strange creatures of Nature.
    partial_custom_tok_func_call: Any
        Partial call for a tokenization function, with all necessary parameters passed to it.
    max_length_summary: int
        Max length of the summaries. Useful for summarization datasets.
    min_length_summary : int
        Min length of the summaries. Useful for summarization datasets.
    no_repeat_ngram_size: int
        Number of n-grams to don't repeat when doing summarization.
    early_stopping_summarization: bool
        Whether to have early stopping when doing summarization tasks.
    length_penalty: float
        Length penalty for summarization tasks.
    num_beams: int
        Number of beams in beam search for summarization tasks.
    dropout_field_name: str
        Name for the dropout field in the pooler layer.
    n_trials : int
        Number of trials (trainings) to carry out with this model.
    random_init_trials: int
        Argument for optuna sampler, to control number of initial trials to run randomly.
    trainer_cls_summarization: Any
        Class for the trainer. Useful when it is desired to override the default trainer cls for summarization.
    model_cls_summarization: Any
        Class for the trainer. Useful when it is desired to override the default trainer cls for summarization.
    custom_tokenization_func: Any
        Custom tokenization function for processing texts. When the user does not want to use the default tokenization function for the task at hand, one can create a custom tokenization function. This function must receive samples from a dataset, a tokenizer and a dataset config.
    only_test: bool
        Whether to only test, not train (for already trained models).
    test_batch_size: int
        Batch size for test; only used when doing only testing.
    overwrite_training_args: Dict
        Arguments to overwrite the default arguments for the trainer, for example to change the optimizer for this concrete model.
    save_dir: str
        The directory to save the trained model.
    push_to_hub: bool
        Whether to push the best model to the hub.
    additional_params_tokenizer: Dict
        Additional arguments to pass to the tokenizer.
    resume_from_checkpoint: bool
        Whether to resume from checkpoint to continue training.
    config_problem_type: str
        The type of the problem, for loss fct.
    custom_trainer_cls: Any
        Custom trainer class to override the current one.
    do_nothing: bool
        Whether to do nothing or not. If true, will not train nor predict.
    custom_params_config_model: Dict
        Dictionary with custom parameters for loading AutoConfig.
    generation_params: Dict
        Parameters for generative tasks, for the generate call.
    hf_hub_username: str
        Username in HF Hub, to push models to hub.
    custom_results_getter: Any
        Custom class to get test results after training.
    quantization_config: BitsAndBytesConfig
        Configuration for quantizing models.
    custom_params_model: Dict
        Dictionary with custom parameters for initializing the model.
    model_init_wrap_cls: Any
        Wrap class with a `__call__` method for the `model_init` function, useful for example for preparing quantized models, etc.
    peft_config: Any
        Configuration class from Peft. For example, `LoraConfig`.
    func_modify_tokenizer: Any
        Function to modify the behavior of the tokenizer, for example to set the pad token id (for decoder-only models this is typically mandatory) or for changing the padding size.

    Examples
    --------
    With the following lines you can create a ModelConfig for bert-base-cased model.

    >>> from autotransformers import ModelConfig

    >>> from autotransformers.default_param_spaces import hp_space_base

    >>> model_config = ModelConfig(name='bert-base-cased', save_name='bert', hp_space=hp_space_base)
    """

    name: str = field(
        metadata={
            "help": "Name of the model, either in the HF hub or a path to the local directory where it is stored."
        }
    )
    save_name: str = field(
        metadata={"help": "Alias for the model, used for saving it."}
    )
    hp_space: Any = field(
        default=None,
        metadata={
            "help": "The hyperparameter space for hyperparameter search with optuna. Must be a function receiving a trial and returning a dictionary with the corresponding suggest_categorical and float fields."
        },
    )
    dropout_vals: List = field(
        default_factory=list, metadata={"help": "Dropout values to try."}
    )
    custom_config_class: PretrainedConfig = field(
        default=None,
        metadata={
            "help": "Custom configuration for a model. Useful for training ensembles of transformers."
        },
    )
    custom_model_class: PreTrainedModel = field(
        default=None,
        metadata={
            "help": "Custom model. None by default. Only used for ensemble models and other strange creatures of Nature."
        },
    )
    custom_tokenization_func: Any = field(
        default=None,
        metadata={
            "help": "Custom tokenization function for processing texts. When the user does not want to use the default tokenization function for the task at hand, one can create a custom tokenization function. This function must receive samples from a dataset, a tokenizer and a dataset config."
        },
    )
    max_length_summary: int = field(
        default=128,
        metadata={
            "help": "Max length of the summaries. Useful for summarization datasets."
        },
    )
    min_length_summary: int = field(
        default=10,
        metadata={
            "help": "Min length of the summaries. Useful for summarization datasets."
        },
    )
    no_repeat_ngram_size: int = field(
        default=3,
        metadata={
            "help": "Number of n-grams to don't repeat when doing summarization."
        },
    )
    early_stopping_summarization: bool = field(
        default=True,
        metadata={
            "help": "Whether to have early stopping when doing summarization tasks."
        },
    )
    length_penalty: float = field(
        default=2.0, metadata={"help": "Length penalty for summarization tasks."}
    )
    num_beams: int = field(
        default=1,
        metadata={"help": "number of beams in beam search for summarization tasks."},
    )
    dropout_field_name: str = field(
        default="cls_dropout",
        metadata={"help": "Name for the dropout field in the pooler layer."},
    )
    n_trials: int = field(
        default=20,
        metadata={"help": "Number of trials (trainings) to carry out with this model."},
    )
    random_init_trials: int = field(
        default=10,
        metadata={
            "help": "Argument for optuna sampler, to control number of initial trials to run randomly."
        },
    )
    trainer_cls_summarization: Any = field(
        default=None,
        metadata={
            "help": "Class for the trainer. Useful when it is desired to override the default trainer cls for summarization."
        },
    )
    model_cls_summarization: Any = field(
        default=None,
        metadata={
            "help": "Class for the trainer. Useful when it is desired to override the default trainer cls for summarization."
        },
    )
    only_test: bool = field(
        default=False,
        metadata={
            "help": "Whether to only test, not train (for already trained models)."
        },
    )
    test_batch_size: int = field(
        default=32,
        metadata={"help": "Batch size for test; only used when doing only testing."},
    )
    overwrite_training_args: Dict = field(
        default=None,
        metadata={
            "help": "Arguments to overwrite the default arguments for the trainer, for example to change the optimizer for this concrete model."
        },
    )
    save_dir: str = field(
        default=".", metadata={"help": "The directory to save the trained model."}
    )
    push_to_hub: bool = field(
        default=False, metadata={"help": "Whether to push the best model to the hub."}
    )
    additional_params_tokenizer: Dict = field(
        default=None,
        metadata={"help": "Additional arguments to pass to the tokenizer."},
    )
    resume_from_checkpoint: bool = field(
        default=False,
        metadata={"help": "Whether to resume from checkpoint to continue training."},
    )
    custom_trainer_cls: Any = field(
        default=None,
        metadata={"help": "Custom trainer class to override the current one."},
    )
    do_nothing: bool = field(
        default=False,
        metadata={
            "help": "Whether to do nothing or not. If true, will not train nor predict."
        },
    )
    custom_params_config_model: Dict = field(
        default=None,
        metadata={"help": "Dictionary with custom parameters for loading AutoConfig."},
    )
    generation_params: Dict = field(
        default=None,
        metadata={"help": "Parameters for generative tasks, for the generate call."},
    )
    hf_hub_username: str = field(
        default=None,
        metadata={"help": "Username in the Huggingface hub, to push models to hub."},
    )
    custom_results_getter: Any = field(
        default=None,
        metadata={"help": "Custom class to get test results after training."},
    )
    quantization_config: BitsAndBytesConfig = field(
        default=None,
        metadata={
            "help": "Quantization config for 8bit or 4bit quantization of models."
        },
    )
    custom_params_model: Dict = field(
        default=None,
        metadata={
            "help": """Dictionary with additional parameters for initializing the model. For example: {"trust_remote_code":True}."""
        },
    )
    model_init_wrap_cls: Any = field(
        default=None,
        metadata={
            "help": "Wrap class with a __call__ method for the `model_init` function, useful for example for preparing quantized models, etc."
        },
    )
    peft_config: Any = field(
        default=None,
        metadata={"help": "Configuration class from Peft. For example, `LoraConfig`."},
    )
    func_modify_tokenizer: Any = field(
        default=None,
        metadata={
            "help": "Function to modify the behavior of the tokenizer, for example to set the pad token id (for decoder-only models this is typically mandatory) or for changing the padding size."
        },
    )
    optuna_sampler: Any = field(
        default=None,
        metadata={
            "help": "Initialized Optuna Sampler to substitute TPESampler, which is the default sampler used."
        },
    )
    neftune_noise_alpha: int = field(
        default=None,
        metadata={
            "help": "Alpha parameter for NefTune (Noise Embeddings) technique for instruction tuning."
        },
    )
    alignment_config: Dict = field(
        default=None,
        metadata={"help": "Configuration for alignment tasks."},
    )

    def __str__(
        self,
    ):
        """Representation of model config in str."""
        self_as_dict = asdict(self)
        attrs_as_str = [f"{k}={v},\n" for k, v in sorted(self_as_dict.items())]
        return f"{self.__class__.__name__}(\n{''.join(attrs_as_str)})"
