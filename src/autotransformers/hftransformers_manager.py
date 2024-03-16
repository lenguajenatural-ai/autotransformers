from transformers import (
    AutoTokenizer,
    EncoderDecoderModel,
    Seq2SeqTrainer,
    Trainer,
    Seq2SeqTrainingArguments,
    TrainingArguments,
    AutoModelForSequenceClassification,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    AutoConfig,
    AutoModelForTokenClassification,
    AutoModelForQuestionAnswering,
    DataCollatorForTokenClassification,
    DataCollatorForSeq2Seq,
)
import torch
from functools import partial
from .dataset_config import DatasetConfig
from .model_config import ModelConfig
from typing import Dict
from trl import DPOTrainer


class MultilabelTrainer(Trainer):
    """Version of the trainer used for multilabel setting."""

    def compute_loss(self, model, inputs, return_outputs=False):
        """Compute loss of the model.

        Parameters
        ----------
        model : transformers.PreTrainedModel
            Model to compute loss.
        inputs : torch.Tensor
            Model inputs.
        return_outputs : bool
            Wether or not to return model outputs.
        """
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = torch.nn.BCEWithLogitsLoss()
        loss = loss_fct(
            logits.view(-1, self.model.config.num_labels),
            labels.float().view(-1, self.model.config.num_labels),
        )
        return (loss, outputs) if return_outputs else loss


map_trainer_cls = {
    "classification": Trainer,
    "ner": Trainer,
    "qa": Trainer,
    "chatbot": Trainer,
    "multilabel": MultilabelTrainer,
    "seq2seq": Seq2SeqTrainer,
    "alignment": DPOTrainer,
}

map_model_cls = {
    "classification": AutoModelForSequenceClassification,
    "ner": AutoModelForTokenClassification,
    "qa": AutoModelForQuestionAnswering,
    "seq2seq": AutoModelForSeq2SeqLM,
    "chatbot": AutoModelForCausalLM,
    "alignment": AutoModelForCausalLM,
}


class HFTransformersManager:
    """
    Utility for loading HF Transformers' objects, using a dataset config and a model config.

    Parameters
    ----------
    model_config: autotransformers.ModelConfig
        Configuration for the model.
    dataset_config: autotransformers.DatasetConfig
        Configuration for the dataset
    """

    def __init__(
        self,
        model_config: ModelConfig = None,
        dataset_config: DatasetConfig = None,
        use_auth_token: bool = True,
    ):
        self.model_config = model_config
        self.dataset_config = dataset_config
        self.use_auth_token = use_auth_token

    def load_config(self, tag2id: Dict, dropout: float = 0.0):
        """
        Load configuration for the model depending on the type of task we are doing.

        Parameters
        ----------
        tag2id: Dict
            Dictionary mapping labels to indices of those labels in the network output layer.
        dropout: float
            Dropout percentage to apply at classification head.

        Returns
        -------
        config: transformers.PretrainedConfig
            Configuration for use in the transformers module.
        """
        task = self.dataset_config.task
        model_name = self.model_config.name
        if task in ["qa", "multiple_choice"]:
            config = AutoConfig.from_pretrained(
                model_name,
                use_auth_token=self.use_auth_token,
                trust_remote_code=True,
                **{self.model_config.dropout_field_name: dropout},
            )
        elif task in ["seq2seq", "chatbot"]:
            config = AutoConfig.from_pretrained(
                model_name,
                use_auth_token=self.use_auth_token,
                trust_remote_code=True,
            )
        else:
            if not self.model_config.custom_config_class:
                config = AutoConfig.from_pretrained(
                    model_name,
                    use_auth_token=self.use_auth_token,
                    trust_remote_code=True,
                    num_labels=(
                        len(tag2id)
                        if not self.dataset_config.config_num_labels
                        else self.dataset_config.config_num_labels
                    ),
                    **{self.model_config.dropout_field_name: dropout},
                )
            else:
                config = self.model_config.custom_config_class
            if not self.dataset_config.config_num_labels == 1:
                setattr(config, "label2id", tag2id)
                setattr(config, "id2label", {i: tag for tag, i in tag2id.items()})
        if self.dataset_config.model_config_problem_type:
            setattr(
                config, "problem_type", self.dataset_config.model_config_problem_type
            )
        if self.model_config.custom_params_config_model is not None:
            for param, val in self.model_config.custom_params_config_model.items():
                setattr(config, param, val)
        return config

    def load_data_collator(self, tokenizer):
        """
        Load data collator depending on the type of task we are doing.

        Parameters
        ----------
        tokenizer: transformers.PretrainedTokenizer
            Tokenizer to process data.

        Returns
        -------
        data_collator: transformers.DataCollator
            DataCollator for use in the transformers library.
        """
        data_collator = None
        if self.dataset_config.task == "ner":
            data_collator = DataCollatorForTokenClassification(tokenizer)
        elif self.dataset_config.task in ["seq2seq", "chatbot"]:
            data_collator = DataCollatorForSeq2Seq(
                tokenizer, return_tensors="pt", padding=True
            )
        return data_collator

    def load_tokenizer(
        self,
    ):
        """
        Load tokenizer for the given model config and model name.

        Returns
        -------
        tokenizer:
            Loaded tokenizer.
        """
        if self.model_config.additional_params_tokenizer:
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_config.name,
                do_lower_case=False,
                add_prefix_space=True,
                use_fast=True,
                use_auth_token=self.use_auth_token,
                **self.model_config.additional_params_tokenizer,
            )
        else:
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_config.name,
                do_lower_case=False,
                add_prefix_space=True,
                use_fast=True,
                use_auth_token=self.use_auth_token,
            )
        if self.model_config.func_modify_tokenizer is not None:
            tokenizer = self.model_config.func_modify_tokenizer(tokenizer)
        return tokenizer

    def get_model_cls(self):
        """
        Get the class to use for a model.

        Returns
        -------
        model_cls:
            Class for the model.
        """
        if not self.model_config.custom_model_class:
            model_cls = (
                self.model_config.model_cls_summarization
                if self.dataset_config.task == "seq2seq"
                and self.model_config.model_cls_summarization
                else map_model_cls[self.dataset_config.task]
            )
            return model_cls
        else:
            return self.model_config.custom_model_class

    def load_trainer(
        self,
        dataset,
        tokenizer,
        args,
        model_init,
        data_collator,
        compute_metrics_func,
        config,
    ):
        """
        Load an instantiated Trainer object depending on the configuration.

        Parameters
        ----------
        dataset: datasets.DatasetDict
            Dataset with train and validation splits.
        tokenizer: transformers.PretrainedTokenizer
            Tokenizer from transformers.
        args: transformers.TrainingArguments
            TrainingArguments for the Trainer.
        model_init: Any
            Function that loads the model.
        data_collator: Any
            Data Collator to use inside Trainer.
        compute_metrics_func: Any
            Function to compute metrics.
        config: transformers.PretrainedConfig
            Configuration for the model in Huggingface Transformers.

        Returns
        -------
        Trainer: transformers.Trainer
            Trainer object loaded with the given configuration.
        """
        trainer_cls = (
            self.model_config.custom_trainer_cls
            if self.model_config.custom_trainer_cls
            else map_trainer_cls[self.dataset_config.task]
        )
        trainer_params = {
            "args": args,
            "train_dataset": dataset[
                "train" if not self.model_config.only_test else "test"
            ],
            "eval_dataset": dataset[
                "validation" if not self.model_config.only_test else "test"
            ],
            "data_collator": data_collator,
            "tokenizer": tokenizer,
            "compute_metrics": (
                partial(
                    compute_metrics_func,
                    tokenizer=tokenizer,
                    id2tag=config.id2label if config else None,
                )
                if compute_metrics_func
                else None
            ),
            "callbacks": self.dataset_config.callbacks,
        }
        if self.dataset_config.task == "alignment":
            trainer_params["model"] = model_init()
            if not self.model_config.peft_config:
                trainer_params["ref_model"] = model_init()
            else:
                trainer_params["ref_model"] = None
                trainer_params["peft_config"] = self.model_config.peft_config

            if self.model_config.alignment_config:
                alignment_config = self.model_config.alignment_config
            else:
                alignment_config = {
                    "beta": 0.5,
                    "max_target_length": 1024,
                    "max_prompt_length": 4096 - 1024,
                    "generate_during_eval": False,
                }
            trainer_params.update(alignment_config)
        elif self.dataset_config.task != "chatbot":
            trainer_params["model_init"] = model_init
        else:
            model = model_init()
            trainer_params["model"] = model
            if self.model_config.neftune_noise_alpha is not None:
                trainer_params[
                    "neftune_noise_alpha"
                ] = self.model_config.neftune_noise_alpha
        trainer = trainer_cls(**trainer_params)
        return trainer

    def load_train_args(self, output_dir):
        """
        Load training args depending on the task.

        Parameters
        ----------
        output_dir: str
            Local directory name to save the model.

        Returns
        -------
        args: transformers.TrainingArguments
            Arguments for training.
        """
        self.dataset_config.fixed_training_args.update(
            {
                "metric_for_best_model": self.dataset_config.metric_optimize,
                "greater_is_better": self.dataset_config.direction_optimize
                == "maximize",
            }
        )
        fixed_training_args = self.dataset_config.fixed_training_args.copy()
        if self.model_config.overwrite_training_args:
            for k, v in self.model_config.overwrite_training_args.items():
                fixed_training_args.update({k: v})
        if self.dataset_config.task != "seq2seq":
            args = TrainingArguments(
                output_dir=output_dir,
                run_name=output_dir,
                overwrite_output_dir=True,
                report_to=["tensorboard"],
                **fixed_training_args,
            )
        else:
            args = Seq2SeqTrainingArguments(
                output_dir=output_dir,
                run_name=output_dir,
                overwrite_output_dir=True,
                predict_with_generate=True,
                report_to=["tensorboard"],
                **fixed_training_args,
            )
        return args

    def load_model_init(self, model_cls, config, tokenizer):
        """
        Load the model init function.

        This function is useful for the Transformers integration with Optuna.

        Parameters
        ----------
        model_cls
            Class for the model.
        config: AutoConfig
            Configuration for the model.
        tokenizer: transformers.PretrainedTokenizer
            Tokenizer to preprocess text data.

        Returns
        -------
        model_init
            Function for initializing the model. Furtherly passed to the Trainer.
        """
        if not self.model_config.custom_model_class:

            def model_init(trial=None):
                """Model init function needed by optuna to initialize the model."""
                if len(self.model_config.dropout_vals) > 1 and trial:
                    dropout = trial.suggest_float(
                        "cls_dropout",
                        self.model_config.dropout_vals[0],
                        self.model_config.dropout_vals[1],
                    )
                    self.model_config.custom_params_model.update(
                        {self.model_config.dropout_field_name: dropout}
                    )
                if self.dataset_config.is_multilabel:
                    config.update(
                        {
                            "problem_type": "multi_label_classification",
                            "num_labels": self.dataset_config.config_num_labels,
                        }
                    )
                if self.model_config.custom_params_model is None:
                    self.model_config.custom_params_model = {
                        "use_auth_token": self.use_auth_token
                    }
                else:
                    self.model_config.custom_params_model[
                        "use_auth_token"
                    ] = self.use_auth_token
                return model_cls.from_pretrained(
                    self.model_config.name,
                    config=config,
                    quantization_config=self.model_config.quantization_config,
                    **self.model_config.custom_params_model,
                )

        else:
            if self.model_config.custom_config_class:
                # for monster models (ensembles of transformers and similar).

                def model_init(trial=None):
                    if len(self.model_config.dropout_vals) > 1 and trial:
                        dropout = trial.suggest_float(
                            "cls_dropout",
                            self.model_config.dropout_vals[0],
                            self.model_config.dropout_vals[1],
                        )
                        self.model_config.custom_params_model.update(
                            {self.model_config.dropout_field_name: dropout}
                        )
                    return self.model_config.custom_model_class(
                        self.model_config.custom_config_class,
                        **self.model_config.custom_params_model,
                    )

            else:

                def model_init(trial=None):
                    if len(self.model_config.dropout_vals) > 1 and trial:
                        dropout = trial.suggest_float(
                            "cls_dropout",
                            self.model_config.dropout_vals[0],
                            self.model_config.dropout_vals[1],
                        )
                        self.model_config.custom_params_model.update(
                            {self.model_config.dropout_field_name: dropout}
                        )
                    return self.model_config.custom_model_class.from_pretrained(
                        self.model_config.name,
                        config=config,
                        **self.model_config.custom_params_model,
                    )

        if self.model_config.model_init_wrap_cls is not None:
            tokenizer = self.load_tokenizer()
            return self.model_config.model_init_wrap_cls(
                model_init, self.model_config, tokenizer
            )
        return model_init
