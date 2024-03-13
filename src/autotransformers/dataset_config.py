from dataclasses import dataclass, asdict, field
from typing import Dict, List, Any


@dataclass
class DatasetConfig:
    """
    Configure a dataset for use within the AutoTrainer class.

    This determines how to load the dataset,
    whether local files are needed, whether additional splits are needed (for example when the original
    dataset only has train-test and we want also validation), and so on.

    Parameters
    ----------
    dataset_name: str
        The name of the dataset.
    alias: str
        Alias for the dataset, for saving it.
    task: str
        The task of the dataset. Currenlty, only classification, ner and qa (question answering) are available.
    fixed_training_args: Dict
        The training arguments (to use in transformers.TrainingArguments) for every model on this dataset, in dictionary format.
    is_multilabel: bool
        Whether it is multilabel classification
    multilabel_label_names: List
        Names of the labels for multilabel training.
    hf_load_kwargs: Dict
        Arguments for loading the dataset from the huggingface datasets' hub. Example: {'path': 'wikiann', 'name': 'es'}.
        If None, it is assumed that all necessary files exist locally and are passed in the files field.
    type_load: str
        The type of load to perform in load_dataset; for example, if your data is in csv format (d = load_dataset('csv', ...)), this should be csv.
    files: Dict
        Files to load the dataset from, in Huggingface's datasets format. Possible keys are train, validation and test.
    data_field: str
        Field to load data from in the case of jsons loading in datasets.
    partial_split: bool
        Wheter a partial split is needed, that is, if you only have train and test sets, this should be True so that a new validation set is created.
    split: bool
        This should be true when you only have one split, that is, a big train set; this creates new validation and test sets.
    label_col: str
        Name of the label column.
    val_size: float
        In case no validation split is provided, the proportion of the training data to leave for validation.
    test_size: float
        In case no test split is provided, the proportion of the total data to leave for testing.
    pre_func
        Function to perform previous transformations. For example, if your dataset lacks a field (like xquad with title field for example), you can fix it in a function provided here.
    squad_v2: bool
        Only useful for question answering. Whether it is squad v2 format or not. Default is false.
    text_field: str
        The name of the field containing the text. Useful only in case of unique-text-field datasets,like most datasets are. In case of 2-sentences datasets like xnli or paws-x this is not useful. Default is text.
    is_2sents: bool
        Whether it is a 2 sentence dataset. Useful for processing datasets like xnli or paws-x.
    sentence1_field: str
        In case this is a 2 sents dataset, the name of the first sentence field.
    sentence2_field: str
        In case this is a 2 sents dataset, the name of the second sentence field.
    summary_field: str = field(
        The name of the field with summaries (we assume the long texts are in the text_field field). Only useful for summarization tasks. Default is summary.
    callbacks: List
        Callbacks to use inside transformers.
    metric_optimize: str
        Name of the metric you want to optimize in the hyperparameter search.
    direction_optimize : str
        Direction of the optimization problem. Whether you want to maximize or minimize metric_optimize.
    custom_eval_func: Any
        In case we want a special evaluation function, we can provide it here. It must receive EvalPredictions by trainer, like any compute_metrics function in transformers.
    seed : int
        Seed for optuna sampler.
    max_length_summary: int
        Max length of the summaries, for tokenization purposes. It will be changed depending on the ModelConfig.
    num_proc : int
        Number of processes to preprocess data.
    loaded_dataset: Any
        In case you want to do weird things like concatenating datasets or things like that, you can do that here, by passing a (non-tokenized) dataset in this field.
    additional_metrics: List
        List of additional metrics loaded from datasets, to compute over the test part.
    retrain_at_end: bool
        whether to retrain with the best performing model. In most cases this should be True, except when training 1 model with 1 set of hyperparams.
    config_num_labels: int
        Number of labels to set for the config, if None it will be computed based on number of labels detected.
    smoke_test: bool
        Whether to select only top 10 rows of the dataset for smoke testing purposes.
    augment_data: bool
        Whether to augment_data or not.
    data_augmentation_steps: List
        List of data augmentation techniques to use from NLPAugPipeline.
    pretokenized_dataset: Any
        Pre-tokenized dataset, to avoid tokenizing inside AutoTrainer, which may cause memory issues with huge datasets.

    Examples
    --------
    One can easily create a DatasetConfig for dataset conll2002 just with the following:

    >>> from autotransformers import DatasetConfig

    >>> config={'fixed_training_args': {}, 'dataset_name': 'conll2002', 'alias': 'conll2002', 'task': 'ner', 'hf_load_kwargs': {'path': 'conll2002', 'name': 'es'}, 'label_col':'ner_tags'}

    >>> config = DatasetConfig(**config)
    """

    dataset_name: str = field(metadata={"help": "The name of the dataset"})
    alias: str = field(metadata={"help": "Alias for the dataset, for saving it."})
    task: str = field(
        metadata={
            "help": "The task of the dataset. Currenlty, only classification, ner and qa (question answering) are available."
        }
    )
    fixed_training_args: Dict = field(
        metadata={
            "help": "The training arguments (to use in transformers.TrainingArguments) for every model on this dataset, in dictionary format."
        }
    )
    is_multilabel: bool = field(
        default=False, metadata={"help": "Whether it is multilabel classification"}
    )
    multilabel_label_names: List = field(
        default_factory=list,
        metadata={"help": "Names of the labels for multilabel training."},
    )
    hf_load_kwargs: Dict = field(
        default=None,
        metadata={
            "help": (
                "arguments for loading the dataset from the huggingface datasets' hub. Example: {'path': 'wikiann', 'name': 'es'}."
                "if None, it is assumed that all necessary files exist locally and are passed in the files field."
            )
        },
    )
    type_load: str = field(
        default="json",
        metadata={
            "help": "The type of load to perform in load_dataset; for example, if your data is in csv format (d = load_dataset('csv', ...)), this should be csv."
        },
    )
    files: Dict = field(
        default=None,
        metadata={
            "help": "Files to load the dataset from, in Huggingface's datasets format. Possible keys are train, validation and test"
        },
    )
    data_field: str = field(
        default="data",
        metadata={
            "help": "Field to load data from in the case of jsons loading in datasets. "
        },
    )
    partial_split: bool = field(
        default=False,
        metadata={
            "help": "Wheter a partial split is needed, that is, if you only have train and test sets, this should be True so that a new validation set is created."
        },
    )
    split: bool = field(
        default=False,
        metadata={
            "help": "This should be true when you only have one split, that is, a big train set; this creates new validation and test sets."
        },
    )
    label_col: str = field(
        default="label_list", metadata={"help": "Name of the label column."}
    )
    val_size: float = field(
        default=0.15,
        metadata={
            "help": "In case no validation split is provided, the proportion of the training data to leave for validation."
        },
    )
    test_size: float = field(
        default=0.15,
        metadata={
            "help": "In case no test split is provided, the proportion of the total data to leave for testing."
        },
    )
    pre_func: Any = field(
        default=None,
        metadata={
            "help": "function to perform previous transformations. For example, if your dataset lacks a field (like xquad with title field for example), you can fix it in a function provided here."
        },
    )
    remove_fields_pre_func: bool = field(
        default=False,
        metadata={"help": "Whether to remove fields after pre_func is applied."},
    )
    squad_v2: bool = field(
        default=False,
        metadata={
            "help": "Only useful for question answering. Whether it is squad v2 format or not. Default is false"
        },
    )
    text_field: str = field(
        default="text",
        metadata={
            "help": "The name of the field containing the text. Useful only in case of unique-text-field datasets,like most datasets are. In case of 2-sentences datasets like xnli or paws-x this is not useful."
        },
    )
    is_2sents: bool = field(
        default=False,
        metadata={
            "help": "Whether it is a 2 sentence dataset. Useful for processing datasets like xnli or paws-x."
        },
    )
    sentence1_field: str = field(
        default=None,
        metadata={
            "help": "In case this is a 2 sents dataset, the name of the first sentence field."
        },
    )
    sentence2_field: str = field(
        default=None,
        metadata={
            "help": "In case this is a 2 sents dataset, the name of the second sentence field."
        },
    )
    summary_field: str = field(
        default="summary",
        metadata={
            "help": "The name of the field with summaries (we assume the long texts are in the text_field field). Only useful for summarization tasks."
        },
    )
    callbacks: List = field(
        default_factory=list, metadata={"help": "Callbacks to use inside transformers."}
    )
    metric_optimize: str = field(
        default="eval_loss",
        metadata={
            "help": "Name of the metric you want to optimize in the hyperparameter search."
        },
    )
    direction_optimize: str = field(
        default="minimize",
        metadata={
            "help": "Direction of the optimization problem. Whether you want to maximize or minimize metric_optimize."
        },
    )
    custom_eval_func: Any = field(
        default=None,
        metadata={
            "help": "In case we want a special evaluation function, we can provide it here. It must receive EvalPredictions by trainer, like any compute_metrics function in transformers."
        },
    )
    seed: int = field(default=420, metadata={"help": "Seed for optuna sampler. "})
    max_length_summary: int = field(
        default=120,
        metadata={
            "help": "Max length of the summaries, for tokenization purposes. It will be changed depending on the ModelConfig."
        },
    )
    num_proc: int = field(
        default=4, metadata={"help": "Number of processes to preprocess data."}
    )
    loaded_dataset: Any = field(
        default=None,
        metadata={
            "help": "In case you want to do weird things like concatenating datasets or things like that, you can do that here, by passing a (non-tokenized) dataset in this field."
        },
    )
    additional_metrics: List = field(
        default=None,
        metadata={
            "help": "List of additional metrics loaded from datasets, to compute over the test part."
        },
    )
    retrain_at_end: bool = field(
        default=True,
        metadata={
            "help": "whether to retrain with the best performing model. In most cases this should be True, except when you're only training 1 model with 1 set of hyperparams."
        },
    )
    config_num_labels: int = field(
        default=None,
        metadata={
            "help": "Number of labels to set for the config, if None it will be computed based on number of labels detected."
        },
    )
    smoke_test: bool = field(
        default=False,
        metadata={
            "help": "Whether to select only top 10 rows of the dataset for smoke testing purposes"
        },
    )
    augment_data: bool = field(
        default=False, metadata={"help": "Whether to augment_data or not."}
    )
    data_augmentation_steps: List = field(
        default_factory=list,
        metadata={
            "help": "List of data augmentation techniques to use from NLPAugPipeline."
        },
    )
    id_field_qa: str = field(
        default="id",
        metadata={
            "help": "Name of the field with the unique id of the examples in a question answering dataset."
        },
    )
    pretokenized_dataset: Any = field(
        default=None,
        metadata={
            "help": "Pre-tokenized dataset, to avoid tokenizing inside AutoTrainer, which may cause memory issues with huge datasets."
        },
    )
    model_config_problem_type: str = field(
        default=None,
        metadata={
            "help": "Problem type to set for the model's config. This depends on the dataset task."
        },
    )
    chat_field: str = field(
        default="messages",
        metadata={
            "help": "The fieldname for the column in the dataset containing the messages between the user and the assistant."
        },
    )

    def __str__(
        self,
    ):
        """Representation of dataset config in str."""
        self_as_dict = asdict(self)
        attrs_as_str = [f"{k}={v},\n" for k, v in sorted(self_as_dict.items())]
        return f"{self.__class__.__name__}(\n{''.join(attrs_as_str)})"
