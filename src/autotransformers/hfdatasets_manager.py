from datasets import load_dataset, Value
import datasets
from .utils import (
    get_tags,
    _tokenize_dataset,
)
from .tokenization_functions import (
    tokenize_ner,
    tokenize_squad,
    tokenize_summarization,
    tokenize_classification,
    tokenize_chatbot,
)
from .augmentation import NLPAugPipeline
from transformers import PreTrainedTokenizer


tok_func_map = {
    "ner": tokenize_ner,
    "qa": tokenize_squad,
    "seq2seq": tokenize_summarization,
    "classification": tokenize_classification,
    "chatbot": tokenize_chatbot,
    "alignment": None,
}


class HFDatasetsManager:
    """
    Utility for loading HF Datasets' objects, using a DatasetConfig and a ModelConfig.

    Parameters
    ----------
    dataset_config: autotransformers.DatasetConfig
        Configuration for the dataset
    model_config: autotransformers.ModelConfig
        Configuration for the model.
    """

    def __init__(self, dataset_config, model_config):
        self.dataset_config = dataset_config
        self.model_config = model_config

    def get_dataset_and_tag2id(self, tokenizer: PreTrainedTokenizer):
        """
        Get dataset and tag2id depending on dataset and model config.

        Using dataset config (task, etc), a preprocessing is applied to
        the dataset, tokenizing text data, returning a processed dataset
        ready for the configured task.

        Parameters
        ----------
        tokenizer: transformers.PretrainedTokenizer
            Tokenizer to process data.

        Returns
        -------
        dataset: datasets.DatasetDict
            Tokenized dataset.
        tag2id: Dict
            Dictionary with tags (labels) and their indexes.
        """
        if self.dataset_config.pretokenized_dataset is None:
            dataset, tag2id = self._generic_load_dataset(tokenizer)
        else:
            dataset = self.dataset_config.pretokenized_dataset
            tag2id = {}
        if "test" not in dataset:
            dataset["test"] = dataset["validation"]
        return dataset, tag2id

    def _generic_load_dataset(self, tokenizer: PreTrainedTokenizer):
        """
        Load a generic dataset.

        Load the dataset and process it depending on the dataset configuration,
        and get the tag2id (map of labels to ids of the labels).

        Parameters
        ----------
        tokenizer: transformers.PretrainedTokenizer
            Tokenizer to process data.

        Returns
        -------
        dataset: Union[datasets.Dataset,datasets.DatasetDict]
            Dataset containing data for training, evaluation and testing.
        tag2id: Dict
            Dictionary mapping the label names to their numerical ids.
        """
        dataset = self._basic_dataset_loading()
        dataset = self._smoke_test_filter(dataset)
        if self.dataset_config.pre_func is not None:
            dataset = dataset.map(
                self.dataset_config.pre_func,
                remove_columns=(
                    dataset["train"].column_names
                    if self.dataset_config.remove_fields_pre_func
                    else None
                ),
            )
        dataset = self._resplit_dataset(dataset)
        if self.dataset_config.task == "qa":
            test_dataset = dataset["test"]
        tag2id = {}
        tasks_excluded_tag2id = ["chatbot", "seq2seq"]
        if not (
            self.dataset_config.config_num_labels == 1
            or self.dataset_config.model_config_problem_type == "regression"
            or self.dataset_config.task in tasks_excluded_tag2id
        ):
            tags = get_tags(dataset, self.dataset_config)
            tag2id = {t: i for i, t in enumerate(sorted(tags))}
            dataset = self._general_label_mapper(tag2id, dataset)
        dataset = self._augment_dataset(dataset)
        if self.dataset_config.task in tok_func_map and tok_func_map[self.dataset_config.task]:
            dataset = _tokenize_dataset(
                tokenizer, tok_func_map, dataset, self.dataset_config, self.model_config
            )
            if self.dataset_config.task == "qa":
                dataset["test"] = test_dataset
            dataset = self._parse_types_dataset(dataset)
        return dataset, tag2id

    def _parse_types_dataset(self, dataset):
        """
        Parse the types of the dataset if needed from int to float for regression.

        Parameters
        ----------
        dataset: datasets.DatasetDict
            Dataset to process.

        Returns
        -------
        dataset: datasets.DatasetDict
            Dataset with correct types.
        """
        features_names = [k for k in dataset["train"].features.keys()]
        labcol = (
            self.dataset_config.label_col
            if self.dataset_config.label_col in features_names
            else "labels"
        )
        if self.dataset_config.config_num_labels == 1 and not isinstance(
            dataset["train"][0][labcol], float
        ):
            features = dataset["train"].features.copy()
            features[labcol] = Value("float")
            dataset = dataset.cast(features)
        return dataset

    def _smoke_test_filter(self, dataset):
        """
        Filter dataset if smoke test.

        Parameters
        ----------
        dataset: datasets.DatasetDict
            Dataset to filter.

        Returns
        -------
        dataset: datasets.DatasetDict
            Dataset filtered if necessary.
        """
        if self.dataset_config.smoke_test:
            for split in dataset:
                dataset[split] = dataset[split].select([i for i in range(10)])
        return dataset

    def _basic_dataset_loading(self):
        """
        Load the raw dataset based on dataset config.

        Returns
        -------
        dataset: datasets.DatasetDict
            Raw dataset.
        """
        if not self.dataset_config.loaded_dataset:
            if self.dataset_config.hf_load_kwargs is not None:
                dataset = load_dataset(**self.dataset_config.hf_load_kwargs)
            else:
                if self.dataset_config.type_load == "json":
                    dataset = load_dataset(
                        self.dataset_config.type_load,
                        data_files=self.dataset_config.files,
                        field=self.dataset_config.data_field or None,
                    )
                elif self.dataset_config.type_load == "csv":
                    dataset = load_dataset(
                        self.dataset_config.type_load,
                        data_files=self.dataset_config.files,
                    )
        else:
            dataset = self.dataset_config.loaded_dataset
        return dataset

    def _augment_dataset(self, dataset):
        """
        Augment dataset based on dataset config.

        Parameters
        ----------
        dataset: datasets.DatasetDict
            Dataset to tokenize.

        Returns
        -------
        dataset: datasets.DatasetDict
            Augmented dataset.
        """
        if self.dataset_config.augment_data:
            aug_pipeline = NLPAugPipeline(
                steps=self.dataset_config.data_augmentation_steps,
                text_field=self.dataset_config.text_field,
            )
            dataset["train"] = dataset["train"].map(
                aug_pipeline.augment, batched=True, batch_size=64
            )
        return dataset

    def _resplit_dataset(self, dataset):
        """
        Re-split dataset based on dataset config.

        Parameters
        ----------
        dataset: datasets.DatasetDict
            Dataset to tokenize.

        Returns
        -------
        dataset: datasets.DatasetDict
            Re-splitted dataset.
        """
        if self.dataset_config.partial_split and not self.dataset_config.split:
            dataset = self._partial_split(dataset)
        elif self.dataset_config.split and not self.dataset_config.partial_split:
            dataset = self._complete_split(dataset)
        return dataset

    def _partial_split(self, dataset):
        """
        Split the train part of the dataset to create a validation split which did not exist.

        Parameters
        ----------
        dataset: Union[datasets.Dataset, datasets.DatasetDict]
            Dataset containing data for training and testing.

        Returns
        -------
        dataset: datasets.Dataset or datasets.DatasetDict
            Dataset containing data for training, evaluation and testing.
        """
        dataset_train_val = dataset["train"].train_test_split(
            test_size=self.dataset_config.val_size, seed=self.dataset_config.seed
        )
        dataset = datasets.DatasetDict(
            {
                "train": dataset_train_val["train"],
                "validation": dataset_train_val["test"],
                "test": dataset["test" if "test" in dataset else "validation"],
            }
        )
        return dataset

    def _complete_split(self, dataset):
        """
        Split the train part of the dataset to create a validation split and test split which did not exist.

        Parameters
        ----------
        dataset: Union[datasets.Dataset, datasets.DatasetDict]
            Dataset containing data for training.

        Returns
        -------
        dataset: datasets.Dataset or datasets.DatasetDict
            Dataset containing data for training, evaluation and testing.
        """
        dataset_train_test = dataset["train"].train_test_split(
            test_size=self.dataset_config.test_size, seed=self.dataset_config.seed
        )
        dataset_train_val = dataset_train_test["train"].train_test_split(
            test_size=self.dataset_config.val_size, seed=self.dataset_config.seed
        )
        dataset = datasets.DatasetDict(
            {
                "train": dataset_train_val["train"],
                "validation": dataset_train_val["test"],
                "test": dataset_train_test["test"],
            }
        )
        return dataset

    def _general_label_mapper(self, tag2id, dataset):
        """
        Transcript the labels from label names to label ids, for classification and ner.

        Parameters
        ----------
        tag2id: Dict
            Dictionary with the map of tag to id of those tags.
        dataset: datasets.Dataset or datasets.DatasetDict
            Dataset containing data for training, evaluation and testing.

        Returns
        -------
        dataset: datasets.Dataset or datasets.DatasetDict
            Processed dataset, with labels mapped to their ids.
        """

        def label_mapper_ner(example):
            example[self.dataset_config.label_col] = [
                tag2id[label] for label in example[self.dataset_config.label_col]
            ]
            return example

        def label_mapper_class(example):
            example[self.dataset_config.label_col] = tag2id[
                example[self.dataset_config.label_col]
            ]
            return example

        if self.dataset_config.task == "ner":
            dataset = dataset.map(label_mapper_ner)
        elif (
            self.dataset_config.task == "classification"
            and not self.dataset_config.is_multilabel
            and not (
                self.dataset_config.config_num_labels == 1
                or self.dataset_config.model_config_problem_type == "regression"
            )
        ):
            dataset = dataset.map(label_mapper_class)
        return dataset
