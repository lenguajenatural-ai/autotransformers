from functools import partial
import numpy as np
import collections
import evaluate
from .utils import match_questions_multiple_answers
from tqdm import tqdm
from .dataset_config import DatasetConfig
from .model_config import ModelConfig
from typing import Any


class ResultsGetter:
    """
    Retrieve results on the test set for different tasks (seq2seq, different forms of classification, NER, QA...).

    Parameters
    ----------
    dataset_config: autotransformers.DatasetConfig
        Configuration for the dataset.
    model_config: autotransformers.ModelConfig
        Configuration for the model.
    compute_metrics_func: Any
        Function to compute metrics.
    """

    def __init__(
        self,
        dataset_config: DatasetConfig,
        model_config: ModelConfig,
        compute_metrics_func: Any,
    ):
        self.dataset_config = dataset_config
        self.model_config = model_config
        self.compute_metrics_func = compute_metrics_func

    def __call__(self, trainer, test_dataset):
        """
        Get results for test dataset, using a trained transformers.Trainer.

        Parameters
        ----------
        trainer: transformers.Trainer
            Trainer trained, to get raw predictions on the test dataset.
        test_dataset: datasets.Dataset
            Test dataset for inference. Metrics are computed on this dataset.

        Returns
        -------
        test_results: Dict
            Dictionary with test results.
        """
        if self.dataset_config.task == "qa":
            test_results = self.get_test_results_qa(
                test_dataset,
                trainer,
                self.dataset_config.squad_v2,
            )
        elif self.dataset_config.task == "seq2seq":
            test_results = self.get_test_results_summarization(
                test_dataset,
                trainer,
                self.compute_metrics_func,
                additional_metrics=self.dataset_config.additional_metrics,
            )
        else:
            test_results = self.general_get_test_results(
                test_dataset,
                trainer,
                self.compute_metrics_func,
            )
        return test_results

    def get_test_results_summarization(
        self, test_dataset, trainer, compute_metrics_func, additional_metrics=None
    ):
        """
        Compute and get the results in test for summarization tasks.

        Parameters
        ----------
        test_dataset: datasets.Dataset
            Test dataset.
        trainer: transformers.Trainer
            HF's transformers trainer.
        compute_metrics_func: Any
            Function to compute metrics.
        model_config: autotransformers.ModelConfig
            Configuration for the model.
        additional_metrics: List
            List with additional metrics to compute.

        Returns
        -------
        metrics: Dict
            Dictionary with metrics for the summarization task.
        """
        if self.model_config.generation_params is None:
            preds = trainer.predict(
                test_dataset,
                max_length=self.model_config.max_length_summary,
                num_beams=self.model_config.num_beams,
            )
        else:
            preds = trainer.predict(
                test_dataset,
                max_length=self.model_config.max_length_summary,
                num_beams=self.model_config.num_beams,
                **self.model_config.generation_params,
            )
        metrics = compute_metrics_func(
            preds, tokenizer=trainer.tokenizer, additional_metrics=additional_metrics
        )
        return metrics

    def general_get_test_results(
        self, test_dataset, trainer, compute_metrics_func, additional_metrics=None
    ):
        """
        Compute metrics in general for every NLU task except for QA.

        Parameters
        ----------
        test_dataset: datasets.Dataset
            Dataset  on any task except for QA.
        trainer: transformers.Trainer
            Trainer trained on a dataset that is not a QA dataset.

        Returns
        -------
        metrics: Dict
            Metrics for the test dataset.
        """
        preds = trainer.predict(test_dataset)
        if hasattr(preds, "metrics"):
            return preds.metrics
        metrics = compute_metrics_func(
            preds,
            tokenizer=trainer.tokenizer,
            id2tag=trainer.model.config.id2label,
            additional_metrics=additional_metrics,
        )
        return metrics

    def get_test_results_qa(
        self, test_dataset, trainer, squad_v2=False, additional_metrics=None
    ):
        """
        Compute metrics on test for QA datasets.

        Parameters
        ----------
        test_dataset: datasets.Dataset
            QA dataset.
        trainer: transformers.Trainer
            Trainer trained on QA dataset.
        squad_v2: bool
            Whether the dataset is in squad v2 format or not.

        Returns
        -------
        metrics: Dict
            Metrics for the test dataset.
        """
        validation_features = test_dataset.map(
            partial(
                self.prepare_validation_features_squad,
                tokenizer=trainer.tokenizer,
            ),
            batched=True,
            remove_columns=test_dataset.column_names,
        )
        raw_predictions = trainer.predict(validation_features)
        validation_features.set_format(
            type=validation_features.format["type"],
            columns=list(validation_features.features.keys()),
        )
        final_predictions = self.postprocess_qa_predictions(
            test_dataset,
            validation_features,
            raw_predictions.predictions,
            tokenizer=trainer.tokenizer,
        )
        if isinstance(final_predictions, tuple):
            final_predictions = final_predictions[0]

        metric, formatted_predictions = self._get_metric_and_formatted_predictions(
            final_predictions, squad_v2
        )

        references = [{"id": ex["id"], "answers": ex["answers"]} for ex in test_dataset]

        references = match_questions_multiple_answers(formatted_predictions, references)

        metrics = metric.compute(
            predictions=formatted_predictions, references=references
        )
        return metrics

    def prepare_validation_features_squad(self, examples, tokenizer, pad_on_right=True):
        """
        Process features for validating on squad-like datasets.

        Parameters
        ----------
        examples: datasets.Dataset
            Samples from datasets.Dataset.
        tokenizer: tokenizers.Tokenizer
            Instance of hf's tokenizer.
        pad_on_right: bool
            Whether or not to pad the samples on the right side. True for most models.

        Returns
        -------
        tokenized_examples:
            Tokenized samples.
        """
        id_field = (
            self.dataset_config.id_field_qa if self.dataset_config is not None else "id"
        )
        # Some of the questions have lots of whitespace on the left, which is not useful and will make the
        # truncation of the context fail (the tokenized question will take a lots of space). So we remove that
        # left whitespace
        examples["question"] = [q.lstrip() for q in examples["question"]]

        # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
        # in one example possible giving several features when a context is long, each of those features having a
        # context that overlaps a bit the context of the previous feature.
        tokenized_examples = tokenizer(
            examples["question" if pad_on_right else "context"],
            examples["context" if pad_on_right else "question"],
            truncation="only_second" if pad_on_right else "only_first",
            max_length=512,
            stride=128,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
            is_split_into_words=False,
        )
        # Since one example might give us several features if it has a long context, we need a map from a feature to
        # its corresponding example. This key gives us just that.
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

        # We keep the example_id that gave us this feature and we will store the offset mappings.
        tokenized_examples["example_id"] = []

        for i in range(len(tokenized_examples["input_ids"])):
            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            sequence_ids = tokenized_examples.sequence_ids(i)
            context_index = 1 if pad_on_right else 0

            # One example can give several spans, this is the index of the example containing this span of text.
            sample_index = sample_mapping[i]
            tokenized_examples["example_id"].append(
                examples[id_field][sample_index]
            )  # id

            # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
            # position is part of the context or not.
            tokenized_examples["offset_mapping"][i] = [
                (o if sequence_ids[k] == context_index else None)
                for k, o in enumerate(tokenized_examples["offset_mapping"][i])
            ]

        return tokenized_examples

    def postprocess_qa_predictions(
        self,
        examples,
        features,
        raw_predictions,
        tokenizer,
        n_best_size=20,
        max_answer_length=30,
        squad_v2=False,
        min_score=None,
    ):
        """
        Process raw predictions of a QA model.

        Parameters
        ----------
        examples: datasets.Dataset
            Samples from datasets.Dataset.
        features:
            Validation features as processed by prepare_validation_features_squad.
        raw_predictions:
            Predictions by trainer.
        tokenizer: tokenizers.Tokenizer
            Instance of hf's tokenizer.
        n_best_size: int
            Number of best answers to get (maximum).
        max_answer_length: int
            Maximum answer length in number of characters. Answer longer than this are not even considered.
        squad_v2: bool
            Whether the dataset is in squad v2 format or not.

        Returns
        -------
        predictions: collections.OrderedDict
            An ordered dict with the predictions formatted so that we can compute metrics easily.
        """
        # After raw predictions are taken by a QA model, this function processes them
        # and sorts them in terms of score etc. It also takes the concrete text that
        # was predicted given the predicted start and end tokens.
        id_field = (
            self.dataset_config.id_field_qa if self.dataset_config is not None else "id"
        )
        all_start_logits, all_end_logits = raw_predictions
        # Build a map example to its corresponding features.
        example_id_to_index = {k: i for i, k in enumerate(examples[id_field])}  # id
        features_per_example = collections.defaultdict(list)
        for i, feature in enumerate(features):
            features_per_example[example_id_to_index[feature["example_id"]]].append(i)

        # The dictionaries we have to fill.
        predictions = collections.OrderedDict()
        scores = collections.OrderedDict()

        # Logging.
        print(
            f"Post-processing {len(examples)} example predictions split into {len(features)} features."
        )

        # Let's loop over all the examples!
        for example_index, example in enumerate(tqdm(examples)):
            # Those are the indices of the features associated to the current example.
            # example["id"]]#
            feature_indices = features_per_example[example_index]

            # min_score  # Only used if squad_v2 is True.
            min_null_score = None
            valid_answers = []

            cls_scores = []

            context = example["context"]
            # input_ids = example["input_ids"]
            # Looping through all the features associated to the current example.
            for feature_index in feature_indices:
                # We grab the predictions of the model for this feature.
                start_logits = all_start_logits[feature_index]
                end_logits = all_end_logits[feature_index]
                # This is what will allow us to map some the positions in our logits to span of texts in the original
                # context.
                offset_mapping = features[feature_index]["offset_mapping"]
                # Update minimum null prediction.
                cls_index = features[feature_index]["input_ids"].index(
                    tokenizer.cls_token_id
                )
                feature_null_score = start_logits[cls_index] + end_logits[cls_index]
                cls_scores.append(feature_null_score)
                if min_null_score is None or min_null_score < feature_null_score:
                    min_null_score = feature_null_score

                # Go through all possibilities for the `n_best_size` greater start and end logits.
                start_indexes = np.argsort(start_logits)[
                    -1 : -n_best_size - 1 : -1
                ].tolist()
                end_indexes = np.argsort(end_logits)[
                    -1 : -n_best_size - 1 : -1
                ].tolist()
                for start_index in start_indexes:
                    for end_index in end_indexes:
                        # Don't consider out-of-scope answers, either because the indices are out of bounds or correspond
                        # to part of the input_ids that are not in the context.
                        if (
                            start_index >= len(offset_mapping)
                            or end_index >= len(offset_mapping)
                            or offset_mapping[start_index] is None
                            or offset_mapping[end_index] is None
                        ):
                            continue
                        # Don't consider answers with a length that is either < 0 or > max_answer_length.
                        if (
                            end_index < start_index
                            or end_index - start_index + 1 > max_answer_length
                        ):
                            continue

                        start_char = offset_mapping[start_index][0]
                        end_char = offset_mapping[end_index][1]
                        text = context[start_char:end_char]
                        valid_answers.append(
                            {
                                "score": start_logits[start_index]
                                + end_logits[end_index],
                                "text": text,
                            }
                        )
            if len(valid_answers) > 0:
                best_answer = sorted(
                    valid_answers, key=lambda x: x["score"], reverse=True
                )[0]
            else:
                # In the very rare edge case we have not a single non-null prediction, we create a fake prediction to avoid
                # failure.
                best_answer = {"text": "", "score": 0.0}

            # Let's pick our final answer: the best one or the null answer (only for squad_v2)
            if not squad_v2:
                predictions[example["id"]] = best_answer["text"]
            else:
                if min_score is not None:
                    thres = min([min(cls_scores), min_score])
                else:
                    thres = min(cls_scores)
                answer = best_answer["text"] if best_answer["score"] > thres else ""
                if example["id"] not in predictions:
                    predictions[example["id"]] = answer
                    scores[example["id"]] = best_answer["score"]

        return predictions

    def _get_metric_and_formatted_predictions(self, final_predictions, squad_v2):
        """
        Get the metric from evaluate and the final predictions formatted.

        Parameters
        ----------
        final_predictions: Dict
            Predictions postprocessed.
        squad_v2: bool
            Whether it is squad_v2 mode or not.

        Returns
        -------
        metric: evaluate.Metric
            Metric from the evaluate library.
        formatted_predictions: Dict
            Predictions in the correct format for the metric.
        """
        if not squad_v2:
            metric = evaluate.load("squad")
            formatted_predictions = [
                {"id": k, "prediction_text": v} for k, v in final_predictions.items()
            ]
        else:
            metric = evaluate.load("squad_v2")
            formatted_predictions = [
                {"id": k, "prediction_text": v, "no_answer_probability": 0.0}
                for k, v in final_predictions.items()
            ]
        return metric, formatted_predictions
