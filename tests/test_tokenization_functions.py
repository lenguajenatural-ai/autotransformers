from autotransformers import (
    tokenize_classification,
    tokenize_ner,
    tokenize_summarization,
    tokenize_squad,
)
from datasets import load_dataset, Dataset, DatasetDict
from autotransformers import DatasetConfig
from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    Trainer,
    TrainingArguments,
)
from functools import partial
import pandas as pd
from autotransformers.utils import (
    dict_to_list,
    get_tags,
    match_questions_multiple_answers,
)

# from collections import OrderedDict
from autotransformers.results_getter import ResultsGetter


def _get_feature_names(dataset_split):
    """Get feature names for a dataset split."""
    return [k for k in dataset_split.features.keys()]


def _label_mapper_ner(example, tag2id):
    """Map the labels for NER to use ints."""
    example["label_list"] = [tag2id[label] for label in example["label_list"]]
    return example


def _create_fake_dataset():
    """Create a fake dataset to test dict_to_list."""
    data_dict = {
        "sentence": "Hola me llamo Pedro",
        "entities": [
            {
                "start_character": 14,
                "end_character": 19,
                "ent_label": "PER",
                "ent_text": "Pedro",
            }
        ],
    }
    data = [data_dict] * 3
    df = pd.DataFrame(data)
    dataset = Dataset.from_pandas(df)
    return dataset


tokenizer = AutoTokenizer.from_pretrained("CenIA/albert-tiny-spanish")
tokenizer2 = AutoTokenizer.from_pretrained("dccuchile/bert-base-spanish-wwm-cased")

qa_model = AutoModelForQuestionAnswering.from_pretrained("CenIA/albert-tiny-spanish")
args_qa = TrainingArguments(output_dir=".", no_cuda=True)


def test_tokenize_classification():
    """Test that tokenize_classification produces the relevant fields."""
    wnli = load_dataset("IIC/wnli_tests")
    wnli_config = DatasetConfig(
        dataset_name="wnli",
        alias="wnli",
        task="classification",
        fixed_training_args={},
        text_field="sentence1",
        label_col="label",
    )
    tokenized_wnli = wnli.map(
        partial(
            tokenize_classification, tokenizer=tokenizer, dataset_config=wnli_config
        ),
        batched=True,
        remove_columns=wnli["train"].column_names,
    )
    names_old = _get_feature_names(wnli["train"])
    names_new = _get_feature_names(tokenized_wnli["train"])
    assert names_old != names_new, "Names should be different after tokenizing."


def test_tokenize_classification_2sents():
    """Test that tokenize_classification produces the relevant fields."""
    wnli = load_dataset("IIC/wnli_tests")
    wnli_config = DatasetConfig(
        dataset_name="wnli",
        alias="wnli",
        task="classification",
        fixed_training_args={},
        text_field="sentence1",
        label_col="label",
        is_2sents=True,
        sentence1_field="sentence1",
        sentence2_field="sentence2",
    )
    tokenized_wnli = wnli.map(
        partial(
            tokenize_classification, tokenizer=tokenizer, dataset_config=wnli_config
        ),
        batched=True,
        remove_columns=wnli["train"].column_names,
    )
    names_old = _get_feature_names(wnli["train"])
    names_new = _get_feature_names(tokenized_wnli["train"])
    assert names_old != names_new, "Names should be different after tokenizing."
    assert isinstance(
        tokenized_wnli["train"][0]["input_ids"][0], int
    ), f"Input ids should be ints"


def test_tokenize_ner():
    """Test the tokenization of a ner task."""
    dataset = _create_fake_dataset()
    dataset = DatasetDict({"train": dataset})
    feat_names_prev = _get_feature_names(dataset["train"])
    dataset = dataset.map(dict_to_list, batched=False)
    dataconfig = DatasetConfig(
        dataset_name="prueba",
        alias="prueba",
        task="ner",
        fixed_training_args={},
        num_proc=1,
        text_field="token_list",
        label_col="label_list",
    )
    tags = get_tags(dataset, dataconfig)
    tag2id = {t: i for i, t in enumerate(sorted(tags))}
    dataset = dataset.map(partial(_label_mapper_ner, tag2id=tag2id))
    tokenized_dataset = dataset.map(
        partial(tokenize_ner, tokenizer=tokenizer, dataset_config=dataconfig),
        batched=True,
        remove_columns=dataset["train"].column_names,
    )
    feat_names_post = _get_feature_names(tokenized_dataset["train"])
    assert (
        feat_names_post != feat_names_prev
    ), f"Posterior names: \n {feat_names_post} \n should be different from pre: \n {feat_names_prev}"
    assert isinstance(
        tokenized_dataset["train"][0]["input_ids"][0], int
    ), f"Input ids should be ints"


def test_tokenize_qa():
    """Test the tokenization functions of QA tasks."""
    dataset = load_dataset("IIC/sqac_tests")
    feat_names_prev = _get_feature_names(dataset["train"])
    dataset_tokenized = dataset.map(
        partial(
            tokenize_squad,
            tokenizer=tokenizer,
        ),
        batched=True,
    )
    tokenized_alternative = tokenize_squad(
        dataset["train"][:], tokenizer
    )  # to test the separate call.
    feat_names_post = _get_feature_names(dataset_tokenized["train"])
    assert "input_ids" in tokenized_alternative, "Input ids has not been created."
    assert (
        feat_names_post != feat_names_prev
    ), f"Posterior names: \n {feat_names_post} \n should be different from pre: \n {feat_names_prev}"
    dataset_tokenized2 = dataset.map(
        partial(
            tokenize_squad,
            tokenizer=tokenizer2,
        ),
        batched=True,
    )  # de esta forma probamos que funcione bien con bpe y no bpe tokenizers.
    feat_names_post2 = _get_feature_names(dataset_tokenized2["train"])
    assert (
        feat_names_post2 != feat_names_prev
    ), f"Posterior names: \n {feat_names_post2} \n should be different from pre: \n {feat_names_prev}"
    results_getter = ResultsGetter(None, None, None)
    validation_features = results_getter.prepare_validation_features_squad(
        dataset["test"][:], tokenizer
    )
    assert (
        "input_ids" in validation_features
    ), "Input_ids is not in validation features."
    validation_features = dataset["test"].map(
        partial(
            results_getter.prepare_validation_features_squad,
            tokenizer=tokenizer,
        ),
        batched=True,
        remove_columns=dataset["test"].column_names,
    )

    names_valid_feats = _get_feature_names(validation_features)
    names_test_split = _get_feature_names(dataset["test"])
    assert (
        names_valid_feats != names_test_split
    ), "Validation features have not been processed."
    trainer_qa = Trainer(
        args=args_qa,
        model=qa_model,
        train_dataset=dataset_tokenized["train"],
        eval_dataset=dataset_tokenized["test"],
        data_collator=None,
        tokenizer=tokenizer,
    )
    raw_predictions = trainer_qa.predict(validation_features)
    validation_features.set_format(
        type=validation_features.format["type"],
        columns=list(validation_features.features.keys()),
    )
    final_predictions = results_getter.postprocess_qa_predictions(
        dataset["test"],
        validation_features,
        raw_predictions.predictions,
        tokenizer=tokenizer,
    )
    final_predictions_squadv2 = results_getter.postprocess_qa_predictions(
        dataset["test"],
        validation_features,
        raw_predictions.predictions,
        tokenizer=tokenizer,
        squad_v2=True,
    )
    assert final_predictions != raw_predictions, "Predictions were not changed."
    assert final_predictions_squadv2 != raw_predictions, "Predictions were not changed."

    (
        metric,
        formatted_predictions,
    ) = results_getter._get_metric_and_formatted_predictions(final_predictions, False)
    references = [{"id": ex["id"], "answers": ex["answers"]} for ex in dataset["test"]]
    references = match_questions_multiple_answers(formatted_predictions, references)
    # references[-1]["answers"]["text"].append("Trucamiento de respuesta.")
    lastref = references[-1]
    lastref["answers"]["text"][0] = "Movida rara."
    references = references + [lastref]
    references_augmented_correct = match_questions_multiple_answers(
        formatted_predictions, references
    )
    assert (
        references_augmented_correct != references
    ), "When more that one id is presented, this should change"
    assert formatted_predictions != final_predictions, "Predictions were not formatted."
    metrics = metric.compute(
        predictions=formatted_predictions, references=references_augmented_correct
    )
    assert isinstance(metrics, dict), "Metrics should be a dict."
    assert any(
        ["f1" in metricname for metricname in metrics]
    ), "There is no metric related to f1 which should be."


def test_tokenize_summarization():
    """Test the tokenization of a summarization task."""
    dataset = load_dataset("IIC/sqac_tests")
    dataconfig = DatasetConfig(
        "name",
        "alias",
        "summarization",
        {},
        text_field="question",
        summary_field="title",
    )
    feat_names_prev = _get_feature_names(dataset["train"])
    tokenized_dataset = dataset.map(
        partial(tokenize_summarization, tokenizer=tokenizer, dataset_config=dataconfig),
        batched=True,
        remove_columns=dataset["train"].column_names,
    )
    tokenized_dataset_pure = tokenize_summarization(
        dataset["train"][:5], tokenizer, dataconfig
    )
    feat_names_post = _get_feature_names(tokenized_dataset["train"])
    assert "input_ids" in tokenized_dataset_pure, "Input ids were not created."
    assert (
        feat_names_post != feat_names_prev
    ), f"Posterior names: \n {feat_names_post} \n should be different from pre: \n {feat_names_prev}"
    assert "labels" in feat_names_post, "Labels should be in the new dataset fields."
