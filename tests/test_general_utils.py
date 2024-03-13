from autotransformers.utils import (
    dict_to_list,
    _tokenize_dataset,
    get_tags,
    _load_json,
    _save_json,
    _parse_modelname,
    _fix_json,
    joinpaths,
    filter_empty,
    get_windowed_match_context_answer,
    _save_metrics,
    _unwrap_reference,
)
from autotransformers.tokenization_functions import (
    tokenize_ner,
    tokenize_classification,
    tokenize_squad,
    tokenize_summarization,
)
from autotransformers import ModelConfig, DatasetConfig
from datasets import Dataset, DatasetDict, load_dataset
import pandas as pd
from transformers import AutoTokenizer
from functools import partial
import os

tok_func_map = {
    "ner": tokenize_ner,
    "qa": tokenize_squad,
    "summarization": tokenize_summarization,
    "classification": tokenize_classification,
}


def _get_feature_names(dataset_split):
    """Get feature names for a dataset split."""
    return [k for k in dataset_split.features.keys()]


def _label_mapper_ner(example, dataset_config, tag2id):
    """Map the labels for NER to use ints."""
    example[dataset_config.label_col] = [
        tag2id[label] for label in example[dataset_config.label_col]
    ]
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
    data_dict2 = {
        "sentence": "Hola me llamo Pedro y ya está.",
        "entities": [
            {
                "start_character": 14,
                "end_character": 19,
                "ent_label": "PER",
                "ent_text": "Pedro",
            }
        ],
    }
    data_dict3 = {
        "sentence": "Hola me llamo Pedro y él Manuel.",
        "entities": [
            {
                "start_character": 14,
                "end_character": 19,
                "ent_label": "PER",
                "ent_text": "Pedro",
            },
            {
                "start_character": 25,
                "end_character": 31,
                "ent_label": "PER",
                "ent_text": "Pedro",
            },
        ],
    }
    data_dict_empty = {
        "sentence": "Hola me llamo Pedro",
        "entities": [],
    }
    data = [data_dict] * 3
    data.append(data_dict_empty)  # for testing empty data also.
    data.append(data_dict2)
    data.append(data_dict3)
    df = pd.DataFrame(data)
    dataset = Dataset.from_pandas(df)
    return dataset


def test_save_load_json():
    """Test load and save functions for jsons."""
    d = {"a": "a"}
    _save_json(d, "prueba.json")
    assert os.path.exists("prueba.json"), "No funciona el guardado de json"
    b = _load_json("prueba.json")
    assert b == d, "The saved and loaded objects are not equal."


def test_joinpaths():
    """Test whether joinpaths work correctly."""
    p1 = "a"
    p2 = "b"
    ptotal = "a/b"
    result = joinpaths(p1, p2)
    assert result == ptotal, f"The obtained path: {result} doesn't coincide: {ptotal}"


def test_filter_empty():
    """Test that filter_empty filters empty chars."""
    lista = ["a", "", " ", "b"]
    result = list(filter(filter_empty, lista))
    assert len(result) < len(lista), "The length of the new list should be shorter."
    assert all([c not in result for c in ["", " "]]), "There are empty characters."


def test_dict_to_list():
    """Test dict_to_list function to parse NER tasks data."""
    dataset = _create_fake_dataset()
    dataset = dataset.map(dict_to_list, batched=False)
    assert "token_list" in dataset[0], "token list should be in dataset."
    assert "label_list" in dataset[0], "label list should be in dataset."
    assert dataset[0]["token_list"][-1] == "Pedro", "Pedro should be the last token."
    assert dataset[0]["label_list"][0] == "O", "First label should be O."
    assert dataset[0]["label_list"][-1] == "PER", "Last label should be PER."


def test_get_tags():
    """Test get tags function."""
    dataset = _create_fake_dataset()
    dataset = dataset.map(dict_to_list, batched=False)
    dataset = DatasetDict({"train": dataset})
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
    entities_should_be = ["O", "PER"]
    assert (
        len(tags) == 2
    ), f"Only 2 different labels were presented, but length of tags is {len(tags)}"
    assert all(
        [ent in tags for ent in entities_should_be]
    ), "Not all entities were captured by get tags."


def test_tokenize_dataset():
    """Test that _tokenize_dataset effectively tokenizes the dataset."""
    tokenizer = AutoTokenizer.from_pretrained("CenIA/albert-tiny-spanish")
    modelconfig = ModelConfig(
        save_name="prueba_tokenize_dataset",
        name="prueba_tokenize_dataset",
        hp_space=lambda trial: trial,
    )
    dataconfig = DatasetConfig(
        dataset_name="prueba",
        alias="prueba",
        task="ner",
        fixed_training_args={},
        num_proc=1,
        text_field="token_list",
        label_col="label_list",
    )
    dataset = _create_fake_dataset()
    dataset = DatasetDict({"train": dataset})
    feat_names_prev = _get_feature_names(dataset["train"])
    dataset = dataset.map(dict_to_list, batched=False)
    tags = get_tags(dataset, dataconfig)
    tag2id = {t: i for i, t in enumerate(sorted(tags))}
    dataset = dataset.map(
        partial(_label_mapper_ner, dataset_config=dataconfig, tag2id=tag2id)
    )
    tokenized_dataset = _tokenize_dataset(
        tokenizer, tok_func_map, dataset, dataconfig, modelconfig
    )
    feat_names_post = _get_feature_names(tokenized_dataset["train"])
    assert (
        feat_names_post != feat_names_prev
    ), f"Posterior names: \n {feat_names_post} \n should be different from pre: \n {feat_names_prev}"
    assert isinstance(
        tokenized_dataset["train"][0]["input_ids"][0], int
    ), f"Input ids should be ints"
    partial_custom_tok_func_call = partial(
        tokenize_ner, tokenizer=tokenizer, dataset_config=dataconfig
    )
    setattr(modelconfig, "partial_custom_tok_func_call", partial_custom_tok_func_call)
    tokenized_alternative = _tokenize_dataset(
        tokenizer, tok_func_map, dataset, dataconfig, modelconfig
    )
    feat_names_post2 = _get_feature_names(tokenized_alternative["train"])
    assert (
        feat_names_post2 != feat_names_prev
    ), f"Posterior names: \n {feat_names_post} \n should be different from pre: \n {feat_names_prev}"
    assert isinstance(
        tokenized_alternative["train"][0]["input_ids"][0], int
    ), "Input ids should be ints."


def test_get_windowed_match_context_answer():
    """Test that the matching of context-answer works."""
    context = "La respuesta a cuál es el rey de España es Juan Carlos Mencía según dicen algunos expertos en la materia que se hacen llamar mencistas."
    answer = "Juan Carlos I."
    beg, end, new_answer = get_windowed_match_context_answer(
        context, answer, maxrange=4
    )
    assert isinstance(beg, int), "Beginning index should be int."
    assert isinstance(end, int), "Ending index should be int."
    assert isinstance(new_answer, str), "The new answer should be a str"
    assert "Juan Carlos" in new_answer, "Juan Carlos should be in new answer."


def test_fix_json():
    """Test if jsons are fixed."""
    metrics = [{"metric": 1}]
    metrics_fixed = _fix_json(metrics)
    assert isinstance(
        metrics_fixed[0]["metric"], float
    ), "Ints were not converted to float."
    metrics2 = [{"metric": {"metric": 1}}]
    metrics_fixed = _fix_json(metrics2)
    assert isinstance(
        metrics_fixed[0]["metric"]["metric"], float
    ), "Ints were not converted to float."


def test_parse_modelname():
    """Test if model names are correctly parsed."""
    modname = "hola/cocacola"
    parsed = _parse_modelname(modname)
    assert "/" not in parsed, "/ should not be in parsed name."


def test_save_metrics():
    """Test that metrics are saved."""
    metrics = {"rouge2": 0.12, "rouge1": 0.30}
    metricsdir = "pruebametrics"
    os.makedirs(metricsdir, exist_ok=True)
    _save_metrics(metrics, "modelometrics", "datasetmetrics", metricsdir)
    assert os.path.exists(
        joinpaths(metricsdir, "modelometrics#datasetmetrics.json")
    ), "El fichero de metrics no ha sido guardado."
    try:
        _save_metrics(metrics, "modelometrics", "datasetmetrics", "metricsfalso")
    except Exception as e:
        print("Ha fallado save metrics donde tiene que fallar.")


def test_unwrap_reference():
    """Test the unwrapping of a QA reference."""
    reference_simple = {"id": "A", "answers": "A"}
    unwrapped_simple = _unwrap_reference(reference_simple)
    assert isinstance(
        unwrapped_simple, list
    ), f"should return list when dict is passed but is: {type(reference_simple)}"
    assert (
        unwrapped_simple[0] == reference_simple
    ), "This should just be a list around the dict."
    reference_multiple = [
        {"id": "A", "answers": {"text": "a", "start": 0}},
        {"id": "A", "answers": {"text": "b", "start": 2}},
    ]
    unwrapped_complex = _unwrap_reference(reference_multiple)
    assert (
        len(unwrapped_complex) == 2
    ), f"The length of unwrapped complex should be 2 and is {len(unwrapped_complex)}"
