import json
import os
from more_itertools import windowed
from functools import partial
import itertools
from collections.abc import Iterable
import nltk
import polyfuzz as pf
from polyfuzz.models import TFIDF

nltk.download("punkt")


def _load_json(filename):
    """Load a json file."""
    with open(filename, "r") as f:
        return json.load(f)


def _save_json(data, filename):
    """Save data in a json file."""
    with open(filename, "w") as f:
        json.dump(data, f)


def joinpaths(*paths):
    """Join all paths passed as args."""
    return os.path.join(*paths)


def filter_empty(string_list):
    """
    Remove empty characters and spaces from list.

    Parameters
    ----------
    string: str
        String to filter.

    Returns
    -------
    result: bool
        Whether string is not in the empty characters list.
    """
    return string_list not in ["", " "]


def dict_to_list(
    example,
    nulltoken="O",
    entities_field="entities",
    sentence_field="sentence",
):
    """
    Transform a dictionary of entities in the default format.

    With start and end characters for each entity, into lists of words and
    labels, having one label per word. This is useful for NER tasks
    when we usually have this format (ent_label, start_char, end_char)
    and we need to have 2 equally-sized lists of words and labels for
    passing them to the tokenizer.

    Parameters
    ----------
    example:
        Sample of huggingface Dataset, with an entities field containing
        the entities in the format mentioned above.
    nulltoken: Union[str, int]
        Default token for the "no-entities". Usually O is used for this,
        which is the default value.
    entities_field: str
        Name of the field which contains entities in (ent_label, start_char, end_char)
        format. Usually "entities" is used for this, which is the default value.
    sentence_field: str
        Name of the field which contains the sentence. Usually "sentence" is used
        for this, which is the default value.

    Returns
    -------
    example:
        Sample of huggingface dataset with 2 new fields: token list and
        label list.
    """
    if len(example[entities_field]) == 0:
        token_list = example[sentence_field].split(" ")
        label_list = [nulltoken] * len(token_list)
        example["token_list"] = token_list
        example["label_list"] = label_list
        return example
    token_list = []
    label_list = []
    if example[entities_field][0]["start_character"] > 0:
        text_prev = example[sentence_field][
            : example[entities_field][0]["start_character"]
        ].split(" ")
        text_prev = list(filter(filter_empty, text_prev))
        token_list.extend(text_prev)
        label_list.extend([nulltoken] * len(text_prev))
    last_pos = example[entities_field][0]["start_character"]
    for entity in example[entities_field]:
        start_char, end_char = (
            entity["start_character"],
            entity["end_character"],
        )
        if start_char > last_pos:
            text_between = example[sentence_field][last_pos:start_char].split(" ")
            text_between = list(filter(filter_empty, text_between))
            token_list.extend(text_between)
            label_list.extend([nulltoken] * len(text_between))
        text_entity = example[sentence_field][start_char:end_char]
        text_entity_sp = text_entity.split(" ")
        token_list.extend(text_entity_sp)
        label_list.extend([entity["ent_label"]] * len(text_entity_sp))
        last_pos = end_char
    if entity["end_character"] < len(example[sentence_field]):
        text_last = example[sentence_field][entity["end_character"] :].split(" ")
        text_last = list(filter(filter_empty, text_last))
        token_list.extend(text_last)
        label_list.extend([nulltoken] * len(text_last))
    assert len(token_list) == len(
        label_list
    ), "Token list and label list should have the same length."
    example["token_list"] = token_list
    example["label_list"] = label_list
    return example


def get_windowed_match_context_answer(context, answer, maxrange=100):
    """
    Find the best possible match for an answer in the context.

    Useful for translated QA datasets, where we don't have exact translations
    of the answers and they do not exist in the context anymore. This could also
    happen because of encodings, or other reasons, which cause that the answer
    does not start at the string index that appears in the dataset.

    Parameters
    ----------
    context: str
        Context where we want to find the answer.
    answer: str
        Answer that we want to find in the context.
    maxrange: int
        Maximum size of the windows for matching, in number of words.

    Returns
    -------
    beg: int
        Beginning character index of the answer.
    end: int
        Ending character index for tha answer.
    new_answer: str
        Answer found in the context.
    """
    context_list = context.split(" ")
    answer_list = answer.split(" ")
    total_list = []
    for n in [i for i in range(1, maxrange, 1)]:
        total_list.extend(windowed(context_list, n=n))
    total_list = list(filter(lambda x: len(x) > 0, total_list))
    total_list = list(
        map(
            lambda window: [word for word in window if isinstance(word, str)],
            total_list,
        )
    )
    total_list = [" ".join(window) for window in total_list]
    tfidf = TFIDF(
        n_gram_range=(1, 3),
        min_similarity=0,
        top_n=1,
        clean_string=False,
    )
    model = pf.PolyFuzz(tfidf).match(from_list=[answer], to_list=total_list)
    matches = model.get_matches()
    new_answer = matches.loc[0, "To"]
    beg = context.find(new_answer)
    end = beg + len(new_answer)
    return beg, end, new_answer


def _fix_json(metrics):
    """
    Fix a json that has incorrect data types.

    Parameters
    ----------
    metrics: List
        List with metrics.

    Returns
    -------
    metrics: List
        List with metrics in a correct data type.
    """
    for i in range(len(metrics)):
        for key in metrics[i]:
            if isinstance(metrics[i][key], int):
                metrics[i][key] = float(metrics[i][key])
            elif isinstance(metrics[i][key], dict):
                for subd in metrics[i][key]:
                    if isinstance(metrics[i][key][subd], int):
                        metrics[i][key][subd] = float(metrics[i][key][subd])
                    elif isinstance(metrics[i][key][subd], dict):
                        for susubd in metrics[i][key][subd]:
                            if isinstance(metrics[i][key][subd][susubd], int):
                                metrics[i][key][subd][susubd] = float(
                                    metrics[i][key][subd][susubd]
                                )
    return metrics


def _parse_modelname(modelname):
    """Fix a modelname if it has "/" instead of "-" ."""
    if "/" in modelname:
        modelname = modelname.replace("/", "-")
    return modelname


def _save_metrics(metrics, model_name, dataset_name, metrics_dir):
    """
    Save metrics in the metrics directory.

    Parameters
    ----------
    metrics: Dict
        Dictionary with metrics.
    model_name: str
        Name of the model.
    dataset_name: str
        Name of the dataset.
    metrics_dir: str
        Name of the metrics dir to store the metrics file.
    """
    model_name = _parse_modelname(model_name)
    name = joinpaths(metrics_dir, f"{model_name}#{dataset_name}.json")
    try:
        with open(name, "w") as f:
            json.dump(metrics, f)
    except Exception as e:
        print(f"Saving metrics failed with error: \n {e} \n The json will be fixed...")
        try:
            metrics = _fix_json(metrics)
            with open(name, "w") as f:
                json.dump(metrics, f)
        except Exception as e:
            print(
                f"Fixing json did not work: \n {e} \n So metrics will be saved in .txt"
            )
            with open(name.replace(".json", ".txt"), "w") as f:
                f.write(str(metrics))


def _unwrap_reference(reference):
    """Unwraps a reference into multiple ones if the question has more than one answer."""
    new_references = []
    if isinstance(reference, list):
        answers = [ref["answers"] for ref in reference]
        for answer in answers:
            new_references.append({"id": reference[0]["id"], "answers": [answer]})
        return new_references
    elif isinstance(reference, dict):
        return [reference]


def match_questions_multiple_answers(formatted_predictions, references):
    """
    Check if any of the given answers for a question coincides with our answer.

    Parameters
    ----------
    formatted_predictions: List
        List with the predictions.
    references: List
        All references with real answers for the questions. Possibly more than one
        answer per question, which we need to unify previously with the same id.

    Returns
    -------
    final_references: List
        Final references for the questions, so that if we get right questions with
        more than one possible answers, it counts as a right guess.
    """
    all_ids = list(sorted(set([ref["id"] for ref in references])))
    final_references = []
    for id_ in all_ids:
        all_refs_id = [ref for ref in references if ref["id"] == id_]
        total_refs_this_id = []
        for reference in all_refs_id:
            unwrapped = _unwrap_reference(reference)
            total_refs_this_id.extend(unwrapped)
        if len(total_refs_this_id) == 1:
            final_references.append(total_refs_this_id[0])
        else:
            my_answer = [
                pred["prediction_text"]
                for pred in formatted_predictions
                if pred["id"] == id_
            ][0]
            refs_answers = [ref["answers"]["text"][0] for ref in total_refs_this_id]
            if my_answer in refs_answers:
                final_stay = [
                    ref
                    for ref in total_refs_this_id
                    if ref["answers"]["text"][0] == my_answer
                ][0]
                final_references.append(final_stay)
            else:
                final_stay = total_refs_this_id[0]
                final_references.append(final_stay)
    return final_references


def get_tags(dataset, dataset_config):
    """
    Get the list of unique tags for a dataset.

    Parameters
    ----------
    dataset: datasets.DatasetDict
        Dataset to tokenize.
    dataset_config: autotransformers.DatasetConfig
        Dataset configuration.

    Returns
    -------
    tags: List
        List of unique labels for the dataset.
    """
    total_tags = []
    for split in dataset:
        if isinstance(
            dataset[split][0][dataset_config.label_col], Iterable
        ) and not isinstance(dataset[split][0][dataset_config.label_col], str):
            tags = list(
                set(
                    list(
                        itertools.chain.from_iterable(
                            dataset[split][:][dataset_config.label_col]
                        )
                    )
                )
            )
        else:
            tags = list(set(dataset[split][:][dataset_config.label_col]))
        total_tags.extend(tags)
    total_tags = list(set(total_tags))
    return total_tags


def _tokenize_dataset(tokenizer, tok_func_map, dataset, dataset_config, model_config):
    """
    Tokenize dataset, depending on the configuration of dataset and model config.

    Parameters
    ----------
    dataset: datasets.DatasetDict
        Dataset to tokenize.
    dataset_config: autotransformers.DatasetConfig
        Dataset configuration.
    model_config: autotransformers.ModelConfig
        Model configuration.

    Returns
    -------
    dataset: datasets.DatasetDict
        Tokenized dataset.
    """
    dataset = dataset.map(
        (
            partial(
                model_config.custom_tokenization_func,
                tokenizer=tokenizer,
                dataset_config=dataset_config,
            )
            if model_config.custom_tokenization_func
            else partial(
                tok_func_map[dataset_config.task],
                tokenizer=tokenizer,
                dataset_config=dataset_config,
            )
        ),
        batched=True,
        remove_columns=dataset["train"].column_names,
        num_proc=dataset_config.num_proc,
    )
    return dataset


def chunks(lst, n):
    """
    Split a list into n-sized chunks.

    Parameters
    ----------
    lst: List
        List containing any type of elements.
    n: int
        Size of the chunks

    Returns
    -------
    Chunks:
        Generates n-sized chunks.
    """
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def fix_eval_results_dict(metrics: dict) -> dict:
    """
    Remove test or eval set prefix from names in the metrics dict.

    Parameters
    ----------
    metrics: Dict
        Dictionary with metrics.

    Returns
    -------
    new_metrics: Dict
        Fixed metrics dictionary.
    """
    prefixes = ["validation", "eval", "test"]
    new_metrics = {}
    for k, v in metrics.items():
        for prefix in prefixes:
            if k.startswith(prefix):
                new_metrics[k.replace(f"{prefix}_", "")] = v
    return new_metrics
