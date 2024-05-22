from sklearn.metrics import classification_report
import numpy as np
import nltk
import itertools
from typing import List
import torch
import evaluate

nltk.download("punkt")

metric_sum = evaluate.load("rouge")
metric_seqeval = evaluate.load("seqeval")


def compute_metrics_classification(
    pred, tokenizer=None, id2tag=None, additional_metrics=None
):
    """
    Compute metrics for classification (multi-class or binary) tasks.

    Parameters
    ----------
    pred: transformers.EvalPrediction
        Prediction as output by transformers.Trainer
    tokenizer: transformers.Tokenizer
        Tokenizer from huggingface.
    id2tag: Dict
        Dictionary mapping label ids to label names.
    additional_metrics: List
        List with additional metrics to compute.

    Returns
    -------
    metrics: Dict
        Dictionary with metrics. For information regarding the exact metrics
        received in it, see the documentation for sklearn.metrics.classification_report.
    """
    preds, labels = pred.predictions, pred.label_ids
    preds = np.argmax(preds, axis=1)
    class_report = classification_report(labels, preds, output_dict=True)
    metrics = class_report["macro avg"]
    return metrics


def compute_metrics_multilabel(
    pred, tokenizer=None, id2tag=None, additional_metrics=None
):
    """
    Compute the metrics for a multilabel task.

    Parameters
    ----------
    pred: transformers.EvalPrediction
        Prediction as output by transformers.Trainer
    tokenizer: transformers.Tokenizer
        Tokenizer from huggingface.
    id2tag: Dict
        Dictionary mapping label ids to label names.
    additional_metrics: List
        List with additional metrics to compute.

    Returns
    -------
    best_metrics: Dict
        Dictionary with best metrics, after trying different thresholds.
    """
    preds, labels = pred.predictions, pred.label_ids
    preds = torch.sigmoid(torch.from_numpy(preds)).numpy()
    thresholds = np.arange(0.1, 0.9, 0.1)
    best_metrics, best_metric, best_threshold = {}, 0, None

    for thres in thresholds:
        preds = preds >= thres
        preds = preds.astype(int)
        labels = labels.astype(int)
        class_report = classification_report(
            labels,
            preds,
            output_dict=True,
        )
        metrics = class_report["macro avg"]
        f1 = metrics["f1-score"]
        if f1 > best_metric:
            best_metrics = metrics
            best_metric = f1
            best_threshold = thres
    print(f"*** The best threshold is {best_threshold} ***")
    return best_metrics


def compute_metrics_ner(p, tokenizer=None, id2tag=None, additional_metrics=None):
    """
    Compute metrics for ner.

    Use seqeval metric from HF Evaluate. Get the predicted label for each instance,
    then skip padded tokens and finally use seqeval metric, which takes into account
    full entities, not individual tokens, when computing the metrics.

    Parameters
    ----------
    p: transformers.EvalPrediction
        Instance of EvalPrediction from transformers.
    tokenizer: transformers.Tokenizer
        Tokenizer from huggingface.
    id2tag: Dict
        Dictionary mapping label ids to label names.
    additional_metrics: List
        List with additional metrics to compute.

    Returns
    -------
    Metrics
        Complete dictionary with all computed metrics on eval data.
    """
    predictions, labels = p.predictions, p.label_ids

    try:
        predictions = np.argmax(predictions, axis=2)
    except Exception:
        print("The output shape is not logits-like, but directly targets.")
        predictions = predictions.astype("int")

    # Remove ignored index (special tokens)
    true_predictions = [
        [str(id2tag[p]) for (p, i) in zip(prediction, label) if i != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [str(id2tag[i]) for (p, i) in zip(prediction, label) if i != -100]
        for prediction, label in zip(predictions, labels)
    ]
    metrics = metric_seqeval.compute(
        predictions=true_predictions, references=true_labels
    )
    metrics["f1-score"] = metrics["overall_f1"]
    return metrics


def compute_metrics_summarization(
    eval_pred, tokenizer, id2tag=None, additional_metrics: List = None
):
    """
    Compute metrics for summarization tasks, by using rouge metrics in datasets library.

    Parameters
    ----------
    eval_pred: transformers.EvalPrediction
        Prediction as output by transformers.Trainer
    tokenizer:
        Tokenizer from huggingface.
    id2tag: Dict
        Dictionary mapping label ids to label names.
    additional_metrics: List
        List with additional metrics to compute.

    Returns
    -------
    metrics: Dict
        Dictionary with relevant metrics for summarization.
    """
    predictions, labels = eval_pred.predictions, eval_pred.label_ids
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Rouge expects a newline after each sentence
    decoded_preds = [
        "\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds
    ]
    decoded_labels = [
        "\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels
    ]

    result = metric_sum.compute(
        predictions=decoded_preds, references=decoded_labels, use_stemmer=True
    )
    # Extract a few results
    result = {key: value * 100 for key, value in result.items()}

    # Add mean generated length
    prediction_lens = [
        np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions
    ]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    if additional_metrics:
        other_results = []
        for metric in additional_metrics:
            subre = metric.compute(predictions=decoded_preds, references=decoded_labels)
            other_results.append(subre)
        print(f"Other results for this dataset: \n {other_results}")
        result["other_metrics"] = other_results
    return result
