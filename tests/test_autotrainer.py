from autotransformers import (
    AutoTrainer,
    DatasetConfig,
    ModelConfig,
    joinpaths,
    ResultsPlotter,
)
from transformers import (
    Seq2SeqTrainer,
    AutoModelForSeq2SeqLM,
    AutoConfig,
    Trainer,
    AutoTokenizer,
    AlbertForSequenceClassification,
)
import os
import pandas as pd
from transformers.training_args import TrainingArguments
from datasets import load_dataset
from autotransformers.hftransformers_manager import MultilabelTrainer
from autotransformers.hftransformers_manager import HFTransformersManager
from autotransformers.hfdatasets_manager import HFDatasetsManager
from autotransformers.results_getter import ResultsGetter
from autotransformers.skip_mix import SkipMix
import shutil as sh


def _create_multilabel_dataset(savename):
    """Create fake multilabel dataset."""
    datarow = {"text": "hola", "labelA": 1, "labelB": 0, "labelC": 1}
    data = [datarow, datarow, datarow]
    df = pd.DataFrame(data)
    for split in ["train", "validation", "test"]:
        df.to_csv(f"{savename}_{split}.csv", header=True, index=False)


def _silly_pre_func(example):
    """Naive function to prove working of pre_func."""
    return example


def hp_space(trial):
    """Hyperameter space used in all tests."""
    return {
        "learning_rate": trial.suggest_categorical(
            "learning_rate", [1.5e-5, 2e-5, 3e-5, 4e-5]
        ),
        "num_train_epochs": trial.suggest_categorical("num_train_epochs", [1]),
        "per_device_train_batch_size": trial.suggest_categorical(
            "per_device_train_batch_size", [1]
        ),
        "per_device_eval_batch_size": trial.suggest_categorical(
            "per_device_eval_batch_size", [1]
        ),
        "gradient_accumulation_steps": trial.suggest_categorical(
            "gradient_accumulation_steps", [1]
        ),
        "warmup_steps": trial.suggest_categorical("warmup_steps", [50, 100, 500, 1000]),
        "weight_decay": trial.suggest_categorical("weight_decay", [0.0]),
    }


fixed_train_args = {
    "evaluation_strategy": "steps",
    "num_train_epochs": 1,
    "do_train": True,
    "do_eval": True,
    "logging_strategy": "steps",
    "eval_steps": 1,
    "save_steps": 1,
    "logging_steps": 1,
    "save_strategy": "steps",
    "save_total_limit": 1,
    "seed": 69,
    "fp16": False,
    "no_cuda": True,
    "dataloader_num_workers": 1,
    "load_best_model_at_end": False,
    "per_device_eval_batch_size": 1,
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 1,
    "eval_accumulation_steps": 1,
    "adam_epsilon": 1e-6,
    "adam_beta1": 0.9,
    "adam_beta2": 0.999,
    "max_steps": 1,
}


def test_autotrainer_no_train():
    """Test autotrainer without training, only getting test results."""

    conll2002_config = {
        "seed": 44,
        "direction_optimize": "maximize",
        "metric_optimize": "eval_f1-score",
        "callbacks": [],
        "fixed_training_args": fixed_train_args,
        "dataset_name": "conll2002",
        "alias": "conll2002",
        "task": "ner",
        "text_field": "tokens",
        "hf_load_kwargs": {"path": "conll2002", "subset": "es"},
        "label_col": "ner_tags",
        "smoke_test": False,
        "retrain_at_end": False,
    }

    conll2002_config = DatasetConfig(**conll2002_config)
    model_config = ModelConfig(
        name="LenguajeNaturalAI/tiny-albert-testing",
        save_name="bsc@roberta",
        hp_space=hp_space,
        only_test=True,
        save_dir="test_models",
        n_trials=1,
        random_init_trials=1,
        custom_params_config_model={"model_type": "bert"},
        overwrite_training_args={"seed": 69},
    )
    autotrainer = AutoTrainer(
        model_configs=[model_config],
        dataset_configs=[conll2002_config],
        metrics_dir="test_notrain",
        use_auth_token=False,
        clean=False,
    )
    all_results = autotrainer()
    assert hasattr(autotrainer, "trainer"), "AutoTrainer should have a trainer."
    assert isinstance(
        all_results, dict
    ), f"All_results should be a Dict, and is {type(all_results)}"
    assert (
        model_config.save_name.replace("/", "-") in all_results
    ), "The results from bsc model should be there but they are not."
    assert isinstance(
        all_results[model_config.save_name.replace("/", "-")], dict
    ), "The results from the model should come in Dict format."


def test_autotrainer_skip_mix():
    """Test autotrainer by skipping mixes of datasets and models we dont want to train."""

    conll2002_config = {
        "seed": 44,
        "direction_optimize": "maximize",
        "metric_optimize": "eval_f1-score",
        "callbacks": [],
        "fixed_training_args": fixed_train_args,
        "dataset_name": "conll2002",
        "alias": "conll2002",
        "task": "ner",
        "text_field": "tokens",
        "hf_load_kwargs": {"path": "conll2002", "subset": "es"},
        "label_col": "ner_tags",
        "smoke_test": False,
        "retrain_at_end": False,
    }

    conll2002_config = DatasetConfig(**conll2002_config)
    model_config = ModelConfig(
        name="LenguajeNaturalAI/tiny-albert-testing",
        save_name="bsc@roberta",
        hp_space=hp_space,
        only_test=True,
        save_dir="test_models",
        n_trials=1,
        random_init_trials=1,
        custom_params_config_model={"model_type": "bert"},
        overwrite_training_args={"seed": 69},
    )
    skip_mix = SkipMix(
        dataset_name=conll2002_config.alias, model_name=model_config.save_name
    )
    autotrainer = AutoTrainer(
        model_configs=[model_config],
        dataset_configs=[conll2002_config],
        metrics_dir="test_notrain",
        use_auth_token=False,
        clean=False,
        skip_mixes=[skip_mix],
    )
    all_results = autotrainer()
    assert not hasattr(
        autotrainer, "trainer"
    ), "AutoTrainer should not have a trainer, as the experiment was skipped."
    assert isinstance(
        all_results, dict
    ), f"All_results should be a Dict, and is {type(all_results)}"


def test_autotrainer_smoke_test():
    """Test autotrainer on smoke test mode."""

    conll2002_config = {
        "seed": 44,
        "direction_optimize": "maximize",
        "metric_optimize": "eval_f1-score",
        "callbacks": [],
        "fixed_training_args": fixed_train_args,
        "dataset_name": "conll2002",
        "alias": "conll2002",
        "task": "ner",
        "text_field": "tokens",
        "hf_load_kwargs": {"path": "conll2002", "subset": "es"},
        "label_col": "ner_tags",
        "smoke_test": True,
        "retrain_at_end": False,
    }

    conll2002_config = DatasetConfig(**conll2002_config)
    model_config = ModelConfig(
        name="LenguajeNaturalAI/tiny-albert-testing",
        save_name="bsc@roberta",
        hp_space=hp_space,
        only_test=True,
        save_dir="test_models",
        n_trials=1,
        random_init_trials=1,
    )
    autotrainer = AutoTrainer(
        model_configs=[model_config],
        dataset_configs=[conll2002_config],
        metrics_dir="test_smoke",
        use_auth_token=False,
        clean=False,
    )
    all_results = autotrainer()
    assert hasattr(autotrainer, "trainer"), "AutoTrainer should have a trainer."
    assert isinstance(
        all_results, dict
    ), f"All_results should be a Dict, and is {type(all_results)}"
    assert (
        model_config.save_name.replace("/", "-") in all_results
    ), "The results from bsc model should be there but they are not."
    assert isinstance(
        all_results[model_config.save_name.replace("/", "-")], dict
    ), "The results from the model should come in Dict format."


def test_autotrainer_train():
    """Test autotrainer training for 1 step only."""
    wnli_config = {
        "seed": 44,
        "direction_optimize": "maximize",
        "metric_optimize": "eval_f1-score",
        "callbacks": [],
        "fixed_training_args": fixed_train_args,
        "dataset_name": "wnli",
        "alias": "wnli",
        "task": "classification",
        "text_field": "sentence1",
        "hf_load_kwargs": {"path": "LenguajeNaturalAI/wnli_testing"},
        "label_col": "label",
        "is_2sents": True,
        "sentence1_field": "sentence1",
        "sentence2_field": "sentence2",
        "pre_func": _silly_pre_func,
        "smoke_test": False,
        "retrain_at_end": True,
    }

    wnli_config = DatasetConfig(**wnli_config)
    model_config = ModelConfig(
        name="LenguajeNaturalAI/tiny-albert-testing",
        save_name="bsc@roberta",
        hp_space=hp_space,
        only_test=False,
        save_dir="test_models",
        n_trials=1,
        random_init_trials=1,
    )
    autotrainer = AutoTrainer(
        model_configs=[model_config],
        dataset_configs=[wnli_config],
        metrics_dir="test_train",
        use_auth_token=False,
        clean=False,
    )
    all_results = autotrainer()
    assert hasattr(autotrainer, "trainer"), "AutoTrainer should have a trainer."
    assert isinstance(
        all_results, dict
    ), f"All_results should be a Dict, and is {type(all_results)}: \n {all_results}"
    assert (
        model_config.save_name.replace("/", "-") in all_results
    ), "The results from bsc model should be there but they are not."
    assert isinstance(
        all_results[model_config.save_name.replace("/", "-")], dict
    ), "The results from the model should come in Dict format."
    dropout = (
        model_config.dropout_vals[0] if len(model_config.dropout_vals) > 0 else 0.0
    )
    assert os.path.exists(
        joinpaths(
            "test_models/",
            f"best_optuna_{model_config.save_name}-{wnli_config.alias}-dropout_{dropout}",
        )
    ), "No folder was created for the model."
    assert os.path.exists(
        joinpaths(
            autotrainer.metrics_dir,
            f"{model_config.save_name}#{wnli_config.alias}.json",
        )
    ), "There is no file with test results."


def test_autotrainer_train_normal_classification():
    """Test autotrainer training on classification for 1 step only."""
    wnli_config = {
        "seed": 44,
        "direction_optimize": "maximize",
        "metric_optimize": "eval_f1-score",
        "callbacks": [],
        "fixed_training_args": fixed_train_args,
        "dataset_name": "wnli",
        "alias": "wnli",
        "task": "classification",
        "text_field": "sentence1",
        "hf_load_kwargs": {"path": "LenguajeNaturalAI/wnli_testing"},
        "label_col": "label",
        "is_2sents": False,
        "pre_func": _silly_pre_func,
        "smoke_test": False,
        "retrain_at_end": False,
    }

    wnli_config = DatasetConfig(**wnli_config)
    model_config = ModelConfig(
        name="LenguajeNaturalAI/tiny-albert-testing",
        save_name="bsc@roberta",
        hp_space=hp_space,
        only_test=False,
        save_dir="test_models",
        n_trials=1,
        random_init_trials=1,
    )
    autotrainer = AutoTrainer(
        model_configs=[model_config],
        dataset_configs=[wnli_config],
        metrics_dir="test_classification",
        use_auth_token=False,
        clean=False,
    )
    all_results = autotrainer()
    assert hasattr(autotrainer, "trainer"), "AutoTrainer should have a trainer."
    assert isinstance(
        all_results, dict
    ), f"All_results should be a Dict, and is {type(all_results)}: \n {all_results}"
    assert (
        model_config.save_name.replace("/", "-") in all_results
    ), "The results from bsc model should be there but they are not."
    assert isinstance(
        all_results[model_config.save_name.replace("/", "-")], dict
    ), "The results from the model should come in Dict format."
    dropout = (
        model_config.dropout_vals[0] if len(model_config.dropout_vals) > 0 else 0.0
    )
    assert os.path.exists(
        joinpaths(
            "test_models/",
            f"best_optuna_{model_config.save_name}-{wnli_config.alias}-dropout_{dropout}",
        )
    ), "No folder was created for the model."
    assert os.path.exists(
        joinpaths(
            autotrainer.metrics_dir,
            f"{model_config.save_name}#{wnli_config.alias}.json",
        )
    ), "There is no file with test results."


def test_autotrainer_train_fixed_params():
    """Test autotrainer training without hparams search."""

    wnli_config = {
        "seed": 44,
        "direction_optimize": "maximize",
        "metric_optimize": "eval_f1-score",
        "callbacks": [],
        "fixed_training_args": fixed_train_args,
        "dataset_name": "wnli",
        "alias": "wnli",
        "task": "classification",
        "text_field": "sentence1",
        "hf_load_kwargs": {"path": "LenguajeNaturalAI/wnli_testing"},
        "label_col": "label",
        "is_2sents": True,
        "sentence1_field": "sentence1",
        "sentence2_field": "sentence2",
        "pre_func": _silly_pre_func,
        "smoke_test": False,
        "retrain_at_end": False,
    }

    wnli_config = DatasetConfig(**wnli_config)
    model_config = ModelConfig(
        name="LenguajeNaturalAI/tiny-albert-testing",
        save_name="bsc@roberta",
        hp_space=hp_space,
        only_test=False,
        save_dir="test_models",
        n_trials=1,
        random_init_trials=1,
    )
    autotrainer = AutoTrainer(
        model_configs=[model_config],
        dataset_configs=[wnli_config],
        metrics_dir="test_fixed",
        hp_search_mode="fixed",
        use_auth_token=False,
        clean=False,
    )
    all_results = autotrainer()
    assert hasattr(autotrainer, "trainer"), "AutoTrainer should have a trainer."
    assert isinstance(
        all_results, dict
    ), f"All_results should be a Dict, and is {type(all_results)}"
    assert (
        model_config.save_name.replace("/", "-") in all_results
    ), "The results from bsc model should be there but they are not."
    assert isinstance(
        all_results[model_config.save_name.replace("/", "-")], dict
    ), "The results from the model should come in Dict format."
    assert os.path.exists(
        joinpaths(
            autotrainer.metrics_dir,
            f"{model_config.save_name}#{wnli_config.alias}.json",
        )
    ), "There is no file with test results."


def test_autotrainer_qa_train():
    """Test autotrainer training on QA."""
    fixed_train_args = {
        "evaluation_strategy": "steps",
        "num_train_epochs": 1,
        "do_train": True,
        "do_eval": True,
        "logging_strategy": "steps",
        "eval_steps": 1,
        "save_steps": 1,
        "logging_steps": 1,
        "save_strategy": "steps",
        "save_total_limit": 1,
        "seed": 69,
        "fp16": False,
        "no_cuda": True,
        "dataloader_num_workers": 1,
        "load_best_model_at_end": False,
        "per_device_train_batch_size": 1,
        "per_device_eval_batch_size": 1,
        "adam_epsilon": 1e-6,
        "adam_beta1": 0.9,
        "adam_beta2": 0.999,
        "max_steps": 1,
        "greater_is_better": False,
    }

    sqac = {
        "seed": 44,
        "direction_optimize": "minimize",
        "metric_optimize": "eval_loss",
        "callbacks": [],
        "fixed_training_args": fixed_train_args,
        "dataset_name": "sqac",
        "alias": "sqac",
        "task": "qa",
        "text_field": "context",
        "hf_load_kwargs": {"path": "avacaondata/sqac_fixed"},
        "label_col": "answers",
        "pre_func": _silly_pre_func,
        "smoke_test": False,
        "squad_v2": False,
        "retrain_at_end": False,
    }

    sqac = DatasetConfig(**sqac)
    model_config = ModelConfig(
        name="LenguajeNaturalAI/tiny-albert-testing",
        save_name="bsc@roberta",
        hp_space=hp_space,
        only_test=False,
        save_dir="test_models",
        n_trials=1,
        random_init_trials=1,
    )
    autotrainer = AutoTrainer(
        model_configs=[model_config],
        dataset_configs=[sqac],
        metrics_dir="test_qa_train",
        use_auth_token=False,
        clean=False,
    )
    all_results = autotrainer()

    datasets_manager = HFDatasetsManager(sqac, model_config)

    dataset, _ = datasets_manager.get_dataset_and_tag2id(autotrainer.tokenizer)
    results_getter = ResultsGetter(
        autotrainer.dataset_configs[0], autotrainer.model_configs[0], None
    )

    results_alternative = results_getter.get_test_results_qa(
        dataset["test"],
        autotrainer.trainer,
        autotrainer.dataset_configs[0].squad_v2,
    )

    dmap = {"sqac": "qa"}

    plotter = ResultsPlotter(
        metrics_dir=autotrainer.metrics_dir,
        model_names=model_config.save_name,
        dataset_to_task_map=dmap,
        metric_field="f1",
    )
    ax = plotter.plot_metrics()
    ax.figure.savefig("test_results.png")
    assert isinstance(
        results_alternative, dict
    ), "Metrics returned from get_test_results_qa should be a dict."
    assert os.path.exists(
        "test_results.png"
    ), "El gráfico de resultados no se ha guardado correctamente."
    assert hasattr(autotrainer, "trainer"), "AutoTrainer should have a trainer."
    assert isinstance(
        all_results, dict
    ), f"All_results should be a Dict, and is {type(all_results)}"
    assert (
        model_config.save_name.replace("/", "-") in all_results
    ), "The results from bsc model should be there but they are not."
    assert isinstance(
        all_results[model_config.save_name.replace("/", "-")], dict
    ), "The results from the model should come in Dict format."
    assert os.path.exists(
        joinpaths(
            autotrainer.metrics_dir, f"{model_config.save_name}#{sqac.alias}.json"
        )
    ), "There is no file with test results."


def test_autotrainer_qa_train_squadv2():
    """Test autotrainer training on QA."""
    fixed_train_args = {
        "evaluation_strategy": "steps",
        "num_train_epochs": 1,
        "do_train": True,
        "do_eval": True,
        "logging_strategy": "steps",
        "eval_steps": 1,
        "save_steps": 1,
        "logging_steps": 1,
        "save_strategy": "steps",
        "save_total_limit": 1,
        "seed": 69,
        "fp16": False,
        "no_cuda": True,
        "dataloader_num_workers": 1,
        "load_best_model_at_end": False,
        "per_device_train_batch_size": 1,
        "per_device_eval_batch_size": 1,
        "adam_epsilon": 1e-6,
        "adam_beta1": 0.9,
        "adam_beta2": 0.999,
        "max_steps": 1,
        "greater_is_better": False,
    }

    sqac = {
        "seed": 44,
        "direction_optimize": "minimize",
        "metric_optimize": "eval_loss",
        "callbacks": [],
        "fixed_training_args": fixed_train_args,
        "dataset_name": "sqac",
        "alias": "sqac",
        "task": "qa",
        "text_field": "context",
        "hf_load_kwargs": {"path": "avacaondata/sqac_fixed"},
        "label_col": "answers",
        "pre_func": _silly_pre_func,
        "smoke_test": False,
        "squad_v2": True,
        "retrain_at_end": False,
    }

    sqac = DatasetConfig(**sqac)
    model_config = ModelConfig(
        name="LenguajeNaturalAI/tiny-albert-testing",
        save_name="bsc@roberta",
        hp_space=hp_space,
        only_test=False,
        save_dir="test_models",
        n_trials=1,
        random_init_trials=1,
    )
    autotrainer = AutoTrainer(
        model_configs=[model_config],
        dataset_configs=[sqac],
        metrics_dir="test_notrain",
        use_auth_token=False,
        clean=False,
    )
    all_results = autotrainer()

    datasets_manager = HFDatasetsManager(sqac, model_config)

    dataset, _ = datasets_manager.get_dataset_and_tag2id(autotrainer.tokenizer)

    results_getter = ResultsGetter(
        autotrainer.dataset_configs[0], autotrainer.model_configs[0], None
    )

    results_alternative = results_getter.get_test_results_qa(
        dataset["test"],
        autotrainer.trainer,
        autotrainer.dataset_configs[0].squad_v2,
    )

    assert isinstance(
        results_alternative, dict
    ), "Metrics returned from get_test_results_qa should be a dict."
    assert hasattr(autotrainer, "trainer"), "AutoTrainer should have a trainer."
    assert isinstance(
        all_results, dict
    ), f"All_results should be a Dict, and is {type(all_results)}"
    assert (
        model_config.save_name.replace("/", "-") in all_results
    ), "The results from bsc model should be there but they are not."
    assert isinstance(
        all_results[model_config.save_name.replace("/", "-")], dict
    ), "The results from the model should come in Dict format."
    assert os.path.exists(
        joinpaths(
            autotrainer.metrics_dir, f"{model_config.save_name}#{sqac.alias}.json"
        )
    ), "There is no file with test results."


def test_metrics_plotter():
    """Test autotrainer plotting of results."""
    fixed_train_args = {
        "evaluation_strategy": "steps",
        "num_train_epochs": 1,
        "do_train": True,
        "do_eval": True,
        "logging_strategy": "steps",
        "eval_steps": 1,
        "save_steps": 1,
        "logging_steps": 1,
        "save_strategy": "steps",
        "save_total_limit": 1,
        "seed": 69,
        "fp16": False,
        "no_cuda": True,
        "dataloader_num_workers": 1,
        "load_best_model_at_end": False,
        "per_device_train_batch_size": 1,
        "per_device_eval_batch_size": 1,
        "adam_epsilon": 1e-6,
        "adam_beta1": 0.9,
        "adam_beta2": 0.999,
        "max_steps": 1,
        "greater_is_better": False,
    }

    sqac = {
        "seed": 44,
        "direction_optimize": "minimize",
        "metric_optimize": "eval_loss",
        "callbacks": [],
        "fixed_training_args": fixed_train_args,
        "dataset_name": "sqac",
        "alias": "sqac",
        "task": "qa",
        "text_field": "context",
        "hf_load_kwargs": {"path": "avacaondata/sqac_fixed"},
        "label_col": "answers",
        "pre_func": _silly_pre_func,
        "smoke_test": False,
        "squad_v2": True,
        "retrain_at_end": False,
    }

    sqac = DatasetConfig(**sqac)
    model_config = ModelConfig(
        name="LenguajeNaturalAI/tiny-albert-testing",
        save_name="bsc@roberta",
        hp_space=hp_space,
        only_test=False,
        save_dir="test_models",
        n_trials=1,
        random_init_trials=1,
    )
    autotrainer = AutoTrainer(
        model_configs=[model_config],
        dataset_configs=[sqac],
        metrics_dir="test_notrain",
        use_auth_token=False,
        clean=False,
    )
    _ = autotrainer()

    dmap = {"sqac": "qa"}

    plotter = ResultsPlotter(
        metrics_dir=autotrainer.metrics_dir,
        model_names=model_config.save_name,
        dataset_to_task_map=dmap,
        metric_field="f1",
    )
    ax = plotter.plot_metrics()
    ax.figure.savefig("test_results.png")
    assert os.path.exists(
        "test_results.png"
    ), "El gráfico de resultados no se ha guardado correctamente."


def test_autotrainer_qa_no_train():
    """Test autotrainer on QA with no bpe model."""
    fixed_train_args = {
        "evaluation_strategy": "steps",
        "num_train_epochs": 1,
        "do_train": True,
        "do_eval": True,
        "logging_strategy": "steps",
        "eval_steps": 1,
        "save_steps": 1,
        "logging_steps": 1,
        "save_strategy": "steps",
        "save_total_limit": 1,
        "seed": 69,
        "fp16": False,
        "no_cuda": True,
        "dataloader_num_workers": 1,
        "load_best_model_at_end": False,
        "per_device_train_batch_size": 1,
        "per_device_eval_batch_size": 1,
        "adam_epsilon": 1e-6,
        "adam_beta1": 0.9,
        "adam_beta2": 0.999,
        "max_steps": 1,
        "greater_is_better": False,
    }

    sqac = {
        "seed": 44,
        "direction_optimize": "minimize",
        "metric_optimize": "eval_loss",
        "callbacks": [],
        "fixed_training_args": fixed_train_args,
        "dataset_name": "sqac",
        "alias": "sqac",
        "task": "qa",
        "text_field": "context",
        "hf_load_kwargs": {"path": "avacaondata/sqac_fixed"},
        "label_col": "answers",
        "pre_func": _silly_pre_func,
        "smoke_test": False,
        "squad_v2": False,
        "retrain_at_end": False,
    }

    sqac = DatasetConfig(**sqac)
    model_config = ModelConfig(
        name="dccuchile/bert-base-spanish-wwm-cased",
        save_name="bsc@roberta",
        hp_space=hp_space,
        only_test=True,
        save_dir="test_models",
        n_trials=1,
        random_init_trials=1,
    )
    autotrainer = AutoTrainer(
        model_configs=[model_config],
        dataset_configs=[sqac],
        metrics_dir="test_qa_notrain",
        use_auth_token=False,
        clean=False,
    )
    all_results = autotrainer()
    dmap = {"sqac": "qa"}

    plotter = ResultsPlotter(
        metrics_dir=autotrainer.metrics_dir,
        model_names=model_config.save_name,
        dataset_to_task_map=dmap,
        metric_field="f1",
    )
    ax = plotter.plot_metrics()
    ax.figure.savefig("test_results.png")
    assert os.path.exists(
        "test_results.png"
    ), "El gráfico de resultados no se ha guardado correctamente."
    assert hasattr(autotrainer, "trainer"), "AutoTrainer should have a trainer."
    assert isinstance(
        all_results, dict
    ), f"All_results should be a Dict, and is {type(all_results)}"
    assert (
        model_config.save_name.replace("/", "-") in all_results
    ), "The results from bsc model should be there but they are not."
    assert isinstance(
        all_results[model_config.save_name.replace("/", "-")], dict
    ), "The results from the model should come in Dict format."
    assert os.path.exists(
        joinpaths(
            autotrainer.metrics_dir, f"{model_config.save_name}#{sqac.alias}.json"
        )
    ), "There is no file with test results."


def test_autotrainer_summarization():
    """Test autotrainer training on summarization."""
    fixed_train_args = {
        "evaluation_strategy": "steps",
        "num_train_epochs": 1,
        "do_train": True,
        "do_eval": True,
        "logging_strategy": "steps",
        "eval_steps": 1,
        "save_steps": 1,
        "logging_steps": 1,
        "save_strategy": "steps",
        "save_total_limit": 2,
        "seed": 69,
        "fp16": False,
        "no_cuda": True,
        "dataloader_num_workers": 2,
        "load_best_model_at_end": True,
        "per_device_eval_batch_size": 16,
        "adam_epsilon": 1e-6,
        "adam_beta1": 0.9,
        "adam_beta2": 0.999,
        "max_steps": 1,
        "greater_is_better": False,
    }

    sqac = {
        "seed": 44,
        "direction_optimize": "minimize",
        "metric_optimize": "eval_loss",
        "callbacks": [],
        "fixed_training_args": fixed_train_args,
        "dataset_name": "sqac",
        "alias": "sqac",
        "task": "seq2seq",
        "text_field": "context",
        "hf_load_kwargs": {"path": "avacaondata/sqac_fixed"},
        "label_col": "title",
        "pre_func": _silly_pre_func,
        "smoke_test": False,
        "summary_field": "title",
        "retrain_at_end": False,
        "split": True,
    }
    sqac = DatasetConfig(**sqac)
    model_config = ModelConfig(
        name="patrickvonplaten/t5-tiny-random",
        save_name="t5",
        hp_space=hp_space,
        only_test=False,
        save_dir="test_models",
        n_trials=1,
        random_init_trials=1,
        trainer_cls_summarization=Seq2SeqTrainer,
        model_cls_summarization=AutoModelForSeq2SeqLM,
    )
    autotrainer = AutoTrainer(
        model_configs=[model_config],
        dataset_configs=[sqac],
        metrics_dir="test_summarization",
        use_auth_token=False,
        clean=False,
    )
    all_results = autotrainer()
    dmap = {"sqac": "seq2seq"}
    plotter = ResultsPlotter(
        metrics_dir=autotrainer.metrics_dir,
        model_names=model_config.save_name,
        dataset_to_task_map=dmap,
        metric_field="rouge2",
    )
    ax = plotter.plot_metrics()

    ax.figure.savefig("test_results.png")
    assert os.path.exists(
        "test_results.png"
    ), "El gráfico de resultados no se ha guardado correctamente."
    assert hasattr(autotrainer, "trainer"), "AutoTrainer should have a trainer."
    assert isinstance(
        all_results, dict
    ), f"All_results should be a Dict, and is {type(all_results)}"
    assert (
        model_config.save_name.replace("/", "-") in all_results
    ), "The results from bsc model should be there but they are not."
    assert isinstance(
        all_results[model_config.save_name.replace("/", "-")], dict
    ), "The results from the model should come in Dict format."
    assert os.path.exists(
        joinpaths(
            autotrainer.metrics_dir, f"{model_config.save_name}#{sqac.alias}.json"
        )
    ), "There is no file with test results."


def test_autotrainer_multilabel_train():
    """Test training on multilabel task."""
    savename = "testdf"
    _create_multilabel_dataset(savename)
    dataset_config = {
        "seed": 44,
        "direction_optimize": "maximize",
        "metric_optimize": "eval_f1-score",
        "callbacks": [],
        "fixed_training_args": fixed_train_args,
        "dataset_name": "wnli",
        "alias": "wnli",
        "task": "classification",
        "text_field": "text",
        "label_col": "labelA",
        "is_2sents": False,
        "smoke_test": False,
        "is_multilabel": True,
        "partial_split": True,
        "config_num_labels": 3,
        "type_load": "csv",
        "retrain_at_end": False,
        "files": {
            split: f"{savename}_{split}.csv"
            for split in ["train", "validation", "test"]
        },
    }
    dataset_config = DatasetConfig(**dataset_config)
    model_config = ModelConfig(
        name="LenguajeNaturalAI/tiny-albert-testing",
        save_name="bsc@roberta",
        hp_space=hp_space,
        only_test=False,
        save_dir="test_models",
        n_trials=1,
        random_init_trials=1,
    )
    autotrainer = AutoTrainer(
        model_configs=[model_config],
        dataset_configs=[dataset_config],
        metrics_dir="test_notrain",
        use_auth_token=False,
        clean=False,
    )
    all_results = autotrainer()
    assert hasattr(autotrainer, "trainer"), "AutoTrainer should have a trainer."
    assert isinstance(
        all_results, dict
    ), f"All_results should be a Dict, and is {type(all_results)}: \n {all_results}"
    assert (
        model_config.save_name.replace("/", "-") in all_results
    ), "The results from bsc model should be there but they are not."
    assert isinstance(
        all_results[model_config.save_name.replace("/", "-")], dict
    ), "The results from the model should come in Dict format."
    dropout = (
        model_config.dropout_vals[0] if len(model_config.dropout_vals) > 0 else 0.0
    )
    assert os.path.exists(
        joinpaths(
            "test_models/",
            f"best_optuna_{model_config.save_name}-{dataset_config.alias}-dropout_{dropout}",
        )
    ), "No folder was created for the model."
    assert os.path.exists(
        joinpaths(
            autotrainer.metrics_dir,
            f"{model_config.save_name}#{dataset_config.alias}.json",
        )
    ), "There is no file with test results."


def test_autotrainer_multilabel_trainer_train():
    """Test training on multilabel task with MultilabelTrainer."""
    savename = "testdf"
    _create_multilabel_dataset(savename)
    dataset_config = {
        "seed": 44,
        "direction_optimize": "maximize",
        "metric_optimize": "eval_f1-score",
        "callbacks": [],
        "fixed_training_args": fixed_train_args,
        "dataset_name": "wnli",
        "alias": "wnli",
        "task": "classification",
        "text_field": "text",
        "label_col": "labelA",
        "is_2sents": False,
        "smoke_test": False,
        "is_multilabel": True,
        "config_num_labels": 3,
        "type_load": "csv",
        "retrain_at_end": False,
        "files": {
            split: f"{savename}_{split}.csv"
            for split in ["train", "validation", "test"]
        },
    }
    dataset_config = DatasetConfig(**dataset_config)
    model_config = ModelConfig(
        name="LenguajeNaturalAI/tiny-albert-testing",
        save_name="bsc@roberta",
        hp_space=hp_space,
        only_test=False,
        save_dir="test_models",
        n_trials=1,
        random_init_trials=1,
        custom_trainer_cls=MultilabelTrainer,
    )
    autotrainer = AutoTrainer(
        model_configs=[model_config],
        dataset_configs=[dataset_config],
        metrics_dir="test_multilabel",
        use_auth_token=False,
        clean=False,
    )
    all_results = autotrainer()
    assert hasattr(autotrainer, "trainer"), "AutoTrainer should have a trainer."
    assert isinstance(
        all_results, dict
    ), f"All_results should be a Dict, and is {type(all_results)}: \n {all_results}"
    assert (
        model_config.save_name.replace("/", "-") in all_results
    ), "The results from bsc model should be there but they are not."
    assert isinstance(
        all_results[model_config.save_name.replace("/", "-")], dict
    ), "The results from the model should come in Dict format."
    dropout = (
        model_config.dropout_vals[0] if len(model_config.dropout_vals) > 0 else 0.0
    )
    assert os.path.exists(
        joinpaths(
            "test_models/",
            f"best_optuna_{model_config.save_name}-{dataset_config.alias}-dropout_{dropout}",
        )
    ), "No folder was created for the model."
    assert os.path.exists(
        joinpaths(
            autotrainer.metrics_dir,
            f"{model_config.save_name}#{dataset_config.alias}.json",
        )
    ), "There is no file with test results."


def test_autotrainer_do_nothing():
    """Test AutoTrainer in do_nothing mode."""
    wnli_config = {
        "seed": 44,
        "direction_optimize": "maximize",
        "metric_optimize": "eval_f1-score",
        "callbacks": [],
        "fixed_training_args": fixed_train_args,
        "dataset_name": "wnli",
        "alias": "wnli",
        "task": "classification",
        "text_field": "sentence1",
        "hf_load_kwargs": {"path": "LenguajeNaturalAI/wnli_testing"},
        "label_col": "label",
        "is_2sents": True,
        "sentence1_field": "sentence1",
        "sentence2_field": "sentence2",
        "pre_func": _silly_pre_func,
        "smoke_test": False,
        "retrain_at_end": False,
    }

    wnli_config = DatasetConfig(**wnli_config)
    model_config = ModelConfig(
        name="LenguajeNaturalAI/tiny-albert-testing",
        save_name="bsc@roberta",
        hp_space=hp_space,
        only_test=False,
        save_dir="test_models",
        n_trials=1,
        random_init_trials=1,
        do_nothing=True,
    )
    autotrainer = AutoTrainer(
        model_configs=[model_config],
        dataset_configs=[wnli_config],
        metrics_dir="test_notrain",
        use_auth_token=False,
        clean=False,
    )
    all_results = autotrainer()
    assert hasattr(autotrainer, "trainer"), "AutoTrainer should have a trainer."
    assert isinstance(
        all_results, dict
    ), f"All_results should be a Dict, and is {type(all_results)}: \n {all_results}"
    assert (
        model_config.save_name.replace("/", "-") in all_results
    ), "The results from bsc model should be there but they are not."
    assert isinstance(
        all_results[model_config.save_name.replace("/", "-")], dict
    ), "The results from the model should come in Dict format."


def test_load_train_args():
    """Test small methods of autotrainer."""
    wnli_config = {
        "seed": 44,
        "direction_optimize": "maximize",
        "metric_optimize": "eval_f1-score",
        "callbacks": [],
        "fixed_training_args": fixed_train_args,
        "dataset_name": "wnli",
        "alias": "wnli",
        "task": "classification",
        "text_field": "sentence1",
        "hf_load_kwargs": {"path": "LenguajeNaturalAI/wnli_testing"},
        "label_col": "label",
        "is_2sents": True,
        "sentence1_field": "sentence1",
        "sentence2_field": "sentence2",
        "pre_func": _silly_pre_func,
        "smoke_test": False,
        "retrain_at_end": False,
    }

    wnli_config = DatasetConfig(**wnli_config)
    model_config = ModelConfig(
        name="LenguajeNaturalAI/tiny-albert-testing",
        save_name="bsc@roberta",
        hp_space=hp_space,
        only_test=False,
        save_dir="test_models",
        n_trials=1,
        random_init_trials=1,
        do_nothing=True,
    )
    transformers_manager = HFTransformersManager(model_config, wnli_config)
    train_args = transformers_manager.load_train_args(".")
    assert isinstance(
        train_args, TrainingArguments
    ), "Training Arguments does not have the correct data type."


def test_model_init():
    """Test the creation of a model init function."""
    wnli_config = {
        "seed": 44,
        "direction_optimize": "maximize",
        "metric_optimize": "eval_f1-score",
        "callbacks": [],
        "fixed_training_args": fixed_train_args,
        "dataset_name": "wnli",
        "alias": "wnli",
        "task": "classification",
        "text_field": "sentence1",
        "hf_load_kwargs": {"path": "LenguajeNaturalAI/wnli_testing"},
        "label_col": "label",
        "is_2sents": True,
        "sentence1_field": "sentence1",
        "sentence2_field": "sentence2",
        "pre_func": _silly_pre_func,
        "smoke_test": False,
        "retrain_at_end": False,
    }

    wnli_config = DatasetConfig(**wnli_config)
    model_config = ModelConfig(
        name="LenguajeNaturalAI/tiny-albert-testing",
        save_name="bsc@roberta",
        hp_space=hp_space,
        only_test=False,
        save_dir="test_models",
        n_trials=1,
        random_init_trials=1,
        do_nothing=True,
    )
    transformers_manager = HFTransformersManager(model_config, wnli_config)
    config = AutoConfig.from_pretrained("LenguajeNaturalAI/tiny-albert-testing")
    model_cls = transformers_manager.get_model_cls()
    tokenizer = transformers_manager.load_tokenizer()
    model_init = transformers_manager.load_model_init(model_cls, config, tokenizer)
    model = model_init()
    assert isinstance(
        model, AlbertForSequenceClassification
    ), "Model should be AlbertForSequenceClassification."


def test_load_data_collator():
    """Test the loading of the DataCollator object."""
    wnli_config = {
        "seed": 44,
        "direction_optimize": "maximize",
        "metric_optimize": "eval_f1-score",
        "callbacks": [],
        "fixed_training_args": fixed_train_args,
        "dataset_name": "wnli",
        "alias": "wnli",
        "task": "classification",
        "text_field": "sentence1",
        "hf_load_kwargs": {"path": "LenguajeNaturalAI/wnli_testing"},
        "label_col": "label",
        "is_2sents": True,
        "sentence1_field": "sentence1",
        "sentence2_field": "sentence2",
        "pre_func": _silly_pre_func,
        "smoke_test": False,
        "retrain_at_end": False,
    }

    wnli_config = DatasetConfig(**wnli_config)
    model_config = ModelConfig(
        name="LenguajeNaturalAI/tiny-albert-testing",
        save_name="bsc@roberta",
        hp_space=hp_space,
        only_test=False,
        save_dir="test_models",
        n_trials=1,
        random_init_trials=1,
        do_nothing=True,
    )
    transformers_manager = HFTransformersManager(model_config, wnli_config)
    tokenizer = transformers_manager.load_tokenizer()
    data_collator = transformers_manager.load_data_collator(tokenizer)
    assert data_collator is None, "DataCollator should be None"


def test_load_trainer():
    """Test the loading of the trainer object."""
    wnli_config = {
        "seed": 44,
        "direction_optimize": "maximize",
        "metric_optimize": "eval_f1-score",
        "callbacks": [],
        "fixed_training_args": fixed_train_args,
        "dataset_name": "wnli",
        "alias": "wnli",
        "task": "classification",
        "text_field": "sentence1",
        "hf_load_kwargs": {"path": "LenguajeNaturalAI/wnli_testing"},
        "label_col": "label",
        "is_2sents": True,
        "sentence1_field": "sentence1",
        "sentence2_field": "sentence2",
        "pre_func": _silly_pre_func,
        "smoke_test": False,
        "retrain_at_end": False,
    }

    wnli_config = DatasetConfig(**wnli_config)
    model_config = ModelConfig(
        name="LenguajeNaturalAI/tiny-albert-testing",
        save_name="bsc@roberta",
        hp_space=hp_space,
        only_test=False,
        save_dir="test_models",
        n_trials=1,
        random_init_trials=1,
        do_nothing=True,
    )
    autotrainer = AutoTrainer(
        model_configs=[model_config],
        dataset_configs=[wnli_config],
        metrics_dir="test_notrain",
        use_auth_token=False,
        clean=False,
    )
    transformers_manager = HFTransformersManager(model_config, wnli_config)
    train_args = transformers_manager.load_train_args(".")
    config = AutoConfig.from_pretrained("LenguajeNaturalAI/tiny-albert-testing")
    model_cls = transformers_manager.get_model_cls()
    tokenizer = transformers_manager.load_tokenizer()
    model_init = transformers_manager.load_model_init(model_cls, config, tokenizer)
    compute_metrics_func = autotrainer._get_compute_metrics(wnli_config)

    data_collator = transformers_manager.load_data_collator(tokenizer)
    dataset = load_dataset("LenguajeNaturalAI/wnli_testing")
    tokenizer = AutoTokenizer.from_pretrained("LenguajeNaturalAI/tiny-albert-testing")
    autotrainer.tokenizer = tokenizer
    trainer = transformers_manager.load_trainer(
        dataset,
        tokenizer,
        train_args,
        model_init,
        data_collator,
        compute_metrics_func,
        config,
    )
    assert isinstance(trainer, Trainer), "Trainer was not correctly created."
