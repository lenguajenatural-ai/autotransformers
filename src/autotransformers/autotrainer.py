from typing import List
import traceback
import os
from tqdm import tqdm
from .metrics import (
    compute_metrics_ner,
    compute_metrics_classification,
    compute_metrics_summarization,
    compute_metrics_multilabel,
)
from .utils import _save_metrics, joinpaths, fix_eval_results_dict
from copy import deepcopy
from .ckpt_cleaner import CkptCleaner
from optuna.samplers import TPESampler
from apscheduler.schedulers.background import BackgroundScheduler
from .hftransformers_manager import HFTransformersManager
from .hfdatasets_manager import HFDatasetsManager
from .results_getter import ResultsGetter

metric_func_map = {
    "ner": compute_metrics_ner,
    "classification": compute_metrics_classification,
    "qa": None,
    "seq2seq": compute_metrics_summarization,
    "multilabel": compute_metrics_multilabel,
    "chatbot": None,
}


class AutoTrainer:
    """
    Main class of autotransformers. Fine-tune and evaluate several models on several datasets.

    Useful for performing benchmarking of different models on the same datasets. The behavior
    of `AutoTrainer` is mainly configured through `model_configs` and `dataset_configs`, which
    define the datasets and the models to be used.

    Parameters
    ----------
    model_configs: List[autotransformers.ModelConfig]
        Configurations for the models, instances of ModelConfig, each describing their
        names in the hub or local directory, the name to save the model, the dropout
        values to use, and a long etc.
    dataset_configs: List[autotransformers.DatasetConfig]
        Configurations for the datasets, instances of DatasetConfig, each describing
        how each dataset should be processed.
    metrics_dir: str
        Directory to save the metrics for the experiments, as returned by `autotransformers.ResultsGetter`.
    hp_search_mode: str
        Mode for hyperparameter search; possibilities are `optuna` or `fixed`. If `fixed`,
        no hyperparameter tuning is carried out.
    clean: bool
        Whether to clean checkpoints every 10 minutes to avoid using too much disk, by
        using autotransformers.CkptCleaner. Best model checkpoint is also saved when unuseful
        checkpoints are deleted.
    metrics_cleaner: str
        Path to the folder where the metrics of the checkpoint cleaner should be stored.
        These metrics are used to decide which checkpoints should be removed. Note: if the
        experiment fails for some reason, and you re-launch it, please remove this folder
        before doing so. Otherwise there will probably be an error, as the checkpoint cleaner
        will use metrics from past experiments, not the running one, so there will be incorrect
        checkpoint removals.
    use_auth_token: bool
        Whether to use auth token to load datasets and models.
    skip_mixes: List[autotransformers.SkipMix]
        List of SkipMix instances with combinations of datasets and models that must be skipped.
    """

    def __init__(
        self,
        model_configs: List,
        dataset_configs: List,
        metrics_dir: str = "tmp_experiments_metrics",
        hp_search_mode: str = "optuna",
        clean: bool = True,
        metrics_cleaner: str = "tmp_metrics_cleaner",
        use_auth_token: bool = False,
        skip_mixes: List = None,
    ):
        self.model_configs = model_configs
        self.dataset_configs = dataset_configs
        self.metrics_dir = metrics_dir
        self.hp_search_mode = hp_search_mode
        self.metrics_cleaner = metrics_cleaner
        self.clean = clean
        self.use_auth_token = use_auth_token
        os.makedirs(self.metrics_dir, exist_ok=True)
        self.use_auth_token = use_auth_token
        self.skip_mixes = skip_mixes

    def __call__(
        self,
    ):
        """
        Use `train_with_fixed_params` or `optuna_hp_search` to carry out hyperparameter search defined in init.

        Check the documentation of those methods for more information.
        """
        if self.hp_search_mode == "optuna":
            return self.optuna_hp_search()
        elif self.hp_search_mode == "fixed":
            return self.train_with_fixed_params()

    def train_with_fixed_params(
        self,
    ):
        """
        Train without hyperparameter search, with a fixed set of params.

        The default parameters are defined in the fixed_train_args of DatasetConfig.
        However, we can use ModelConfig.overwrite_training_args to change this,
        by passing a dictionary with the new parameters that we want to use for a model.
        """
        all_results = {}
        for dataset_config in tqdm(
            self.dataset_configs, desc="Iterating over datasets..."
        ):
            for model_config in tqdm(
                self.model_configs,
                desc=f"Trying models on dataset {dataset_config.dataset_name}",
            ):
                if self.skip_mixes is not None:
                    if any(
                        [
                            skip_mix.dataset_name == dataset_config.alias
                            and skip_mix.model_name == model_config.save_name
                            for skip_mix in self.skip_mixes
                        ]
                    ):
                        continue
                transformers_manager = HFTransformersManager(
                    model_config, dataset_config, use_auth_token=self.use_auth_token
                )
                datasets_manager = HFDatasetsManager(dataset_config, model_config)

                try:
                    self.tokenizer = transformers_manager.load_tokenizer()
                    model_config, dataset_config = self._adapt_objects_summarization(
                        model_config, dataset_config
                    )
                    dataset, tag2id = datasets_manager.get_dataset_and_tag2id(
                        deepcopy(self.tokenizer)
                    )
                    data_collator = transformers_manager.load_data_collator(
                        self.tokenizer
                    )
                    if len(model_config.dropout_vals) == 0:
                        model_config.dropout_vals = [0.0]
                    config = transformers_manager.load_config(
                        tag2id, model_config.dropout_vals[0]
                    )
                    output_dir = joinpaths(
                        model_config.save_dir,
                        f"fixedparams_{model_config.save_name}-{dataset_config.alias}",
                    )
                    args = transformers_manager.load_train_args(output_dir)
                    model_cls = transformers_manager.get_model_cls()
                    model_init = transformers_manager.load_model_init(
                        model_cls, config, self.tokenizer
                    )
                    compute_metrics_func = self._get_compute_metrics(dataset_config)
                    self.trainer = transformers_manager.load_trainer(
                        dataset,
                        self.tokenizer,
                        args,
                        model_init,
                        data_collator,
                        compute_metrics_func,
                        config,
                    )
                    test_results = self.train_one_model_fixed_params(
                        model_config,
                        dataset_config,
                        compute_metrics_func,
                        dataset["test"],
                    )
                    test_results = fix_eval_results_dict(test_results)
                except Exception as e:
                    print(f"Error occurred: {e}; don't worry, we skip this model.")
                    trace_error = "".join(
                        traceback.format_exception(e, value=e, tb=e.__traceback__)
                    )  # etype=type(e), # NOTE: in new version throws error
                    test_results = {"error": trace_error}
                    _save_metrics(
                        test_results,
                        model_config.save_name,
                        dataset_config.alias,
                        self.metrics_dir,
                    )
                all_results[model_config.save_name.replace("/", "-")] = test_results
        return all_results

    def optuna_hp_search(
        self,
    ):
        """
        Carry out hyperparameter search with Optuna.

        Use `model_configs` and `dataset_configs` passed in init. Iterate over
        each dataset, and then over each model, with hyperparameter tuning.
        Metrics over the test dataset are gathered and then saved
        in the `metrics_dir` specified in init for each of those models, for later comparison.

        Returns
        -------
        all_results: Dict
            Dictionary with results from the experiments.
        """
        all_results = {}
        for dataset_config in tqdm(
            self.dataset_configs, desc="Iterating over datasets..."
        ):
            for model_config in tqdm(
                self.model_configs,
                desc=f"Trying models on dataset {dataset_config.dataset_name}",
            ):
                if self.skip_mixes is not None:
                    if any(
                        [
                            skip_mix.dataset_name == dataset_config.alias
                            and skip_mix.model_name == model_config.save_name
                            for skip_mix in self.skip_mixes
                        ]
                    ):
                        continue
                if len(model_config.dropout_vals) == 0:
                    model_config.dropout_vals = [0.0]
                transformers_manager = HFTransformersManager(
                    model_config, dataset_config, use_auth_token=self.use_auth_token
                )
                datasets_manager = HFDatasetsManager(dataset_config, model_config)
                self.tokenizer = transformers_manager.load_tokenizer()
                model_config, dataset_config = self._adapt_objects_summarization(
                    model_config, dataset_config
                )
                dataset, tag2id = datasets_manager.get_dataset_and_tag2id(
                    deepcopy(self.tokenizer)
                )
                data_collator = transformers_manager.load_data_collator(self.tokenizer)
                config = transformers_manager.load_config(tag2id)

                output_dir = joinpaths(
                    model_config.save_dir,
                    f"best_optuna_{model_config.save_name}-{dataset_config.alias}",
                )
                args = transformers_manager.load_train_args(output_dir)
                model_cls = transformers_manager.get_model_cls()

                model_init = transformers_manager.load_model_init(
                    model_cls, config, self.tokenizer
                )

                compute_metrics_func = self._get_compute_metrics(dataset_config)
                self.trainer = transformers_manager.load_trainer(
                    dataset,
                    self.tokenizer,
                    args,
                    model_init,
                    data_collator,
                    compute_metrics_func,
                    config,
                )

                def compute_objective(metrics):
                    return metrics[dataset_config.metric_optimize]

                test_results = self.train_one_model_optuna(
                    model_config,
                    dataset_config,
                    compute_objective,
                    compute_metrics_func,
                    output_dir,
                    dataset["test"],
                )
                test_results = fix_eval_results_dict(test_results)
                all_results[model_config.save_name.replace("/", "-")] = test_results
        return all_results

    def train_one_model_fixed_params(
        self, model_config, dataset_config, compute_metrics_func, test_dataset
    ):
        """
        Train one model with fixed params in one dataset, without tuning parameters.

        Parameters
        ----------
        model_config: autotransformers.ModelConfig
            Configuration for the model.
        dataset_config: autotransformers.DatasetConfig,
            Configuration for the dataset.
        compute_metrics_func: Any
            Function to compute metrics.
        test_dataset: datasets.Dataset
            Test dataset to get metrics on.

        Returns
        -------
        test_results: Dict
            Dictionary with results over the test set after training with fixed params.
        """
        if not model_config.only_test:
            self.trainer.train()

        test_results = self._get_test_results(
            dataset_config, compute_metrics_func, model_config, test_dataset
        )
        _save_metrics(
            test_results,
            model_config.save_name,
            dataset_config.alias,
            self.metrics_dir,
        )
        if model_config.push_to_hub and model_config.hf_hub_username is not None:
            self.trainer.push_to_hub(
                f"{model_config.hf_hub_username}/{model_config.save_name}",
                private=True,
            )
        test_results["model_name"] = model_config.save_name
        test_results["dataset_name"] = dataset_config.alias
        return test_results

    def train_one_model_optuna(
        self,
        model_config,
        dataset_config,
        compute_objective,
        compute_metrics_func,
        output_dir,
        test_dataset,
    ):
        """
        Train one model in one dataset, with hyperparameter tuning, using Optuna.

        Load a checkpoint cleaner in the background to clean bad performing checkpoints
        every 10 minutes, also saving the best performing checkpoint. Then, carry out
        hyperparameter search and, if configured (see `DatasetConfig`), retrain at end
        with the best hyperparameters again. After that, results on the test set are
        obtained. For that, `ResultsGetter` is used for dataset processing, prediction
        and metrics gathering. If desired, the user may change the behavior
        of this part by creating a custom `ResultsGetter` overriding the desired
        methods, and passing it to `DatasetConfig` as a `custom_results_getter`.
        Metrics are saved in json or txt format, and, if configured, the model
        is pushed to the hub.

        Parameters
        ----------
        model_config: autotransformers.ModelConfig
            Configuration for the model.
        dataset_config: autotransformers.DatasetConfig,
            Configuration for the dataset.
        compute_objective: Any
            Function to return the computed metric objective.
        compute_metrics_func: Any
            Function to compute metrics.
        output_dir: str
            Directory where the model is saved.
        test_dataset: datasets.Dataset
            Test dataset to get metrics on.

        Returns
        -------
        test_results: Dict
            Dictionary with the results in the test set.
        """
        if not model_config.do_nothing:
            if not model_config.only_test:
                scheduler = BackgroundScheduler()
                cleaner_callable = self._create_clean_job(
                    output_dir,
                    model_config.save_dir,
                    mode=(
                        "max"
                        if dataset_config.direction_optimize == "maximize"
                        else "min"
                    ),
                    metrics_save_dir=self.metrics_cleaner,
                    modelname=f"{model_config.save_name}-{dataset_config.alias}",
                )
                scheduler.add_job(cleaner_callable, "interval", seconds=600)
                scheduler.start()
                if not model_config.resume_from_checkpoint:
                    best_run = self.trainer.hyperparameter_search(
                        direction=dataset_config.direction_optimize,
                        hp_space=model_config.hp_space,
                        n_trials=model_config.n_trials,
                        compute_objective=compute_objective,
                        sampler=(
                            TPESampler(
                                seed=dataset_config.seed,
                                n_startup_trials=model_config.random_init_trials,
                            )
                            if not model_config.optuna_sampler
                            else model_config.optuna_sampler
                        ),
                    )
                    if dataset_config.retrain_at_end:
                        for n, v in best_run.hyperparameters.items():
                            setattr(self.trainer.args, n, v)
                        self.trainer.train()
                else:
                    self.trainer.train(model_config.name)

            test_results = self._get_test_results(
                dataset_config, compute_metrics_func, model_config, test_dataset
            )
            _save_metrics(
                test_results,
                model_config.save_name,
                dataset_config.alias,
                self.metrics_dir,
            )
            if model_config.push_to_hub and model_config.hf_hub_username is not None:
                self.trainer.push_to_hub(
                    f"{model_config.hf_hub_username}/{model_config.save_name}",
                    private=True,
                )
            if not model_config.only_test:
                scheduler.shutdown()
                cleaner_callable(skip_last=False)
            test_results["model_name"] = model_config.save_name
            test_results["dataset_name"] = dataset_config.alias
            return test_results
        else:
            return {
                "Do nothing": True,
                "model_name": model_config.save_name,
                "dataset_name": dataset_config.alias,
            }

    def _get_test_results(
        self, dataset_config, compute_metrics_func, model_config, test_dataset
    ):
        """
        Get results for the test set. Metrics vary depending on the task.

        Use `ResultsGetter` for dataset processing, obtaining the predictions
        and getting the metrics. If desired, the user may change the behavior
        of this part by creating a custom `ResultsGetter` overriding the desired
        methods.

        Parameters
        ----------
        dataset_config: autotransformers.DatasetConfig,
            Configuration for the dataset.
        compute_metrics_func: Any
            Function to compute metrics.
        model_config: autotransformers.ModelConfig
            Configuration for the model.
        test_dataset: datasets.Dataset
            Test dataset to get metrics on.

        Returns
        -------
        test_results: Dict
            Dictionary with the results in the test set.
        """
        if model_config.custom_results_getter is None:
            results_getter = ResultsGetter(
                dataset_config, model_config, compute_metrics_func
            )
        else:
            results_getter = model_config.custom_results_getter(
                dataset_config, model_config, compute_metrics_func
            )
        if dataset_config.task != "chatbot":
            test_results = results_getter(self.trainer, test_dataset)
        else:
            test_results = self.trainer.evaluate(test_dataset, metric_key_prefix="test")
        return test_results

    def _create_clean_job(
        self,
        output_dir,
        dataset_folder,
        mode,
        metrics_save_dir,
        modelname,
        try_mode=False,
    ):
        """
        Create a job to schedule cleaning process with CkptCleaner.

        Initialize a checkpoint cleaner with class CkptCleaner,
        with parameters passed in this function call. This callable class
        is used as a job to be scheduled, so that checkpoints are cleaned
        every 10 minutes.

        Parameters
        ----------
        output_dir: str
            Directory where models are being saved.
        dataset_folder: str
            The name of the dataset models.
        mode: str
            max or min are allowed.
        metrics_save_dir: str
            Directory to save metrics.
        modelname: str
            Name of the current model.
        try_mode: bool
            Default is False. This is to test the checkpoint cleaner without removing checkpoints.

        Returns
        -------
        ckpt_cleaner: CkptCleaner
            Instance of CkptCleaner to clean the checkpoints for the current model.
        """
        ckpt_cleaner = CkptCleaner(
            current_folder_clean=output_dir,
            current_dataset_folder=dataset_folder,
            metrics_save_dir=metrics_save_dir,
            modelname=modelname,
            mode=mode,
            try_mode=try_mode,
        )
        return ckpt_cleaner

    def _adapt_tokenizer_summarization(
        self,
    ):
        """Add bos and eos tokens to the tokenizer for summarization, in EncoderDecoder models."""
        self.tokenizer.bos_token = self.tokenizer.cls_token
        self.tokenizer.eos_token = self.tokenizer.sep_token

    def _get_compute_metrics(self, dataset_config):
        """
        Get the function to compute metrics with.

        Parameters
        ----------
        dataset_config: autotransformers.DatasetConfig
            Configuration for the dataset.

        Returns
        -------
        compute_metrics_func: Any
            Function to compute metrics.
        """
        compute_metrics_func = (
            metric_func_map[dataset_config.task]
            if not dataset_config.custom_eval_func
            else dataset_config.custom_eval_func
        )
        if dataset_config.is_multilabel:
            compute_metrics_func = metric_func_map["multilabel"]
        return compute_metrics_func

    def _adapt_objects_summarization(self, model_config, dataset_config):
        """
        Adapt dataset config and model config, along with the tokenizer, for seq2seq tasks.

        Parameters
        ----------
        model_config: autotransformers.ModelConfig
            Configuration for the model.
        dataset_config: autotransformers.DatasetConfig
            Configuration for the dataset.

        Returns
        -------
        model_config: autotransformers.ModelConfig
            Adjusted configuration for the model.
        dataset_config: autotransformers.DatasetConfig
            Adjusted configuration for the dataset.
        """
        if dataset_config.task == "seq2seq":
            dataset_config.max_length_summary = model_config.max_length_summary
            if not model_config.model_cls_summarization:
                self._adapt_tokenizer_summarization()
        return model_config, dataset_config
