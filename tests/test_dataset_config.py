from autotransformers import DatasetConfig


def test_dataset_config():
    """Test that dataset config save the correct parameters."""
    fixed_train_args = {
        "evaluation_strategy": "epoch",
        "num_train_epochs": 10,
        "do_train": True,
        "do_eval": False,
        "logging_strategy": "epoch",
        "save_strategy": "epoch",
        "save_total_limit": 10,
        "seed": 69,
        "bf16": True,
        "dataloader_num_workers": 16,
        "adam_epsilon": 1e-8,
        "adam_beta1": 0.9,
        "adam_beta2": 0.999,
        "group_by_length": True,
        "lr_scheduler_type": "linear",
        "learning_rate": 1e-4,
        "per_device_train_batch_size": 10,
        "per_device_eval_batch_size": 10,
        "gradient_accumulation_steps": 6,
        "warmup_ratio": 0.08,
    }

    dataset_config = {
        "seed": 44,
        "direction_optimize": "minimize",
        "metric_optimize": "eval_loss",
        "callbacks": [],
        "fixed_training_args": fixed_train_args,
        "dataset_name": "en_es_legal",
        "alias": "en_es_legal",
        "task": "summarization",
        "hf_load_kwargs": {"path": "avacaondata/wnli", "use_auth_token": False},
        "label_col": "target_es",
        "retrain_at_end": False,
        "num_proc": 12,
        "custom_eval_func": lambda p: p,
    }
    dataset_config = DatasetConfig(**dataset_config)
    assert (
        dataset_config.task == "summarization"
    ), f"Task should be summarization and is {dataset_config.task}"
    assert (
        dataset_config.label_col == "target_es"
    ), f"Label col should be target_es and is {dataset_config.label_col}"
    assert (
        dataset_config.fixed_training_args["evaluation_strategy"] == "epoch"
    ), "Evaluation strategy should be epoch."
    assert dataset_config.fixed_training_args["seed"] == 69, "Seed should be 69."
