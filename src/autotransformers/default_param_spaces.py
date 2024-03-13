def hp_space_base(trial):
    """Hyperparameter space in Optuna format for base-sized models (e.g. bert-base)."""
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 7e-5, log=True),
        "num_train_epochs": trial.suggest_categorical(
            "num_train_epochs", [3, 5, 7, 10, 15, 20, 30]
        ),
        "per_device_train_batch_size": trial.suggest_categorical(
            "per_device_train_batch_size", [8, 16]
        ),
        "per_device_eval_batch_size": trial.suggest_categorical(
            "per_device_eval_batch_size", [32]
        ),
        "gradient_accumulation_steps": trial.suggest_categorical(
            "gradient_accumulation_steps", [1, 2, 3, 4]
        ),
        "warmup_ratio": trial.suggest_float("warmup_ratio", 0.01, 0.10, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-10, 0.3, log=True),
        "adam_epsilon": trial.suggest_float("adam_epsilon", 1e-10, 1e-6, log=True),
    }


def hp_space_large(trial):
    """Hyperparameter space in Optuna format for large-sized models (e.g. bert-large)."""
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-6, 1e-4, log=True),
        "num_train_epochs": trial.suggest_categorical(
            "num_train_epochs", [3, 5, 7, 10, 15, 20, 30, 40, 50]
        ),
        "per_device_train_batch_size": trial.suggest_categorical(
            "per_device_train_batch_size", [4]
        ),
        "per_device_eval_batch_size": trial.suggest_categorical(
            "per_device_eval_batch_size", [16]
        ),
        "gradient_accumulation_steps": trial.suggest_categorical(
            "gradient_accumulation_steps", [4, 8, 12, 16]
        ),
        "warmup_ratio": trial.suggest_float("warmup_ratio", 0.01, 0.10, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-10, 0.3, log=True),
        "adam_epsilon": trial.suggest_float("adam_epsilon", 1e-10, 1e-6, log=True),
        "adam_beta2": trial.suggest_float("adam_beta2", 0.98, 0.999, log=True),
    }
