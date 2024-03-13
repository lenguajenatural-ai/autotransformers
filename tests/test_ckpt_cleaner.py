from autotransformers.ckpt_cleaner import CkptCleaner
from autotransformers.utils import _save_json, joinpaths
import os


def _create_trainer_state(best_model_checkpoint, best_metric):
    """Create fake trainer_state dict for checkpoint cleaner to use."""
    trainer_state = {
        "best_model_checkpoint": best_model_checkpoint,
        "best_metric": best_metric,
    }
    return trainer_state


def _get_total_files_directory(directory):
    """Get total count of files in a directory."""
    return len([x[0] for x in os.walk(directory)])


def _create_current_folder_clean(current_folder_clean):
    """Create a folder with fake runs."""
    numruns = 5
    num_ckpts = 10
    os.makedirs(current_folder_clean, exist_ok=True)
    for run in range(numruns):
        trainer_state = _create_trainer_state(
            joinpaths(
                current_folder_clean, f"run-{numruns-1}", f"checkpoint-{num_ckpts-1}"
            ),
            200,
        )
        for ckpt in range(num_ckpts):
            path = joinpaths(current_folder_clean, f"run-{run}", f"checkpoint-{ckpt}")
            os.makedirs(path, exist_ok=True)
            _save_json(trainer_state, joinpaths(path, "trainer_state.json"))


def test_ckpt_cleaner():
    """Test that CkptCleaner removes the correct folders."""
    folder_clean = "tmp_clean_folder"
    _create_current_folder_clean(folder_clean)
    prev_files_count = _get_total_files_directory(folder_clean)
    dataset_folder = "tmp_dataset_folder"
    os.makedirs(dataset_folder, exist_ok=True)
    ckpt_cleaner = CkptCleaner(
        current_folder_clean=folder_clean,
        current_dataset_folder=dataset_folder,
        modelname="test_modelname",
        metrics_save_dir="test_metricsdir",
    )
    ckpt_cleaner()
    ckpt_cleaner()
    ckpt_cleaner()
    ckpt_cleaner()
    post_files_count = _get_total_files_directory(folder_clean)
    assert (
        post_files_count < prev_files_count
    ), f"Count for post is {post_files_count}, for prev is {prev_files_count}"
    assert os.path.exists(
        joinpaths(dataset_folder, "best_ckpt_test_modelname")
    ), "Path for best ckpt should exist but it does not."
