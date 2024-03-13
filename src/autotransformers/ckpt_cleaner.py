import shutil
import os
from tqdm import tqdm
from .utils import _load_json, _save_json
from typing import List, Dict


class CkptCleaner:
    """
    Clean all checkpoints that are no longer useful.

    Use a metrics dictionary to check the results of all runs of a model
    for a dataset, then sort these metrics to decide which checkpoints are
    removable and which are among the four best. When called, only those
    are kept, and all the other checkpoints are removed. This enables the
    user to effectively use their computer resources, so there is no need to
    worry about the disk usage, which is a typical concern when running multiple
    transformer models.
    """

    def __init__(
        self,
        current_folder_clean: str,
        current_dataset_folder: str,
        metrics_save_dir: str,
        modelname: str,
        mode: str = "max",
        try_mode: bool = False,
    ):
        self.current_folder_clean = current_folder_clean
        self.current_dataset_folder = current_dataset_folder
        self.modelname = modelname
        self.metrics_save_dir = metrics_save_dir
        self.mode = mode
        self.try_mode = try_mode
        self.last_saved_ckpt = ""

        os.makedirs(self.metrics_save_dir, exist_ok=True)

    def __call__(self, skip_last: bool = True):
        """
        Check the metrics folder and remove checkpoints of models not performing well (all except 4 best).

        Called by a scheduler, to eventually remove the undesired checkpoints.
        """
        metricsname = f"{self.modelname}.json"
        if metricsname in os.listdir(self.metrics_save_dir):
            metrics = _load_json(os.path.join(self.metrics_save_dir, metricsname))
        else:
            metrics = {}
        lista = os.listdir(self.current_folder_clean)
        runs_dirs = [folder for folder in lista if "run-" in folder]
        runs_dirs = list(
            sorted(
                runs_dirs,
                key=lambda x: int(x.split("-")[-1]),
            )
        )
        if skip_last:
            runs_dirs = runs_dirs[:-2]
        for run_dir in tqdm(runs_dirs):
            checkpoint_dirs = [
                folder
                for folder in os.listdir(
                    os.path.join(self.current_folder_clean, run_dir)
                )
                if "checkpoint-" in folder
            ]
            if len(checkpoint_dirs) > 0:
                checkpoint_dirs = list(
                    sorted(
                        checkpoint_dirs,
                        key=lambda x: int(x.split("-")[-1]),
                    )
                )
                last = checkpoint_dirs[-1]
                trainer_state = _load_json(
                    os.path.join(
                        self.current_folder_clean, run_dir, last, "trainer_state.json"
                    )
                )
                best_model_checkpoint = trainer_state["best_model_checkpoint"]
                if best_model_checkpoint not in metrics:
                    metrics[best_model_checkpoint] = float(trainer_state["best_metric"])
                    _save_json(
                        metrics, os.path.join(self.metrics_save_dir, metricsname)
                    )
                checkpoint_dirs = [
                    os.path.join(self.current_folder_clean, run_dir, checkpoint)
                    for checkpoint in checkpoint_dirs
                ]
                bname = self.get_best_name(metrics)
                checkpoint_dirs = [
                    ckpt
                    for ckpt in checkpoint_dirs
                    if ckpt
                    not in [self.fix_dir(best_model_checkpoint), self.fix_dir(bname)]
                ]
                if bname != self.last_saved_ckpt:
                    print("saving new best checkpoint...")
                    # don't need to receive the target.
                    _ = self.save_best(bname)
                    self.last_saved_ckpt = bname
                else:
                    print("will save nothing as best model has not changed...")
                assert (
                    bname not in checkpoint_dirs
                ), "best_model_checkpoint should not be in checkpoint dirs."
                assert (
                    best_model_checkpoint not in checkpoint_dirs
                ), "best_model_checkpoint should not be in checkpoint dirs."
                self.remove_dirs(checkpoint_dirs)
        sorted_metrics = sorted(metrics, key=metrics.get, reverse=self.mode == "max")
        if len(sorted_metrics) > 0:
            print(
                f"For model {self.current_folder_clean} the best metric is {metrics[sorted_metrics[0]]} and the worst is {metrics[sorted_metrics[-1]]}"
            )
            best_ckpt = sorted_metrics[0]
            _ = self.save_best(best_ckpt)
            if len(sorted_metrics) > 4:
                dirs_to_remove = sorted_metrics[
                    4:
                ]  # REMOVE ALL BUT BEST 4 CHECKPOINTS.
                self.remove_dirs(dirs_to_remove)

    def get_best_name(self, metrics: Dict):
        """
        Get the path of the best performing model.

        Parameters
        ----------
        metrics: Dict
            Metrics of all models in a dictionary.

        Returns
        -------
        best: str
            Path to the best performing model.
        """
        sorted_metrics = sorted(metrics, key=metrics.get, reverse=self.mode == "max")
        best = sorted_metrics[0]
        return best

    def save_best(
        self,
        best_model: str,
    ):
        """
        Save best model.

        Parameters
        ----------
        best_model: str
            Path of the best performing model.

        Returns
        -------
        target: str
            Complete path to the target directory where the best model has been copied.
        """
        target = os.path.join(
            self.current_dataset_folder, f"best_ckpt_{self.modelname}"
        )
        if os.path.exists(target) and os.path.exists(best_model):
            if not self.try_mode:
                shutil.rmtree(target)
            else:
                print(
                    f"Al estar en try mode se hace como que se elimina el directorio {target}"
                )
        print(f"Copiando {best_model} a {target}")
        if os.path.exists(best_model):
            if not self.try_mode:
                shutil.copytree(
                    best_model, target, ignore=shutil.ignore_patterns("*optimizer*")
                )
        if not self.try_mode:
            assert os.path.exists(target), "TARGET DOES NOT EXIST..."
        return target

    def fix_dir(self, dir: str):
        """
        Fix directory path for windows file systems.

        Parameters
        ----------
        dir: str
            Directory to fix.

        Returns
        -------
        dir: str
            Fixed directory.
        """
        return dir.replace("D:\\", "D:")

    def remove_dirs(self, checkpoint_dirs: List):
        """
        Delete checkpoint directories.

        Parameters
        ----------
        checkpoint_dirs: List
            List with the checkpoint directories to remove.
        """
        for ckpt_dir in tqdm(checkpoint_dirs, desc="deleting models..."):
            try:
                if not self.try_mode:
                    shutil.rmtree(ckpt_dir)
                else:
                    print(
                        f"Al estar en try mode se hace como que se elimina el directorio {ckpt_dir}"
                    )
            except FileNotFoundError:
                print(f"Se intentó eliminar el directorio {ckpt_dir} y no se pudo")
