import os
import torch
import re
import shutil

from typing import Optional, Tuple, Dict, Any, List
from torch import nn

from src.common import MODELS_PATH, cwd_relative_path


class TrainingManager:
    _CHECKPOINT_REGEX = re.compile(r"(0[1-9]|\d{2,})_acc_\d{3}")

    def __init__(self, model_name: str, keep_last_checkpoints: int=1) -> None:
        self._model_name: str = model_name
        self._keep_last_checkpoints: int = keep_last_checkpoints
        self._model_path: str = os.path.join(MODELS_PATH, self._model_name)
        self._epoch: Optional[int] = None

    @property
    def epoch_num(self) -> int:
        return self._epoch

    def start_training(self, model: nn.Module) -> None:
        last_checkpoint = self._load_last_checkpoint()

        if last_checkpoint:
            model_weights, epoch = last_checkpoint
            self._epoch = epoch + 1
            model.load_state_dict(model_weights)

            print(f"Resuming training from epoch {self._epoch}")
        else:
            self._epoch = 1
            os.makedirs(self._model_path, exist_ok=True)
            print("Starting new training")

    def step(self, model: nn.Module, val_accuracy: float, save_checkpoint: bool = True) -> None:
        assert self._epoch, "Training not started"

        if save_checkpoint:
            self._save_checkpoint(model, val_accuracy)

        self._epoch += 1

    def finalize(self, model: nn.Module) -> None:
        filename = self._model_name.replace('/', '_')
        dest_abs_path = os.path.join(self._model_path, filename)

        best_checkpoint = self._get_best_checkpoint()
        assert best_checkpoint, "Model not trained (no checkpoints found)"
        checkpoint, epoch = best_checkpoint

        best_checkpoint_abs_path = os.path.join(self._model_path, checkpoint)
        shutil.move(best_checkpoint_abs_path, dest_abs_path)

        self._remove_checkpoints(self._get_list_of_checkpoints())
        print(f"Saved best model (epoch {epoch}) to: {cwd_relative_path(dest_abs_path)}")

    def _save_checkpoint(self, model: nn.Module, val_accuracy: float) -> None:
        epoch_str = f"{self._epoch:02}"
        val_accuracy_str = f"{val_accuracy:.3f}".split('.')[1]
        filename = f'{epoch_str}_acc_{val_accuracy_str}'
        assert self._CHECKPOINT_REGEX.fullmatch(filename), f"Invalid filename: {filename}"

        abs_path = os.path.join(self._model_path, filename)
        torch.save(model.state_dict(), abs_path)
        print(f"Saved checkpoint to {cwd_relative_path(abs_path)}")
        self._remove_old_checkpoints()

    def _get_list_of_checkpoints(self) -> List[str]:
        if not os.path.exists(self._model_path):
            return []

        files = os.listdir(self._model_path)
        checkpoint_files = [f for f in files if self._CHECKPOINT_REGEX.fullmatch(f)]

        return checkpoint_files if len(checkpoint_files) > 0 else []

    def _load_last_checkpoint(self) -> Optional[Tuple[Dict[str, Any], int]]:
        checkpoint_files = self._get_list_of_checkpoints()
        if not checkpoint_files:
            return None

        last_checkpoint = max(checkpoint_files)
        abs_path = os.path.join(self._model_path, last_checkpoint)
        model_weights = torch.load(abs_path, weights_only=True)
        epoch = int(last_checkpoint.split('_')[0])
        return model_weights, epoch

    def _get_best_checkpoint(self) -> Optional[Tuple[str, int]]:
        checkpoint_files = self._get_list_of_checkpoints()
        if not checkpoint_files:
            return None

        best_checkpoint_file = max(checkpoint_files, key=lambda x: x.split('_')[1])
        epoch = int(best_checkpoint_file.split('_')[0])
        return best_checkpoint_file, epoch

    def _remove_old_checkpoints(self) -> None:
        checkpoint_files = self._get_list_of_checkpoints()
        if not checkpoint_files:
            return

        checkpoints_to_remove = sorted(checkpoint_files)[:-self._keep_last_checkpoints]
        self._remove_checkpoints(checkpoints_to_remove)

    def _remove_checkpoints(self, checkpoints_to_remove: List[str]) -> None:
        for checkpoint in checkpoints_to_remove:
            abs_path = os.path.join(self._model_path, checkpoint)
            os.remove(abs_path)
            print(f"Removed checkpoint: {cwd_relative_path(abs_path)}")


