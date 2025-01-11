import os
import torch
import re

from typing import Optional, Tuple, Dict, Any
from torch import nn

from src.common import MODELS_PATH, CHECKPOINTS_RELATIVE_PATH


class TrainingManager:
    _CHECKPOINT_REGEX = re.compile(r"(0[1-9]|\d{2,})_\d{3}")

    def __init__(self, model_name: str) -> None:
        self._model_name: str = model_name
        self._model_path: str = os.path.join(MODELS_PATH, self._model_name)
        self._checkpoints_path: str = os.path.join(self._model_path, CHECKPOINTS_RELATIVE_PATH)
        self._epoch: Optional[int] = None

    @property
    def epochs(self) -> int:
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
            os.makedirs(self._checkpoints_path, exist_ok=True)
            print("Starting new training")

    def step(self, model: nn.Module, val_accuracy: float, save_checkpoint: bool = True) -> None:
        assert self._epoch, "Training not started"

        if save_checkpoint:
            self._save_checkpoint(model, val_accuracy)

        self._epoch += 1

    def save_final_model(self, model: nn.Module) -> None:
        filename = self._model_name.replace('/', '_')
        abs_path = os.path.join(self._model_path, filename)
        self._save_model(model, abs_path)

    def _save_checkpoint(self, model: nn.Module, val_accuracy: float) -> None:
        epoch_str = f"{self._epoch:02}"
        val_accuracy_str = f"{val_accuracy:.3f}".split('.')[1]
        filename = f'{epoch_str}_{val_accuracy_str}'
        assert self._CHECKPOINT_REGEX.fullmatch(filename), f"Invalid filename: {filename}"

        abs_path = os.path.join(self._checkpoints_path, filename)
        self._save_model(model, abs_path)

    @staticmethod
    def _save_model(model: nn.Module, abs_path: str) -> None:
        torch.save(model.state_dict(), abs_path)

    def _load_last_checkpoint(self) -> Optional[Tuple[Dict[str, Any], int]]:
        if not os.path.exists(self._checkpoints_path):
            return None

        checkpoint_files = self._get_list_of_checkpoints()
        if not checkpoint_files:
            return None

        last_checkpoint = max(checkpoint_files)
        abs_path = os.path.join(self._checkpoints_path, last_checkpoint)
        model_weights = torch.load(abs_path, weights_only=True)
        epoch = int(last_checkpoint.split('_')[0])
        return model_weights, epoch

    def _get_list_of_checkpoints(self):
        checkpoint_files = os.listdir(self._checkpoints_path)
        assert all(self._CHECKPOINT_REGEX.fullmatch(f) for f in checkpoint_files), \
            f"Invalid checkpoint filenames in {self._checkpoints_path}"
        return checkpoint_files
