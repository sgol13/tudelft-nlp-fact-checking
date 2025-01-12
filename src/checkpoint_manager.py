import os
import torch
import re
import shutil

from datetime import datetime
from typing import Optional, Tuple, Dict, Any, List

from src.common import MODELS_PATH

ModelWeights = Dict[str, Any]

class CheckpointsManager:
    _LAST_CHECKPOINT_REGEX = re.compile(r"epoch_(0[1-9]|\d{2,})")
    _BEST_MODEL_REGEX = re.compile(r"best_model_(0[1-9]|\d{2,})")

    def __init__(self, model_name: str) -> None:
        self._model_name: str = model_name
        self._model_dir_path: str = os.path.join(MODELS_PATH, self._model_name)
        os.makedirs(self._model_dir_path, exist_ok=True)

        self._best_val_accuracy: float = 0.0

    def clear_model_dir(self) -> None:
        last_checkpoint_file = self._get_last_checkpoint_name()
        if last_checkpoint_file:
            self._remove_checkpoint(last_checkpoint_file)

        best_model_file = self._get_best_model_name()
        if best_model_file:
            self._remove_checkpoint(best_model_file)

    def load_last_checkpoint(self) -> Optional[Tuple[ModelWeights, int]]:
        filenames = self._get_filenames(self._LAST_CHECKPOINT_REGEX)
        if not filenames:
            return None

        assert len(filenames) <= 1, f"Multiple epoch checkpoints found: {filenames}"
        name = filenames[0]
        epoch = int(name.split('_')[1])
        return self._load_model(name), epoch

    def load_best_model(self) -> Optional[ModelWeights]:
        filenames = self._get_filenames(self._BEST_MODEL_REGEX)
        if not filenames:
            return None

        assert len(filenames) == 1, f"Multiple best model checkpoints found: {filenames}"
        return self._load_model(filenames[0])

    def load_model(self, name: str) -> Optional[ModelWeights]:
        return self._load_model(name)

    def save_checkpoint(self, checkpoint: ModelWeights, epoch: int, val_accuracy: float) -> None:
        self._update_last_checkpoint(checkpoint, epoch)

        if val_accuracy > self._best_val_accuracy:
            self._best_val_accuracy = val_accuracy
            self._update_best_model(epoch)

    def store_best_model(self) -> None:
        best_model_name = self._get_best_model_name()
        assert best_model_name, "No best model found"

        time_str = datetime.now().strftime("%d-%m_%H-%M")
        permament_name = f"{self._model_name.replace('/', '_')}_{time_str}"
        shutil.copy(self._abs_path(best_model_name), self._abs_path(permament_name))
        print(f"Stored best model as: {permament_name}")

    def _update_last_checkpoint(self, checkpoint: ModelWeights, epoch: int) -> None:
        previous_checkpoint_name = self._get_last_checkpoint_name()

        filename = f"epoch_{epoch:02}"
        torch.save(checkpoint, self._abs_path(filename))
        print(f"Saved checkpoint: {filename}")

        if previous_checkpoint_name:
            self._remove_checkpoint(previous_checkpoint_name)

    def _update_best_model(self, epoch: int) -> None:
        last_checkpoint_file = self._get_last_checkpoint_name()
        previous_best_model_file = self._get_best_model_name()

        filename = f"best_model_{epoch:02}"
        shutil.copy(self._abs_path(last_checkpoint_file), self._abs_path(filename))
        print(f"Saved best model: {filename}")

        if previous_best_model_file:
            self._remove_checkpoint(previous_best_model_file)

    def _load_model(self, name: str) -> Optional[ModelWeights]:
        print(f"Loading model: {name}")
        path = self._abs_path(name)
        return torch.load(path, weights_only=True)

    def _get_last_checkpoint_name(self) -> Optional[str]:
        filenames = self._get_filenames(self._LAST_CHECKPOINT_REGEX)
        if not filenames:
            return None

        assert len(filenames) == 1, f"Multiple epoch checkpoints found: {filenames}"
        return filenames[0]

    def _get_best_model_name(self) -> Optional[str]:
        filenames = self._get_filenames(self._BEST_MODEL_REGEX)
        if not filenames:
            return None

        assert len(filenames) == 1, f"Multiple best model checkpoints found: {filenames}"
        return filenames[0]

    def _get_filenames(self, regex_pattern: re.Pattern) -> List[str]:
        all_files = os.listdir(self._model_dir_path)
        matching_files = [f for f in all_files if regex_pattern.fullmatch(f)]
        return matching_files

    def _remove_checkpoint(self, name: str) -> None:
        path = self._abs_path(name)
        os.remove(path)
        print(f"Removed: {name}")

    def _abs_path(self, name: str) -> str:
        return os.path.join(self._model_dir_path, name)