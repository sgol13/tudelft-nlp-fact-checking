import json
import os
from typing import Union, Dict, List

from src.common import MODELS_PATH
from matplotlib import pyplot as plt

class StatsManager:
    _STATS_FILE = "stats.json"

    def __init__(self, model_name: str):
        self._stats_file_path = os.path.join(MODELS_PATH, model_name, self._STATS_FILE)
        self._stats = []

    def load(self):
        try:
            with open(self._stats_file_path, "r") as file:
                self._stats = json.load(file)
        except FileNotFoundError:
            self._stats = []
            print(f"{self._STATS_FILE} not found.")

    def clear(self):
        self._stats = []
        if os.path.exists(self._stats_file_path):
            os.remove(self._stats_file_path)

    def update(self, epoch_stats: Dict[str, Union[float, List[float]]]) -> None:
        self._stats.append(epoch_stats)
        self._save()

    def print_current_epoch(self):
        assert self._stats, "No stats for current epoch."
        epoch_stats = self._stats[-1]

        print("          loss  | accuracy")
        print(f"  train:  {epoch_stats['train_loss']:.3f} | {100*epoch_stats['train_accuracy']:.2f}")
        print(f"   eval:  {epoch_stats['val_loss']:.3f} | {100*epoch_stats['val_accuracy']:.2f}")

    def plot(self):
        is_first_epoch = len(self._stats) == 1

        first_tick = 0 if is_first_epoch else 1
        xticks = range(first_tick, max(10, len(self._stats)))

        epochs = [stat['epoch'] for stat in self._stats]
        train_accuracies = [stat['train_accuracy'] for stat in self._stats]
        val_accuracies = [stat['val_accuracy'] for stat in self._stats]
        train_losses = [stat['train_loss'] for stat in self._stats]
        val_losses = [stat['val_loss'] for stat in self._stats]

        fig, axs = plt.subplots(2, 1, figsize=(8, 5), constrained_layout=True)

        for stat in self._stats[first_tick:]:
            batch_losses = stat.get('train_loss_batches', [])
            if batch_losses:
                batch_epochs = [stat['epoch'] - 1 + i / len(batch_losses) for i in range(len(batch_losses))]
                axs[1].scatter(batch_epochs, batch_losses, color='blue', alpha=0.3, s=1)

        axs[0].plot(epochs, train_accuracies, label='train accuracy', marker='o')
        axs[0].plot(epochs, val_accuracies, label='val accuracy', marker='o')
        axs[0].set_ylabel('Accuracy', fontsize=14)
        axs[0].legend()
        axs[0].grid(True)
        axs[0].set_xticks(xticks)

        axs[1].plot(epochs, train_losses, label='train loss', marker='o')
        axs[1].plot(epochs, val_losses, label='val loss', marker='o')
        axs[1].set_ylabel('Loss', fontsize=14)
        axs[1].legend()
        axs[1].grid(True)
        axs[1].set_xticks(xticks)

        plt.show()

    def get_epoch_stats(self, epoch: int) -> Dict[str, Union[float, List[float]]]:
        return self._stats[epoch - 1]

    def _save(self):
        with open(self._stats_file_path, "w") as file:
            json.dump(self._stats, file, indent=4)



