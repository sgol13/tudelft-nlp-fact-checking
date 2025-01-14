import os
from typing import Optional, Dict, Union, List

import pandas as pd
import torch
import numpy as np
import random

from sklearn.metrics import accuracy_score
from torch import nn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm.auto import tqdm

from src.common import QTDataset, OUTPUT_PATH, cwd_relative_path
from src.early_stopping import EarlyStopping
from src.checkpoint_manager import CheckpointsManager
from src.f1_evaluation import calculate_f1_scores
from src.quantemp_processor import qt_veracity_label_encoder
from src.stats_manager import StatsManager


class ClassificationTraining:
    def __init__(self,
                 *,
                 model_name: str,
                 train_dataset: torch.utils.data.TensorDataset,
                 val_dataset: torch.utils.data.TensorDataset,
                 model: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 loss_function: torch.nn.Module,
                 batch_size: int,
                 device: torch.device,
                 random_state: int,
                 ):
        self._model_name: str = model_name
        self._model: torch.nn.Module = model
        self._optimizer: torch.optim.Optimizer = optimizer
        self._loss_func: torch.nn.Module = loss_function
        self._device: torch.device = device

        self._train_dataloader: torch.utils.data.DataLoader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=batch_size)
        self._val_dataloader: torch.utils.data.DataLoader = DataLoader(val_dataset, sampler=SequentialSampler(val_dataset), batch_size=batch_size)

        random.seed(random_state)
        np.random.seed(random_state)
        torch.manual_seed(random_state)
        torch.cuda.manual_seed_all(random_state)

        self._checkpoint_manager: CheckpointsManager = CheckpointsManager(model_name)
        self._stats_manager: StatsManager = StatsManager(model_name)
        self._early_stopping: Optional[EarlyStopping] = None

        self._epoch: Optional[int] = None
        self._allow_new_training: bool = False

    def start_new_training(self) -> None:
        self._checkpoint_manager.clear_model_dir()
        self._stats_manager.clear()

        self._epoch = 1
        self._allow_new_training = True
        print("Starting new training from epoch 1")

    def resume_training(self) -> None:
        last_checkpoint = self._checkpoint_manager.load_last_checkpoint()
        assert last_checkpoint, "No checkpoint found."
        self._stats_manager.load()

        weights, last_epoch = last_checkpoint
        self._epoch = last_epoch + 1
        self._allow_new_training = True
        print(f"Resuming training from epoch {self._epoch}")

    def load_best_model(self) -> int:
        best_model = self._checkpoint_manager.load_best_model()
        assert best_model, "No best model found."
        self._stats_manager.load()

        weights, best_epoch = best_model
        self._model.load_state_dict(weights)
        self._allow_new_training = False
        return best_epoch

    def load_model(self, name: str) -> None:
        weights = self._checkpoint_manager.load_model(name)
        assert weights, f"Model {name} not found."

        self._model.load_state_dict(weights)
        self._allow_new_training = False

    def store_best_model(self) -> None:
        self._checkpoint_manager.store_best_model()

    def train(self, epochs: int, patience: int=None) -> None:
        if patience:
            self._early_stopping = EarlyStopping(patience)

        assert self._allow_new_training, (
            "Incorrect state: training is not allowed. Start new training or resume from the last checkpoint.")

        for _ in range(epochs):
            print(f"\nEPOCH {self._epoch}")

            train_stats = self._train_epoch()
            val_stats = self._validate()

            self._stats_manager.update({"epoch": self._epoch} | train_stats | val_stats)
            self._stats_manager.print_current_epoch()

            self._checkpoint_manager.save_checkpoint(self._model.state_dict(), self._epoch, val_stats["val_accuracy"])

            self._epoch += 1
            if self._early_stopping and self._early_stopping(val_stats["val_loss"]):
                break

            self._stats_manager.plot()

    def plot_stats(self) -> None:
        self._stats_manager.plot()

    def evaluate_best_model(self, test_claims: QTDataset, test_dataset: torch.utils.data.TensorDataset) -> None:
        epoch = self.load_best_model()

        predictions = self._predict(test_dataset)
        test_accuracy = self._accuracy(test_dataset.tensors[2], torch.tensor(predictions))

        stats = self._stats_manager.get_epoch_stats(epoch)
        f1_scores: List[float] = calculate_f1_scores(test_claims, predictions)

        # save output stats
        table_row = [epoch, stats['train_accuracy'], stats['val_accuracy'], test_accuracy] + f1_scores
        with open(os.path.join(OUTPUT_PATH, f'{self._model_name}.txt'), 'w') as file:
            file.write(f"{table_row[0]} ")
            file.write(" ".join(f"{x:.2f}" for x in table_row[1:]))

        # save predictions
        df = pd.DataFrame({
            "claim": [claim["claim"] for claim in test_claims],
            "verdict": qt_veracity_label_encoder.inverse_transform(predictions)
        })
        abs_path = os.path.join(OUTPUT_PATH, f'{self._model_name}.csv')
        df.to_csv(abs_path, index=False)
        print(f"Saved to {cwd_relative_path(abs_path)}/txt")


    def _train_epoch(self) -> Dict[str, Union[float, List[float]]]:
        self._model.train()

        total_train_accuracy = 0
        total_train_loss = 0
        train_loss_batches = []

        for b_input_tokens, b_input_mask, b_labels in tqdm(self._train_dataloader, desc='train'):
            b_input_tokens = b_input_tokens.to(self._device)
            b_input_mask = b_input_mask.to(self._device)
            b_labels = b_labels.to(self._device)

            self._optimizer.zero_grad()

            b_logits = self._model(b_input_tokens, b_input_mask)

            loss = self._loss_func(b_logits, b_labels)
            loss.backward()
            self._optimizer.step()

            total_train_accuracy += self._accuracy(b_logits.cpu(), b_labels.cpu())
            total_train_loss += loss.item()
            train_loss_batches.append(loss.item())

        return {
            "train_accuracy": total_train_accuracy / len(self._train_dataloader),
            "train_loss": total_train_loss / len(self._train_dataloader),
            "train_loss_batches": train_loss_batches
        }

    def _validate(self) -> Dict[str, float]:
        self._model.eval()

        total_val_accuracy = 0
        total_val_loss = 0

        for b_input_tokens, b_input_mask, b_labels in tqdm(self._val_dataloader, desc='val'):
            b_input_tokens = b_input_tokens.to(self._device)
            b_input_mask = b_input_mask.to(self._device)
            b_labels = b_labels.to(self._device)

            with torch.no_grad():
                b_logits = self._model(b_input_tokens, b_input_mask)

            loss = self._loss_func(b_logits, b_labels)

            total_val_accuracy += self._accuracy(b_logits.cpu(), b_labels.cpu())
            total_val_loss += loss.item()

        return {
            "val_accuracy": total_val_accuracy / len(self._val_dataloader),
            "val_loss": total_val_loss / len(self._val_dataloader),
        }

    def _predict(self, dataset: torch.utils.data.TensorDataset) -> List[int]:
        self._model.eval()

        dataloader = DataLoader(dataset, sampler=SequentialSampler(dataset), batch_size=64, drop_last=False)

        predictions = []
        for b_input_tokens, b_input_mask, _ in tqdm(dataloader):
            b_input_tokens = b_input_tokens.to(self._device)
            b_input_mask = b_input_mask.to(self._device)

            with torch.no_grad():
                b_logits = self._model(b_input_tokens, b_input_mask)

            b_predictions = torch.argmax(b_logits, dim=1).flatten().cpu().tolist()
            predictions.extend(b_predictions)

        return predictions

    @staticmethod
    def _accuracy(output: torch.Tensor, labels: torch.Tensor) -> float:
        if len(output.shape) == 2:
            output = torch.argmax(output, dim=1).flatten()

        labels_flat = labels.flatten()
        return torch.sum(output == labels_flat).item() / len(labels_flat)
