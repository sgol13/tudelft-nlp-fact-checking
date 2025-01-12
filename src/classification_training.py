from typing import Tuple

import torch
import numpy as np
import random

from torch import nn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm.auto import tqdm

from src.early_stopping import EarlyStopping
from src.checkpoint_manager import CheckpointsManager


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
                 early_stopping_patience: int,
                 device: torch.device,
                 random_state: int,
                 ):
        self._model_name: str = model_name
        self._model: torch.nn.Module = model
        self._optimizer: torch.optim.Optimizer = optimizer
        self._loss_func: torch.nn.Module = loss_function
        self._device: torch.device = device

        self._train_dataloader: torch.utils.data.DataLoader = DataLoader(
            train_dataset,
            sampler=RandomSampler(train_dataset),
            batch_size=batch_size
        )

        self._val_dataloader: torch.utils.data.DataLoader = DataLoader(
            val_dataset,
            sampler=SequentialSampler(val_dataset),
            batch_size=batch_size
        )

        random.seed(random_state)
        np.random.seed(random_state)
        torch.manual_seed(random_state)
        torch.cuda.manual_seed_all(random_state)

        self._checkpoint_manager = CheckpointsManager(model_name, keep_last_checkpoints=early_stopping_patience)
        self._checkpoint_manager.start_training(model)

        self._early_stopping: EarlyStopping = EarlyStopping(early_stopping_patience)

    def start_new_training(self) -> None:
        # start_new_training()
        pass

    def resume_training(self) -> None:
        # load_last_checkpoint()
        pass

    def select_final_model(self) -> None:
        # select_final_model()
        pass

    def load_final_model(self) -> None:
        # load_final_model()
        pass

    def train(self, epochs: int) -> None:
        for _ in range(epochs):
            print(f"\nEPOCH {self._checkpoint_manager.epoch_num}")

            avg_train_accuracy, avg_train_loss = self._train_epoch()
            avg_val_accuracy, avg_val_loss = self._evaluate()

            print(f"    train accuracy: {avg_train_accuracy:.3f}")
            print(f"     eval accuracy: {avg_val_accuracy:.3f}\n")
            print(f"    avg train loss: {avg_train_loss:.3f}")
            print(f"     avg eval loss: {avg_val_loss:.3f}\n")

            self._checkpoint_manager.step(self._model, avg_val_accuracy)

            if self._early_stopping(avg_val_loss):
                self.finalize()
                break

    def _train_epoch(self) -> Tuple[float, float]:
        self._model.train()

        total_train_accuracy = 0
        total_train_loss = 0

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

        avg_train_accuracy = total_train_accuracy / len(self._train_dataloader)
        avg_train_loss = total_train_loss / len(self._train_dataloader)

        return avg_train_accuracy, avg_train_loss


    def _evaluate(self) -> Tuple[float, float]:
        self._model.eval()

        total_eval_accuracy = 0
        total_eval_loss = 0

        for b_input_tokens, b_input_mask, b_labels in tqdm(self._val_dataloader, desc='eval'):
            b_input_tokens = b_input_tokens.to(self._device)
            b_input_mask = b_input_mask.to(self._device)
            b_labels = b_labels.to(self._device)

            with torch.no_grad():
                b_logits = self._model(b_input_tokens, b_input_mask)

            loss = self._loss_func(b_logits, b_labels)

            total_eval_accuracy += self._accuracy(b_logits.cpu(), b_labels.cpu())
            total_eval_loss += loss.item()

        avg_val_accuracy = total_eval_accuracy / len(self._val_dataloader)
        avg_val_loss = total_eval_loss / len(self._val_dataloader)

        return avg_val_accuracy, avg_val_loss

    @staticmethod
    def _accuracy(output: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        pred_flat = torch.argmax(output, dim=1).flatten()
        labels_flat = labels.flatten()
        return torch.sum(pred_flat == labels_flat) / len(labels_flat)
