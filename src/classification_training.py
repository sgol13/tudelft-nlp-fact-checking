import torch
import numpy as np
import random

from torch import nn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm.auto import tqdm


class ClassificationTraining:
    def __init__(self,
                 *,
                 model_name: str,
                 train_dataset: torch.utils.data.TensorDataset,
                 val_dataset: torch.utils.data.TensorDataset,
                 model: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 loss_function: torch.nn.Module,
                 batch_size: int,
                 device: torch.device,
                 random_state: int = None,
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

    def train(self, epochs: int) -> None:
        pass

    def load_model(self, checkpoint_name: str) -> None:
        pass

    def save_model(self) -> None:
        pass

    def _train_epoch(self):
        self._model.train()

        total_train_accuracy = 0
        total_train_loss = 0
        for batch in tqdm(self._train_dataloader):
            batch = batch.to(self._device)
            b_input_ids, b_input_mask, b_labels = batch

            self._model.zero_grad()

            probas = self._model(b_input_ids, b_input_mask)

            loss = self._loss_func(probas, b_labels)
            total_train_loss += loss.item()

            loss.backward()

            self._optimizer.step()

            logits = probas.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()
            total_train_accuracy += accuracy(logits, label_ids)

        avg_train_accuracy = total_train_accuracy / len(self._train_dataloader)
        print(" Train Accuracy: {0:.2f}".format(avg_train_accuracy))

        avg_train_loss = total_train_loss / len(self._train_dataloader)

        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))


    def _evaluate(self):
        model.eval()

        total_eval_accuracy = 0
        total_eval_loss = 0
        nb_eval_steps = 0

        for batch in tqdm(self._validation_dataloader):
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            with torch.no_grad():
                logits = model(b_input_ids, b_input_mask)

            loss = loss_func(logits, b_labels)
            total_eval_loss += loss.item()

            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            total_eval_accuracy += accuracy(logits, label_ids)

        avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
        print("  Accuracy: {0:.2f}".format(avg_val_accuracy))

        avg_val_loss = total_eval_loss / len(validation_dataloader)

        print("  Validation Loss: {0:.2f}".format(avg_val_loss))
        print("  Validation took: {:}".format(validation_time))

    @staticmethod
    def _accuracy(output: np.ndarray, labels: np.ndarray) -> float:
        pred_flat = np.argmax(output, axis=1).flatten()
        labels_flat = labels.flatten()
        return np.sum(pred_flat == labels_flat) / len(labels_flat)
