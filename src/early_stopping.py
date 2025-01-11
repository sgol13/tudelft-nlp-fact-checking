import torch
import numpy as np

class EarlyStopping:
    def __init__(self, patience=7, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """

        self._patience = patience
        self._counter = 0
        self._best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta

    def __call__(self, val_loss, model):

        score = -val_loss

        if self._best_score is None:
            self._best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self._best_score + self.delta:
            self._counter += 1
            self.trace_func(f'EarlyStopping counter: {self._counter} out of {self._patience}')
            if self._counter >= self._patience:
                self.early_stop = True
        else:
            self._best_score = score
            self.save_checkpoint(val_loss, model)
            self._counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss