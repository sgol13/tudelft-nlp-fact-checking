class EarlyStopping:
    def __init__(self, patience: int):
        self._patience = patience
        self._counter = 0
        self._highest_accuracy = None

    def __call__(self, val_accuracy) -> bool:
        if self._highest_accuracy is None:
            self._highest_accuracy = val_accuracy
        elif val_accuracy > self._highest_accuracy:
            self._highest_accuracy = val_accuracy
            self._counter = 0
            return False
        else:
            self._counter += 1

            print(f'Early stopping counter: {self._counter}/{self._patience}')
            return self._counter >= self._patience
