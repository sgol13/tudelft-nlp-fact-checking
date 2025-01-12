class EarlyStopping:
    def __init__(self, patience: int = 3):
        self._patience = patience
        self._counter = 0
        self._lowest_loss = None

    def __call__(self, val_loss) -> bool:
        score = -val_loss

        if self._lowest_loss is None:
            self._lowest_loss = val_loss
        elif val_loss > self._lowest_loss:
            self._counter += 1

            print(f'Early stopping counter: {self._counter}/{self._patience}')

            return self._counter >= self._patience
        else:
            self._lowest_loss = score
            self._counter = 0
            return False
