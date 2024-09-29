import warnings

class EarlyStopping():
    def __init__(self, patience: int = 1, min_delta: float = 0.0, maximize: bool = False):
        if patience < 1:
            warnings.warn('Warning: Setting patience to a value smaller than 1 will result in True being returned for any call')
        self.patience = patience
        self.min_delta = min_delta
        self.maximize = maximize

        self.counter = 0
        if self.maximize:
            self.best_score = 0.0
        else:
            self.best_score = float('inf')

    def early_stop(self, score):
        if self.maximize:
            if score > self.best_score:
                self.best_score = score
                self.counter = 0
            elif score < (self.best_score - self.min_delta):
                self.counter += 1
        else:
            if score < self.best_score:
                self.best_score = score
                self.counter = 0
            elif score > (self.best_score + self.min_delta):
                self.counter += 1

        if self.counter >= self.patience:
            return True
        else:
            return False