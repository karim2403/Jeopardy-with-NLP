"""Validation-loss early stopping with serializable state."""
import math


class EarlyStopping:
    def __init__(self, patience=2, min_delta=0.0, state=None):
        if patience < 0 or isinstance(patience, bool):
            raise ValueError('Early-stopping patience must be a nonnegative integer')
        if min_delta < 0 or not math.isfinite(min_delta):
            raise ValueError('Early-stopping min_delta must be nonnegative and finite')
        self.patience = patience
        self.min_delta = min_delta
        self.best = math.inf
        self.bad_epochs = 0
        if state:
            self.best = state['best']
            self.bad_epochs = state['bad_epochs']

    @property
    def enabled(self):
        return self.patience > 0

    def update(self, value):
        if not math.isfinite(value):
            raise ValueError('Early-stopping metric must be finite')
        improved = value < self.best - self.min_delta
        if improved:
            self.best = value
            self.bad_epochs = 0
        elif self.enabled:
            self.bad_epochs += 1
        return {'improved': improved, 'bad_epochs': self.bad_epochs,
                'patience': self.patience,
                'should_stop': self.enabled and self.bad_epochs >= self.patience}

    def state_dict(self):
        return {'best': self.best, 'bad_epochs': self.bad_epochs,
                'patience': self.patience, 'min_delta': self.min_delta}
