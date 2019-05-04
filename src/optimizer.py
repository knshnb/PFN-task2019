import numpy as np
from parameter import ALPHA, ETA

class SGD:
    def update(self, params, grad):
        for key in params:
            params[key] -= ALPHA * grad[key]

class Momentum:
    def __init__(self):
        self.w = None

    def update(self, params, grad):
        if self.w is None:
            self.w = {}
            for key, val in params.items():
                self.w[key] = np.zeros_like(val)
        
        for key in params:
            self.w[key] = -ALPHA * grad[key] + ETA * self.w[key]
            params[key] += self.w[key]