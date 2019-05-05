import numpy as np
from parameter import ALPHA, ETA, BETA1, BETA2, ADAM_EPS

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

class Adam:
    def __init__(self):
        self.m = None
        self.v = None
        self.t = 0

    def update(self, params, grad):
        if self.m is None or self.v is None:
            self.m = {}
            self.v = {}
            for key, val in params.items():
                self.m[key] = np.zeros_like(val)
                self.v[key] = np.zeros_like(val)

        self.t += 1
        for key in params:
            self.m[key] = BETA1 * self.m[key] + (1 - BETA1) * grad[key]
            self.v[key] = BETA2 * self.v[key] + (1 - BETA2) * (grad[key] ** 2)
            m_ = self.m[key] / (1 - BETA1 ** self.t)
            v_ = self.v[key] / (1 - BETA2 ** self.t)
            params[key] -= ALPHA * m_ / (np.sqrt(v_) + ADAM_EPS)