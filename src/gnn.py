import numpy as np
from parameter import ALPHA, EPS, SIGMA, ETA, S_LIMIT
from optimizer import SGD, Momentum

def ReLU(x):
    return np.maximum(0.0, x)
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))
def binary_cross_entropy(s, y):
    # sが大きいときはオーバーフロー対策で近似
    term1 = y * (-s if -s > S_LIMIT else np.log(1 + np.exp(-s)))
    term2 = (1 - y) * (s if s > S_LIMIT else np.log(1 + np.exp(s)))
    return term1 + term2

class GraphNeuralNetwork:
    def __init__(self, optimizer=Momentum(), D=8, T=2):
        # aggregation層の深さ
        self.T = T
        # 訓練パラメータ
        self.params = {}
        self.params["W"] = SIGMA * np.random.randn(D, D)
        self.params["A"] = SIGMA * np.random.randn(D)
        self.params["b"] = np.array([0.0])
        # 最適化手法
        self.optimizer = optimizer

    # 集約1, 集約2
    def _aggregate(self, adj, H):
        support = np.dot(adj, H)
        return ReLU(np.dot(support, self.params["W"]))

    # READOUT
    def _readout(self, H):
        return H.sum(axis=0)
    
    # 縮約を繰り返しREADOUTすることでh_Gを得る
    def _get_embedding(self, adj):
        # 特徴ベクトルは初めの値のみ0、それ以外は1に設定
        H = np.zeros((adj.shape[0], self.params["W"].shape[0]))
        H[:, 0] = 1
        for _ in range(self.T):
            H = self._aggregate(adj, H)
        return self._readout(H)

    # グラフの2値分類問題の確率値pを得る
    def _get_p(self, adj):
        s = np.dot(self.params["A"], self._get_embedding(adj)) + self.params["b"]
        return sigmoid(s)

    def predict(self, adj):
        return 1 if self._get_p(adj) > 0.5 else 0

    # sから損失を直接求める
    def loss(self, adj, y):
        s = np.dot(self.params["A"], self._get_embedding(adj)) + self.params["b"]
        return binary_cross_entropy(s, y)

    # param全てについてloss関数を数値微分
    def _numerical_grad(self, adj, y):
        grad = {}
        for key, param in self.params.items():
            grad[key] = np.empty_like(param)
            for idx, _ in np.ndenumerate(param):
                param[idx] += EPS
                tmp = self.loss(adj, y)
                param[idx] -= EPS
                grad[key][idx] = (tmp - self.loss(adj, y)) / EPS
        return grad

    # 複数データについての勾配の平均
    def _numerical_grad_minibatch(self, data):
        grads = [self._numerical_grad(d[0], d[1]) for d in data]
        return {key: np.mean([grad[key] for grad in grads]) for key in self.params}

    def gradient_descent(self, data):
        self.optimizer.update(self.params, self._numerical_grad_minibatch(data))