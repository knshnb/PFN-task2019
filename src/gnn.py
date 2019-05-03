import numpy as np
from parameter import ALPHA, EPS, SIGMA, ETA, S_LIMIT

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
    def __init__(self, D=8, T=2):
        # aggregation層の深さ
        self.T = T
        # 訓練パラメータ
        self.W = SIGMA * np.random.randn(D, D)
        self.A = SIGMA * np.random.randn(D)
        self.b = 0.0
        # momentum用
        self.Ww = np.zeros_like(self.W)
        self.Aw = np.zeros_like(self.A)
        self.bw = 0.0

    # 集約1, 集約2
    def aggregate(self, adj, H):
        support = np.dot(adj, H)
        return ReLU(np.dot(support, self.W))

    # READOUT
    def readout(self, H):
        return H.sum(axis=0)
    
    # 縮約を繰り返しREADOUTすることでh_Gを得る
    def get_embedding(self, adj):
        # 特徴ベクトルは初めの値のみ0、それ以外は1に設定
        H = np.zeros((adj.shape[0], self.W.shape[0]))
        H[:, 0] = 1
        for _ in range(self.T):
            H = self.aggregate(adj, H)
        return self.readout(H)

    # グラフの2値分類問題の確率値pを得る
    def get_p(self, adj):
        s = np.dot(self.A, self.get_embedding(adj)) + self.b
        return sigmoid(s)

    def predict(self, adj):
        return 1 if self.get_p(adj) > 0.5 else 0

    # sから損失を直接求める
    def loss(self, adj, y):
        s = np.dot(self.A, self.get_embedding(adj)) + self.b
        return binary_cross_entropy(s, y)

    # W, A, bについてloss関数を数値微分
    def numerical_grad_W(self, adj, y):
        W_diff = np.empty_like(self.W)
        for i in range(self.W.shape[0]):
            for j in range(self.W.shape[1]):
                self.W[i][j] += EPS
                tmp = self.loss(adj, y)
                self.W[i][j] -= EPS
                W_diff[i][j] = (tmp - self.loss(adj, y)) / EPS
        return W_diff

    def numerical_grad_A(self, adj, y):
        A_diff = np.empty_like(self.A)
        for i in range(self.A.shape[0]):
            self.A += EPS
            tmp = self.loss(adj, y)
            self.A -= EPS
            A_diff[i] = (tmp - self.loss(adj, y)) / EPS
        return A_diff

    def numerical_grad_b(self, adj, y):
        b_diff = np.empty_like(self.b)
        self.b += EPS
        tmp = self.loss(adj, y)
        self.b -= EPS
        b_diff = (tmp - self.loss(adj, y)) / EPS
        return b_diff
    
    def gradient_descent(self, data):
        W_diff = np.mean([self.numerical_grad_W(d[0], d[1]) for d in data])
        A_diff = np.mean([self.numerical_grad_A(d[0], d[1]) for d in data])
        b_diff = np.mean([self.numerical_grad_b(d[0], d[1]) for d in data])
        self.W -= ALPHA * W_diff
        self.A -= ALPHA * A_diff
        self.b -= ALPHA * b_diff
    
    def momentum(self, data):
        W_diff = np.mean([self.numerical_grad_W(d[0], d[1]) for d in data])
        A_diff = np.mean([self.numerical_grad_A(d[0], d[1]) for d in data])
        b_diff = np.mean([self.numerical_grad_b(d[0], d[1]) for d in data])
        self.Ww = -ALPHA * W_diff + ETA * self.Ww
        self.Aw = -ALPHA * A_diff + ETA * self.Aw
        self.bw = -ALPHA * b_diff + ETA * self.bw
        self.W += self.Ww
        self.A += self.Aw
        self.b += self.bw