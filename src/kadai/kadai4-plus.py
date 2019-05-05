import os
import sys
sys.path.append("src/")
import pickle
import numpy as np

from gnn import GraphNeuralNetwork
from optimizer import Adam
from train import read_train_data, read_test_data, train

# 隣接行列を次数により正規化
def normalize(graph):
    deg = np.zeros_like(graph, dtype=float)
    for i in range(graph.shape[0]):
        graph[i][i] = 1
        deg[i][i] = 1 / np.sqrt(graph[i].sum())
    return np.dot(np.dot(deg, graph), deg)

if __name__ == "__main__":
    train_data = np.array([(normalize(g), l) for g, l in read_train_data()])
    test_data = np.array([(normalize(g), l) for g, l in read_test_data()])

    for i in range(5):
        print("{}th model".format(i))
        gnn = GraphNeuralNetwork(Adam())
        train(gnn, train_data, test_data, epoch_num=100)
        path_name = "model/normalized_model{}.pickle".format(i)
        os.makedirs(os.path.dirname(path_name), exist_ok=True)
        with open(path_name, mode="wb") as f:
            pickle.dump(gnn, f)
        print("{} saved!".format(path_name))