import pickle
import numpy as np

import sys
sys.path.append("src/")
from gnn import GraphNeuralNetwork

def read_test_graph(idx):
    filename = "datasets/test/{}_graph.txt".format(idx)
    return np.loadtxt(filename, skiprows=1)

if __name__ == "__main__":
    with open("model/best_model.pickle", mode="rb") as f:
        gnn = pickle.load(f)
    for i in range(500):
        graph = read_test_graph(i)
        print(gnn.predict(graph))
    