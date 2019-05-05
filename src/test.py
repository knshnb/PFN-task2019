import unittest
import numpy as np
from gnn import GraphNeuralNetwork

IDENTITY_WEIGHT = np.array([
    [1, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 1],
], dtype=np.float64)

SAMPLE_GRAPH = np.array([
    [0, 1, 0, 0],
    [1, 0, 1, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 0],
])

SAMPLE_OUT = np.array([
    [
        [1, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0],
    ],
    [
        [1, 0, 0, 0, 0, 0, 0, 0],
        [2, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
    ],
    [
        [2, 0, 0, 0, 0, 0, 0, 0],
        [2, 0, 0, 0, 0, 0, 0, 0],
        [2, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
    ],
    [
        [2, 0, 0, 0, 0, 0, 0, 0],
        [4, 0, 0, 0, 0, 0, 0, 0],
        [2, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
    ],
])

class TestGraphNeuralNetwork(unittest.TestCase):
    def test_aggregate(self):
        gnn = GraphNeuralNetwork()
        gnn.params["W"] = IDENTITY_WEIGHT
        H = np.zeros((4, 8))
        H[:, 0] = 1
        for t in range(4):
            self.assertTrue(np.array_equal(H, SAMPLE_OUT[t]))
            H = gnn._aggregate(SAMPLE_GRAPH, H)

    def test_get_embedding(self):
        gnn = GraphNeuralNetwork()
        gnn.params["W"] = IDENTITY_WEIGHT
        for t in range(4):
            gnn.T = t
            out = gnn._get_embedding(SAMPLE_GRAPH)
            self.assertTrue(np.array_equal(out, gnn._readout(SAMPLE_OUT[t])))

if __name__ == "__main__":
    unittest.main()