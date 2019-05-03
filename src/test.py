import unittest
from factory import *

from gnn import GraphNeuralNetwork

class TestGraphNeuralNetwork(unittest.TestCase):
    def test_aggregate(self):
        gnn = GraphNeuralNetwork()
        gnn.W = IDENTITY_WEIGHT
        H = np.zeros((4, 8))
        H[:, 0] = 1
        for t in range(4):
            self.assertTrue(np.array_equal(H, SAMPLE_OUT[t]))
            H = gnn.aggregate(SAMPLE_GRAPH, H)

    def test_get_embedding(self):
        gnn = GraphNeuralNetwork()
        gnn.W = IDENTITY_WEIGHT
        for t in range(4):
            gnn.T = t
            out = gnn.get_embedding(SAMPLE_GRAPH)
            self.assertTrue(np.array_equal(out, gnn.readout(SAMPLE_OUT[t])))

if __name__ == "__main__":
    unittest.main()