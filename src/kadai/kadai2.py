import sys
sys.path.append("src/")
from gnn import GraphNeuralNetwork
from optimizer import SGD
from train import read_graph, read_label

if __name__ == "__main__":
    graph = read_graph(0)
    label = read_label(0)
    print("label: {}".format(label))
    gnn = GraphNeuralNetwork(optimizer=SGD())
    for i in range(1000):
        print("[{}th iteration] loss: {}, p: {}".format(
            i,
            gnn.loss(graph, label)[0],
            gnn.get_p(graph)[0],
        ))
        gnn.gradient_descent([(graph, label)])