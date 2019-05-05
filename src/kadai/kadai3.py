import sys
sys.path.append("src/")

from gnn import GraphNeuralNetwork
from optimizer import SGD, Momentum
from train import read_train_data, read_test_data, train

if __name__ == "__main__":
    train_data = read_train_data()
    test_data = read_test_data()
    optimizers = {"SGD": SGD, "Momentum": Momentum}
    for i in range(5):
        for name, Optimizer in optimizers.items():
            print("{}th {} start!".format(i, name))
            gnn = GraphNeuralNetwork(Optimizer())
            train(gnn, train_data, test_data, epoch_num=100, print_train_loss=True)