import sys
sys.path.append("src/")

from gnn import GraphNeuralNetwork
from optimizer import SGD, Momentum
from train import train

if __name__ == "__main__":
    optimizers = {"SGD": SGD, "Momentum": Momentum}
    for i in range(5):
        for name, Optimizer in optimizers.items():
            print("{}th {} start!".format(i, name))
            gnn = GraphNeuralNetwork(Optimizer())
            train(gnn, epoch_num=100, print_train_loss=True)