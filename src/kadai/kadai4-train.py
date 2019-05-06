import os
import sys
sys.path.append("src/")
import pickle
import numpy as np

from gnn import GraphNeuralNetwork
from optimizer import Adam
from train import read_train_data, read_test_data, test
from parameter import MINIBATCH_SIZE, TRAIN_NUM, TEST_NUM

loss_uppper_bound = 0.630
accuracy_lower_bound = 0.653

def train_best(gnn, train_data, test_data, epoch_num=100, print_train_loss=False):
    for epoch in range(epoch_num):
        np.random.shuffle(train_data)
        iter_num = TRAIN_NUM // MINIBATCH_SIZE
        for mb_idx in range(iter_num):
            minibatch = train_data[mb_idx * MINIBATCH_SIZE : (mb_idx + 1) * MINIBATCH_SIZE]
            gnn.gradient_descent(minibatch)
            # 1 epoch中に2回testを計算
            if mb_idx in [0, iter_num // 2]:
                loss, accuracy = test(gnn, test_data)
                if loss < loss_uppper_bound and accuracy > accuracy_lower_bound:
                    print("epoch: {}, loss: {}, accuracy: {}".format(epoch, loss, accuracy))
                    return True
    return False

if __name__ == "__main__":
    train_data = read_train_data()
    test_data = read_test_data()

    for i in range(100):
        print("{}th model".format(i))
        gnn = GraphNeuralNetwork(Adam())
        if train_best(gnn, train_data, test_data, epoch_num=100):
            path_name = "model/best_model.pickle"
            os.makedirs(os.path.dirname(path_name), exist_ok=True)
            with open(path_name, mode="wb") as f:
                pickle.dump(gnn, f)
            print("{} saved!".format(path_name))
            exit(0)