import numpy as np

from gnn import GraphNeuralNetwork
from utils import read_graph, read_label
from parameter import MINIBATCH_SIZE, TRAIN_NUM, TEST_NUM

def test(gnn):
    loss_acc = 0.0
    accuracy_acc = 0.0
    for i in range(TRAIN_NUM, TRAIN_NUM + TEST_NUM):
        graph = read_graph(i)
        label = read_label(i)
        loss_acc += gnn.loss(graph, label)
        accuracy_acc += gnn.predict(graph) == label
    return loss_acc / TEST_NUM, accuracy_acc / TEST_NUM

def SGD(gnn, times=10):
    random_idx = np.arange(TRAIN_NUM)
    for epoch in range(times):
        print("epoch: {}".format(epoch))
        np.random.shuffle(random_idx)
        for mb_idx in range(TRAIN_NUM // MINIBATCH_SIZE):
            minibatch = []
            for i in range(mb_idx * MINIBATCH_SIZE, (mb_idx + 1) * MINIBATCH_SIZE):
                filenum = random_idx[i]
                minibatch.append((read_graph(filenum), read_label(filenum)))
            gnn.gradient_descent(minibatch)
            print(test(gnn))

def SGD_momentum(gnn, times=100):
    random_idx = np.arange(TRAIN_NUM)
    for epoch in range(times):
        print("epoch: {}".format(epoch))
        np.random.shuffle(random_idx)
        iter_num = TRAIN_NUM // MINIBATCH_SIZE
        for mb_idx in range(iter_num):
            minibatch = []
            for i in range(mb_idx * MINIBATCH_SIZE, (mb_idx + 1) * MINIBATCH_SIZE):
                filenum = random_idx[i]
                minibatch.append((read_graph(filenum), read_label(filenum)))
            gnn.momentum(minibatch)
            if mb_idx in [0, iter_num // 2]:
                print(test(gnn))

if __name__ == "__main__":
    gnn = GraphNeuralNetwork()
    SGD_momentum(gnn)

