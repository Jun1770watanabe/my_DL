import numpy as np
from dataset.mnist import load_mnist
from network import TwoLayerNet

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

nw = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

iters_num = 10000
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1

train_loss_list = []
train_acc_list = []
test_acc_list = []

epoch = max(train_size / batch_size, 1)

for i in range(iters_num):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    grad = nw.gradient(x_batch, t_batch)

    for key in grad.keys():
        nw.params[key] -= learning_rate * grad[key]

    loss = nw.loss(x_batch, t_batch)
    train_loss_list.append(loss)

    if i % epoch == 0:
        train_acc = nw.accuracy(x_train, t_train)
        test_acc = nw.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print(i, train_acc*100, test_acc*100)