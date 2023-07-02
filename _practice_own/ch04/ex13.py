import numpy as np
from dataset.mnist import load_mnist
from ex11 import TwoLayerNet

import matplotlib.pylab as plt

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

train_loss_list = []

#checking overfitting
train_acc_list = []
test_acc_list = []

iters_num = 50000
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.02

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

iter_per_epoch = max(train_size / batch_size, 1)

#personal thoughts 02
#network_escape = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

for i in range(iters_num):
    #persomal thoughts 01
    if i < 2:
        learning_rate = 0.1
    else:
        learning_rate = 0.1 * train_loss_list[i - 1] / train_loss_list[i - 2]

    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    grad = network.gradient(x_batch, t_batch)

    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate *grad[key]
    '''
    #personal thoughts 02######################################
    if i > 100 and np.amax(train_loss_list[i - 100:i - 1], axis=0) < 1.5 * np.amin(train_loss_list[i - 100:i - 1], axis=0):
        np.random.seed(i)
        W1_escape=network.params['W1'] + np.random.normal(size=np.shape(network.params['W1']))
        b1_escape=network.params['b1'] + np.random.normal(size=np.shape(network.params['b1']))
        W2_escape=network.params['W2'] + np.random.normal(size=np.shape(network.params['W2']))
        b2_escape=network.params['b2'] + np.random.normal(size=np.shape(network.params['b2']))

        network_escape.params['W1'] = W1_escape
        network_escape.params['b1'] = b1_escape
        network_escape.params['W2'] = W2_escape
        network_escape.params['b2'] = b2_escape

        grad_escape = network_escape.gradient(x_batch, t_batch)

        loss_escape = network_escape.loss(x_batch, t_batch)
        
        if loss_escape < network.loss(x_batch, t_batch):
            for key in ('W1', 'b1', 'W2', 'b2'):
                network.params[key] = network_escape.params[key]

        grad = network.gradient(x_batch, t_batch)
                
    ###########################################################
    
    else:
    '''
    loss = network.loss(x_batch, t_batch)

    train_loss_list.append(loss)

    # Why divide by iter_per_epoch?
    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print("step{:6}| loss: {:5.10f} | train acc: {:.6f} | test acc: {:.6f} | ".format(i, train_loss_list[i], train_acc_list[int(i / iter_per_epoch)], test_acc_list[int(i / iter_per_epoch)]))

plt.plot(range(iters_num), train_loss_list)
plt.show()

plt.plot(range(int(iters_num / iter_per_epoch) + 1), train_acc_list, color='r', label='train acc')
plt.plot(range(int(iters_num / iter_per_epoch) + 1), test_acc_list, color='g', label= 'test acc')
plt.ylim(-0.1, 1.1)
plt.legend()
plt.show()