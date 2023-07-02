import numpy as np
from ex11 import *
from ex11 import TwoLayerNet

net = TwoLayerNet(input_size=784, hidden_size=100, output_size=10)
print(net.params['W1'].shape, net.params['b1'].shape, net.params['W2'].shape, net.params['b2'].shape)

x = np.random.rand(100, 784)
y = net.predict(x)

t = np.random.rand(100, 10)

grads = net.numerical_gradient(x, t)

print(grads['W1'].shape, grads['b1'].shape, grads['W2'].shape, grads['b2'].shape)