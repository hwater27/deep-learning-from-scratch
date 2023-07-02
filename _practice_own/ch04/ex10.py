import sys, os
sys.path.append(os.pardir)
import numpy as np
from ex08 import numerical_gradient
from ex03 import cross_entropy_error
from ex09 import simpleNet, net, x, t

f = lambda w: net.loss(x, t)
dW = numerical_gradient(f, net.W)
print(dW)