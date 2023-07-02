import numpy as np

def step_function(x, threshold=0):
    return (x > threshold)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def relu(x):
    return np.max(x, 0)

