import numpy as np
import matplotlib.pylab as plt

class fu(layer = 2 , ):
    classification = True

    if classification == False:
        regression = True

    layer = 2
    bias = np.zeros_like(np.arange(layer))
    Weight_set_tensor = np.empty(shape = layer, dtype = int)

    dataset = 0
    batch_size = 0

    for making_layer in range(layer):
        Weight_set_tensor.append()


    def __init__(self):
        self.layer = 2
        return self
    
    def setdata(self, layer, classification, weight_of_eachlayer):
        self.layer = layer
    
    def step_function():
        pass

    def sigmoid(x):

        pass

    def softmax():
        pass

    def pardiff(function, X, x_i):
        h = 1e-4; reverse_h = 10000
        former, further = np.zeros_like(X), np.zeros_like(X)

        for i in range(X.size):
            former[i] = X[i]
            further[i] = X[i]

        former[x_i] = X[i] - h
        further[x_i] = X[i] + h

        return ( function(further) - function(former) ) * reverse_h * 0.5

    def loss_function():
        pass

    def gradient_descent(loss_function, weight_set, learning_rate = 0.01):
        

        return weight_set - learning_rate * fu.pardiff(loss_function(), )
    
    def neural_network(self, input, using_function = sigmoid, ):
        using_function()

        output = 1

        return output

    def feedback(self):
        pass

    def learning(self, epochs):

        for i in range(epochs + 1):
            if i % 100 == 0:
                print("{} | {}".format(fu.loss_function(), i))
        pass

    


    
