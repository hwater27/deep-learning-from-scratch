import numpy as np
import matplotlib.pylab as plt

def numerical_diff(f, x):
    h = 1e-4
    return (f(x + h) - f(x - h)) / (2*h)

def function_1(x):
    return 0.01*x**2 + 0.1*x

x = np.arange(0.0, 20.0, 0.1)
y = function_1(x)
plt.xlabel("x")
plt.ylabel("y")
plt.plot(x, y)
#plt.show()

def numerical_pardiff(f, x, index):
    h = 1e-4
    former = np.zeros_like(x)
    further = np.zeros_like(x)
    for i in range(x.size):
        former[i] = x[i]
        further[i] = x[i]
    former[index] = x[index] - h
    further[index] = x[index] + h
    return (f(further) - f(former) ) / (2*h)

def function_2(x):
    return x[0]**2 + x[1]**2

#print(numerical_pardiff(function_2, [3,4], 0))

def numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)

    for idx in range(len(x)):
        tmp_val = x[idx]
        x[idx] = tmp_val + h
        fxh1 = f(x)

        x[idx] = tmp_val - h
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2*h)
        x[idx] = tmp_val

    return grad