import numpy as np

def sum_squares_error(y, t):
    return 0.5 * np.sum((y-t)**2)