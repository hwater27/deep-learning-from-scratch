import ex04
import numpy as np
import matplotlib.pyplot as plt

x = np.arange(-5.0, 5.0, 0.1)
y = ex04.sigmoid(x)
plt.plot(x, y)
plt.ylim(-0.1, 1.1)
plt.show()