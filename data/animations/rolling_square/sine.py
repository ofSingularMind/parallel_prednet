import numpy as np
import matplotlib.pyplot as plt

s = 2
A = (s/2) * (np.sqrt(2) - 1)
B = 4

x = np.linspace(0, np.pi, 500)
y = np.abs(A*np.sin(B*x))

plt.plot(x, y)

plt.show()