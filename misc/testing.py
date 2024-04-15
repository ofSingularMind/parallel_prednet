import matplotlib.pyplot as plt
import numpy as np

import os
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")
# or '2' to filter out INFO messages too
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf

# x = np.linspace(0, 6*np.pi, 100)
# y = np.sin(x)

# # You probably won't need this if you're embedding things in a tkinter plot...
# plt.ion()

# fig = plt.figure()
# ax = fig.add_subplot(111)
# line1, = ax.plot(x, y, 'r-') # Returns a tuple of line objects, thus the comma

# for phase in np.linspace(0, 10*np.pi, 500):
#     line1.set_ydata(np.sin(x + phase))
#     fig.canvas.draw()
#     fig.canvas.flush_events()

a = tf.data.Dataset.range(1, 4)  # ==> [ 1, 2, 3 ]
b = tf.data.Dataset.range(4, 8)  # ==> [ 4, 5, 6, 7 ]
ds = a.concatenate(b).shuffle(10)
print(a)
print(b)
print(ds)
print(list(ds.as_numpy_iterator()))

# The input dataset and dataset to be concatenated should have
# compatible element specs.
# c = tf.data.Dataset.zip((a, b))
# a.concatenate(c)



# d = tf.data.Dataset.from_tensor_slices(["a", "b", "c"])
# a.concatenate(d)