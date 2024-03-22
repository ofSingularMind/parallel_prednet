import os
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or '2' to filter out INFO messages too

import numpy as np
import keras
from keras import layers

# main_input = layers.Input(shape = (10,), name='main_input')

layer_1 = layers.Dense(10, activation='relu', name='L1')
layer_2 = layers.Dense(20, activation='relu', name='L2')
layer_3 = layers.Dense(30, activation='relu', name='L3')
layer_4 = layers.Dense(40, activation='relu', name='L4')

# initialize the layers
# i# = initial predictions
i4 = layer_4(np.random.rand(1, 30))
i3 = layer_3(np.random.rand(1, 60))
i2 = layer_2(np.random.rand(1, 40))
i1 = layer_1(np.random.rand(1, 30))

for i in range(10):
    # Form top-down predictions from bottom-up input
    # n# = new predictions
    n4 = layer_4(i3)
    n3 = layer_3(layers.concatenate([i2, n4])) # new i4 but old i2
    n2 = layer_2(layers.concatenate([i1, n3]))
    bu_inp = np.random.rand(1, 10)
    n1 = layer_1(layers.concatenate([bu_inp, n2]))
    print(f"Iteration {i}")
    (i1, i2, i3, i4) = (n1, n2, n3, n4)