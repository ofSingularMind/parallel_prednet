from keras import layers
from keras import backend as K
import keras
import tensorflow as tf
import numpy as np
import os
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")
# or '2' to filter out INFO messages too
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
# ,\r?\n
# ,\s{2,}


class Target(keras.layers.Layer):
    def __init__(self, output_channels, layer_num, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.output_channels = output_channels
        # Add Conv
        self.conv = layers.Conv2D(self.output_channels, (3, 3), padding="same", activation="relu", name=f"Target_Conv_Layer{layer_num}")
        # Add Pool
        self.pool = layers.MaxPooling2D((2, 2), padding="valid", name=f"Target_Pool_Layer{layer_num}")

    def call(self, inputs):
        x = self.conv(inputs)
        return self.pool(x)


class Prediction(keras.layers.Layer):
    def __init__(self, output_channels, num_P_CNN, layer_num, activation='relu', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.output_channels = output_channels
        self.num_P_CNN = num_P_CNN
        self.conv_layers = []
        for i in range(num_P_CNN):
            self.conv_layers.append(layers.Conv2D(self.output_channels, (3, 3), padding="same", activation=activation, name=f"Prediction_Conv{i}_Layer{layer_num}"))

    def call(self, inputs):
        out = inputs
        for i in range(self.num_P_CNN):
            out = self.conv_layers[i](out)
        return out


class Error(keras.layers.Layer):
    def __init__(self, layer_num, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.layer_num = layer_num
        # Add Subtract
        # Add ReLU

    def call(self, predictions, targets):
        # compute errors
        e_down = keras.backend.relu(targets - predictions)
        e_up = keras.backend.relu(predictions - targets)
        return keras.layers.Concatenate(axis=-1)([e_down, e_up])


class Representation(keras.layers.Layer):
    def __init__(self, output_channels, num_R_CLSTM, layer_num, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Add ConvLSTM, being sure to pass previous states in OR use stateful=True
        self.num_R_CLSTM = num_R_CLSTM
        self.conv_lstm_layers = []
        for i in range(num_R_CLSTM):
            conv_i = layers.Conv2D(output_channels, (3, 3), padding="same", activation="hard_sigmoid", name=f"Representation_Conv_i_{i}_Layer{layer_num}")
            conv_f = layers.Conv2D(output_channels, (3, 3), padding="same", activation="hard_sigmoid", name=f"Representation_Conv_f_{i}_Layer{layer_num}")
            conv_o = layers.Conv2D(output_channels, (3, 3), padding="same", activation="hard_sigmoid", name=f"Representation_Conv_o_{i}_Layer{layer_num}")
            conv_c = layers.Conv2D(output_channels, (3, 3), padding="same", activation="tanh", name=f"Representation_Conv_c_{i}_Layer{layer_num}")
            convs = {"conv_i": conv_i, "conv_f": conv_f, "conv_o": conv_o, "conv_c": conv_c}
            self.conv_lstm_layers.append(convs)

    def call(self, inputs, initial_states=None):
        outs = []
        states = []
        for j in range(self.num_R_CLSTM):
            i = self.conv_lstm_layers[j]["conv_i"](inputs)
            f = self.conv_lstm_layers[j]["conv_f"](inputs)
            o = self.conv_lstm_layers[j]["conv_o"](inputs)
            h, c = initial_states[j] if initial_states is not None else 2 * [tf.zeros(f.shape, dtype=tf.float32)]
            c = f * c + i * self.conv_lstm_layers[j]["conv_c"](inputs)
            h = o * keras.activations.tanh(c)
            outs.append(h)
            states.append([h, c])
        output = keras.layers.Concatenate(axis=-1)(outs) if self.num_R_CLSTM > 1 else outs[0]
        return output, states

