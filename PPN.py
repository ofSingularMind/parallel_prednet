import os
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or '2' to filter out INFO messages too

import numpy as np
import tensorflow as tf
import keras
from keras import layers
from kitti_settings import *
from data_utils import SequenceGenerator

class Target(keras.layers.Layer):
    def __init__(self, output_channels):
        super().__init__()
        self.output_channels = output_channels
        # Add Conv
        self.conv = layers.Conv2D(self.output_channels, (3, 3), padding='same', activation='relu')
        # Add Pool
        self.pool = None
        # self.pool = layers.MaxPooling2D((2, 2), padding='same')

    def call(self, inputs):
        x = self.conv(inputs)
        if self.pool is not None:
            return self.pool(x)
        else:
            return x

class Prediction(keras.layers.Layer):
    def __init__(self, output_channels):
        super().__init__()
        self.output_channels = output_channels
        # Add Conv
        self.conv = layers.Conv2D(self.output_channels, (3, 3), padding='same', activation='relu')

    def call(self, inputs):
        return self.conv(inputs)

class Error(keras.layers.Layer):
    def __init__(self):
        super().__init__()
        # Add Subtract
        # Add ReLU

    def call(self, predictions, targets):
        # compute errors
        e_down = keras.backend.relu(targets - predictions)
        e_up = keras.backend.relu(predictions - targets)
        return keras.layers.Concatenate(axis=-1)([e_down, e_up])

class Representation(keras.layers.Layer):
    def __init__(self, output_channels):
        super().__init__()
        # Add ConvLSTM, being sure to pass previous states in OR use stateful=True
        self.conv_lstm = layers.ConvLSTM2D(output_channels, (3, 3), padding='same', return_sequences=False, activation='relu')

    def call(self, inputs):
        return self.conv_lstm(inputs)

class PredLayer(keras.layers.Layer):
    def __init__(self, im_height, im_width, output_channels, top_layer=False, *args, **kwargs):
        super(PredLayer, self).__init__(*args, **kwargs)
        self.im_height = im_height
        self.im_width = im_width
        self.output_channels = output_channels
        # R = Representation, P = Prediction, T = Target, E = Error, and P == A_hat and T == A
        self.states = {'R': None, 'P': None, 'T': None, 'E': None, 'TD_Inp': None, 'L_Inp': None}
        # self.input_shape=(None, nt, im_height, im_width, output_channels)
        # self.inputs = keras.Input(shape=self.input_shape)
        self.representation = Representation(output_channels)
        self.prediction = Prediction(output_channels)
        self.target = Target(output_channels)
        self.error = Error()
        self.upsample = layers.UpSampling2D((2, 2))
        self.top_layer = top_layer

    def initialize_states(self, batch_size):
        # Initialize internal layer states
        self.states['R'] = tf.zeros((batch_size, self.im_height, self.im_width, self.output_channels))
        self.states['P'] = tf.zeros((batch_size, self.im_height, self.im_width, self.output_channels))
        self.states['T'] = tf.zeros((batch_size, self.im_height, self.im_width, self.output_channels))
        self.states['E'] = tf.zeros((batch_size, self.im_height, self.im_width, self.output_channels))
        self.states['TD_Inp'] = None
        self.states['L_Inp'] = None

    def call(self, inputs=None):
        # PredLayer should update internal states when called with new TD and BU inputs
        # inputs[0] = BU, inputs[1] = TD
        target = inputs[0] # (batch_size, im_height, im_width, output_channels)
        self.states['T'] = self.target(target)
        if not self.top_layer:
            self.states['TD_Inp'] = inputs[1] # self.upsample(inputs[1])
            # self.states['L_Inp'] = self.upsample(inputs[2])
            R_inp = keras.layers.Concatenate()([self.states['E'], self.states['R'], self.states['TD_Inp']])
            R_inp = tf.expand_dims(R_inp, axis=1)
            self.states['R'] = self.representation(R_inp)
        else:
            R_inp = keras.layers.Concatenate()([self.states['E'], self.states['R']])
            R_inp = tf.expand_dims(R_inp, axis=1)
            self.states['R'] = self.representation(R_inp)
        self.states['P'] = self.prediction(self.states['R'])
        self.states['E'] = self.error(self.states['P'], self.states['T'])
        return self.states['E']

class PredNet(keras.Model):
    def __init__(self, batch_size=4, *args, **kwargs):
        super(PredNet, self).__init__(*args, **kwargs)
        self.batch_size = batch_size
        self.im_height = 128
        self.im_width = 160
        self.num_layers = 4
        self.layer_input_channels = [2*3+48, 2*48+96, 2*96+192, 96+192]
        self.layer_output_channels = [3, 48, 96, 192]
        self.layer_weights = [1, 0.1, 0.1, 0.1]
        self.predlayers = []
        for l, c in enumerate(self.layer_output_channels):
            self.predlayers.append(PredLayer(self.im_height, self.im_width, c, top_layer=(l==self.num_layers-1), name=f'PredLayer_{l}'))
            # initialize layer states
            self.predlayers[-1].initialize_states(self.batch_size)
            temp = np.random.rand(self.batch_size, self.im_height, self.im_width, self.layer_input_channels[l])
            temp = self.predlayers[l](2*[temp])
            
    # def initialize_states(self, batch_size):
    #     for layer in self.predlayers:
    #         layer.initialize_states(batch_size)

    def call(self, inputs):
        # inputs will be a sequence of video frames
        errors = []
        # iterate through the time-steps manually
        for t in inputs.shape[1]:
            # Starting from the top layer
            for l, layer in enumerate(reversed(self.predlayers)):
                if l == self.num_layers - 1:
                    error = layer(self.predlayers[l-1].states['E'])
                elif l < self.num_layers - 1 and l > 0:
                    BU_inp = self.predlayers[l-1].states['E']
                    TD_inp = self.predlayers[l+1].states['R']
                    error = layer([BU_inp, TD_inp]) #, self.predlayers[l+1].states['L_Inp']])
                else:
                    BU_inp = inputs[:,t,...] # (self.batch_size, self.im_height, self.im_width, self.layer_input_channels[0])
                    TD_inp = self.predlayers[l+1].states['R']
                    error = layer([BU_inp, TD_inp]) #, self.predlayers[l+1].states['L_Inp']])
                errors.append(keras.ops.mean(error))
                # self.add_loss(keras.ops.sum(layer.states['E']) * self.layer_weights[l])
                
        return self.layers[0].states['P']

# Run code

# Data files
train_file = os.path.join(DATA_DIR, 'X_train.hkl')
train_sources = os.path.join(DATA_DIR, 'sources_train.hkl')
val_file = os.path.join(DATA_DIR, 'X_val.hkl')
val_sources = os.path.join(DATA_DIR, 'sources_val.hkl')
    
# Training parameters
nt = 10
nb_epoch = 150 # 150
batch_size = 4 # 4
samples_per_epoch = 200 # 500
N_seq_val = 100  # number of sequences to use for validation

train_generator = SequenceGenerator(val_file, val_sources, nt, batch_size=batch_size, shuffle=True) # TODO: change to train
val_generator = SequenceGenerator(val_file, val_sources, nt, batch_size=batch_size, N_seq=N_seq_val)

PPN = PredNet(batch_size=batch_size)
PPN.compile(optimizer='adam')
history = PPN.fit(train_generator, steps_per_epoch=samples_per_epoch / batch_size, epochs=nb_epoch, callbacks=None,
                validation_data=val_generator, validation_steps=N_seq_val / batch_size)