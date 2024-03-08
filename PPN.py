import os
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or '2' to filter out INFO messages too

import numpy as np
import tensorflow as tf
import keras
from keras import backend as K
from keras import layers
from kitti_settings import *


class Target(keras.layers.Layer):
    def __init__(self, output_channels):
        super().__init__()
        self.output_channels = output_channels
        # Add Conv
        self.conv = layers.Conv2D(self.output_channels, (3, 3), padding='same', activation='relu')
        # Add Pool
        # self.pool = None
        self.pool = layers.MaxPooling2D((2, 2), padding='same')

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
    def __init__(self, im_height, im_width, output_channels, bottom_layer=False, top_layer=False, *args, **kwargs):
        super(PredLayer, self).__init__(*args, **kwargs)
        self.im_height = im_height
        self.im_width = im_width
        self.pixel_max = 1
        self.output_channels = output_channels
        self.top_layer = top_layer
        self.bottom_layer = bottom_layer
        # R = Representation, P = Prediction, T = Target, E = Error, and P == A_hat and T == A
        self.states = {'R': None, 'P': None, 'T': None, 'E': None, 'TD_Inp': None, 'L_Inp': None}
        # self.input_shape=(None, nt, im_height, im_width, output_channels)
        # self.inputs = keras.Input(shape=self.input_shape)
        self.representation = Representation(output_channels)
        self.prediction = Prediction(output_channels)
        if not self.bottom_layer:
            self.target = Target(output_channels)
        self.error = Error()
        self.upsample = layers.UpSampling2D((2, 2))

    def initialize_states(self, batch_size):
        # Initialize internal layer states
        self.states['R'] = tf.zeros((batch_size, self.im_height, self.im_width, self.output_channels))
        self.states['P'] = tf.zeros((batch_size, self.im_height, self.im_width, self.output_channels))
        self.states['T'] = tf.zeros((batch_size, self.im_height, self.im_width, self.output_channels))
        self.states['E'] = tf.zeros((batch_size, self.im_height, self.im_width, 4*self.output_channels))
        self.states['E_current'] = tf.zeros((batch_size, self.im_height, self.im_width, 2*self.output_channels))
        self.states['E_last'] = tf.zeros((batch_size, self.im_height, self.im_width, 2*self.output_channels))
        self.states['E_delta'] = tf.zeros((batch_size, self.im_height, self.im_width, 2*self.output_channels))
        self.states['TD_Inp'] = None
        self.states['L_Inp'] = None

    def clear_states(self):
        # Clear internal layer states
        self.states['R'] = None
        self.states['P'] = None
        self.states['T'] = None
        self.states['E'] = None
        self.states['TD_Inp'] = None
        self.states['L_Inp'] = None

    def call(self, inputs=None):
        # print(f"Calling PredLayer... {self.name}")
        # PredLayer should update internal states when called with new TD and BU inputs, inputs[0] = BU, inputs[1] = TD
        
        # UPDATE REPRESENTATION
        if self.top_layer:
            R_inp = keras.layers.Concatenate()([self.states['E'], self.states['R']])
            R_inp = tf.expand_dims(R_inp, axis=1)
            self.states['R'] = self.representation(R_inp)
        else:
            self.states['TD_Inp'] = self.upsample(inputs[1])
            # self.states['L_Inp'] = self.upsample(inputs[2])
            R_inp = keras.layers.Concatenate()([self.states['E'], self.states['R'], self.states['TD_Inp']])
            R_inp = tf.expand_dims(R_inp, axis=1)
            self.states['R'] = self.representation(R_inp)
        
        # FORM PREDICTION
        self.states['P'] = K.minimum(self.prediction(self.states['R']), self.pixel_max)

        # RETRIEVE TARGET
        target = inputs[0] # (batch_size, im_height, im_width, output_channels)
        if self.bottom_layer:
            self.states['T'] = target
        else:
            self.states['T'] = self.target(target)
        
        # COMPUTE ERROR
        self.states['E_current'] = self.error(self.states['P'], self.states['T'])
        self.states['E_delta'] = self.states['E_current'] - self.states['E_last']
        self.states['E_last'] = self.states['E_current']
        self.states['E'] = keras.layers.Concatenate(axis=-1)([self.states['E_current'], self.states['E_delta']])

        # Print out shapes of all states:
        # print(f"R: {self.states['R'].shape}, P: {self.states['P'].shape}, T: {self.states['T'].shape}, E: {self.states['E'].shape}")
        return self.states['E']

class ParaPredNet(keras.Model):
    def __init__(self, batch_size=4, nt=10, output_mode='Error', *args, **kwargs):
        super(ParaPredNet, self).__init__(*args, **kwargs)
        self.batch_size = batch_size
        self.nt = nt
        self.im_height = 128
        self.im_width = 160
        self.num_layers = 4
        self.layer_output_channels = [3, 24, 48, 96]
        self.layer_input_channels = [3, 4*3, 4*24, 4*48] # The 4x comes from the pos/neg error channels and pos/neg error-delta channels
        self.layer_weights = [1, 0.1, 0.1, 0.1]
        self.time_loss_weights = 1./ (self.nt - 1) * np.ones((self.nt,1))  # equally weight all timesteps except the first
        self.time_loss_weights[0] = 0
        self.output_mode = output_mode
        self.predlayers = []
        for l, c in enumerate(self.layer_output_channels):
            self.predlayers.append(PredLayer(self.im_height // 2**l, self.im_width // 2**l, c, bottom_layer=(l==0), top_layer=(l==self.num_layers-1), name=f'PredLayer_{l}'))
            # initialize layer states
            self.predlayers[-1].initialize_states(self.batch_size)
            # build layers
            if l == 0:
                temp_BU = tf.random.uniform((self.batch_size, self.im_height, self.im_width, self.layer_input_channels[l]), maxval=255, dtype=tf.float32)
            else:
                temp_BU = tf.random.uniform((self.batch_size, self.im_height // 2**(l-1), self.im_width // 2**(l-1), self.layer_input_channels[l]), maxval=255, dtype=tf.float32)
            if l < self.num_layers - 1:
                temp_TD = tf.random.uniform((self.batch_size, self.im_height // 2**(l+1), self.im_width // 2**(l+1), self.layer_output_channels[l+1]), maxval=255, dtype=tf.float32)
            else:
                temp_TD = None
            temp_out = self.predlayers[l]([temp_BU, temp_TD])
            
    def call(self, inputs):
        # print("Calling PredNet...")
        # inputs will be a sequence of video frames

        # Initialize layer states
        for layer in self.predlayers:
            layer.initialize_states(self.batch_size)
        
        # Iterate through the time-steps manually
        for t in range(self.nt):
            # print(f"...Time-step: {t}")
            # Starting from the top layer
            for l, layer in reversed(list(enumerate(self.predlayers))):
                if l == self.num_layers - 1:
                    error = layer([self.predlayers[l-1].states['E'], None])
                elif l < self.num_layers - 1 and l > 0:
                    BU_inp = self.predlayers[l-1].states['E'] # TODO: Confirm that this is appropriate iteration's data to compare
                    TD_inp = self.predlayers[l+1].states['R']
                    error = layer([BU_inp, TD_inp]) #, self.predlayers[l+1].states['L_Inp']])
                else:
                    BU_inp = inputs[:,t,...] # (self.batch_size, self.im_height, self.im_width, self.layer_input_channels[0])
                    TD_inp = self.predlayers[l+1].states['R']
                    error = layer([BU_inp, TD_inp]) #, self.predlayers[l+1].states['L_Inp']])
                if self.output_mode == 'Error':
                    layer_error = self.layer_weights[l] * K.mean(K.batch_flatten(error), axis=-1, keepdims=True) # (batch_size, 1)
                    all_error = layer_error if l == self.num_layers - 1 else tf.add(all_error, layer_error) # (batch_size, 1)
            if self.output_mode == 'Error':
                if t == 0:
                    all_errors_over_time = self.time_loss_weights[t] * all_error
                else:
                    all_errors_over_time = tf.add(all_errors_over_time, self.time_loss_weights[t] * all_error) # (batch_size, 1)
            elif self.output_mode == 'Prediction':    
                if t == 0:
                    all_predictions = tf.expand_dims(self.predlayers[0].states['P'], axis=1)
                else:
                    all_predictions = tf.concat([all_predictions, tf.expand_dims(self.predlayers[0].states['P'], axis=1)], axis=1)

        if self.output_mode == 'Error':
            output = all_errors_over_time * 100
        elif self.output_mode == 'Prediction':
            output = all_predictions
        
        # Clear states from computation graph
        for layer in self.predlayers:
            layer.clear_states()
                
        return output