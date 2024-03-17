from kitti_settings import *
from keras import layers
from keras import backend as K
import keras
import tensorflow as tf
import numpy as np
import os
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')
# or '2' to filter out INFO messages too
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class Target(keras.layers.Layer):
    def __init__(self, output_channels, Layernum, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.output_channels = output_channels
        # Add Conv
        self.conv = layers.Conv2D(self.output_channels, (3, 3), padding='same', activation='relu', name=f'Target_Conv_Layer{Layernum}')
        # Add Pool
        # self.pool = None
        self.pool = layers.MaxPooling2D((2, 2), padding='valid', name=f'Target_Pool_Layer{Layernum}')

    def call(self, inputs):
        x = self.conv(inputs)
        if self.pool is not None:
            return self.pool(x)
        else:
            return x


class Prediction(keras.layers.Layer):
    def __init__(self, output_channels, num_P_CNN, Layernum, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.output_channels = output_channels
        self.num_P_CNN = num_P_CNN
        self.conv_layers = []
        for i in range(num_P_CNN):
            self.conv_layers.append(layers.Conv2D(self.output_channels, (3, 3), padding='same', activation='relu', name=f'Prediction_Conv{i}_Layer{Layernum}'))

    def call(self, inputs):
        out = inputs
        for i in range(self.num_P_CNN):
            out = self.conv_layers[i](out)
        return out


class Error(keras.layers.Layer):
    def __init__(self, Layernum, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.Layernum = Layernum
        # Add Subtract
        # Add ReLU

    def call(self, predictions, targets):
        # compute errors
        e_down = keras.backend.relu(targets - predictions)
        e_up = keras.backend.relu(predictions - targets)
        return keras.layers.Concatenate(axis=-1)([e_down, e_up])


class Representation(keras.layers.Layer):
    def __init__(self, output_channels, num_R_CLSTM, Layernum, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Add ConvLSTM, being sure to pass previous states in OR use stateful=True
        self.num_R_CLSTM = num_R_CLSTM
        self.conv_lstm_layers = []
        for i in range(num_R_CLSTM):
            self.conv_lstm_layers.append(layers.ConvLSTM2D(output_channels, (3, 3), padding='same', return_sequences=False, activation='tanh', recurrent_activation='hard_sigmoid', return_state=True, name=f'Representation_ConvLSTM{i}_Layer{Layernum}'))

    def call(self, inputs, initial_states=None):
        outs = []
        states = []
        out = inputs
        for i in range(self.num_R_CLSTM):
            out, h, c = self.conv_lstm_layers[i](
                out if i==0 else tf.expand_dims(out, axis=1), 
                initial_state=initial_states[i] if initial_states is not None else None)
            outs.append(out)
            states.append([h, c])
        output = keras.layers.Concatenate(axis=-1)(outs) if self.num_R_CLSTM > 1 else outs[0]
        return output, states

    def reset_states(self):
        for layer in self.conv_lstm_layers:
            layer.reset_states()


class PredLayer(keras.layers.Layer):
    def __init__(self, im_height, im_width, num_P_CNN, num_R_CLSTM, output_channels, Layernum, bottom_layer=False, top_layer=False, *args, **kwargs):
        super(PredLayer, self).__init__(*args, **kwargs)
        self.Layernum = Layernum
        self.im_height = im_height
        self.im_width = im_width
        self.num_P_CNN = num_P_CNN
        self.num_R_CLSTM = num_R_CLSTM
        self.pixel_max = 1
        self.output_channels = output_channels
        self.top_layer = top_layer
        self.bottom_layer = bottom_layer
        # R = Representation, P = Prediction, T = Target, E = Error, and P == A_hat and T == A
        self.states = {'R': None, 'P': None, 'T': None,
                       'E': None, 'TD_Inp': None, 'L_Inp': None}
        self.representation = Representation(output_channels, num_R_CLSTM, Layernum=self.Layernum, name=f'Representation_Layer{self.Layernum}')
        self.prediction = Prediction(output_channels, num_P_CNN, Layernum=self.Layernum, name=f'Prediction_Layer{self.Layernum}')
        if not self.bottom_layer:
            self.target = Target(output_channels, Layernum=self.Layernum, name=f'Target_Layer{self.Layernum}')
        self.error = Error(Layernum=self.Layernum, name=f'Error_Layer{self.Layernum}')
        self.upsample = layers.UpSampling2D((2, 2), name=f'Upsample_Layer{self.Layernum}')

    def initialize_states(self, batch_size):
        # Initialize internal layer states
        self.states['R'] = tf.random.uniform(
            (batch_size, self.im_height, self.im_width, self.num_R_CLSTM*self.output_channels))
        self.states['P'] = tf.random.uniform(
            (batch_size, self.im_height, self.im_width, self.output_channels))
        self.states['T'] = tf.random.uniform(
            (batch_size, self.im_height, self.im_width, self.output_channels))
        self.states['E'] = tf.random.uniform(
            (batch_size, self.im_height, self.im_width, 2*self.output_channels))
        self.states['TD_Inp'] = None
        self.states['L_Inp'] = None
        self.states['lstm'] = None

    def clear_states(self):
        # Clear internal layer states
        self.states['R'] = None
        self.states['P'] = None
        self.states['T'] = None
        self.states['E'] = None
        self.states['TD_Inp'] = None
        self.states['L_Inp'] = None
        self.states['lstm'] = None
        # self.representation.reset_states()

    def call(self, inputs=None, direction='top_down', paddings=None):
        # print(f"Calling PredLayer... {self.name}")
        # PredLayer should update internal states when called with new TD and BU inputs, inputs[0] = BU, inputs[1] = TD

        if direction == 'top_down':
            # UPDATE REPRESENTATION
            if self.top_layer:
                R_inp = tf.expand_dims(keras.layers.Concatenate()(
                    [self.states['E'], self.states['R']]), axis=1)
            else:
                self.states['TD_Inp'] = self.upsample(inputs[1])
                self.states['TD_Inp'] = keras.layers.ZeroPadding2D(
                    paddings)(self.states['TD_Inp'])
                R_inp = tf.expand_dims(keras.layers.Concatenate()(
                    [self.states['E'], self.states['R'], self.states['TD_Inp']]), axis=1)

            if self.states['lstm'] is None:
                self.states['R'], self.states['lstm'] = self.representation(R_inp)
            else:
                self.states['R'], new_lstm_states = self.representation(
                    R_inp, initial_states=self.states['lstm'])
                self.states['lstm'] = new_lstm_states

            # FORM PREDICTION
            self.states['P'] = K.minimum(
                self.prediction(self.states['R']), self.pixel_max)

        elif direction == 'bottom_up':
            # RETRIEVE TARGET (bottom-up input)
            # (batch_size, im_height, im_width, output_channels)
            target = inputs[0]
            if self.bottom_layer:
                self.states['T'] = target
            else:
                self.states['T'] = self.target(target)

            # COMPUTE ERROR
            self.states['E'] = self.error(self.states['P'], self.states['T'])

            # Print out shapes of all states:
            # print(f"R: {self.states['R'].shape}, P: {self.states['P'].shape}, T: {self.states['T'].shape}, E: {self.states['E'].shape}")
            return self.states['E']

        else:
            raise ValueError(
                "Invalid direction. Must be 'top_down' or 'bottom_up'.")


class ParaPredNet(keras.Model):
    def __init__(self, batch_size=4, nt=10, im_height=540, im_width=960, num_P_CNN=3, num_R_CLSTM=3, output_channels=[3, 48, 96, 192], output_mode='Error', *args, **kwargs):
        super(ParaPredNet, self).__init__(*args, **kwargs)
        self.batch_size = batch_size
        self.nt = nt
        self.im_height = im_height
        self.im_width = im_width
        self.num_P_CNN = num_P_CNN
        self.num_R_CLSTM = num_R_CLSTM
        self.Layeroutput_channels = output_channels
        self.num_layers = len(self.Layeroutput_channels)
        self.resolutions = self.calculate_resolutions(
            self.im_height, self.im_width, self.num_layers)
        self.paddings = self.calculate_padding(
            self.im_height, self.im_width, self.num_layers)
        self.Layerinput_channels = [0] * self.num_layers
        for i in range(len(self.Layeroutput_channels)):
            if i == 0:
                self.Layerinput_channels[i] = self.Layeroutput_channels[i]
            else:
                self.Layerinput_channels[i] = 2 * \
                    self.Layeroutput_channels[i-1]
        # weighting for each layer's contribution to the loss
        self.Layerweights = [1] + [0.1] * (self.num_layers - 1)
        # equally weight all timesteps except the first
        self.time_loss_weights = 1. / (self.nt - 1) * np.ones((self.nt, 1))
        self.time_loss_weights[0] = 0
        self.output_mode = output_mode
        self.predlayers = []
        for l, c in enumerate(self.Layeroutput_channels):
            self.predlayers.append(PredLayer(self.resolutions[l, 0], self.resolutions[l, 1], self.num_P_CNN, self.num_R_CLSTM, c, l, 
                                             bottom_layer=(l == 0), top_layer=(l == self.num_layers-1), name=f'PredLayer{l}'))
            # initialize layer states
            self.predlayers[-1].initialize_states(self.batch_size)
            # build layers
            if l == 0:
                temp_BU = tf.random.uniform(
                    (self.batch_size, self.resolutions[l, 0], self.resolutions[l, 1], self.Layerinput_channels[l]), maxval=255, dtype=tf.float32)
            else:
                temp_BU = tf.random.uniform(
                    (self.batch_size, self.resolutions[l, 0], self.resolutions[l, 1], self.Layerinput_channels[l]), maxval=255, dtype=tf.float32)
            if l < self.num_layers - 1:
                temp_TD = tf.random.uniform(
                    (self.batch_size, self.resolutions[l+1, 0], self.resolutions[l+1, 1], self.num_R_CLSTM*self.Layeroutput_channels[l+1]), maxval=255, dtype=tf.float32)
            else:
                temp_TD = None
            temp_out = self.predlayers[l](
                [temp_BU, temp_TD], paddings=self.paddings[l])

    def call(self, inputs):
        # print("Calling PredNet...")
        # inputs will be a tuple of batches of sequences of video frames
        # [-1] represents the PNG image source
        inputs_single_source = inputs[-1]

        # Initialize layer states
        for layer in self.predlayers:
            layer.initialize_states(self.batch_size)

        # Iterate through the time-steps manually
        for t in range(self.nt):
            # Perform top-down pass, starting from the top layer
            for l, layer in reversed(list(enumerate(self.predlayers))):
                # Top layer
                if l == self.num_layers - 1:
                    BU_inp = None
                    TD_inp = None
                    layer([BU_inp, TD_inp], direction='top_down',
                          paddings=self.paddings[l])
                # Bottom and Middle layers
                else:
                    BU_inp = None
                    TD_inp = self.predlayers[l+1].states['R']
                    layer([BU_inp, TD_inp], direction='top_down',
                          paddings=self.paddings[l])
            # Perform bottom-up pass, starting from the bottom layer
            for l, layer in list(enumerate(self.predlayers)):
                # Bottom layer
                if l == 0:
                    # (self.batch_size, self.im_height, self.im_width, self.Layerinput_channels[0])
                    BU_inp = inputs_single_source[:, t, ...]
                    TD_inp = None
                    error = layer([BU_inp, TD_inp], direction='bottom_up')
                # Middle and Top layers
                else:
                    BU_inp = self.predlayers[l-1].states['E']
                    TD_inp = None
                    error = layer([BU_inp, TD_inp], direction='bottom_up')
                # Update error in bottom-up pass
                if self.output_mode == 'Error':
                    Layererror = self.Layerweights[l] * K.mean(
                        K.batch_flatten(error), axis=-1, keepdims=True)  # (batch_size, 1)
                    all_error = Layererror if l == 0 else tf.add(
                        all_error, Layererror)  # (batch_size, 1)

            # save outputs over time
            if self.output_mode == 'Error':
                if t == 0:
                    all_errors_over_time = self.time_loss_weights[t] * all_error
                else:
                    all_errors_over_time = tf.add(
                        all_errors_over_time, self.time_loss_weights[t] * all_error)  # (batch_size, 1)
            elif self.output_mode == 'Prediction':
                if t == 0:
                    all_predictions = tf.expand_dims(
                        self.predlayers[0].states['P'], axis=1)
                else:
                    all_predictions = tf.concat([all_predictions, tf.expand_dims(
                        self.predlayers[0].states['P'], axis=1)], axis=1)

        if self.output_mode == 'Error':
            output = all_errors_over_time * 100
        elif self.output_mode == 'Prediction':
            output = all_predictions

        # Clear states from computation graph
        for layer in self.predlayers:
            layer.clear_states()

        return output

    def calculate_resolutions(self, im_height, im_width, num_layers):
        # Calculate resolutions for each layer
        resolutions = np.array([[im_height, im_width]])
        for i in range(num_layers-1):
            resolutions = np.concatenate((resolutions, np.array(
                [[resolutions[-1][0]//2, resolutions[-1][1]//2]])), axis=0)
        return resolutions

    def calculate_padding(self, im_height, im_width, num_layers):
        # Calculate padding for the input image to be divisible by 2**num_layers
        paddings = np.array([[[0, 0], [0, 0]] for _ in range(num_layers)])

        pooled_sizes = np.array([[im_height, im_width]])
        for i in range(num_layers-1):
            # Going down:
            pooled_sizes = np.concatenate((pooled_sizes, np.array(
                [[pooled_sizes[-1][0]//2, pooled_sizes[-1][1]//2]])), axis=0)

        upsampled_sizes = np.array([pooled_sizes[-1]])
        for i in reversed(range(num_layers-1)):
            # Going up:
            upsampled_sizes = np.concatenate((np.array(
                [[upsampled_sizes[0][0]*2, upsampled_sizes[0][1]*2]]), upsampled_sizes), axis=0)
            diff = (pooled_sizes[i][0] - upsampled_sizes[0]
                    [0], pooled_sizes[i][1] - upsampled_sizes[0][1])
            paddings[i] = [[0, diff[0]], [0, diff[1]]]
            upsampled_sizes[0] += np.array(diff)

        # print(np.concatenate((np.array(pooled_sizes), np.array(upsampled_sizes)), axis=1))
        # print(np.array(paddings))

        return paddings
