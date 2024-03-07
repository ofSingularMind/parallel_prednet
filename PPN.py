import os
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or '2' to filter out INFO messages too

import numpy as np
import tensorflow as tf
import shutil
import keras
from keras import backend as K
from keras import layers
from kitti_settings import *
from data_utils import SequenceGenerator#, MyCustomCallback
from keras.callbacks import LearningRateScheduler, ModelCheckpoint, TensorBoard

# if results directory already exists, then delete it
if os.path.exists(RESULTS_SAVE_DIR): shutil.rmtree(RESULTS_SAVE_DIR)
if os.path.exists(LOG_DIR): shutil.rmtree(LOG_DIR)
os.mkdir(LOG_DIR)

save_model = True  # if weights will be saved
plot_intermediate = False  # if the intermediate model predictions will be plotted
tensorboard = True  # if the Tensorboard callback will be used
weights_checkpoint_file = os.path.join(WEIGHTS_DIR, 'tensorflow_weights/para_prednet_kitti_weights.hdf5')  # where weights are loaded prior to training
weights_file = os.path.join(WEIGHTS_DIR, 'tensorflow_weights/para_prednet_kitti_weights.hdf5')  # where weights will be saved
json_file = os.path.join(WEIGHTS_DIR, 'para_prednet_kitti_model_ALEX.json')


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
        self.states['E'] = tf.zeros((batch_size, self.im_height, self.im_width, 2*self.output_channels))
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
        
        # DEFINE TARGETS
        target = inputs[0] # (batch_size, im_height, im_width, output_channels)
        if self.bottom_layer:
            self.states['T'] = target
        else:
            self.states['T'] = self.target(target)
        
        # UPDATE STATES
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
        self.states['P'] = self.prediction(self.states['R'])    
        self.states['E'] = self.error(self.states['P'], self.states['T'])

        # Print out shapes of all states:
        # print(f"R: {self.states['R'].shape}, P: {self.states['P'].shape}, T: {self.states['T'].shape}, E: {self.states['E'].shape}")
        return self.states['E']

class PredNet(keras.Model):
    def __init__(self, batch_size=4, nt=10, *args, **kwargs):
        super(PredNet, self).__init__(*args, **kwargs)
        self.batch_size = batch_size
        self.nt = nt
        self.im_height = 128
        self.im_width = 160
        self.num_layers = 4
        self.layer_input_channels = [3, 2*3, 2*48, 2*96]
        self.layer_output_channels = [3, 48, 96, 192]
        self.layer_weights = [1, 0.1, 0.1, 0.1]
        self.time_loss_weights = 1./ (self.nt - 1) * np.ones((self.nt,1))  # equally weight all timesteps except the first
        self.time_loss_weights[0] = 0
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
        print("Calling PredNet...")
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
                layer_error = self.layer_weights[l] * K.mean(K.batch_flatten(error), axis=-1, keepdims=True) # (batch_size, 1)
                all_error = layer_error if l == self.num_layers - 1 else tf.add(all_error, layer_error) # (batch_size, 1)
            all_errors_over_time = self.time_loss_weights[t] * all_error if t == 0 else tf.add(all_errors_over_time, self.time_loss_weights[t] * all_error) # (batch_size, 1)

        # Clear states from computation graph
        for layer in self.predlayers:
            layer.clear_states()
                
        return all_errors_over_time


# Run code

# Data files
train_file = os.path.join(DATA_DIR, 'X_train.hkl')
train_sources = os.path.join(DATA_DIR, 'sources_train.hkl')
val_file = os.path.join(DATA_DIR, 'X_val.hkl')
val_sources = os.path.join(DATA_DIR, 'sources_val.hkl')
    
# Training parameters
nt = 10
nb_epoch = 150 # 150
batch_size = 2 # 4
samples_per_epoch = 500 # 500
N_seq_val = 100  # number of sequences to use for validation

train_generator = SequenceGenerator(train_file, train_sources, nt, batch_size=batch_size, shuffle=True)
val_generator = SequenceGenerator(val_file, val_sources, nt, batch_size=batch_size, N_seq=N_seq_val)

PPN = PredNet(batch_size=batch_size, nt=nt)
PPN.compile(optimizer='adam', loss='mean_squared_error')
print("PredNet compiled...")

lr_schedule = lambda epoch: 0.001 if epoch < 75 else 0.0001    # start with lr of 0.001 and then drop to 0.0001 after 75 epochs
callbacks = [LearningRateScheduler(lr_schedule)]
if save_model:
    if not os.path.exists(WEIGHTS_DIR): os.mkdir(WEIGHTS_DIR)
    callbacks.append(ModelCheckpoint(filepath=weights_file, monitor='val_loss', save_best_only=True))
# if plot_intermediate:
#     callbacks.append(MyCustomCallback())
if tensorboard:
    callbacks.append(TensorBoard(log_dir=LOG_DIR, histogram_freq=1, write_graph=True, write_images=False))

history = PPN.fit(train_generator, steps_per_epoch=samples_per_epoch / batch_size, epochs=nb_epoch, callbacks=callbacks, 
                validation_data=val_generator, validation_steps=N_seq_val / batch_size)