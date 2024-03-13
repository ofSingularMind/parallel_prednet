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
from data_utils import SequenceGenerator, MyCustomCallback
from keras.callbacks import LearningRateScheduler, ModelCheckpoint, TensorBoard
from PPN import ParaPredNet

# use mixed precision for faster runtimes and lower memory usage
# keras.mixed_precision.set_global_policy("mixed_float16")

# if results directory already exists, then delete it
if os.path.exists(RESULTS_SAVE_DIR): shutil.rmtree(RESULTS_SAVE_DIR)
if os.path.exists(LOG_DIR): shutil.rmtree(LOG_DIR)
os.mkdir(LOG_DIR)

save_model = True  # if weights will be saved
plot_intermediate = True  # if the intermediate model predictions will be plotted
tensorboard = True  # if the Tensorboard callback will be used
weights_checkpoint_file = os.path.join(WEIGHTS_DIR, 'tensorflow_weights/para_prednet_kitti_weights.hdf5')  # where weights are loaded prior to training
weights_file = os.path.join(WEIGHTS_DIR, 'tensorflow_weights/para_prednet_kitti_weights.hdf5')  # where weights will be saved
json_file = os.path.join(WEIGHTS_DIR, 'para_prednet_kitti_model_ALEX.json')
if os.path.exists(weights_file): os.remove(weights_file) # Careful: this will delete the weights file

# Run code

# Data files
train_file = os.path.join(DATA_DIR, 'X_train.hkl')
train_sources = os.path.join(DATA_DIR, 'sources_train.hkl')
val_file = os.path.join(DATA_DIR, 'X_val.hkl')
val_sources = os.path.join(DATA_DIR, 'sources_val.hkl')
    
# Training parameters
nt = 10  # number of time steps
nb_epoch = 150 # 150
batch_size = 2 # 4
samples_per_epoch = 100 # 500
N_seq_val = 20  # number of sequences to use for validation
output_channels = [3, 48, 96, 192]
im_shape = (128, 160, 3)

# train_generator = SequenceGenerator(train_file, train_sources, nt, batch_size=batch_size, shuffle=True)
# val_generator = SequenceGenerator(val_file, val_sources, nt, batch_size=batch_size, N_seq=N_seq_val)

# Dataset for images
train_ds = tf.keras.utils.image_dataset_from_directory(
    "/home/evalexii/remote_datasets/",
    validation_split=0.2,
    subset="training",
    batch_size=batch_size,
    image_size=(im_shape[0], im_shape[1])
)

PPN = ParaPredNet(batch_size=batch_size, nt=nt, output_channels=output_channels) # [3, 48, 96, 192]
PPN.compile(optimizer='adam', loss='mean_squared_error')
PPN.build(input_shape=(None, nt) + im_shape)
print("ParaPredNet compiled...")
print(PPN.summary())
num_layers = len(output_channels)  # number of layers in the architecture
print(f"Top layer resolution: {int(im_shape[0]/(2**(num_layers-1)))} x {int(im_shape[1]/(2**(num_layers-1)))}")

# load previously saved weights
if os.path.exists(weights_checkpoint_file):
    PPN.load_weights(weights_checkpoint_file)

lr_schedule = lambda epoch: 0.01 if epoch < 5 else 0.0001    # start with lr of 0.001 and then drop to 0.0001 after 75 epochs
callbacks = [LearningRateScheduler(lr_schedule)]
if save_model:
    if not os.path.exists(WEIGHTS_DIR): os.mkdir(WEIGHTS_DIR)
    callbacks.append(ModelCheckpoint(filepath=weights_file, monitor='val_loss', save_best_only=True, save_weights_only=True))
if plot_intermediate:
    callbacks.append(MyCustomCallback(batch_size=batch_size, nt=nt, output_channels=output_channels))
if tensorboard:
    callbacks.append(TensorBoard(log_dir=LOG_DIR, histogram_freq=1, write_graph=True, write_images=False))

history = PPN.fit(train_generator, steps_per_epoch=samples_per_epoch / batch_size, epochs=nb_epoch, callbacks=callbacks, 
                validation_data=val_generator, validation_steps=N_seq_val / batch_size)