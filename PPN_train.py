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
from data_utils import SequenceGenerator, MyCustomCallback, create_dataset_from_generator, create_dataset_from_serialized_generator
from keras.callbacks import LearningRateScheduler, ModelCheckpoint, TensorBoard
from PPN import ParaPredNet
import matplotlib.pyplot as plt

# use mixed precision for faster runtimes and lower memory usage
# keras.mixed_precision.set_global_policy("mixed_float16")

# if results directory already exists, then delete it
if os.path.exists(RESULTS_SAVE_DIR): shutil.rmtree(RESULTS_SAVE_DIR)
if os.path.exists(LOG_DIR): shutil.rmtree(LOG_DIR)
os.mkdir(LOG_DIR)

save_model = True  # if weights will be saved
plot_intermediate = False  # if the intermediate model predictions will be plotted
tensorboard = True  # if the Tensorboard callback will be used
weights_checkpoint_file = os.path.join(WEIGHTS_DIR, 'tensorflow_weights/para_prednet_monkaa_weights.hdf5')  # where weights are loaded prior to training
weights_file = os.path.join(WEIGHTS_DIR, 'tensorflow_weights/para_prednet_monkaa_weights.hdf5')  # where weights will be saved
json_file = os.path.join(WEIGHTS_DIR, 'para_prednet_monkaa_model_ALEX.json')
if os.path.exists(weights_file): os.remove(weights_file) # Careful: this will delete the weights file

# Training data
pfm_paths = []
pfm_paths.append('/home/evalexii/remote_dataset/disparity/family_x2/left/')
pfm_paths.append('/home/evalexii/remote_dataset/material_index/family_x2/left/')
pfm_paths.append('/home/evalexii/remote_dataset/object_index/family_x2/left/')
pfm_paths.append('/home/evalexii/remote_dataset/optical_flow/family_x2/into_future/left/')
pgm_paths = []
pgm_paths.append('/home/evalexii/remote_dataset/motion_boundaries/family_x2/into_future/left/')
png_paths = []
png_paths.append('/home/evalexii/remote_dataset/frames_cleanpass/family_x2/left')
    
# Training parameters
nt = 10  # number of time steps
nb_epoch = 150 # 150
batch_size = 2 # 4
samples_per_epoch = 100 # 500
N_seq_val = 20  # number of sequences to use for validation
output_channels = [3, 12, 24, 48] # [3, 48, 96, 192]
im_shape = (540, 960, 3)

#  Create and split dataset
dataset, length = create_dataset_from_serialized_generator(pfm_paths, pgm_paths, png_paths, 
                                                           im_height=im_shape[0], im_width=im_shape[1], 
                                                           batch_size=batch_size, nt=nt, reserialize=False, shuffle=True)

ts = 0.7
vs = (1 - ts) / 2
train_size = int(ts * length)
val_size = int(vs * length)
test_size = int(vs * length)

train_dataset = dataset.take(train_size)
test_dataset = dataset.skip(train_size)
val_dataset = test_dataset.skip(val_size)
test_dataset = test_dataset.take(test_size)

# # Iterate over the dataset
# for b, batch in enumerate(dataset):
#     lth = len(batch)
#     for bs, batch_seq in enumerate(batch):
#         # print(item.shape)
#         fig, axes = plt.subplots(batch_size, nt, figsize=(15, 5))
#         for i in range(batch_size):
#             for j in range(nt):
#                 axes[i,j].imshow(batch_seq[i,j])
#         plt.savefig(f'./images/test_{b}_{bs}_{i}_{j}.png')

PPN = ParaPredNet(batch_size=batch_size, nt=nt, im_height=im_shape[0], im_width=im_shape[1], output_channels=output_channels) 
PPN.compile(optimizer='adam', loss='mean_squared_error')
PPN.build(input_shape=(None, nt) + im_shape)
print("ParaPredNet compiled...")
print(PPN.summary())
num_layers = len(output_channels)  # number of layers in the architecture
print(f"Top layer resolution: {int(im_shape[0]//(2**(num_layers-1)))} x {int(im_shape[1]//(2**(num_layers-1)))}")

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

history = PPN.fit(train_dataset, steps_per_epoch=samples_per_epoch / batch_size, epochs=nb_epoch, callbacks=callbacks, 
                validation_data=val_dataset, validation_steps=N_seq_val / batch_size)
        
