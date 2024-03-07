import os
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or '2' to filter out INFO messages too
import hickle as hkl

import numpy as np
from six.moves import cPickle as pickle
import subprocess
from kitti_settings import *
import keras
from keras import layers

# # Construct the input layer with no definite frame size.
# inp = layers.Input(shape=(None, 64, 64, 1))

# # We will construct 3 `ConvLSTM2D` layers with batch normalization,
# # followed by a `Conv3D` layer for the spatiotemporal outputs.
# x = layers.ConvLSTM2D(
#     filters=2,
#     kernel_size=(5, 5),
#     padding="same",
#     return_sequences=True,
#     activation="relu",
# )(inp)

# # Next, we will build the complete model and compile it.
# model = keras.models.Model(inp, x)
# model.compile(
#     loss=keras.losses.binary_crossentropy,
#     optimizer=keras.optimizers.Adam(),
# )
# print(model.summary())
# nt = 1
# a = model(np.random.random((4, nt, 64, 64, 1)).astype(np.float32))
# print(a.shape)


# Construct the input layer with no definite frame size.
# inp = layers.Input(shape=(None, 64, 64, 1))

# We will construct 3 `ConvLSTM2D` layers with batch normalization,
# followed by a `Conv3D` layer for the spatiotemporal outputs.
x = layers.ConvLSTM2D(
    filters=2,
    kernel_size=(3, 3),
    padding="same",
    return_sequences=True,
    activation="relu",
)
y = layers.Conv2D(
    filters=1,
    kernel_size=(3, 3),
    padding="same",
    activation="relu",
)

# Next, we will build the complete model and compile it.
# model = keras.models.Model(inp, x)
# model.compile(
#     loss=keras.losses.binary_crossentropy,
#     optimizer=keras.optimizers.Adam(),
# )
# print(model.summary())
nt = 1
a = y(np.random.random((4, 64, 64, 1)).astype(np.float32))
print(a[0].shape)

















# # Data files
# train_file = os.path.join(DATA_DIR, 'X_train.hkl')
# train_sources = os.path.join(DATA_DIR, 'sources_train.hkl')
# val_file = os.path.join(DATA_DIR, 'X_val.hkl')
# val_sources = os.path.join(DATA_DIR, 'sources_val.hkl')

# files = [train_file, train_sources, val_file, val_sources]
# file_names = [os.path.join(DATA_DIR, 'X_train.pkl'), os.path.join(DATA_DIR, 'sources_train.pkl'), os.path.join(DATA_DIR, 'X_val.pkl'), os.path.join(DATA_DIR, 'sources_val.pkl')]


# def fix_my_hickle_files(data_files):
#     for data_file, file_name in zip(data_files, file_names):
#         data = hkl.load(data_file)
#         print("gets here")
#         pickle.dump(data, open(file_name, 'w'))

# def rehickling(data_files, file_names):
#     for data_file, file_name in zip(data_files, file_names):
#         with open(file_name, 'rb') as file:
#             data = pickle.load(file)
#             print("gets here")
#             hkl.dump(data, data_file)

# def hickle_swap(data_files):
#     for data_file in data_files:
#         with open(data_file, 'r') as file:
#             print("opens file")
#             data = hkl.load(file)
#             print("loads file")

#             uninstall_result = subprocess.call(["pip", "uninstall", "hickle"])
#             if uninstall_result == 0:
#                 print("Old package uninstalled successfully.")
#             else:
#                 print("Error uninstalling old package.")

#             install_result = subprocess.call(["pip", "install", "hickle"])
#             if install_result == 0:
#                 print("New package installed successfully.")
#             else:
#                 print("Error installing new package.")

#             # Continue with the rest of your script logic
#             hkl.dump(data, data_file)
#             print("dumps file")

#             uninstall_result = subprocess.call(["pip", "uninstall", "hickle"])
#             if uninstall_result == 0:
#                 print("Old package uninstalled successfully.")
#             else:
#                 print("Error uninstalling old package.")

#             install_result = subprocess.call(["pip", "install", "hickle==2.1.0"])
#             if install_result == 0:
#                 print("New package installed successfully.")
#             else:
#                 print("Error installing new package.")

# def test_hickle(data_files):
#     # uninstall_result = subprocess.call(["pip", "uninstall", "hickle"])
#     # if uninstall_result == 0:
#     #     print("Old package uninstalled successfully.")
#     # else:
#     #     print("Error uninstalling old package.")

#     # install_result = subprocess.call(["pip", "install", "hickle"])
#     # if install_result == 0:
#     #     print("New package installed successfully.")
#     # else:
#     #     print("Error installing new package.")
    
#     # ensure each file can load in hickle==3.4.9
#     for data_file in data_files:
#         with open(data_file, 'r') as file:
#             print("opens file")
#             data = hkl.load(file)
#             print("loads file")

# def grab_data_and_save(data_files):
#     files = [
#         "/home/evalexii/Documents/Thesis/code/prednet/kitti_data/X_train.npy",
#         "/home/evalexii/Documents/Thesis/code/prednet/kitti_data/sources_train.npy",
#         "/home/evalexii/Documents/Thesis/code/prednet/kitti_data/X_val.npy",
#         "/home/evalexii/Documents/Thesis/code/prednet/kitti_data/sources_val.npy"
#     ]
#     for i, data_file in enumerate(data_files):
#         data = np.load(files[i], allow_pickle=True)
#         hkl.dump(data, data_file)
#         print("dumps file")
#     for data_file in data_files:
#         data = hkl.load(data_file)
#         print("loads file")
#     while input("Hold: ") != "q":
#         pass

# def grab_single_data_and_save(data_file):
#     data = np.load("/home/evalexii/Documents/Thesis/code/prednet/add.npy", allow_pickle=True)
#     hkl.dump(data, data_file)
#     print("dumps to file")
#     while input("Hold: ") != "q":
#         pass
    
#     # uninstall_result = subprocess.call(["pip", "uninstall", "hickle"])
#     # if uninstall_result == 0:
#     #     print("Old package uninstalled successfully.")
#     # else:
#     #     print("Error uninstalling old package.")

#     # install_result = subprocess.call(["pip", "install", "hickle==2.1.0"])
#     # if install_result == 0:
#     #     print("New package installed successfully.")
#     # else:
#     #     print("Error installing new package.")

# fix_my_hickle_files(files)
# rehickling(files, file_names)
# hickle_swap(files)
# test_hickle(files)
# grab_data_and_save(files)
# grab_single_data_and_save(train_file)

