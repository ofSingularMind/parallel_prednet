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

import keras
from keras import backend as K
import numpy as np
from keras.layers import Input, Dense, Layer
from keras.models import Model
from PPN import ParaPredNet

# Data files
train_file = os.path.join(DATA_DIR, 'X_train.hkl')
train_sources = os.path.join(DATA_DIR, 'sources_train.hkl')
val_file = os.path.join(DATA_DIR, 'X_val.hkl')
val_sources = os.path.join(DATA_DIR, 'sources_val.hkl')
test_file = os.path.join(DATA_DIR, 'X_test.hkl')
test_sources = os.path.join(DATA_DIR, 'sources_test.hkl')

files = [test_file, test_sources]
file_names = [os.path.join(DATA_DIR, 'X_test.npy'), os.path.join(DATA_DIR, 'sources_test.npy')]


def fix_my_hickle_files(data_files):
    for data_file, file_name in zip(data_files, file_names):
        data = hkl.load(data_file)
        print("gets here")
        pickle.dump(data, open(file_name, 'w'))

def rehickling(data_files, file_names):
    for data_file, file_name in zip(data_files, file_names):
        with open(file_name, 'rb') as file:
            data = pickle.load(file)
            print("gets here")
            hkl.dump(data, data_file)

def hickle_swap(data_files):
    for data_file in data_files:
        with open(data_file, 'r') as file:
            print("opens file")
            data = hkl.load(file)
            print("loads file")

            uninstall_result = subprocess.call(["pip", "uninstall", "hickle"])
            if uninstall_result == 0:
                print("Old package uninstalled successfully.")
            else:
                print("Error uninstalling old package.")

            install_result = subprocess.call(["pip", "install", "hickle"])
            if install_result == 0:
                print("New package installed successfully.")
            else:
                print("Error installing new package.")

            # Continue with the rest of your script logic
            hkl.dump(data, data_file)
            print("dumps file")

            uninstall_result = subprocess.call(["pip", "uninstall", "hickle"])
            if uninstall_result == 0:
                print("Old package uninstalled successfully.")
            else:
                print("Error uninstalling old package.")

            install_result = subprocess.call(["pip", "install", "hickle==2.1.0"])
            if install_result == 0:
                print("New package installed successfully.")
            else:
                print("Error installing new package.")

def test_hickle(data_files):
    for data_file in data_files:
        with open(data_file, 'r') as file:
            print("opens file")
            data = hkl.load(file)
            print("loads file")

def grab_data_and_save(data_files):
    files = [
        "/home/evalexii/Documents/Thesis/code/prednet/kitti_data/X_test.npy",
        "/home/evalexii/Documents/Thesis/code/prednet/kitti_data/sources_test.npy",
    ]
    for i, data_file in enumerate(data_files):
        data = np.load(files[i], allow_pickle=True)
        hkl.dump(data, data_file)
        print("dumps file")
    for data_file in data_files:
        data = hkl.load(data_file)
        print("loads file")
    while input("Hold: ") != "q":
        pass

def grab_single_data_and_save(data_file):
    data = np.load("/home/evalexii/Documents/Thesis/code/prednet/add.npy", allow_pickle=True)
    hkl.dump(data, data_file)
    print("dumps to file")
    while input("Hold: ") != "q":
        pass
    
    # uninstall_result = subprocess.call(["pip", "uninstall", "hickle"])
    # if uninstall_result == 0:
    #     print("Old package uninstalled successfully.")
    # else:
    #     print("Error uninstalling old package.")

    # install_result = subprocess.call(["pip", "install", "hickle==2.1.0"])
    # if install_result == 0:
    #     print("New package installed successfully.")
    # else:
    #     print("Error installing new package.")

# fix_my_hickle_files(files)
# rehickling(files, file_names)
# hickle_swap(files)
# test_hickle(files)
# grab_data_and_save(files)
# grab_single_data_and_save(train_file)

# from data_utils import SequenceGenerator
# n_plot = 4 # 40
# batch_size = 4
# nt = 4
# plot_nt = 4
# weights_file = os.path.join(WEIGHTS_DIR, 'tensorflow_weights/para_prednet_kitti_weights.hdf5')
# test_file = os.path.join('', '/home/evalexii/remote_dataset/X_test.hkl')
# test_sources = os.path.join('', '/home/evalexii/remote_dataset/sources_test.hkl')
# test_generator = SequenceGenerator(test_file, test_sources, nt, sequence_start_mode='unique')

# print(f"Shape: {test_generator.create_n(5).shape}")
    
from PIL import Image
import tensorflow as tf
X_test = hkl.load(test_file)
print(X_test.shape)
# for i in range(X_test.shape[0]):
#     img = Image.fromarray(X_test[i])
#     img.save(f"/home/evalexii/test_imgs/img_{i}.png")
#     print(f"Saved img_{i}.png")

