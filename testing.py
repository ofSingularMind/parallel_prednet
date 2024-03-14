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
from data_utils import dir_PFM_to_PNG, create_dataset_from_generator
import matplotlib.pyplot as plt

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

# n_plot = 4 # 40
# batch_size = 4
# nt = 4
# plot_nt = 4
# weights_file = os.path.join(WEIGHTS_DIR, 'tensorflow_weights/para_prednet_kitti_weights.hdf5')
# test_file = os.path.join(DATA_DIR, 'X_test.hkl')
# test_sources = os.path.join(DATA_DIR, 'sources_test.hkl')

# test_PPN = ParaPredNet(batch_size=batch_size, nt=nt)
# test_PPN.output_mode = 'Prediction'
# test_PPN.compile(optimizer='adam', loss='mean_squared_error')
# test_PPN.build(input_shape=(None, nt, 128, 160, 3))
# print("ParaPredNet compiled...")

# print(test_PPN.summary())
    

# dir_PFM_to_PNG("/home/evalexii/Downloads/Sampler/Monkaa/")


# Define paths to .pfm and .png directories
pfm_paths = []
pfm_paths.append('/home/evalexii/remote_dataset/disparity/family_x2/left/')
pfm_paths.append('/home/evalexii/remote_dataset/material_index/family_x2/left/')
pfm_paths.append('/home/evalexii/remote_dataset/object_index/family_x2/left/')
pfm_paths.append('/home/evalexii/remote_dataset/optical_flow/family_x2/into_future/left/')
pgm_paths = []
pgm_paths.append('/home/evalexii/remote_dataset/motion_boundaries/family_x2/into_future/left/')
png_paths = []
png_paths.append('/home/evalexii/remote_dataset/frames_cleanpass/family_x2/left')

batch_size = 3
nt = 10
dataset, length = create_dataset_from_generator(pfm_paths, pgm_paths, png_paths, batch_size=batch_size, nt=nt)
ts = 0.7
vs = (1 - ts) / 2
train_size = int(ts * length)
val_size = int(vs * length)
test_size = int(vs * length)

train_dataset = dataset.take(train_size)
test_dataset = dataset.skip(train_size)
val_dataset = test_dataset.skip(val_size)
test_dataset = test_dataset.take(test_size)

# Iterate over the dataset
for batch in train_dataset:
    for j in range(batch_size):
        fig, axes = plt.subplots(len(batch), nt, figsize=(15, 5))
        for i, image in enumerate(batch):
            print(image.shape)
            for k in range(nt):
                axes[i,k].imshow(image[j,k])
        plt.show()

# def sort_files_by_name(files):
#     return sorted(files, key=lambda x: os.path.basename(x))

# def generator(pfm_paths, pgm_paths, png_paths, nt=10):
#     num_pfm_paths = len(pfm_paths)
#     num_pgm_paths = len(pgm_paths)
#     num_png_paths = len(png_paths)
#     total_paths = num_pfm_paths + num_pgm_paths + num_png_paths
    
#     pfm_sources = []
#     pgm_sources = []
#     png_sources = []

#     for pfm_path in pfm_paths:
#         pfm_sources += [sort_files_by_name(glob.glob(pfm_path + '/*.pfm'))]
#     for pgm_path in pgm_paths:
#         pgm_sources += [sort_files_by_name(glob.glob(pgm_path + '/*.pgm'))]
#     for png_path in png_paths:
#         png_sources += [sort_files_by_name(glob.glob(png_path + '/*.png'))]

#     all_files = np.array(list(zip(*pfm_sources, *pgm_sources, *png_sources)))
#     for i in range(all_files.shape[0]):
        
#         nt_outs = []
#         for j in range(num_pfm_paths):
#             nt_outs.append(np.array([readPFM(all_files[i+k, j]) for k in range(nt)]))
#         for j in range(num_pgm_paths):
#             nt_outs.append(np.array([Image.open(all_files[i+k, j + num_pfm_paths]) for k in range(nt)]) / 255.0)
#         for j in range(num_png_paths):
#             nt_outs.append(np.array([Image.open(all_files[i+k, j + num_pfm_paths + num_pgm_paths]) for k in range(nt)]) / 255.0)
    
#         return nt_outs

# generator = generator(pfm_paths, pgm_paths, png_paths)
# print("asd")