import os
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")
# or '2' to filter out INFO messages too
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import subprocess
from six.moves import cPickle as pickle
import random
import time
from PIL import Image
import glob
import sys
import re
from PPN_models.PPN_Baseline import ParaPredNet

# from monkaa_settings import *
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from keras.layers import Input
from keras.callbacks import Callback
from keras.models import Model, model_from_json
from keras.preprocessing.image import Iterator
from keras import backend as K
import tensorflow as tf
import keras
import numpy as np
import hickle as hkl
from config import update_settings, get_settings
import flow_vis

DATA_DIR, WEIGHTS_DIR, RESULTS_SAVE_DIR, LOG_DIR = get_settings()["dirs"]


# Data generator that creates sequences for input into PredNet.
class SequenceGenerator(Iterator):
    def __init__(self, data_file, source_file, nt, batch_size=8, shuffle=False, seed=None, output_mode="error", sequence_start_mode="all", N_seq=None, data_format=K.image_data_format()):
        # X will be like (n_images, nb_cols, nb_rows, nb_channels)
        self.X = hkl.load(data_file)
        # source for each image so when creating sequences can assure that consecutive frames are from same video
        self.sources = hkl.load(source_file)
        self.nt = nt
        self.batch_size = batch_size
        self.data_format = data_format
        assert sequence_start_mode in {"all", "unique"}, "sequence_start_mode must be in {all, unique}"
        self.sequence_start_mode = sequence_start_mode
        assert output_mode in {"error", "prediction"}, "output_mode must be in {error, prediction}"
        self.output_mode = output_mode

        if self.data_format == "channels_first":
            self.X = np.transpose(self.X, (0, 3, 1, 2))
        self.im_shape = self.X[0].shape

        if self.sequence_start_mode == "all":  # allow for any possible sequence, starting from any frame
            self.possible_starts = np.array([i for i in range(self.X.shape[0] - self.nt) if self.sources[i] == self.sources[i + self.nt - 1]])
        # create sequences where each unique frame is in at most one sequence
        elif self.sequence_start_mode == "unique":
            curr_location = 0
            possible_starts = []
            while curr_location < self.X.shape[0] - self.nt + 1:
                if (self.sources[curr_location]
                    == self.sources[curr_location + self.nt - 1]):
                    possible_starts.append(curr_location)
                    curr_location += self.nt
                else:
                    curr_location += 1
            self.possible_starts = possible_starts

        if shuffle:
            self.possible_starts = np.random.permutation(self.possible_starts)
        # select a subset of sequences if want to
        if N_seq is not None and len(self.possible_starts) > N_seq:
            self.possible_starts = self.possible_starts[:N_seq]
        self.N_sequences = len(self.possible_starts)
        super(SequenceGenerator, self).__init__(len(self.possible_starts), batch_size, shuffle, seed)

    def __getitem__(self, null):
        return self.next()

    def next(self):
        with self.lock:
            current_index = (self.batch_index * self.batch_size) % self.n
            index_array, current_batch_size = (next(self.index_generator), self.batch_size, )
        batch_x = np.zeros((current_batch_size, self.nt) + self.im_shape, np.float32)
        for i, idx in enumerate(index_array):
            idx = self.possible_starts[idx]
            batch_x[i] = self.preprocess(self.X[idx : idx + self.nt])
        if self.output_mode == "error":  # model outputs errors, so y should be zeros
            batch_y = np.zeros(current_batch_size, np.float32)
        elif self.output_mode == "prediction":  # output actual pixels
            batch_y = batch_x
        return batch_x, batch_y

    def preprocess(self, X):
        return X.astype(np.float32) / 255

    def create_n(self, n=1):
        assert n <= len(self.possible_starts), "Can't create more sequences than there are possible starts"
        X_all = np.zeros((n, self.nt) + self.im_shape, np.float32)
        idxes = np.random.choice(len(self.possible_starts), n, replace=False)
        for i, idx in enumerate(np.array(self.possible_starts)[idxes].tolist()):
            X_all[i] = self.preprocess(self.X[idx : idx + self.nt])
        return X_all

    def create_all(self):
        X_all = np.zeros((self.N_sequences, self.nt) + self.im_shape, np.float32)
        for i, idx in enumerate(self.possible_starts):
            X_all[i] = self.preprocess(self.X[idx : idx + self.nt])
        return X_all


class IntermediateEvaluations(Callback):
    def __init__(self, test_dataset, length, batch_size=4, nt=10, output_channels=[3, 48, 96, 192], dataset="kitti", model_choice="baseline"):
        super(IntermediateEvaluations, self).__init__()
        self.test_dataset = test_dataset
        self.dataset_iterator = iter(self.test_dataset)
        self.n_plot = batch_size  # 40
        self.batch_size = batch_size
        self.nt = nt
        self.plot_nt = nt
        self.output_channels = output_channels
        self.dataset = dataset
        self.model_choice = model_choice
        # self.weights_file = os.path.join(WEIGHTS_DIR, "tensorflow_weights/para_prednet_monkaa_weights.hdf5")
        self.rg_colormap = LinearSegmentedColormap.from_list('custom_cmap', [(0, 'red'), (0.5, 'black'), (1, 'green')])
        # Retrieve target sequence (use the same sequence(s) always)
        self.X_test_inputs = [next(self.dataset_iterator) for _ in range(10)][-1][0]  # take just batch_x not batch_y
        self.Xtc = 3 # X_test_channels


        if self.dataset == "kitti":
            self.X_test = self.X_test_inputs
        elif self.dataset in ["monkaa", "driving"] and self.model_choice != "multi_channel":
            self.X_test = self.X_test_inputs[-1] # take only the PNG images for MSE calcs and plotting
        elif self.dataset in ["rolling_square", "rolling_circle", "all_rolling", "ball_collisions", "general_ellipse_vertical", "general_cross_horizontal", "various"] and self.model_choice != "multi_channel":
            self.X_test = self.X_test_inputs # take only the PNG images for MSE calcs and plotting
            self.Xtc = self.X_test.shape[-1] # X_test_channels        
        elif self.dataset == "monkaa" and self.model_choice == "multi_channel":
            # self.X_test_inputs = 
            self.X_test = tf.concat([self.X_test_inputs[0], self.X_test_inputs[3], self.X_test_inputs[5]], axis=-1)
            self.X_test = self.X_test.numpy()
            self.X_test[..., 0] = np.interp(self.X_test[..., 0], (self.X_test[..., 0].min(), self.X_test[..., 0].max()), (0, 1))
            # self.X_test_mat = self.X_test[..., 1].astype(np.int32)
            # self.X_test_obj = self.X_test[..., 2].astype(np.int32)
            # self.X_test_opt = np.zeros_like(self.X_test[..., 3:6], dtype=np.int32)
            for b in range(self.batch_size):
                for t in range(self.nt):
                    self.X_test[b, t, ..., 1:4] = flow_vis.flow_to_color(self.X_test[b, t, ..., 1:3], convert_to_bgr=False).astype(np.float32) // 255.0
            # self.X_test_mot = self.X_test[..., 6]
        elif self.dataset == "driving" and self.model_choice == "multi_channel":
            # self.X_test_inputs = 
            self.X_test = tf.concat([self.X_test_inputs[0], self.X_test_inputs[1], self.X_test_inputs[2]], axis=-1)
            self.X_test = self.X_test.numpy()
            self.X_test[..., 0] = np.interp(self.X_test[..., 0], (self.X_test[..., 0].min(), self.X_test[..., 0].max()), (0, 1))
            # self.X_test_mat = self.X_test[..., 1].astype(np.int32)
            # self.X_test_obj = self.X_test[..., 2].astype(np.int32)
            # self.X_test_opt = np.zeros_like(self.X_test[..., 3:6], dtype=np.int32)
            for b in range(self.batch_size):
                for t in range(self.nt):
                    self.X_test[b, t, ..., 1:4] = flow_vis.flow_to_color(self.X_test[b, t, ..., 1:3], convert_to_bgr=False).astype(np.float32) // 255.0
            # self.X_test_mot = self.X_test[..., 6]

        if not os.path.exists(RESULTS_SAVE_DIR):
            os.makedirs(RESULTS_SAVE_DIR, exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        if epoch % 1 == 0 or epoch == 1:
            self.plot_training_samples(epoch)

    def plot_training_samples(self, epoch):
        """
        Evaluate trained PredNet on KITTI or Monkaa sequences.
        Calculates mean-squared error and plots predictions.
        """

        # Calculate predicted sequence(s)
        #   self.model.layers[-1] is the PredNet layer
        self.model.layers[-1].output_mode = "Error_Images_and_Prediction"
        if self.model_choice == "baseline":
            error_images, X_hat = self.model.layers[-1](self.X_test_inputs)
        elif self.model_choice == "cl_delta":
            error_images, X_hat, X_delta_hat = self.model.layers[-1](self.X_test_inputs)
            X_delta_hat_gray = np.mean(X_delta_hat, axis=-1)
        elif self.model_choice == "cl_recon":
            error_images, X_hat, X_recon_hat = self.model.layers[-1](self.X_test_inputs)
        elif self.model_choice == "multi_channel":
            error_images, X_hat = self.model.layers[-1](self.X_test_inputs)
            X_hat = X_hat.numpy()
            X_hat[..., 0] = np.interp(X_hat[..., 0], (X_hat[..., 0].min(), X_hat[..., 0].max()), (0, 1))
            X_hat[..., -3:] = np.interp(X_hat[..., -3:], (X_hat[..., -3:].min(), X_hat[..., -3:].max()), (0, 1))
            # X_hat_mat = X_hat[..., 1].astype(np.int32)
            # X_hat_obj = X_hat[..., 2].astype(np.int32)
            # X_hat_opt = np.zeros_like(X_hat[..., 1:4], dtype=np.int32)
            for b in range(self.batch_size):
                for t in range(self.nt):
                    X_hat[b, t, ..., 1:4] = flow_vis.flow_to_color(X_hat[b, t, ..., 1:3], convert_to_bgr=False).astype(np.float32) // 255.0
            # X_hat_mot = X_hat[..., 6]

        error_images_gray = np.mean(error_images, axis=-1)
        self.model.layers[-1].output_mode = "Error"
        
        # for c in range(X_test.shape[-1]):
        #     # print min/max values for each channel for both X_test and X_hat
        #     print(f"X_test channel {c} min: {X_test[..., c].min()}, max: {X_test[..., c].max()}, dtype: {X_test[..., c].dtype}")
        #     print(f"X_hat channel {c} min: {X_hat[..., c].min()}, max: {X_hat[..., c].max()}, dtype: {X_hat[..., c].dtype}")

        
        # Compare MSE of PredNet predictions vs. using last frame.  Write results to prediction_scores.txt
        # look at all timesteps except the first
        mse_model = np.mean((self.X_test[:, 1:, ..., -self.Xtc:] - X_hat[:, 1:, ..., -self.Xtc:]) ** 2)
        mse_prev = np.mean((self.X_test[:, :-1, ..., -self.Xtc:] - self.X_test[:, 1:, ..., -self.Xtc:]) ** 2)
        f = open(RESULTS_SAVE_DIR + "training_scores.txt", "a+")
        f.write("======================= %i : Epoch\n" % epoch)
        f.write("%f : Model MSE\n" % mse_model)
        f.write("%f : Previous Frame MSE\n" % mse_prev)
        f.close()

        # Plot some training predictions
        aspect_ratio = float(X_hat.shape[2]) / X_hat.shape[3]
        if self.model_choice=="baseline":
            gs = gridspec.GridSpec(3, self.plot_nt)
            plt.figure(figsize=(3 * self.plot_nt, 10 * aspect_ratio), layout="constrained")
        elif self.model_choice=="cl_delta" or self.model_choice=="cl_recon":
            gs = gridspec.GridSpec(5, self.plot_nt)
            plt.figure(figsize=(3 * self.plot_nt, 15 * aspect_ratio), layout="constrained")
        elif self.model_choice=="multi_channel":
            gs = gridspec.GridSpec(7, self.plot_nt) # 13 for all modalities
            plt.figure(figsize=(3 * self.plot_nt, 20 * aspect_ratio), layout="constrained")
        gs.update(wspace=0.0, hspace=0.0)
        plot_save_dir = os.path.join(RESULTS_SAVE_DIR, "training_plots/")
        if not os.path.exists(plot_save_dir):
            os.makedirs(plot_save_dir, exist_ok=True)
        plot_idx = np.random.permutation(self.X_test.shape[0])[: self.n_plot]
        for i in plot_idx:
            X_test_last = tf.zeros_like(self.X_test[i, -1])
            for t in range(self.plot_nt):
                plt.subplot(gs[t])
                plt.imshow(X_hat[i, t, ..., -self.Xtc:], interpolation="none")
                plt.tick_params(axis="both", which="both", bottom="off", top="off", left="off", right="off", labelbottom="off", labelleft="off", )
                if t == 0:
                    plt.ylabel("Predicted", fontsize=10)

                plt.subplot(gs[t + self.plot_nt])
                plt.imshow(self.X_test[i, t, ..., -self.Xtc:], interpolation="none")
                plt.tick_params(axis="both", which="both", bottom="off", top="off", left="off", right="off", labelbottom="off", labelleft="off", )
                if t == 0:
                    plt.ylabel("Actual", fontsize=10)

                plt.subplot(gs[t + 2 * self.plot_nt])
                plt.imshow(error_images_gray[i, t], cmap=self.rg_colormap)
                plt.tick_params(axis="both", which="both", bottom="off", top="off", left="off", right="off", labelbottom="off", labelleft="off", )
                if t == 0:
                    plt.ylabel("Error", fontsize=10)
                
                # Composite Learning Delta Images
                if self.model_choice == "cl_delta":
                    plt.subplot(gs[t + 3 * self.plot_nt])
                    plt.imshow(X_delta_hat_gray[i, t], cmap=self.rg_colormap)
                    plt.tick_params(axis="both", which="both", bottom="off", top="off", left="off", right="off", labelbottom="off", labelleft="off", )
                    if t == 0:
                        plt.ylabel("Predicted Delta", fontsize=10)

                    plt.subplot(gs[t + 4 * self.plot_nt])
                    if t > 0:
                        plt.imshow(np.mean((self.X_test[i, t] - X_test_last), axis=-1), cmap=self.rg_colormap)
                    else:
                        plt.imshow(np.mean((self.X_test[i, t] - self.X_test[i, t]), axis=-1), cmap=self.rg_colormap)
                    plt.tick_params(axis="both", which="both", bottom="off", top="off", left="off", right="off", labelbottom="off", labelleft="off", )
                    if t == 0:
                        plt.ylabel("Actual Delta", fontsize=10)
                    X_test_last = self.X_test[i, t]
                
                # Composite Learning Reconstructed Images
                elif self.model_choice == "cl_recon":
                    plt.subplot(gs[t + 3 * self.plot_nt])
                    plt.imshow(X_recon_hat[i, t], interpolation="none")
                    plt.tick_params(axis="both", which="both", bottom="off", top="off", left="off", right="off", labelbottom="off", labelleft="off", )
                    if t == 0:
                        plt.ylabel("Predicted Recon", fontsize=10) 

                    plt.subplot(gs[t + 4 * self.plot_nt])
                    plt.imshow(X_test_last, interpolation="none")
                    plt.tick_params(axis="both", which="both", bottom="off", top="off", left="off", right="off", labelbottom="off", labelleft="off", )
                    if t == 0:
                        plt.ylabel("Actual Recon", fontsize=10) 
                    X_test_last = self.X_test[i, t]

                # Multi Channel Images
                elif self.model_choice == "multi_channel":
                    # DISPARITY
                    plt.subplot(gs[t + 3 * self.plot_nt])
                    plt.imshow(X_hat[i, t, ..., 0], interpolation="none")
                    plt.tick_params(axis="both", which="both", bottom="off", top="off", left="off", right="off", labelbottom="off", labelleft="off", )
                    if t == 0:
                        plt.ylabel("Predicted Disp", fontsize=10)

                    plt.subplot(gs[t + 4 * self.plot_nt])
                    plt.imshow(self.X_test[i, t, ..., 0], interpolation="none")
                    plt.tick_params(axis="both", which="both", bottom="off", top="off", left="off", right="off", labelbottom="off", labelleft="off", )
                    if t == 0:
                        plt.ylabel("Actual Disp", fontsize=10)
                    
                    # # MATERIAL SEGMENTATION
                    # plt.subplot(gs[t + 5 * self.plot_nt])
                    # plt.imshow(X_hat_mat[i, t, ...], interpolation="none")
                    # plt.tick_params(axis="both", which="both", bottom="off", top="off", left="off", right="off", labelbottom="off", labelleft="off", )
                    # if t == 0:
                    #     plt.ylabel("Predicted Mat", fontsize=10)

                    # plt.subplot(gs[t + 6 * self.plot_nt])
                    # plt.imshow(self.X_test_mat[i, t, ...], interpolation="none")
                    # plt.tick_params(axis="both", which="both", bottom="off", top="off", left="off", right="off", labelbottom="off", labelleft="off", )
                    # if t == 0:
                    #     plt.ylabel("Actual Mat", fontsize=10)
                    
                    # # OBJECT SEGMENTATION
                    # plt.subplot(gs[t + 7 * self.plot_nt])
                    # plt.imshow(X_hat_obj[i, t, ...], interpolation="none")
                    # plt.tick_params(axis="both", which="both", bottom="off", top="off", left="off", right="off", labelbottom="off", labelleft="off", )
                    # if t == 0:
                    #     plt.ylabel("Predicted Obj", fontsize=10)

                    # plt.subplot(gs[t + 8 * self.plot_nt])
                    # plt.imshow(self.X_test_obj[i, t, ...], interpolation="none")
                    # plt.tick_params(axis="both", which="both", bottom="off", top="off", left="off", right="off", labelbottom="off", labelleft="off", )
                    # if t == 0:
                    #     plt.ylabel("Actual Obj", fontsize=10)
                    
                    # OPTICAL FLOW
                    plt.subplot(gs[t + 5 * self.plot_nt])
                    plt.imshow(X_hat[i, t, ..., 1:4], interpolation="none")
                    plt.tick_params(axis="both", which="both", bottom="off", top="off", left="off", right="off", labelbottom="off", labelleft="off", )
                    if t == 0:
                        plt.ylabel("Predicted Opt", fontsize=10)

                    plt.subplot(gs[t + 6 * self.plot_nt])
                    plt.imshow(self.X_test[i, t, ..., 1:4], interpolation="none")
                    plt.tick_params(axis="both", which="both", bottom="off", top="off", left="off", right="off", labelbottom="off", labelleft="off", )
                    if t == 0:
                        plt.ylabel("Actual Opt", fontsize=10)
                    
                    # # MOTION BOUNDARIES
                    # plt.subplot(gs[t + 11 * self.plot_nt])
                    # plt.imshow(X_hat_mot[i, t], interpolation="none", cmap="gray")
                    # plt.tick_params(axis="both", which="both", bottom="off", top="off", left="off", right="off", labelbottom="off", labelleft="off", )
                    # if t == 0:
                    #     plt.ylabel("Predicted Mot", fontsize=10)

                    # plt.subplot(gs[t + 12 * self.plot_nt])
                    # plt.imshow(self.X_test_mot[i, t], interpolation="none", cmap="gray")
                    # plt.tick_params(axis="both", which="both", bottom="off", top="off", left="off", right="off", labelbottom="off", labelleft="off", )
                    # if t == 0:
                    #     plt.ylabel("Actual Mot", fontsize=10)
                    

            plt.savefig(plot_save_dir + "e" + str(epoch) + "_plot_" + str(i) + ".png")
            plt.clf()


def writePFM(file, image, scale=1):
    file = open(file, "wb")

    color = None

    if image.dtype.name != "float32":
        raise Exception("Image dtype must be float32.")

    image = np.flipud(image)

    if len(image.shape) == 3 and image.shape[2] == 3:  # color image
        color = True
    # greyscale
    elif len(image.shape) == 2 or len(image.shape) == 3 and image.shape[2] == 1:
        color = False
    else:
        raise Exception("Image must have H x W x 3, H x W x 1 or H x W dimensions.")

    file.write("PF\n" if color else "Pf\n")
    file.write("%d %d\n" % (image.shape[1], image.shape[0]))

    endian = image.dtype.byteorder

    if endian == "<" or endian == "=" and sys.byteorder == "little":
        scale = -scale

    file.write("%f\n" % scale)

    image.tofile(file)


def dir_PFM_to_PNG(dir):
    for obj in os.listdir(dir):
        print(f"Processing {obj}. Isdir == {os.path.isdir(dir + obj)}")
        if os.path.isdir(dir + obj):
            dir_PFM_to_PNG(dir + obj + "/")
        elif obj.endswith(".pfm"):
            data, scale = readPFM(dir + obj)
            print(f"Converting {obj} to {obj[:-4]}.png")
            if (np.max(data) > 1) or (np.min(data) < 0):
                data = (data - np.min(data)) / (np.max(data) - np.min(data))
            plt.imsave(dir + obj[:-4] + ".png", data)
            print(f"Processed {obj} to {obj[:-4]}.png")


def readPFM(file):
    file = open(file, "rb")

    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().rstrip()
    if header == b"PF":
        color = True
    elif header == b"Pf":
        color = False
    else:
        raise Exception("Not a PFM file.")

    dim_match = re.match(b"^(\d+)\s(\d+)\s$", file.readline())
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception("Malformed PFM header.")

    scale = float(file.readline().rstrip())
    if scale < 0:  # little-endian
        endian = "<"
        scale = -scale
    else:
        endian = ">"  # big-endian

    data = np.fromfile(file, endian + "f")
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    return data


def sort_files_by_name(files):
    return sorted(files, key=lambda x: os.path.basename(x))


def serialize_dataset(pfm_paths, pgm_paths, png_paths, dataset_name="driving", start_time=time.perf_counter(), single_channel=False):
    print(f"Start to serialize at {time.perf_counter() - start_time} seconds.")
    pfm_sources = []
    pgm_sources = []
    png_sources = []

    for pfm_path in pfm_paths:
        pfm_sources += [sort_files_by_name(glob.glob(pfm_path + "/*.pfm"))]
    for pgm_path in pgm_paths:
        pgm_sources += [sort_files_by_name(glob.glob(pgm_path + "/*.pgm"))]
    for png_path in png_paths:
        png_sources += [sort_files_by_name(glob.glob(png_path + "/*.png"))]

    if single_channel:
        assert len(png_sources) > 0, "Single channel requires at least one PNG source."

    all_files = np.array(list(zip(*pfm_sources, *pgm_sources, *png_sources)))

    # Get the length of the dataset
    length = all_files.shape[0]

    # Store the dataset in list with one key per source
    dataset = [0 for _ in range(all_files.shape[1])]

    # Prepare array for filling with data
    # data = np.zeros_like(all_files, dtype=np.float32)

    if dataset_name == "driving":
        temp = np.minimum(length, 200)
    else:    
        temp = np.minimum(length, 1000)
    print(f"Start to load images at {time.perf_counter() - start_time} seconds.")

    for j in range(len(pfm_paths)):
        l = []
        # last = time.perf_counter()
        for i in range(temp):
            im = readPFM(all_files[i, j])
            if j == 0:
                im = np.interp(im, (im.min(), im.max()), (0, 1))
            if len(im.shape) == 2:
                # expand to include channel dimension
                im = np.expand_dims(im, axis=2)
            l.append(im)
            # l = np.array(np.expand_dims(im, axis=0)) if i == 0 else np.concatenate((l, np.expand_dims(im, axis=0)), axis=0)
            # print(f"PFM image {i+1} of {temp} done in {time.perf_counter() - last} seconds.")
            # last = time.perf_counter()
        dataset[j] = np.array(l)
        print(f"PFM source {j+1} of {len(pfm_paths)} done at {time.perf_counter() - start_time} seconds.")

    for j in range(len(pgm_paths)):
        l = []
        # last = time.perf_counter()
        for i in range(temp):
            im = np.array(Image.open(all_files[i, j + len(pfm_paths)]), dtype=np.float32) / 255.0
            if len(im.shape) == 2:
                # expand to include channel dimension
                im = np.expand_dims(im, axis=2)
            l.append(im)
            # print(f"PGM image {i} of {temp-1} done in {time.perf_counter() - last} seconds.")
            # last = time.perf_counter()
        dataset[j + len(pfm_paths)] = np.array(l)
        print(f"PGM source {j+1} of {len(pgm_paths)} done at {time.perf_counter() - start_time} seconds.")

    for j in range(len(png_paths)):
        l = []
        # last = time.perf_counter()
        for i in range(temp):
            im = (np.array(Image.open(all_files[i, j + len(pfm_paths) + len(pgm_paths)]), dtype=np.float32)
                / 255.0)
            if len(im.shape) == 2:
                # expand to include channel dimension
                im = np.expand_dims(im, axis=2)
            if single_channel:
                im = np.mean(im, axis=2)
                im = np.expand_dims(im, axis=2)
            l.append(im)
            # print(f"PNG image {i} of {temp-1} done in {time.perf_counter() - last} seconds.")
            # last = time.perf_counter()
        dataset[j + len(pfm_paths) + len(pgm_paths)] = np.array(l)
        print(f"PNG source {j+1} of {len(png_paths)} done at {time.perf_counter() - start_time} seconds.")

    # # normalize all image data to float between 0..1
    # for source in dataset:
    #     s_min = np.min(source)
    #     s_max = np.max(source)
    #     print(f"Before: Min: {s_min}, Max: {s_max}")
    #     # -3, 0, 1 -> 0, 3, 4 -> 0, 0.75, 1
    #     source = (source - s_min) / (s_max - s_min)
    #     s_min = np.min(source)
    #     s_max = np.max(source)
    #     print(f"After: Min: {s_min}, Max: {s_max}")
    # print(f"Image normalization complete at {time.perf_counter() - start_time} seconds.")

    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR, exist_ok=True)
    # where weights are loaded prior to training
    dataset_file = os.path.join(DATA_DIR, f"{dataset_name}_train.hkl")

    hkl.dump(dataset, dataset_file, mode="w")
    print(f"HKL dump done at {time.perf_counter() - start_time} seconds.")
    print(f"Dataset serialization complete at {time.perf_counter() - start_time} seconds.")


def create_dataset_from_generator(pfm_paths, pgm_paths, png_paths, im_height=540, im_width=960, batch_size=1, nt=10, shuffle=True, ):
    num_pfm_paths = len(pfm_paths)
    num_pgm_paths = len(pgm_paths)
    num_png_paths = len(png_paths)
    total_paths = num_pfm_paths + num_pgm_paths + num_png_paths

    def generator():
        pfm_sources = []
        pgm_sources = []
        png_sources = []

        for pfm_path in pfm_paths:
            pfm_sources += [sort_files_by_name(glob.glob(pfm_path + "/*.pfm"))]
        for pgm_path in pgm_paths:
            pgm_sources += [sort_files_by_name(glob.glob(pgm_path + "/*.pgm"))]
        for png_path in png_paths:
            png_sources += [sort_files_by_name(glob.glob(png_path + "/*.png"))]

        all_files = np.array(list(zip(*pfm_sources, *pgm_sources, *png_sources)))

        assert (nt <= all_files.shape[0]), "nt must be less than or equal to the number of files in the dataset"

        for i in range(all_files.shape[0] + 1 - nt):
            nt_outs = []
            for j in range(num_pfm_paths):
                nt_outs.append([readPFM(all_files[i + k, j]) for k in range(nt)])
            for j in range(num_pgm_paths):
                nt_outs.append([np.array(Image.open(all_files[i + k, j + num_pfm_paths])) / 255.0 for k in range(nt)])
            for j in range(num_png_paths):
                nt_outs.append([np.array(Image.open(all_files[i + k, j + num_pfm_paths + num_pgm_paths])) / 255.0 for k in range(nt)])

            yield tuple(nt_outs)

    dataset = tf.data.Dataset.from_generator(generator, output_signature=((tf.TensorSpec(shape=(nt, im_height, im_width), dtype=tf.float32)), (tf.TensorSpec(shape=(nt, im_height, im_width), dtype=tf.float32)), (tf.TensorSpec(shape=(nt, im_height, im_width), dtype=tf.float32)), (tf.TensorSpec(shape=(nt, im_height, im_width, 3), dtype=tf.float32)), (tf.TensorSpec(shape=(nt, im_height, im_width), dtype=tf.float32)), (tf.TensorSpec(shape=(nt, im_height, im_width, 3), dtype=tf.float32)), ), )

    # Get the length of the dataset
    length = len(glob.glob(pfm_paths[0] + "/*.pfm"))

    # Shuffle, batch and prefetch the dataset
    if shuffle:
        dataset = dataset.shuffle(buffer_size=length, reshuffle_each_iteration=True)
    dataset = dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

    return dataset, length


def create_dataset_from_serialized_generator(pfm_paths, pgm_paths, png_paths, output_mode="Error", dataset_name="driving", im_height=540, im_width=960, batch_size=4, nt=10, train_split=0.7, reserialize=False, shuffle=True, resize=False, single_channel=False):
    start_time = time.perf_counter()
    if reserialize:
        serialize_dataset(pfm_paths, pgm_paths, png_paths, dataset_name=dataset_name, start_time=start_time, single_channel=single_channel)
        print("Reserialized dataset.")
    else:
        print("Using previously serialized dataset.")
    print(f"Begin tf.data.Dataset creation at {time.perf_counter() - start_time} seconds.")

    num_pfm_paths = len(pfm_paths)
    num_pgm_paths = len(pgm_paths)
    num_png_paths = len(png_paths)
    num_total_paths = num_pfm_paths + num_pgm_paths + num_png_paths

    # list of numpy arrays, one for each source
    all_files = hkl.load(os.path.join(DATA_DIR, f"{dataset_name}_train.hkl"))
    num_samples = all_files[0].shape[0]
    assert all([all_files[i].shape[0] == num_samples for i in range(num_total_paths)]), "All sources must have the same number of samples"

    # Get the length of the dataset (number of unique sequences, nus)
    nus = num_samples + 1 - nt
    length = nus
    train_samples = int(train_split * nus)
    val_samples = int((1 - train_split) / 2 * nus)
    test_samples = int((1 - train_split) / 2 * nus)
    all_details = [(0, train_samples, train_samples), 
                   (train_samples, train_samples + val_samples, val_samples), 
                   (train_samples + val_samples, train_samples + val_samples + test_samples, test_samples)]

    def create_generator(details, shuffle):
        def generator():
            start, stop, num_samples = details
            iterator = (random.sample(range(start, stop), num_samples) if shuffle else range(num_samples + 1 - nt))
            for it, i in enumerate(iterator):
                # print(f"{it}, {i}")
                nt_outs = []
                for j in range(num_total_paths):
                    if resize:
                        nt_outs.append([tf.image.resize(all_files[j][i + k], (im_height, im_width)) for k in range(nt)])
                    else:
                        nt_outs.append([all_files[j][i + k] for k in range(nt)])
                batch_x = tuple(nt_outs) if len(nt_outs) > 1 else tuple(nt_outs)[0] 
                if output_mode == "Error":
                    batch_y = [0.0]
                    yield (batch_x, batch_y)
                elif output_mode == "Prediction":
                    raise NotImplementedError
                    yield (batch_x, batch_x)

        return generator

    datasets = []
    for details in all_details:
        gen = create_generator(details, shuffle)
        if dataset_name == "monkaa":
            dataset = tf.data.Dataset.from_generator(gen, output_signature=((tf.TensorSpec(shape=(nt, im_height, im_width, 1), dtype=tf.float32), tf.TensorSpec(shape=(nt, im_height, im_width, 1), dtype=tf.float32), tf.TensorSpec(shape=(nt, im_height, im_width, 1), dtype=tf.float32), tf.TensorSpec(shape=(nt, im_height, im_width, 3), dtype=tf.float32), tf.TensorSpec(shape=(nt, im_height, im_width, 1), dtype=tf.float32), tf.TensorSpec(shape=(nt, im_height, im_width, 3), dtype=tf.float32)), tf.TensorSpec(shape=(1), dtype=tf.float32)))
        elif dataset_name == "driving":
            dataset = tf.data.Dataset.from_generator(gen, output_signature=((tf.TensorSpec(shape=(nt, im_height, im_width, 1), dtype=tf.float32), tf.TensorSpec(shape=(nt, im_height, im_width, 3), dtype=tf.float32), tf.TensorSpec(shape=(nt, im_height, im_width, 3), dtype=tf.float32)), tf.TensorSpec(shape=(1), dtype=tf.float32)))
        elif dataset_name not in ["all_rolling", "various"]:
            if single_channel:
                dataset = tf.data.Dataset.from_generator(gen, output_signature=(tf.TensorSpec(shape=(nt, im_height, im_width, 1), dtype=tf.float32), tf.TensorSpec(shape=(1), dtype=tf.float32)))
            else:
                dataset = tf.data.Dataset.from_generator(gen, output_signature=(tf.TensorSpec(shape=(nt, im_height, im_width, 3), dtype=tf.float32), tf.TensorSpec(shape=(1), dtype=tf.float32)))
            dataset = (dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE).repeat())
        else:
            if single_channel:
                dataset = tf.data.Dataset.from_generator(gen, output_signature=(tf.TensorSpec(shape=(nt, im_height, im_width, 1), dtype=tf.float32), tf.TensorSpec(shape=(1), dtype=tf.float32)))
            else:
                dataset = tf.data.Dataset.from_generator(gen, output_signature=(tf.TensorSpec(shape=(nt, im_height, im_width, 3), dtype=tf.float32), tf.TensorSpec(shape=(1), dtype=tf.float32)))
        # Batch and prefetch the dataset, and ensure infinite dataset
        # if shuffle: dataset = dataset.shuffle(dataset.cardinality(), reshuffle_each_iteration=True)
        # dataset = (dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE).repeat())
        datasets.append(dataset)

    print(f"{len(datasets)} datasets created.")
    print(f"End tf.data.Dataset creation at {time.perf_counter() - start_time} seconds.")

    return datasets, length


def analyze_dataset(path):
    dataset = hkl.load(path)
    for i, source in enumerate(dataset):
        print(f"Source {i} has shape {source.shape} and dtype {source.dtype}")
        print(f"Min: {np.min(source)}, Max: {np.max(source)}")
        # print(f"Mean: {np.mean(source)}, Std: {np.std(source)}")
        # print(f"Unique values: {np.unique(source)}")
        # print(f"Unique values count: {np.unique(source, return_counts=True)}")
        print("\n")


def fix_my_hickle_files(data_files, file_names):
    for data_file, file_name in zip(data_files, file_names):
        data = hkl.load(data_file)
        print("gets here")
        pickle.dump(data, open(file_name, "w"))


def rehickling(data_files, file_names):
    for data_file, file_name in zip(data_files, file_names):
        with open(file_name, "rb") as file:
            data = pickle.load(file)
            print("gets here")
            hkl.dump(data, data_file)


def hickle_swap(data_files):
    for data_file in data_files:
        with open(data_file, "r") as file:
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
        with open(data_file, "r") as file:
            print("opens file")
            data = hkl.load(file)
            print("loads file")


def grab_data_and_save(data_files):
    files = ["/home/evalexii/Documents/Thesis/code/mod_prednet/kitti_data/X_train.npy", 
             "/home/evalexii/Documents/Thesis/code/mod_prednet/kitti_data/X_val.npy",
             "/home/evalexii/Documents/Thesis/code/mod_prednet/kitti_data/X_test.npy",
             "/home/evalexii/Documents/Thesis/code/mod_prednet/kitti_data/sources_train.npy", 
             "/home/evalexii/Documents/Thesis/code/mod_prednet/kitti_data/sources_val.npy", 
             "/home/evalexii/Documents/Thesis/code/mod_prednet/kitti_data/sources_test.npy", 
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


# Data files
# DATA_DIR = "/home/evalexii/Documents/Thesis/code/parallel_prednet/kitti_data/"
# train_file = os.path.join(DATA_DIR, 'X_train.hkl')
# train_sources = os.path.join(DATA_DIR, 'sources_train.hkl')
# val_file = os.path.join(DATA_DIR, 'X_val.hkl')
# val_sources = os.path.join(DATA_DIR, 'sources_val.hkl')
# test_file = os.path.join(DATA_DIR, 'X_test.hkl')
# test_sources = os.path.join(DATA_DIR, 'sources_test.hkl')

# files = [train_file, val_file, test_file, train_sources, val_sources, test_sources]
# file_names = [os.path.join(DATA_DIR, 'X_test.npy'), os.path.join(DATA_DIR, 'sources_test.npy')]

# fix_my_hickle_files(files)
# rehickling(files, file_names)
# hickle_swap(files)
# test_hickle(files)
# grab_data_and_save(files)
# grab_single_data_and_save(train_file)


def test_dataset():
    # Training data
    pfm_paths = []
    pfm_paths.append("/home/evalexii/local_dataset/disparity/family_x2/left/")
    pfm_paths.append("/home/evalexii/local_dataset/material_index/family_x2/left/")
    pfm_paths.append("/home/evalexii/local_dataset/object_index/family_x2/left/")
    pfm_paths.append("/home/evalexii/local_dataset/optical_flow/family_x2/into_future/left/")
    pgm_paths = []
    pgm_paths.append("/home/evalexii/local_dataset/motion_boundaries/family_x2/into_future/left/")
    png_paths = []
    png_paths.append("/home/evalexii/local_dataset/frames_cleanpass/family_x2/left")
    num_sources = len(pfm_paths) + len(pgm_paths) + len(png_paths)

    # Training parameters
    nt = 10  # number of time steps
    nb_epoch = 150  # 150
    batch_size = 2  # 4
    samples_per_epoch = 100  # 500
    N_seq_val = 20  # number of sequences to use for validation
    output_channels = [3, 12, 24, 48]  # [3, 48, 96, 192]
    original_im_shape = (540, 960, 3)
    downscale_factor = 2
    im_shape = (original_im_shape[0] // downscale_factor, original_im_shape[1] // downscale_factor, 3, )

    #  Create and split dataset
    dataset, length = create_dataset_from_serialized_generator(pfm_paths, pgm_paths, png_paths, output_mode="Error", im_height=im_shape[0], im_width=im_shape[1], batch_size=batch_size, nt=nt, reserialize=False, shuffle=True, resize=True, )

    ts = 0.7
    vs = (1 - ts) / 2
    train_size = int(ts * length)
    val_size = int(vs * length)
    test_size = int(vs * length)
    print(f"Train size: {train_size}")
    print(f"Validation size: {val_size}")
    print(f"Test size: {test_size}")

    train_dataset = dataset.take(train_size)
    test_dataset = dataset.skip(train_size)
    val_dataset = test_dataset.skip(val_size)
    test_dataset = test_dataset.take(test_size)

    # Iterate over the dataset
    for b, batch in enumerate(train_dataset):
        # batch is a tuple of (batch of sequences of images) and (batch of scalar errors)
        for j in range(batch_size):
            fig, axes = plt.subplots(len(batch[0]), nt, figsize=(15, 5))
            for i, image in enumerate(batch[0]):
                print(image.shape)
                for k in range(nt):
                    axes[i, k].imshow(image[j, k])
            plt.savefig(f"./images/test_{b}_{j}.png")


def config_gpus():
    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices("GPU")
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)
