import os
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")
# or '2' to filter out INFO messages too
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import random
import time
from PIL import Image
import glob
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from keras.layers import Input
from keras.callbacks import Callback
from keras.preprocessing.image import Iterator
from keras import backend as K
import tensorflow as tf
import numpy as np
import hickle as hkl
from PN_models.PN_ObjectCentric import SceneDecomposer
import math
from keras.preprocessing.image import load_img, img_to_array

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
    def __init__(self, data_dirs, test_dataset, length, batch_size=4, nt=10, output_channels=[3, 48, 96, 192], dataset="kitti", model_choice="baseline", iteration=0):
        self.DATA_DIR, self.WEIGHTS_DIR, self.RESULTS_SAVE_DIR, self.LOG_DIR = data_dirs
        self.RESULTS_SAVE_DIR = os.path.join(self.RESULTS_SAVE_DIR, f"it#{iteration}/")
        super(IntermediateEvaluations, self).__init__()
        self.test_dataset = test_dataset
        self.dataset_iterator = iter(self.test_dataset)
        batch_size = np.minimum(batch_size, 4)
        self.n_plot = batch_size  # 40
        self.batch_size = batch_size
        self.nt = nt
        self.plot_nt = nt
        self.output_channels = output_channels
        self.dataset = dataset
        self.model_choice = model_choice
        self.rg_colormap = LinearSegmentedColormap.from_list('custom_cmap', [(0, 'red'), (0.5, 'black'), (1, 'green')])
        
        # Retrieve target sequence (use the same sequence(s) always)
        self.X_test_inputs = [next(self.dataset_iterator) for _ in range(10)][-1][0]  # take just batch_x not batch_y
        self.X_test = self.X_test_inputs # take only the PNG images for MSE calcs and plotting
        # self.Xtc = self.test_dataset.element_spec[0].element_spec.shape[-1] # X_test_channels        
        self.Xtc = self.X_test_inputs[0].shape[-1] # X_test_channels

        if not os.path.exists(self.RESULTS_SAVE_DIR):
            os.makedirs(self.RESULTS_SAVE_DIR, exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        if epoch % 1 == 0 or epoch == 1:
            self.plot_training_samples(epoch)

    def plot_training_samples(self, epoch):
        """
        Evaluate trained PredNet on SSM Dataset sequences.
        Calculates mean-squared error and plots predictions.
        """

        # Calculate predicted sequence(s)
        self.model.layers[-1].output_mode = "Error_Images_and_Prediction"
        assert self.model_choice in ["baseline","object_centric"], "This branch is specific to baseline or object-centric PredNet"
        error_images, X_hat = self.model.layers[-1](self.X_test_inputs)

        if self.Xtc == 3:
            error_images_gray = np.mean(error_images, axis=-1)
        else:
            error_images_gray = np.concatenate([np.expand_dims(np.mean(error_images[..., 3*i:3*(i+1)], axis=-1), axis=-1) for i in range(self.Xtc//3)], axis=-1)
        self.model.layers[-1].output_mode = "Error"

        # Compare MSE of PredNet predictions vs. using last frame.  Write results to prediction_scores.txt
        # Look at all timesteps except the first
        mse_model = np.mean((self.X_test[:, 1:, ..., -self.Xtc:] - X_hat[:, 1:, ..., -self.Xtc:]) ** 2)
        mse_prev = np.mean((self.X_test[:, :-1, ..., -self.Xtc:] - self.X_test[:, 1:, ..., -self.Xtc:]) ** 2)
        f = open(self.RESULTS_SAVE_DIR + "training_scores.txt", "a+")
        f.write("======================= %i : Epoch\n" % epoch)
        f.write("%f : Model MSE\n" % mse_model)
        f.write("%f : Previous Frame MSE\n" % mse_prev)
        f.close()

        # Plot some training predictions
        aspect_ratio = float(X_hat.shape[2]) / X_hat.shape[3]
        if self.model_choice == "baseline":
            gs = gridspec.GridSpec(self.Xtc, self.plot_nt)
            plt.figure(figsize=(3 * self.plot_nt, 10 * aspect_ratio), layout="constrained")
        elif self.model_choice == "object_centric":
            gs = gridspec.GridSpec(self.Xtc+2, self.plot_nt)
            plt.figure(figsize=(3 * self.plot_nt, 10 * aspect_ratio), layout="constrained")

        gs.update(wspace=0.0, hspace=0.0)
        plot_save_dir = os.path.join(self.RESULTS_SAVE_DIR, "training_plots/")
        if not os.path.exists(plot_save_dir):
            os.makedirs(plot_save_dir, exist_ok=True)
        plot_idx = np.random.permutation(self.X_test.shape[0])[: self.n_plot]
        if self.model_choice == "baseline":
            for i in plot_idx:
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
                
                plt.savefig(plot_save_dir + "e" + str(epoch) + "_plot_" + str(i) + ".png")
                plt.clf()
        
        elif self.model_choice == "object_centric":
            for i in plot_idx:
                # Plot the decomposed image data
                for j in range(self.Xtc//3):
                    for t in range(self.plot_nt):
                        plt.subplot(gs[t + (3*j)*self.plot_nt])
                        plt.imshow(X_hat[i, t, ..., 3*j:3*(j+1)], interpolation="none")
                        plt.tick_params(axis="both", which="both", bottom="off", top="off", left="off", right="off", labelbottom="off", labelleft="off", )
                        if t == 0:
                            plt.ylabel("Predicted", fontsize=10)

                        plt.subplot(gs[t + (3*j+1)*self.plot_nt])
                        plt.imshow(self.X_test[i, t, ..., 3*j:3*(j+1)], interpolation="none")
                        plt.tick_params(axis="both", which="both", bottom="off", top="off", left="off", right="off", labelbottom="off", labelleft="off", )
                        if t == 0:
                            plt.ylabel("Actual", fontsize=10)

                        plt.subplot(gs[t + (3*j+2) * self.plot_nt])
                        plt.imshow(error_images_gray[i, t, ..., j:j+1], cmap=self.rg_colormap)
                        plt.tick_params(axis="both", which="both", bottom="off", top="off", left="off", right="off", labelbottom="off", labelleft="off", )
                        if t == 0:
                            plt.ylabel("Error", fontsize=10)
                
                # Plot the reconstructed full images
                for t in range(self.plot_nt):
                    plt.subplot(gs[t + self.Xtc*self.plot_nt])
                    reconstructed_image = np.minimum(np.sum([X_hat[i, t, ..., k*3:(k+1)*3] for k in range(self.Xtc//3)], axis=0), 1)
                    plt.imshow(reconstructed_image, interpolation="none")
                    plt.tick_params(axis="both", which="both", bottom="off", top="off", left="off", right="off", labelbottom="off", labelleft="off", )
                    if t == 0:
                        plt.ylabel("Predicted", fontsize=10)

                    plt.subplot(gs[t + (self.Xtc+1)*self.plot_nt])
                    original_image = np.minimum(np.sum([self.X_test[i, t, ..., k*3:(k+1)*3] for k in range(self.Xtc//3)], axis=0), 1)
                    plt.imshow(original_image, interpolation="none")
                    plt.tick_params(axis="both", which="both", bottom="off", top="off", left="off", right="off", labelbottom="off", labelleft="off", )
                    if t == 0:
                        plt.ylabel("Actual", fontsize=10)
                
                plt.savefig(plot_save_dir + "e" + str(epoch) + "_plot_" + str(i) + ".png")
                plt.clf()


def sort_files_by_name(files):
    return sorted(files, key=lambda x: int(os.path.basename(x).split('.')[0]))

def serialize_dataset(data_dirs, png_paths, dataset_name="SSM", test_data=False, start_time=time.perf_counter(), iteration=0, dataset_chunk_size=1000):
    DATA_DIR, WEIGHTS_DIR, RESULTS_SAVE_DIR, LOG_DIR = data_dirs
    print(f"Start to serialize at {time.perf_counter() - start_time} seconds.")

    png_sources = []

    for png_path in png_paths:
        png_sources += [sort_files_by_name(glob.glob(png_path + "/*.png"))]

    all_files = np.array(list(zip(*png_sources)))

    # Get the length of the dataset
    length = all_files.shape[0]

    # Store the dataset in list with one key per source
    dataset = [0 for _ in range(all_files.shape[1])]

    # Dataset chunk details...
    nms = dataset_chunk_size # nominal_subset_max
    subset_max = np.minimum(nms, length - (iteration) * nms)
    assert subset_max >= dataset_chunk_size, "Subset max is less than dataset_chunk_size - create new data and restart training"
    subset_length = np.minimum(nms, length - (iteration) * nms)

    # Select random offset for dataset
    assert (iteration + 1) * subset_length <= length, "Iteration exceeds dataset length."
    offset = iteration * nms
    print(f"Selecting {subset_length} indices {offset} to {offset + subset_length} from dataset of length {length}.")
    print(f"Start to load images at {time.perf_counter() - start_time} seconds.")

    for j in range(len(png_paths)):
        l = []
        for i in range(offset, subset_length+offset):
            im = (np.array(Image.open(all_files[i, j]), dtype=np.float32)
                / 255.0)
            if len(im.shape) == 2:
                # expand to include channel dimension
                im = np.expand_dims(im, axis=2)

            l.append(im)

        dataset[j] = np.array(l)
        print(f"PNG source {j+1} of {len(png_paths)} done at {time.perf_counter() - start_time} seconds.")

    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR, exist_ok=True)
    # where weights are loaded prior to training
    dataset_file = os.path.join(DATA_DIR, f"{dataset_name}_train.hkl") if not test_data else os.path.join(DATA_DIR, f"{dataset_name}_test.hkl")

    if os.path.exists(dataset_file):
        os.remove(dataset_file)

    hkl.dump(dataset, dataset_file, mode="w")
    print(f"HKL dump done at {time.perf_counter() - start_time} seconds.")
    print(f"Dataset serialization complete at {time.perf_counter() - start_time} seconds.")

def create_dataset_from_serialized_generator(data_dirs, png_paths, output_mode="Error", dataset_name="SSM", im_height=540, im_width=960, output_channels=3, batch_size=4, nt=10, train_split=0.7, reserialize=False, shuffle=True, resize=False, single_channel=False, iteration=0, decompose=False, dataset_chunk_size=1000, stage=2):
    DATA_DIR, WEIGHTS_DIR, RESULTS_SAVE_DIR, LOG_DIR = data_dirs
    start_time = time.perf_counter()
    if decompose: sceneDecomposer = SceneDecomposer(n_colors=4, stage=stage)
    if reserialize:
        serialize_dataset(data_dirs, png_paths, dataset_name=dataset_name, start_time=start_time, iteration=iteration, dataset_chunk_size=dataset_chunk_size)
        print("Reserialized dataset.")
    else:
        print("Using previously serialized dataset.")
    print(f"Begin tf.data.Dataset creation at {time.perf_counter() - start_time} seconds.")

    num_png_paths = len(png_paths)
    num_total_paths = num_png_paths

    # list of numpy arrays, one for each source
    all_files = hkl.load(os.path.join(DATA_DIR, f"{dataset_name}_train.hkl"))
    print(f"Data files loaded at {time.perf_counter() - start_time} seconds.")
    num_samples = all_files[0].shape[0]
    assert all([all_files[i].shape[0] == num_samples for i in range(num_total_paths)]), "All sources must have the same number of samples"

    all_files = [sceneDecomposer.process_sequence(all_files[i]) for i in range(num_total_paths)] if decompose else all_files
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
                nt_outs = []
                for j in range(num_total_paths):
                    if resize:
                        nt_out = [tf.image.resize(all_files[j][i + k], (im_height, im_width)) for k in range(nt)]
                    else:
                        nt_out = [all_files[j][i + k] for k in range(nt)]
                    nt_outs.append(nt_out)
                
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
        else:
            if single_channel:
                dataset = tf.data.Dataset.from_generator(gen, output_signature=(tf.TensorSpec(shape=(nt, im_height, im_width, 1), dtype=tf.float32), tf.TensorSpec(shape=(1), dtype=tf.float32)))
            else:
                dataset = tf.data.Dataset.from_generator(gen, output_signature=(tf.TensorSpec(shape=(nt, im_height, im_width, output_channels), dtype=tf.float32), tf.TensorSpec(shape=(1), dtype=tf.float32)))

        datasets.append(dataset)

    print(f"{len(datasets)} datasets created.")
    print(f"End tf.data.Dataset creation at {time.perf_counter() - start_time} seconds.")

    return datasets, length

def create_dataset_from_generator(data_dirs, png_paths, output_mode="Error", dataset_name="SSM", im_height=540, im_width=960, batch_size=4, nt=10, train_split=0.7, reserialize=False, shuffle=True, resize=False, single_channel=False):
    DATA_DIR, WEIGHTS_DIR, RESULTS_SAVE_DIR, LOG_DIR = data_dirs
    start_time = time.perf_counter()
    print(f"Begin tf.data.Dataset creation at {time.perf_counter() - start_time} seconds.")

    num_png_paths = len(png_paths)
    num_total_paths = num_png_paths

    png_sources = []
    for png_path in png_paths:
        png_sources += [sort_files_by_name(glob.glob(png_path + "/*.png"))]

    all_files = np.array(list(zip(*png_sources)))
    num_samples = all_files.shape[0]
    assert (nt <= all_files.shape[0]), "nt must be less than or equal to the number of files in the dataset"
    assert all([all_files[:,i].shape[0] == num_samples for i in range(num_total_paths)]), "All sources must have the same number of samples"

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
                nt_outs = []
                for j in range(num_png_paths):
                    nt_outs.append([np.array(Image.open(all_files[i + k, j])) / 255.0 for k in range(nt)])

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
        elif dataset_name not in ["all_rolling", "SSM"]:
            if single_channel:
                dataset = tf.data.Dataset.from_generator(gen, output_signature=(tf.TensorSpec(shape=(nt, im_height, im_width, 1), dtype=tf.float32), tf.TensorSpec(shape=(1), dtype=tf.float32)))
            else:
                dataset = tf.data.Dataset.from_generator(gen, output_signature=(tf.TensorSpec(shape=(nt, im_height, im_width, 3), dtype=tf.float32), tf.TensorSpec(shape=(1), dtype=tf.float32)))
        else:
            if single_channel:
                dataset = tf.data.Dataset.from_generator(gen, output_signature=(tf.TensorSpec(shape=(nt, im_height, im_width, 1), dtype=tf.float32), tf.TensorSpec(shape=(1), dtype=tf.float32)))
            else:
                dataset = tf.data.Dataset.from_generator(gen, output_signature=(tf.TensorSpec(shape=(nt, im_height, im_width, 3), dtype=tf.float32), tf.TensorSpec(shape=(1), dtype=tf.float32)))
        
        datasets.append(dataset)

    print(f"{len(datasets)} datasets created.")
    print(f"End tf.data.Dataset creation at {time.perf_counter() - start_time} seconds.")

    return datasets, length

class sequence_dataset_creator():
    def __init__(self, training_args):
        self.training_args = training_args
        if self.training_args["decompose_images"]:
            self.sceneDecomposer = SceneDecomposer(n_colors=4, stage=2)
    
    def list_image_files(self, image_folder):
        file_paths = [os.path.join(image_folder, fname) for fname in sorted(os.listdir(image_folder)) if fname.endswith(('jpg', 'jpeg', 'png'))]
        return tf.data.Dataset.from_tensor_slices(file_paths), len(file_paths)

    def decode_image(self, file_path):
        img = tf.io.read_file(file_path)
        img = tf.image.decode_jpeg(img, channels=3)  # Adjust channels as needed (e.g., 1 for grayscale, 3 for RGB)
        img = tf.image.convert_image_dtype(img, tf.float32)  # Convert to float32 [0,1]
        img = tf.image.resize(img, [64, 64])  # Resize if needed
        return img

    def generate_sequence_dataset(self, image_folder, sequence_length, batch_size):
        # Get list of image files
        file_dataset, num_files = self.list_image_files(image_folder)
        
        # Read and decode images
        image_dataset = file_dataset.map(self.decode_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        # Create sequences of images
        def make_sequences(ds, sequence_length):
            ds = ds.window(sequence_length, shift=1, drop_remainder=True)
            ds = ds.flat_map(lambda window: window.batch(sequence_length))
            return ds

        sequence_dataset = make_sequences(image_dataset, sequence_length)

        # Apply the black box function to each sequence
        def decompose_image_sequence(sequence):
            sequence_np = sequence.numpy()  # Convert to numpy
            decomposed_sequence_np = self.sceneDecomposer.process_sequence(sequence_np)  # Apply numpy processing
            return decomposed_sequence_np

        # Convert the numpy function to a tensorflow function
        @tf.function
        def tf_decompose_image_sequence(sequence):
            decomposed_sequence = tf.py_function(func=decompose_image_sequence, inp=[sequence], Tout=tf.float32)
            decomposed_sequence.set_shape((sequence_length, self.training_args["SSM_im_shape"][0], self.training_args["SSM_im_shape"][1], self.training_args["output_channels"][0]))  # Explicitly set the shape
            return decomposed_sequence

        if self.training_args["decompose_images"]:
            sequence_dataset = sequence_dataset.map(tf_decompose_image_sequence, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        # Pair each sequence with a single zero value as ground truth
        def add_ground_truth(sequence):
            target = tf.zeros((1,))  # Ground truth single zero value
            return sequence, target

        sequence_dataset = sequence_dataset.map(add_ground_truth, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        
        # Batch the sequences
        sequence_dataset = sequence_dataset.batch(batch_size)
        sequence_dataset = sequence_dataset.prefetch(tf.data.experimental.AUTOTUNE)

        # Calculate number of sequences and batches
        num_sequences = num_files - sequence_length + 1
        num_batches = tf.math.ceil(num_sequences / batch_size)

        return sequence_dataset, num_sequences, num_batches

class SequenceDataLoader:
    def __init__(self, training_args, folder_path, sequence_length, batch_size, img_height, img_width, processed_img_channels, shuffle=True):
        self.training_args = training_args
        self.folder_path = folder_path
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.img_height = img_height
        self.img_width = img_width
        self.processed_img_channels = processed_img_channels
        self.shuffle = shuffle
        self.img_filenames = sorted([f for f in os.listdir(folder_path) if f.endswith('.png')])
        self.num_images = len(self.img_filenames)
        self.dataset_length = self.num_images - self.sequence_length + 1
        if self.training_args["decompose_images"]:
            self.sceneDecomposer = SceneDecomposer(n_colors=4)
        
    def process_sequence(self, sequence):
        stage = 2 if self.training_args["second_stage"] else 1
        return self.sceneDecomposer.process_sequence(sequence, stage=stage)
    
    def load_image(self, file_path):
        img = Image.open(file_path) # load_img(file_path)
        img_array = np.array(img, dtype=np.float32) / 255.0 # img_to_array(img) / 255.0
        return img_array
    
    def load_sequence(self, start_index):
        sequence = []
        for i in range(self.sequence_length):
            img_path = os.path.join(self.folder_path, self.img_filenames[start_index + i])
            img_array = self.load_image(img_path)
            sequence.append(img_array)
        sequence = np.stack(sequence, axis=0)
        if self.training_args["decompose_images"]:
            sequence = self.process_sequence(sequence)
        return sequence
    
    def generate_batch(self):
        all_indices = np.arange(self.num_images - self.sequence_length + 1)
        np.random.shuffle(all_indices) if self.shuffle else None
        
        for i in range(0, len(all_indices), self.batch_size):
            batch_sequences = []
            for j in range(self.batch_size):
                if i + j < len(all_indices):
                    sequence = self.load_sequence(all_indices[i + j])
                    batch_sequences.append(sequence)
            if batch_sequences:
                yield np.stack(batch_sequences, axis=0), np.zeros((len(batch_sequences), 1))  # Target 0.0 MSE for training

    def create_tf_dataset(self):
        dataset = tf.data.Dataset.from_generator(
            self.generate_batch,
            output_signature=(
                tf.TensorSpec(shape=(None, self.sequence_length, self.img_height, self.img_width, self.processed_img_channels), dtype=tf.float32),
                tf.TensorSpec(shape=(None, 1), dtype=tf.float32)
            )
        )
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        dataset = dataset.repeat()
        return dataset, self.dataset_length

class sequence_dataset_creator_yuck(tf.keras.utils.Sequence):
    def __init__(self, image_folder, sequence_length, batch_size):
        self.image_folder = image_folder
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.file_paths = [os.path.join(image_folder, fname) for fname in sorted(os.listdir(image_folder)) if fname.endswith(('jpg', 'jpeg', 'png'))]
        self.num_files = len(self.file_paths)
        self.possible_starts = np.arange(self.num_files - self.sequence_length + 1)
        np.random.shuffle(self.possible_starts)  # Shuffle once and iterate

    def __len__(self):
        return (self.num_files - self.sequence_length + 1) // self.batch_size

    def on_epoch_end(self):
        np.random.shuffle(self.possible_starts)  # Reshuffle at the end of each epoch

    def __getitem__(self, idx):
        batch_indices = self.possible_starts[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_images = []
        for start_idx in batch_indices:
            batch_seq = self.file_paths[start_idx:start_idx + self.sequence_length]
            batch_images.append([self.load_and_preprocess_image(fp) for fp in batch_seq])
        return np.array(batch_images), np.zeros((self.batch_size, 1))  # Dummy target array for training

    def load_and_preprocess_image(self, file_path):
        img = tf.io.read_file(file_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, [64, 64])  # Resize as needed
        img = tf.image.convert_image_dtype(img, tf.float32) / 255.0  # Normalize to [0, 1]
        return img.numpy()  # Return as numpy array for batch assembling