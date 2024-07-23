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
from data.animations.decompose_images.decomposer import SceneDecomposer

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
        self.Xtc = self.X_test.shape[-1] # X_test_channels        

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

def serialize_dataset(data_dirs, png_paths, dataset_name="SSM", test_data=False, start_time=time.perf_counter(), iteration=0):
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
    nms = 1000 # nominal_subset_max
    subset_max = np.minimum(nms, length - (iteration) * nms)
    assert subset_max >= 1000, "Subset max is less than 1000 - create new data and restart training"
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

def create_dataset_from_serialized_generator(data_dirs, png_paths, output_mode="Error", dataset_name="SSM", im_height=540, im_width=960, output_channels=3, batch_size=4, nt=10, train_split=0.7, reserialize=False, shuffle=True, resize=False, single_channel=False, iteration=0, decompose=False):
    DATA_DIR, WEIGHTS_DIR, RESULTS_SAVE_DIR, LOG_DIR = data_dirs
    start_time = time.perf_counter()
    if decompose: sceneDecomposer = SceneDecomposer()
    if reserialize:
        serialize_dataset(data_dirs, png_paths, dataset_name=dataset_name, start_time=start_time, iteration=iteration)
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

    all_files = [sceneDecomposer.process_dataset(all_files[i]) for i in range(num_total_paths)] if decompose else all_files
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
