import os
import hickle as hkl
import numpy as np
import keras
import tensorflow as tf
from keras import backend as K
from keras.preprocessing.image import Iterator

from keras.models import Model, model_from_json
from keras.callbacks import Callback
from keras.layers import Input
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from keras import backend as K
from monkaa_settings import *

from PPN import ParaPredNet
import re
import sys

import glob
from PIL import Image
import time
import random


# Data generator that creates sequences for input into PredNet.
class SequenceGenerator(Iterator):
    def __init__(self, data_file, source_file, nt,
                 batch_size=8, shuffle=False, seed=None,
                 output_mode='error', sequence_start_mode='all', N_seq=None,
                 data_format=K.image_data_format()):
        self.X = hkl.load(data_file)  # X will be like (n_images, nb_cols, nb_rows, nb_channels)
        self.sources = hkl.load(source_file) # source for each image so when creating sequences can assure that consecutive frames are from same video
        self.nt = nt
        self.batch_size = batch_size
        self.data_format = data_format
        assert sequence_start_mode in {'all', 'unique'}, 'sequence_start_mode must be in {all, unique}'
        self.sequence_start_mode = sequence_start_mode
        assert output_mode in {'error', 'prediction'}, 'output_mode must be in {error, prediction}'
        self.output_mode = output_mode

        if self.data_format == 'channels_first':
            self.X = np.transpose(self.X, (0, 3, 1, 2))
        self.im_shape = self.X[0].shape

        if self.sequence_start_mode == 'all':  # allow for any possible sequence, starting from any frame
            self.possible_starts = np.array([i for i in range(self.X.shape[0] - self.nt) if self.sources[i] == self.sources[i + self.nt - 1]])
        elif self.sequence_start_mode == 'unique':  #create sequences where each unique frame is in at most one sequence
            curr_location = 0
            possible_starts = []
            while curr_location < self.X.shape[0] - self.nt + 1:
                if self.sources[curr_location] == self.sources[curr_location + self.nt - 1]:
                    possible_starts.append(curr_location)
                    curr_location += self.nt
                else:
                    curr_location += 1
            self.possible_starts = possible_starts

        if shuffle:
            self.possible_starts = np.random.permutation(self.possible_starts)
        if N_seq is not None and len(self.possible_starts) > N_seq:  # select a subset of sequences if want to
            self.possible_starts = self.possible_starts[:N_seq]
        self.N_sequences = len(self.possible_starts)
        super(SequenceGenerator, self).__init__(len(self.possible_starts), batch_size, shuffle, seed)

    def __getitem__(self, null):
        return self.next()

    def next(self):
        with self.lock:
            current_index = (self.batch_index * self.batch_size) % self.n
            index_array, current_batch_size = next(self.index_generator), self.batch_size
        batch_x = np.zeros((current_batch_size, self.nt) + self.im_shape, np.float32)
        for i, idx in enumerate(index_array):
            idx = self.possible_starts[idx]
            batch_x[i] = self.preprocess(self.X[idx:idx+self.nt])
        if self.output_mode == 'error':  # model outputs errors, so y should be zeros
            batch_y = np.zeros(current_batch_size, np.float32)
        elif self.output_mode == 'prediction':  # output actual pixels
            batch_y = batch_x
        return batch_x, batch_y

    def preprocess(self, X):
        return X.astype(np.float32) / 255

    def create_n(self, n=1):
        assert n <= len(self.possible_starts), "Can't create more sequences than there are possible starts"
        X_all = np.zeros((n, self.nt) + self.im_shape, np.float32)
        idxes = np.random.choice(len(self.possible_starts), n, replace=False)
        for i, idx in enumerate(np.array(self.possible_starts)[idxes].tolist()):
            X_all[i] = self.preprocess(self.X[idx:idx+self.nt])
        return X_all
    
    def create_all(self):
        X_all = np.zeros((self.N_sequences, self.nt) + self.im_shape, np.float32)
        for i, idx in enumerate(self.possible_starts):
            X_all[i] = self.preprocess(self.X[idx:idx+self.nt])
        return X_all

class MyCustomCallback(Callback):
    def __init__(self, batch_size=4, nt=10, output_channels=[3, 48, 96, 192]):
        super(MyCustomCallback, self).__init__()
        self.n_plot = batch_size # 40
        self.batch_size = batch_size
        self.nt = nt
        self.plot_nt = nt
        self.weights_file = os.path.join(WEIGHTS_DIR, 'tensorflow_weights/para_prednet_kitti_weights.hdf5')
        self.test_file = os.path.join(DATA_DIR, 'X_test.hkl')
        self.test_sources = os.path.join(DATA_DIR, 'sources_test.hkl')

        self.test_PPN = ParaPredNet(batch_size=self.batch_size, nt=self.nt, output_channels=output_channels)
        self.test_PPN.output_mode = 'Prediction'
        self.test_PPN.compile(optimizer='adam', loss='mean_squared_error')
        self.test_PPN.build(input_shape=(None, self.nt, 128, 160, 3))
        print("ParaPredNet compiled...")

        self.test_generator = SequenceGenerator(self.test_file, self.test_sources, self.nt, sequence_start_mode='unique')

        if not os.path.exists(RESULTS_SAVE_DIR): os.mkdir(RESULTS_SAVE_DIR)
        self.test_weights_path = os.path.join(RESULTS_SAVE_DIR, 'tensorflow_weights/')
        if not os.path.exists(self.test_weights_path): os.mkdir(self.test_weights_path)

    def on_epoch_end(self, epoch, logs=None):
        if epoch % 1 == 0 or epoch == 1:
            self.plot_training_samples(epoch)

    def plot_training_samples(self, epoch):
        '''
        Evaluate trained PredNet on KITTI sequences.
        Calculates mean-squared error and plots predictions.
        '''

        # load latest weights and regenerate test model
        if os.path.exists(self.weights_file):
            self.test_PPN.load_weights(self.weights_file)

        X_test = self.test_generator.create_n(self.n_plot)
        # X_hat = tf.cast(self.test_PPN(X_test), dtype=tf.float32)
        X_hat = self.test_PPN(X_test)

        # Compare MSE of PredNet predictions vs. using last frame.  Write results to prediction_scores.txt
        mse_model = np.mean( (X_test[:, 1:] - X_hat[:, 1:])**2 )  # look at all timesteps except the first
        mse_prev = np.mean( (X_test[:, :-1] - X_test[:, 1:])**2 )
        f = open(RESULTS_SAVE_DIR + 'training_scores.txt', 'a+')
        f.write("======================= %i : Epoch\n" % epoch)
        f.write("%f : Model MSE\n" % mse_model)
        f.write("%f : Previous Frame MSE\n" % mse_prev)
        f.close()

        # Plot some training predictions
        aspect_ratio = float(X_hat.shape[2]) / X_hat.shape[3]
        plt.figure(figsize = (2*self.plot_nt, 4*aspect_ratio))
        gs = gridspec.GridSpec(2, self.plot_nt)
        gs.update(wspace=0., hspace=0.)
        plot_save_dir = os.path.join(RESULTS_SAVE_DIR, 'training_plots/')
        if not os.path.exists(plot_save_dir): os.mkdir(plot_save_dir)
        plot_idx = np.random.permutation(X_test.shape[0])[:self.n_plot]
        for i in plot_idx:
            for t in range(self.plot_nt):
                plt.subplot(gs[t])
                plt.imshow(X_hat[i,t], interpolation='none')
                plt.tick_params(axis='both', which='both', bottom='off', top='off', left='off', right='off', labelbottom='off', labelleft='off')
                if t==0: plt.ylabel('Predicted', fontsize=10)
                
                plt.subplot(gs[t + self.plot_nt])
                plt.imshow(X_test[i,t], interpolation='none')
                plt.tick_params(axis='both', which='both', bottom='off', top='off', left='off', right='off', labelbottom='off', labelleft='off')
                if t==0: plt.ylabel('Actual', fontsize=10)

            plt.savefig(plot_save_dir + 'e' + str(epoch) + '_plot_' + str(i) + '.png')
            plt.clf()

def writePFM(file, image, scale=1):
    file = open(file, 'wb')

    color = None

    if image.dtype.name != 'float32':
        raise Exception('Image dtype must be float32.')
      
    image = np.flipud(image)  

    if len(image.shape) == 3 and image.shape[2] == 3: # color image
        color = True
    elif len(image.shape) == 2 or len(image.shape) == 3 and image.shape[2] == 1: # greyscale
        color = False
    else:
        raise Exception('Image must have H x W x 3, H x W x 1 or H x W dimensions.')

    file.write('PF\n' if color else 'Pf\n')
    file.write('%d %d\n' % (image.shape[1], image.shape[0]))

    endian = image.dtype.byteorder

    if endian == '<' or endian == '=' and sys.byteorder == 'little':
        scale = -scale

    file.write('%f\n' % scale)

    image.tofile(file)

def dir_PFM_to_PNG(dir):
    for obj in os.listdir(dir):
        print(f"Processing {obj}. Isdir == {os.path.isdir(dir + obj)}")
        if os.path.isdir(dir + obj):
            dir_PFM_to_PNG(dir + obj + '/')
        elif obj.endswith(".pfm"):
            data, scale = readPFM(dir + obj)
            print(f"Converting {obj} to {obj[:-4]}.png")
            if (np.max(data) > 1) or (np.min(data) < 0):
                data = (data - np.min(data)) / (np.max(data) - np.min(data))
            plt.imsave(dir + obj[:-4] + '.png', data)
            print(f"Processed {obj} to {obj[:-4]}.png")

def readPFM(file):
    file = open(file, 'rb')

    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().rstrip()
    if header == b'PF':
        color = True
    elif header == b'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(b'^(\d+)\s(\d+)\s$', file.readline())
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0: # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>' # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    return data

def sort_files_by_name(files):
    return sorted(files, key=lambda x: os.path.basename(x))

def serialize_dataset(pfm_paths, pgm_paths, png_paths, start_time=time.perf_counter()):
    
    print(f"Start to serialize at {time.perf_counter() - start_time} seconds.")
    pfm_sources = []
    pgm_sources = []
    png_sources = []

    for pfm_path in pfm_paths:
        pfm_sources += [sort_files_by_name(glob.glob(pfm_path + '/*.pfm'))]
    for pgm_path in pgm_paths:
        pgm_sources += [sort_files_by_name(glob.glob(pgm_path + '/*.pgm'))]
    for png_path in png_paths:
        png_sources += [sort_files_by_name(glob.glob(png_path + '/*.png'))]

    all_files = np.array(list(zip(*pfm_sources, *pgm_sources, *png_sources)))

    # Get the length of the dataset
    length = all_files.shape[0]

    # Store the dataset in list with one key per source
    dataset = [0 for _ in range(all_files.shape[1])]

    # Prepare array for filling with data
    # data = np.zeros_like(all_files, dtype=np.float32)

    temp = length
    print(f"Start to load images at {time.perf_counter() - start_time} seconds.")

    for j in range(len(pfm_paths)):
        l = []
        last = time.perf_counter()
        for i in range(temp):
            l.append(readPFM(all_files[i, j]))
            # print(f"PFM image {i} of {temp-1} done in {time.perf_counter() - last} seconds.")
            last = time.perf_counter()
        dataset[j] = np.array(l)
        print(f"PFM source {j+1} of {len(pfm_paths)} done at {time.perf_counter() - start_time} seconds.")
    
    for j in range(len(pgm_paths)):
        l = []
        last = time.perf_counter()
        for i in range(temp):
            l.append(np.array(Image.open(all_files[i, j + len(pfm_paths)])) / 255.0)
            # print(f"PGM image {i} of {temp-1} done in {time.perf_counter() - last} seconds.")
            last = time.perf_counter()
        dataset[j + len(pfm_paths)] = np.array(l)
        # dataset[j + len(pfm_paths)] = np.array([n for i in range(temp)])
        print(f"PGM source {j+1} of {len(pgm_paths)} done at {time.perf_counter() - start_time} seconds.")
    
    for j in range(len(png_paths)):
        l = []
        last = time.perf_counter()
        for i in range(temp):
            l.append(np.array(Image.open(all_files[i, j + len(pfm_paths) + len(pgm_paths)])) / 255.0)
            # print(f"PNG image {i} of {temp-1} done in {time.perf_counter() - last} seconds.")
            last = time.perf_counter()
        dataset[j + len(pfm_paths) + len(pgm_paths)] = np.array(l)
        # dataset[j + len(pfm_paths) + len(pgm_paths)] = np.array([np.array(Image.open(all_files[i, j + len(pfm_paths) + len(pgm_paths)])) / 255.0 for i in range(temp)])
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
    
    if not os.path.exists(DATA_DIR): os.mkdir(DATA_DIR)
    dataset_file = os.path.join(DATA_DIR, 'monkaa_train.hkl')  # where weights are loaded prior to training

    hkl.dump(dataset, dataset_file, mode='w')
    print(f"HKL dump done at {time.perf_counter() - start_time} seconds.")
    print(f"Dataset serialization complete at {time.perf_counter() - start_time} seconds.")

def create_dataset_from_generator(pfm_paths, pgm_paths, png_paths, im_height=540, im_width=960, batch_size=1, nt=10, shuffle=True):

    num_pfm_paths = len(pfm_paths)
    num_pgm_paths = len(pgm_paths)
    num_png_paths = len(png_paths)
    total_paths = num_pfm_paths + num_pgm_paths + num_png_paths

    def generator():
        
        pfm_sources = []
        pgm_sources = []
        png_sources = []

        for pfm_path in pfm_paths:
            pfm_sources += [sort_files_by_name(glob.glob(pfm_path + '/*.pfm'))]
        for pgm_path in pgm_paths:
            pgm_sources += [sort_files_by_name(glob.glob(pgm_path + '/*.pgm'))]
        for png_path in png_paths:
            png_sources += [sort_files_by_name(glob.glob(png_path + '/*.png'))]

        all_files = np.array(list(zip(*pfm_sources, *pgm_sources, *png_sources)))

        assert nt <= all_files.shape[0], "nt must be less than or equal to the number of files in the dataset"
        
        for i in range(all_files.shape[0]+1-nt):
            nt_outs = []
            for j in range(num_pfm_paths):
                nt_outs.append([readPFM(all_files[i+k, j]) for k in range(nt)])
            for j in range(num_pgm_paths):
                nt_outs.append([np.array(Image.open(all_files[i+k, j + num_pfm_paths]))  / 255.0 for k in range(nt)])
            for j in range(num_png_paths):
                nt_outs.append([np.array(Image.open(all_files[i+k, j + num_pfm_paths + num_pgm_paths])) / 255.0 for k in range(nt)])
        
            yield tuple(nt_outs)

    dataset = tf.data.Dataset.from_generator(generator, output_signature=(
        (tf.TensorSpec(shape=(nt, im_height, im_width), dtype=tf.float32)),
        (tf.TensorSpec(shape=(nt, im_height, im_width), dtype=tf.float32)),
        (tf.TensorSpec(shape=(nt, im_height, im_width), dtype=tf.float32)),
        (tf.TensorSpec(shape=(nt, im_height, im_width, 3), dtype=tf.float32)),
        (tf.TensorSpec(shape=(nt, im_height, im_width), dtype=tf.float32)),
        (tf.TensorSpec(shape=(nt, im_height, im_width, 3), dtype=tf.float32))
    ))

    # Get the length of the dataset
    length = len(glob.glob(pfm_paths[0] + '/*.pfm'))

    # Shuffle, batch and prefetch the dataset
    if shuffle:
        dataset = dataset.shuffle(buffer_size=length, reshuffle_each_iteration=True)
    dataset = dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

    return dataset, length


def create_dataset_from_serialized_generator(pfm_paths, pgm_paths, png_paths, im_height=540, im_width=960, batch_size=3, nt=10, reserialize=False, shuffle=True):

    start_time = time.perf_counter()
    if reserialize:
        serialize_dataset(pfm_paths, pgm_paths, png_paths, start_time=start_time)
        print("Reserialized dataset.")
    else:
        print("Using previously serialized dataset.")
    print(f"Begin tf.data.Dataset creation at {time.perf_counter() - start_time} seconds.")


    num_pfm_paths = len(pfm_paths)
    num_pgm_paths = len(pgm_paths)
    num_png_paths = len(png_paths)
    num_total_paths = num_pfm_paths + num_pgm_paths + num_png_paths

    all_files = hkl.load(os.path.join(DATA_DIR, 'monkaa_train.hkl')) # list of numpy arrays, one for each source
    num_samples = all_files[0].shape[0]
    assert all([all_files[i].shape[0] == num_samples for i in range(num_total_paths)]), "All sources must have the same number of samples"

    # Get the length of the dataset
    length = len(glob.glob(pfm_paths[0] + '/*.pfm'))


    def generator():
        last = time.perf_counter()
        # num unique sequences, nus
        nus = num_samples + 1 - nt
        for i in random.sample(range(nus), nus) if shuffle else range(num_samples +1 -nt): # per source
            nt_outs = []
            for j in range(num_total_paths):
                nt_outs.append([all_files[j][i+k] for k in range(nt)])
                last = time.perf_counter()
            yield tuple(nt_outs)

    dataset = tf.data.Dataset.from_generator(generator, output_signature=(
        (tf.TensorSpec(shape=(nt, im_height, im_width), dtype=tf.float32)),
        (tf.TensorSpec(shape=(nt, im_height, im_width), dtype=tf.float32)),
        (tf.TensorSpec(shape=(nt, im_height, im_width), dtype=tf.float32)),
        (tf.TensorSpec(shape=(nt, im_height, im_width, 3), dtype=tf.float32)),
        (tf.TensorSpec(shape=(nt, im_height, im_width), dtype=tf.float32)),
        (tf.TensorSpec(shape=(nt, im_height, im_width, 3), dtype=tf.float32))
    ))

    # Batch and prefetch the dataset

    # if shuffle:
    #     dataset = dataset.shuffle(buffer_size=int(1.2*num_total_paths*length), reshuffle_each_iteration=True)
    dataset = dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    print(f"End tf.data.Dataset creation at {time.perf_counter() - start_time} seconds.")

    return dataset, length

