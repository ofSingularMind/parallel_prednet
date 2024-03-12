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
from kitti_settings import *

from PPN import ParaPredNet

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