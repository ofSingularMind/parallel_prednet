from keras import layers
from keras import backend as K
import keras
import tensorflow as tf
import numpy as np
import os
import warnings
from PPN_models.PPN_Common import Target, Prediction, Error, Representation

# Suppress warnings
warnings.filterwarnings("ignore")
# or '2' to filter out INFO messages too
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
# ,\r?\n
# ,\s{2,}


class PredLayer(keras.layers.Layer):
    def __init__(self, im_height, im_width, num_P_CNN, num_R_CLSTM, output_channels, layer_num, bottom_layer=False, top_layer=False, *args, **kwargs):
        super(PredLayer, self).__init__(*args, **kwargs)
        self.layer_num = layer_num
        self.im_height = im_height
        self.im_width = im_width
        self.num_P_CNN = num_P_CNN
        self.num_R_CLSTM = num_R_CLSTM
        self.pixel_max = 1
        self.output_channels = output_channels
        self.bottom_layer = bottom_layer
        self.top_layer = top_layer
        # R = Representation, P = Prediction, T = Target, E = Error, and P == A_hat and T == A
        self.states = {"R": None, "P": None, "T": None, "E": None, "TD_inp": None, "L_inp": None}
        self.representation = Representation(output_channels, num_R_CLSTM, layer_num=self.layer_num, name=f"Representation_Layer{self.layer_num}")
        self.prediction = Prediction(output_channels, num_P_CNN, layer_num=self.layer_num, name=f"Prediction_Layer{self.layer_num}")
        self.prediction_recon = Prediction(output_channels, num_P_CNN, layer_num=self.layer_num, name=f"Prediction_Recon_Layer{self.layer_num}")
        if not self.bottom_layer:
            self.target = Target(output_channels, layer_num=self.layer_num, name=f"Target_Layer{self.layer_num}")
        self.error = Error(layer_num=self.layer_num, name=f"Error_Layer{self.layer_num}")
        self.upsample = layers.UpSampling2D((2, 2), name=f"Upsample_Layer{self.layer_num}")

    def initialize_states(self, batch_size):
        # Initialize internal layer states
        self.states["R"] = tf.zeros((batch_size, self.im_height, self.im_width, self.num_R_CLSTM * self.output_channels))
        
        self.states["P"] = tf.zeros((batch_size, self.im_height, self.im_width, self.output_channels))
        self.states["P_recon"] = tf.zeros((batch_size, self.im_height, self.im_width, self.output_channels))
        
        self.states["T"] = tf.zeros((batch_size, self.im_height, self.im_width, self.output_channels))
        self.states["T_recon"] = tf.zeros((batch_size, self.im_height, self.im_width, self.output_channels))
        self.states["T_last"] = None
        
        self.states["E"] = tf.zeros((batch_size, self.im_height, self.im_width, 2 * self.output_channels))
        self.states["E_recon"] = tf.zeros((batch_size, self.im_height, self.im_width, 2 * self.output_channels))
        self.states["combined_E"] = tf.zeros((batch_size, self.im_height, self.im_width, 4 * self.output_channels))
        
        self.states["E_raw"] = tf.zeros((batch_size, self.im_height, self.im_width, self.output_channels))
        self.states["E_raw_last"] = tf.zeros((batch_size, self.im_height, self.im_width, self.output_channels))
        
        self.states["TD_inp"] = None
        self.states["L_inp"] = None
        self.states["lstm"] = None

    def clear_states(self):
        # Clear internal layer states
        self.states["R"] = None
        
        self.states["P"] = None
        self.states["P_recon"] = None
        
        self.states["T"] = None
        self.states["T_recon"] = None
        self.states["T_last"] = None
        
        self.states["E"] = None
        self.states["E_recon"] = None
        self.states["combined_E"] = None
        
        self.states["E_raw"] = None
        self.states["E_raw_last"] = None
        
        self.states["TD_inp"] = None
        self.states["L_inp"] = None
        self.states["lstm"] = None

    def call(self, inputs=None, direction="top_down", paddings=None):
        # PredLayer should update internal states when called with new TD and BU inputs, inputs[0] = BU, inputs[1] = TD

        if direction == "top_down":
            # UPDATE REPRESENTATION
            if self.top_layer:
                R_inp = keras.layers.Concatenate(axis=-1)([self.states["E"], self.states["R"]])
            else:
                self.states["TD_inp"] = self.upsample(inputs[1])
                self.states["TD_inp"] = keras.layers.ZeroPadding2D(paddings)(self.states["TD_inp"])
                R_inp = keras.layers.Concatenate(axis=-1)([self.states["E"], self.states["R"], self.states["TD_inp"]])

            if self.states["lstm"] is None:
                self.states["R"], self.states["lstm"] = self.representation(R_inp)
            else:
                self.states["R"], self.states["lstm"] = self.representation(R_inp, initial_states=self.states["lstm"])

            # FORM PREDICTION(S)
            self.states["P"] = K.minimum(self.prediction(self.states["R"]), self.pixel_max) if self.bottom_layer else self.prediction(self.states["R"])
            self.states["P_recon"] = K.minimum(self.prediction_recon(self.states["R"]), self.pixel_max) if self.bottom_layer else self.prediction_recon(self.states["R"])

        elif direction == "bottom_up":
            # RETRIEVE TARGET(S) (bottom-up input) ~ (batch_size, im_height, im_width, output_channels)
            target = inputs[0]
            self.states["T"] = target if self.bottom_layer else self.target(target)
            if self.states["T_last"] is None:
                # First frame, keep reconstruction target as zeros
                self.states["T_last"] = self.states["T"]
            else:
                # Next frames, set reconstruction target as last frame
                self.states["T_recon"] = self.states["T_last"]
            self.states["T_last"] = self.states["T"]

            # COMPUTE TARGET ERROR
            self.states['E'] = self.error(self.states['P'], self.states['T'])
            self.states['E_recon'] = self.error(self.states['P_recon'], self.states['T_recon'])
            self.states['combined_E'] = keras.layers.Concatenate(axis=-1)([self.states['E'], self.states['E_recon']])


            # COMPUTE RAW ERROR FOR PLOTTING
            self.states['E_raw'] = self.states['T'] - self.states['P']

            return self.states['combined_E']

        else:
            raise ValueError("Invalid direction. Must be 'top_down' or 'bottom_up'.")


class ParaPredNet(keras.Model):
    def __init__(self, training_args, im_height=540, im_width=960, *args, **kwargs):
        super(ParaPredNet, self).__init__(*args, **kwargs)
        self.batch_size = training_args['batch_size']
        self.nt = training_args['nt']
        self.im_height = im_height
        self.im_width = im_width
        self.num_P_CNN = training_args['num_P_CNN']
        self.num_R_CLSTM = training_args['num_R_CLSTM']
        self.layer_output_channels = training_args['output_channels']
        self.num_layers = len(self.layer_output_channels)
        self.resolutions = self.calculate_resolutions(self.im_height, self.im_width, self.num_layers)
        self.paddings = self.calculate_padding(self.im_height, self.im_width, self.num_layers)
        self.layer_input_channels = [0] * self.num_layers
        for i in range(len(self.layer_output_channels)):
            if i == 0:
                self.layer_input_channels[i] = self.layer_output_channels[i]
            else:
                self.layer_input_channels[i] = 4 * self.layer_output_channels[i - 1]
        # weighting for each layer's contribution to the loss
        self.layer_weights = [1] + [0.1] * (self.num_layers - 1)
        # equally weight all timesteps except the first
        self.time_loss_weights = 1.0 / (self.nt - 1) * np.ones((self.nt, 1))
        self.time_loss_weights[0] = 0
        self.output_mode = training_args['output_mode']
        self.dataset = training_args['dataset']
        self.num_passes = training_args['num_passes']
        self.predlayers = []
        for l, c in enumerate(self.layer_output_channels):
            self.predlayers.append(PredLayer(self.resolutions[l, 0], self.resolutions[l, 1], self.num_P_CNN, self.num_R_CLSTM, c, l, bottom_layer=(l == 0), top_layer=(l == self.num_layers - 1), name=f"PredLayer{l}"))
            # initialize layer states
            self.predlayers[-1].initialize_states(self.batch_size)
            # build layers
            temp_BU = tf.random.uniform((self.batch_size, self.resolutions[l, 0], self.resolutions[l, 1], self.layer_input_channels[l]), maxval=255, dtype=tf.float32)
            if l < self.num_layers - 1:
                temp_TD = tf.random.uniform((self.batch_size, self.resolutions[l + 1, 0], self.resolutions[l + 1, 1], self.num_R_CLSTM * self.layer_output_channels[l + 1]), maxval=255, dtype=tf.float32)
            else:
                temp_TD = None
            temp_out = self.predlayers[l]([temp_BU, temp_TD], paddings=self.paddings[l])

    def call(self, inputs):
        # print("Calling PredNet...")
        # inputs will be a tuple of batches of sequences of video frames
        # [-1] represents the PNG image source
        if self.dataset == "kitti":
            # inputs = inputs[-1]
            pass
        elif self.dataset == "monkaa":
            inputs = inputs[-1]

        # Initialize layer states
        for layer in self.predlayers:
            layer.initialize_states(self.batch_size)

        # Iterate through the time-steps manually
        for t in range(self.nt):
            for _ in range(self.num_passes):
                """Perform top-down pass, starting from the top layer"""
                for l, layer in reversed(list(enumerate(self.predlayers))):
                    # Top layer
                    if l == self.num_layers - 1:
                        BU_inp = None
                        TD_inp = None
                        layer([BU_inp, TD_inp], direction="top_down", paddings=self.paddings[l])
                    # Bottom and Middle layers
                    else:
                        BU_inp = None
                        TD_inp = self.predlayers[l + 1].states["R"]
                        layer([BU_inp, TD_inp], direction="top_down", paddings=self.paddings[l])

            """ Perform bottom-up pass, starting from the bottom layer """
            for l, layer in list(enumerate(self.predlayers)):
                # Bottom layer
                if l == 0:
                    # (self.batch_size, self.im_height, self.im_width, self.layer_input_channels[0])
                    BU_inp = inputs[:, t, ...]
                    TD_inp = None
                    error = layer([BU_inp, TD_inp], direction="bottom_up")
                # Middle and Top layers
                else:
                    BU_inp = self.predlayers[l - 1].states["combined_E"]
                    TD_inp = None
                    error = layer([BU_inp, TD_inp], direction="bottom_up")
                # Update error in bottom-up pass
                if self.output_mode == "Error":
                    layer_error = self.layer_weights[l] * K.mean(K.batch_flatten(error), axis=-1, keepdims=True)  # (batch_size, 1)
                    all_error = layer_error if l == 0 else tf.add(all_error, layer_error)  # (batch_size, 1)


            # save outputs over time
            if self.output_mode == "Error":
                if t == 0:
                    all_errors_over_time = self.time_loss_weights[t] * all_error
                else:
                    all_errors_over_time = tf.add(all_errors_over_time, self.time_loss_weights[t] * all_error)  # (batch_size, 1)
            elif self.output_mode == "Prediction":
                if t == 0:
                    all_predictions = tf.expand_dims(self.predlayers[0].states["P"], axis=1)
                    all_recon_predictions = tf.expand_dims(self.predlayers[0].states["P_recon"], axis=1)
                else:
                    all_predictions = tf.concat([all_predictions, tf.expand_dims(self.predlayers[0].states["P"], axis=1)], axis=1)
                    all_recon_predictions = tf.concat([all_recon_predictions, tf.expand_dims(self.predlayers[0].states["P_recon"], axis=1)], axis=1)
            elif self.output_mode == "Error_Images_and_Prediction":
                if t == 0:
                    all_error_images = tf.expand_dims(self.predlayers[0].states["E_raw"], axis=1)
                    all_predictions = tf.expand_dims(self.predlayers[0].states["P"], axis=1)
                    all_recon_predictions = tf.expand_dims(self.predlayers[0].states["P_recon"], axis=1)
                else:
                    all_error_images = tf.concat([all_error_images, tf.expand_dims(self.predlayers[0].states["E_raw"], axis=1)], axis=1)
                    all_predictions = tf.concat([all_predictions, tf.expand_dims(self.predlayers[0].states["P"], axis=1)], axis=1)
                    all_recon_predictions = tf.concat([all_recon_predictions, tf.expand_dims(self.predlayers[0].states["P_recon"], axis=1)], axis=1)

        if self.output_mode == "Error":
            output = all_errors_over_time * 100
        elif self.output_mode == "Prediction":
            output = [all_predictions, all_recon_predictions]
        elif self.output_mode == "Error_Images_and_Prediction":
            output = [all_error_images, all_predictions, all_recon_predictions]

        # Clear states from computation graph
        for layer in self.predlayers:
            layer.clear_states()

        return output

    def calculate_resolutions(self, im_height, im_width, num_layers):
        # Calculate resolutions for each layer
        resolutions = np.array([[im_height, im_width]])
        for i in range(num_layers - 1):
            resolutions = np.concatenate((resolutions, np.array([[resolutions[-1][0] // 2, resolutions[-1][1] // 2]])), axis=0)
        return resolutions

    def calculate_padding(self, im_height, im_width, num_layers):
        # Calculate padding for the input image to be divisible by 2**num_layers
        paddings = np.array([[[0, 0], [0, 0]] for _ in range(num_layers)])

        pooled_sizes = np.array([[im_height, im_width]])
        for i in range(num_layers - 1):
            # Going down:
            pooled_sizes = np.concatenate((pooled_sizes, np.array([[pooled_sizes[-1][0] // 2, pooled_sizes[-1][1] // 2]])), axis=0)

        upsampled_sizes = np.array([pooled_sizes[-1]])
        for i in reversed(range(num_layers - 1)):
            # Going up:
            upsampled_sizes = np.concatenate((np.array([[upsampled_sizes[0][0] * 2, upsampled_sizes[0][1] * 2]]), upsampled_sizes), axis=0)
            diff = (pooled_sizes[i][0] - upsampled_sizes[0][0], pooled_sizes[i][1] - upsampled_sizes[0][1])
            paddings[i] = [[0, diff[0]], [0, diff[1]]]
            upsampled_sizes[0] += np.array(diff)

        # print(np.concatenate((np.array(pooled_sizes), np.array(upsampled_sizes)), axis=1))
        # print(np.array(paddings))

        return paddings
