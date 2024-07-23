from keras import layers
from keras import backend as K
import keras
import tensorflow as tf
import numpy as np
import os
import warnings
from PPN_models.PPN_Common import Target, Prediction, Error, Representation, ObjectRepresentation

# Suppress warnings
warnings.filterwarnings("ignore")
# or '2' to filter out INFO messages too
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# for the ablation study:
# 1 layer with representation channels [3-in, 339-out], prediction channels [339-in, 3-out]
    # edits:
    # in PPN_Baseline, PredLayer, self.states["R"] = tf.zeros((batch_size, self.im_height, self.im_width, 339))
    # in PPN_Common, Representation, add line: output_channels = 339 - so representation outputs 339 channels

class PredLayer(keras.Model):
    def __init__(self, training_args, im_height, im_width, output_channels, layer_num, bottom_layer=False, top_layer=False, *args, **kwargs):
        super(PredLayer, self).__init__(*args, **kwargs)
        self.training_args = training_args
        self.layer_num = layer_num
        self.im_height = im_height
        self.im_width = im_width
        self.pixel_max = 1
        self.output_channels = output_channels
        self.bottom_layer = bottom_layer
        self.top_layer = top_layer
        # R = Representation, P = Prediction, T = Target, E = Error, and P == A_hat and T == A
        self.states = {"R": None, "P": None, "PM": None, "T": None, "E": None, "TD_inp": None, "L_inp": None}
        self.representation = Representation(output_channels, layer_num=self.layer_num, name=f"Representation_Layer{self.layer_num}")
        self.prediction = Prediction(output_channels, layer_num=self.layer_num, name=f"Prediction_Layer{self.layer_num}")
        if not self.bottom_layer:
            self.target = Target(output_channels, layer_num=self.layer_num, name=f"Target_Layer{self.layer_num}")
        if self.bottom_layer and self.training_args['object_representations']:
            self.object_representations = ObjectRepresentation(self.training_args, self.training_args['output_channels'][0]//3, self.layer_num, self.im_height, self.im_width, name=f"ObjectRepresentation_Layer{self.layer_num}")
        self.error = Error(layer_num=self.layer_num, name=f"Error_Layer{self.layer_num}")
        self.upsample = layers.UpSampling2D((2, 2), name=f"Upsample_Layer{self.layer_num}")
        self.last_frame = None

    def initialize_states(self, batch_size):
        # Initialize internal layer states
        self.states["R"] = tf.zeros((batch_size, self.im_height, self.im_width, self.output_channels))
        self.states["P_M"] = tf.zeros((batch_size, self.im_height, self.im_width, self.output_channels))
        self.states["P"] = tf.zeros((batch_size, self.im_height, self.im_width, self.output_channels))
        self.states["T"] = tf.zeros((batch_size, self.im_height, self.im_width, self.output_channels))
        self.states["E"] = tf.zeros((batch_size, self.im_height, self.im_width, 2 * self.output_channels)) # double for the pos/neg concatenated error
        self.states["TD_inp"] = None
        self.states["lstm"] = None

        self.last_frame = tf.zeros((batch_size, self.im_height, self.im_width, self.training_args['output_channels'][0]))

    def clear_states(self):
        # Clear internal layer states
        self.states["R"] = None
        self.states["P"] = None
        self.states["P_M"] = None
        self.states["T"] = None
        self.states["E"] = None
        self.states["TD_inp"] = None
        self.states["lstm"] = None
        self.states["E_raw"] = None

        self.clear_last_frame()

    def set_last_frame(self, last_frame):
        for _ in range(self.layer_num):
            last_frame = keras.layers.MaxPool2D((2, 2))(last_frame)
        self.last_frame = last_frame

    def clear_last_frame(self):
        self.last_frame = None

    def call(self, inputs=None, direction="top_down", paddings=None):
        # PredLayer should update internal states when called with new TD and BU inputs, inputs[0] = BU, inputs[1] = TD

        if direction == "top_down":
            # UPDATE REPRESENTATION
            if self.top_layer:
                R_inp = keras.layers.Concatenate()([self.states["E"], self.states["R"]])
            else:
                self.states["TD_inp"] = self.upsample(inputs[1])
                self.states["TD_inp"] = keras.layers.ZeroPadding2D(paddings)(self.states["TD_inp"])
                R_inp = keras.layers.Concatenate()([self.states["E"], self.states["R"], self.states["TD_inp"]])
            if self.bottom_layer and self.training_args['object_representations']:
                object_representations = self.object_representations(self.last_frame)
                R_inp = keras.layers.Concatenate()([R_inp, object_representations])

            if self.states["lstm"] is None:
                self.states["R"], self.states["lstm"] = self.representation(R_inp)
            else:
                self.states["R"], self.states["lstm"] = self.representation(R_inp, initial_states=self.states["lstm"])

            # FORM PREDICTION(S)
            self.states["P"] = K.minimum(self.prediction(self.states["R"]), self.pixel_max) if self.bottom_layer else self.prediction(self.states["R"])

            # RE-COMPUTE TARGET ERROR (this only matters when PredNet num_passes > 1, otherwise is overwritten)
            self.states["E"] = self.error(self.states["P"], self.states["T"])

        elif direction == "bottom_up":
            # RETRIEVE TARGET(S) (bottom-up input) ~ (batch_size, im_height, im_width, output_channels)
            target = inputs[0]
            self.states["T"] = target if self.bottom_layer else self.target(target)

            # COMPUTE TARGET ERROR
            self.states["E"] = self.error(self.states["P"], self.states["T"])

            # COMPUTE RAW ERROR FOR PLOTTING
            self.states["E_raw"] = self.states["T"] - self.states["P"]

            return self.states["E"]

        else:
            raise ValueError("Invalid direction. Must be 'top_down' or 'bottom_up'.")


class ParaPredNet(keras.Model):
    def __init__(self, training_args, im_height=540, im_width=960, *args, **kwargs):
        super(ParaPredNet, self).__init__(*args, **kwargs)
        self.training_args = training_args
        self.batch_size = training_args['batch_size']
        self.nt = training_args['nt']
        self.im_height = im_height
        self.im_width = im_width
        self.layer_output_channels = training_args['output_channels']
        self.num_layers = len(self.layer_output_channels)
        self.resolutions = self.calculate_resolutions(self.im_height, self.im_width, self.num_layers)
        self.paddings = self.calculate_padding(self.im_height, self.im_width, self.num_layers)
        self.layer_input_channels = [0] * self.num_layers
        for i in range(len(self.layer_output_channels)):
            if i == 0:
                self.layer_input_channels[i] = self.layer_output_channels[i]
            else:
                self.layer_input_channels[i] = 2 * self.layer_output_channels[i - 1]
        # weighting for each layer's contribution to the loss
        self.layer_weights = [1] + [0.1] * (self.num_layers - 1)
        # equally weight all timesteps except the first
        self.time_loss_weights = 1.0 / (self.nt - 1) * np.ones((self.nt, 1)) if self.nt > 1 else np.ones((self.nt, 1))
        self.time_loss_weights[0] = 0
        self.output_mode = training_args['output_mode']
        self.dataset = training_args['dataset']
        self.continuous_eval = False

        # perform setup
        self.predlayers = []
        for l, c in enumerate(self.layer_output_channels):
            self.predlayers.append(PredLayer(training_args, self.resolutions[l, 0], self.resolutions[l, 1], c, l, bottom_layer=(l == 0), top_layer=(l == self.num_layers - 1), name=f"PredLayer{l}"))
            # initialize layer states
            self.predlayers[-1].initialize_states(self.batch_size)
            # build layers
            temp_BU = tf.random.uniform((self.batch_size, self.resolutions[l, 0], self.resolutions[l, 1], self.layer_input_channels[l]), maxval=255, dtype=tf.float32)
            if l < self.num_layers - 1:
                temp_TD = tf.random.uniform((self.batch_size, self.resolutions[l + 1, 0], self.resolutions[l + 1, 1], self.layer_output_channels[l + 1]), maxval=255, dtype=tf.float32)
            else:
                temp_TD = None
            temp_out = self.predlayers[l]([temp_BU, temp_TD], paddings=self.paddings[l])
        self.init_layer_states()

    def call(self, inputs):
        # inputs will be a tuple of batches of sequences of video frames
        inputs = self.process_inputs(inputs)

        # Initialize layer states
        if not self.continuous_eval: self.init_layer_states()

        # Iterate through the time-steps manually
        for t in range(self.nt):
            """Perform top-down pass, starting from the top layer"""
            for l, layer in reversed(list(enumerate(self.predlayers))):
                # BU_inp = bottom-up input, TD_inp = top-down input

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
                    BU_inp = self.predlayers[l - 1].states["E"]
                    TD_inp = None
                    error = layer([BU_inp, TD_inp], direction="bottom_up")
                # Update error in bottom-up pass
                if self.output_mode == "Error":
                    layer_error = self.layer_weights[l] * K.mean(K.batch_flatten(error), axis=-1, keepdims=True)  # (batch_size, 1)
                    all_error = layer_error if l == 0 else tf.add(all_error, layer_error)  # (batch_size, 1)

            if self.output_mode == "Error" and self.training_args["decompose_images"]:
                predictions = self.predlayers[0].states["P"]
                targets = self.predlayers[0].states["T"]
                # Calculate reconstruction error
                reconstructed_image = K.minimum(1.0, tf.reduce_sum([predictions[...,i*3:(i+1)*3] for i in range(predictions.shape[-1]//3)], axis=0))
                original_image = K.minimum(1.0, tf.reduce_sum([targets[...,i*3:(i+1)*3] for i in range(targets.shape[-1]//3)], axis=0))
                recon_e_down = keras.backend.relu(original_image - reconstructed_image)
                recon_e_up = keras.backend.relu(reconstructed_image - original_image)
                recon_error = keras.layers.Concatenate(axis=-1)([recon_e_down, recon_e_up])
                recon_error = K.mean(K.batch_flatten(error), axis=-1, keepdims=True)
                all_error = tf.add(all_error, recon_error)  # (batch_size, 1)

            # save outputs over time
            if self.output_mode == "Error":
                if t == 0:
                    all_errors_over_time = self.time_loss_weights[t] * all_error
                else:
                    all_errors_over_time = tf.add(all_errors_over_time, self.time_loss_weights[t] * all_error)  # (batch_size, 1)
            elif self.output_mode == "Prediction":
                if t == 0:
                    all_predictions = tf.expand_dims(self.predlayers[0].states["P"], axis=1)
                else:
                    all_predictions = tf.concat([all_predictions, tf.expand_dims(self.predlayers[0].states["P"], axis=1)], axis=1)
            elif self.output_mode == "Error_Images_and_Prediction":
                if t == 0:
                    all_error_images = tf.expand_dims(self.predlayers[0].states["E_raw"], axis=1)
                    all_predictions = tf.expand_dims(self.predlayers[0].states["P"], axis=1)
                else:
                    all_error_images = tf.concat([all_error_images, tf.expand_dims(self.predlayers[0].states["E_raw"], axis=1)], axis=1)
                    all_predictions = tf.concat([all_predictions, tf.expand_dims(self.predlayers[0].states["P"], axis=1)], axis=1)

        if self.output_mode == "Error":
            output = all_errors_over_time * 100
        elif self.output_mode == "Prediction":
            output = all_predictions
        elif self.output_mode == "Error_Images_and_Prediction":
            output = [all_error_images, all_predictions]
        elif self.output_mode == "Intermediate_Activations":
            out_dict = {}
            for predlayer in self.predlayers:
                out_dict[f"{predlayer.representation.name}"] = predlayer.states["R"]
                out_dict[f"{predlayer.prediction.name}"] = predlayer.states["P"]
            output = out_dict

        # Clear states from computation graph
        if not self.continuous_eval: self.clear_layer_states()

        return output

    def process_inputs(self, inputs):
        return inputs
    
    def init_layer_states(self):
        for layer in self.predlayers:
            layer.initialize_states(self.batch_size)

    def clear_layer_states(self):
        for layer in self.predlayers:
            layer.clear_states()

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

        return paddings