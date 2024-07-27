from keras import layers
from keras import backend as K
import keras
import tensorflow as tf
import numpy as np
import os
import sys
import warnings
from PN_models.PN_Common import Target, Prediction, Error, Representation
from keras.layers import Dense, GlobalAveragePooling2D, ConvLSTM2D, BatchNormalization, Dropout, Conv2D, MaxPooling2D, Flatten
from keras.applications import MobileNetV2
import matplotlib.pyplot as plt

import numpy as np
from PIL import Image
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

# Suppress warnings
warnings.filterwarnings("ignore")
# or '2' to filter out INFO messages too
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# for the ablation study:
# 1 layer with representation channels [3-in, 339-out], prediction channels [339-in, 3-out]
    # edits:
    # in PN_Baseline, PredLayer, self.states["R"] = tf.zeros((batch_size, self.im_height, self.im_width, 339))
    # in PN_Common, Representation, add line: output_channels = 339 - so representation outputs 339 channels

class PredLayer(keras.Model):
    def __init__(self, training_args, im_height, im_width, output_channels, layer_num, bottom_layer=False, top_layer=False, *args, **kwargs):
        super(PredLayer, self).__init__(*args, **kwargs)
        self.training_args = training_args
        self.layer_num = layer_num
        self.batch_size = training_args['batch_size']
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
            self.object_representations = ObjectRepresentation(num_classes=4, batch_size=self.batch_size, im_height=self.im_height, im_width=self.im_width, output_channels=12, name=f"ObjectRepresentation_Layer{self.layer_num}")
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
            if not self.training_args["pretrain_classifier"]:
                # UPDATE REPRESENTATION
                if self.top_layer:
                    R_inp = keras.layers.Concatenate()([self.states["E"], self.states["R"]])
                else:
                    self.states["TD_inp"] = self.upsample(inputs[1])
                    self.states["TD_inp"] = keras.layers.ZeroPadding2D(paddings)(self.states["TD_inp"])
                    R_inp = keras.layers.Concatenate()([self.states["E"], self.states["R"], self.states["TD_inp"]])
            if self.bottom_layer and self.training_args['object_representations']:
                object_representations, classification_diversity_loss = self.object_representations(tf.expand_dims(self.last_frame, axis=1)) # Inputs shape: (bs, 1, h, w, ic), Outputs shape: (bs, h, w, nc*oc)
                if not self.training_args["pretrain_classifier"]:
                    R_inp = keras.layers.Concatenate()([R_inp, object_representations])

            if not self.training_args["pretrain_classifier"]:
                if self.states["lstm"] is None:
                    self.states["R"], self.states["lstm"] = self.representation(R_inp)
                else:
                    self.states["R"], self.states["lstm"] = self.representation(R_inp, initial_states=self.states["lstm"])

                # FORM PREDICTION(S)
                self.states["P"] = K.minimum(self.prediction(self.states["R"]), self.pixel_max) if self.bottom_layer else self.prediction(self.states["R"])

            return classification_diversity_loss if self.bottom_layer and self.training_args['object_representations'] else tf.constant(0.0, shape=(self.batch_size, 1))

        elif direction == "bottom_up":
            # RETRIEVE TARGET(S) (bottom-up input) ~ (batch_size, im_height, im_width, output_channels)
            target = inputs[0]

            if self.bottom_layer: 
                self.set_last_frame(target)
            if not self.training_args["pretrain_classifier"]:

                self.states["T"] = target if self.bottom_layer else self.target(target)

                # COMPUTE TARGET ERROR
                self.states["E"] = self.error(self.states["P"], self.states["T"])

                # COMPUTE RAW ERROR FOR PLOTTING
                self.states["E_raw"] = self.states["T"] - self.states["P"]

            return self.states["E"]

        else:
            raise ValueError("Invalid direction. Must be 'top_down' or 'bottom_up'.")


class PredNet(keras.Model):
    def __init__(self, training_args, im_height=540, im_width=960, *args, **kwargs):
        super(PredNet, self).__init__(*args, **kwargs)
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
        self.classification_diversity_loss_weight = 0.1
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
                if l == self.num_layers - 1 and self.num_layers > 1:
                    BU_inp = None
                    TD_inp = None
                    _ = layer([BU_inp, TD_inp], direction="top_down", paddings=self.paddings[l])

                # Middle layers
                elif l > 0 and l < self.num_layers - 1:
                    BU_inp = None
                    TD_inp = self.predlayers[l + 1].states["R"]
                    _ = layer([BU_inp, TD_inp], direction="top_down", paddings=self.paddings[l])

                # Bottom layer
                else:
                    BU_inp = None
                    TD_inp = self.predlayers[l + 1].states["R"] if self.num_layers > 1 else None
                    classification_diversity_loss = layer([BU_inp, TD_inp], direction="top_down", paddings=self.paddings[l])
                
            # Apply classification diversity loss weight
            classification_diversity_loss = self.classification_diversity_loss_weight * classification_diversity_loss  # (batch_size, 1)

            """ Perform bottom-up pass, starting from the bottom layer """
            for l, layer in list(enumerate(self.predlayers)):
                # Bottom layer
                if l == 0:
                    # (self.batch_size, self.im_height, self.im_width, self.layer_input_channels[0])
                    BU_inp = inputs[:, t, ...]
                    TD_inp = None
                    frame_errors = layer([BU_inp, TD_inp], direction="bottom_up")
                # Middle and Top layers
                else:
                    BU_inp = self.predlayers[l - 1].states["E"]
                    TD_inp = None
                    frame_errors = layer([BU_inp, TD_inp], direction="bottom_up")
                # Update error in bottom-up pass
                if self.output_mode == "Error":
                    layer_frame_errors = self.layer_weights[l] * K.mean(K.batch_flatten(frame_errors), axis=-1, keepdims=True)  # (batch_size, 1)
                    all_frame_errors = layer_frame_errors if l == 0 else tf.add(all_frame_errors, layer_frame_errors)  # (batch_size, 1)

            if self.output_mode == "Error" and self.training_args["decompose_images"] and self.training_args["second_stage"]:
                # reconstruction error doesn't work with the randomized backgrounds in the input images for first stage
                predictions = self.predlayers[0].states["P"]
                targets = self.predlayers[0].states["T"]
                # Calculate reconstruction error
                reconstructed_image = K.minimum(1.0, tf.reduce_sum([predictions[...,i*3:(i+1)*3] for i in range(predictions.shape[-1]//3)], axis=0))
                original_image = K.minimum(1.0, tf.reduce_sum([targets[...,i*3:(i+1)*3] for i in range(targets.shape[-1]//3)], axis=0))
                recon_e_down = keras.backend.relu(original_image - reconstructed_image)
                recon_e_up = keras.backend.relu(reconstructed_image - original_image)
                recon_error = keras.layers.Concatenate(axis=-1)([recon_e_down, recon_e_up])
                recon_error = K.mean(K.batch_flatten(recon_error), axis=-1, keepdims=True)
                total_prediction_errors = all_frame_errors + recon_error # (batch_size, 1)
            elif self.output_mode == "Error" and self.training_args["decompose_images"]:
                total_prediction_errors = all_frame_errors

            # Calculate total error
            if self.output_mode == "Error":
                if self.training_args["pretrain_classifier"]:
                    all_error = classification_diversity_loss # (batch_size, 1)
                else:
                    all_error = total_prediction_errors + classification_diversity_loss # (batch_size, 1)

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



class CustomCNN(keras.layers.Layer):
    def __init__(self, num_classes, num_conv_layers=3, *args, **kwargs):
        super(CustomCNN, self).__init__(*args, **kwargs)
        self.num_classes = num_classes
        self.num_conv_layers = num_conv_layers
        self.conv_layers = []
        
        # Initialize multiple convolution and pooling layers based on layer_num
        for i in range(num_conv_layers):
            self.conv_layers.append(
                Conv2D(32 * (2 ** i), (3, 3), activation='relu', padding='same')
            )
            self.conv_layers.append(
                MaxPooling2D((2, 2))
            )
        
        self.flatten = Flatten()
        self.dense1024 = Dense(1024, activation='relu')  # Logits output
        self.predictions = Dense(self.num_classes, activation=None)

    def call(self, inputs):
        x = inputs
        for layer in self.conv_layers:
            x = layer(x)
        x = self.flatten(x)
        x = self.dense1024(x)
        pre_out = self.predictions(x)
        out = pre_out + tf.random.uniform(tf.shape(pre_out), 0, 1e-6) # Add noise to prevent zero logits
        return out
class CustomMobileNetV2(tf.keras.layers.Layer):
    def __init__(self, num_classes, input_shape, **kwargs):
        super(CustomMobileNetV2, self).__init__(**kwargs)
        self.num_classes = num_classes
        
        # Load the base model with pre-trained weights
        self.base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)
        
        # Freeze the layers of the base model
        for layer in self.base_model.layers:
            layer.trainable = False
        
        # Add custom layers on top of the base model
        self.global_avg_pool = GlobalAveragePooling2D()
        self.dense_1024 = Dense(1024, activation='relu')
        self.batch_norm = BatchNormalization()
        self.dropout = Dropout(0.5)
        self.predictions = Dense(self.num_classes, activation=None)
    
    def compute_output_shape(self, input_shape):
        return (self.num_classes,)

    def call(self, inputs):
        # Convert RGB to Grayscale
        # x = tf.image.rgb_to_grayscale(inputs)
        # # Convert grayscale back to 3 channels by duplicating the single channel three times
        # x = tf.image.grayscale_to_rgb(x)
        x = self.base_model(inputs)
        x = self.global_avg_pool(x)
        x = self.dense_1024(x)
        # x = self.batch_norm(x)
        # x = self.dropout(x)
        pre_out = self.predictions(x)
        out = pre_out + tf.random.uniform(tf.shape(pre_out), 0, 1e-6) # Add noise to prevent zero logits
        return out


class ObjectRepresentation(layers.Layer):
    '''
    Convert images of object masks to class IDs, then update and extract the corresponding object representations
    '''
    def __init__(self, num_classes, batch_size, im_height, im_width, output_channels, **kwargs):
        super(ObjectRepresentation, self).__init__(**kwargs)
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.im_height = im_height
        self.im_width = im_width
        self.output_channels = output_channels
        self.conv_lstm_general = ConvLSTM2D(filters=output_channels, kernel_size=(3, 3), padding='same', return_sequences=False, return_state=True, stateful=False, name='conv_lstm_general')
        self.conv_lstm_class = ConvLSTM2D(filters=output_channels, kernel_size=(3, 3), padding='same', return_sequences=False, return_state=True, stateful=False, name='conv_lstm_class')
        self.classifier = CustomMobileNetV2(num_classes=4, input_shape=(self.im_height, self.im_width, 3))
        # self.classifier = CustomCNN(num_classes=4)

        self.class_states_h = self.add_weight(shape=(num_classes, batch_size, im_height, im_width, output_channels), initializer='zeros', trainable=False, name='class_state_h')
        self.class_states_c = self.add_weight(shape=(num_classes, batch_size, im_height, im_width, output_channels), initializer='zeros', trainable=False, name='class_state_c')
        self.general_states_h = self.add_weight(shape=(1, batch_size, im_height, im_width, output_channels), initializer='zeros', trainable=False, name='general_state_h')
        self.general_states_c = self.add_weight(shape=(1, batch_size, im_height, im_width, output_channels), initializer='zeros', trainable=False, name='general_state_c')

        self.predicted_class_IDs = []


    def diff_gather(self, params, logits, beta=3):
        '''
        Differentiable gather operation.
        Params shape: (num_classes, batch_size, ...)
        Logits shape: (batch_size, num_classes)
        '''
        weights = tf.transpose(tf.nn.softmax(logits * beta), [1, 0]) # (num_classes, batch_size)
        current_weights_shape = weights.shape
        reshaped_weights = weights
        for _ in range(len(params.shape) - len(current_weights_shape)):
            reshaped_weights = tf.expand_dims(reshaped_weights, axis=-1)

        weighted_params = reshaped_weights * params
        weighted_sum = tf.reduce_sum(weighted_params, axis=0)
        return weighted_sum

    def diff_scatter_nd_update(self, A, B, logits, beta=1e10):
        """
        Update tensor A with values from tensor B based on highest indices indicated by a logits matrix.
        Like tf.tensor_scatter_nd_update, but differentiable, in the sense that integer class indices are not required.

        Args:
        A (tf.Tensor): A tensor of shape (nc, bs, h, w, oc).
        B (tf.Tensor): A tensor of shape (bs, h, w, oc).
        logits (tf.Tensor): A logits matrix of shape (bs, nc).

        Returns:
        tf.Tensor: Updated tensor A.
        """
        # Convert logits to one-hot
        one_hot = tf.nn.softmax(logits * beta) # (bs, nc)

        # Check dimensions
        if len(A.shape) != 5 or len(B.shape) != 4 or len(one_hot.shape) != 2:
            raise ValueError("Input tensors must be of the shape (nc, bs, h, w, oc), (bs, h, w, oc), and (bs, nc) respectively.")
        
        # Check dimension matching
        nc, bs, h, w, oc = A.shape
        if ((B.shape != (bs, h, w, oc)) and (B.shape != (None, h, w, oc))) or ((one_hot.shape != (bs, nc)) and (one_hot.shape != (None, nc))):
            raise ValueError("Dimension mismatch among inputs.")

        # Expand the one-hot matrix to match A's dimensions
        mask = tf.expand_dims(tf.expand_dims(tf.expand_dims(one_hot, -1), -1), -1)
        mask = tf.transpose(mask, [1, 0, 2, 3, 4])  # Reshape to (nc, bs, 1, 1, 1)

        # Expand B to broadcast over the nc dimension
        B_expanded = tf.expand_dims(B, 0)

        # Multiply A by (1 - mask) to zero out the update positions
        A_masked = A * (1 - mask)

        # Multiply B by mask to align updates
        B_masked = B_expanded * mask

        # Combine the two components
        A_updated = A_masked + B_masked

        return A_updated

    # def calculate_classification_diversity_loss(self, all_logits):
        # '''
        # Compute the classification diversity loss.
        # Args:
        # all_logits (tf.Tensor): A tensor of shape (num_predictions, batch_size, num_classes).
        # Returns:
        # tf.Tensor: The classification diversity loss as total entropy of the prediction matrix, per batch.
        # '''

        # def entropy(logits, axis):
        #     probs = tf.nn.softmax(logits, axis=axis)
        #     return -tf.reduce_sum(probs * tf.math.log(probs + 1e-9), axis=axis)

        # def total_matrix_entropy(logits):
        #     '''
        #     Args:
        #     logits: (batch_size, num_predictions, num_classes)
        #     Returns:
        #     total_entropy: (batch_size,)
        #     '''
        #     inter_prediction_uncertainty = tf.reduce_sum(entropy(logits, axis=2), axis=-1, keepdims=True) # Shape: (batch_size, 1)
        #     intra_prediction_overlap = tf.reduce_sum(entropy(logits, axis=1), axis=-1, keepdims=True) # Shape: (batch_size, 1)
        #     return intra_prediction_overlap# + 0.1*inter_prediction_uncertainty

        # def distinct_class_penalty(logits):
        #     '''
        #     Penalty for duplicate class predictions.
        #     Args:
        #     logits: (batch_size, num_predictions, num_classes)
        #     Returns:
        #     penalty: (batch_size,)
        #     '''            
        #     # Calculate sharp probability distribution across classes
        #     probs = tf.nn.softmax(logits*3, axis=2) # Shape: (batch_size, num_predictions, num_classes)
            
        #     # Calculate entropy of predictions for each class
        #     intra_prediction_overlap = -tf.reduce_sum(probs * tf.math.log(probs + 1e-9), axis=[1,2]) # Shape: (batch_size, 1)
            
        #     # Calculate penalty using MSE
        #     # penalty = tf.reduce_mean(tf.square(counts - 1), axis=-1, keepdims=True) # Shape: (batch_size, 1)
            
        #     return intra_prediction_overlap*100

        # # form batches of logits matrices
        # batch_logits_matrices = tf.transpose(all_logits, [1, 0, 2]) # Shape: (batch_size, num_predictions, num_classes)

        # # Compute the total entropy of the logits matrix
        # total_entropy = total_matrix_entropy(batch_logits_matrices) # Shape: (batch_size, 1)

        # # Compute the distinct class penalty
        # distinct_penalty = distinct_class_penalty(batch_logits_matrices) # Shape: (batch_size, 1)

        # return total_entropy# + distinct_penalty # Shape: (batch_size, 1)

    def calculate_classification_diversity_loss(self, all_logits):
        '''
        Compute the classification diversity loss.
        Args:
        all_logits (tf.Tensor): A tensor of shape (num_predictions, batch_size, num_classes).
        Returns:
        tf.Tensor: The classification diversity loss as total entropy of the prediction matrix, per batch.
        '''

        def entropy(logits, axis):
            probs = tf.nn.softmax(logits, axis=axis)
            return -tf.reduce_sum(probs * tf.math.log(probs + 1e-9), axis=axis)

        def total_matrix_entropy(logits):
            '''
            Args:
            logits: (batch_size, num_predictions, num_classes)
            Returns:
            total_entropy: (batch_size,)
            '''
            intra_prediction_overlap = tf.reduce_sum(entropy(logits, axis=1), axis=-1) # Shape: (batch_size,)
            inter_prediction_uncertainty = tf.reduce_sum(entropy(logits, axis=2), axis=-1) # Shape: (batch_size,)
            return intra_prediction_overlap# + inter_prediction_uncertainty

        def distinct_class_penalty(logits):
            '''
            Penalty for duplicate class predictions.
            Args:
            logits: (batch_size, num_predictions, num_classes)
            Returns:
            penalty: (batch_size,)
            '''
            probs = tf.nn.softmax(logits, axis=-1)
            max_probs = tf.reduce_max(probs, axis=-1)
            penalty = tf.reduce_sum(tf.square(max_probs - 1.0 / self.num_classes), axis=-1)
            return penalty

        def non_zero_logits_penalty(logits):
            '''
            Penalty to discourage zero logits.
            Args:
            logits: (batch_size, num_predictions, num_classes)
            Returns:
            penalty: (batch_size,)
            '''
            penalty = tf.reduce_sum(tf.nn.relu(-logits), axis=[1, 2])
            return penalty

        entropy_loss = total_matrix_entropy(all_logits)
        distinct_class_penalty_loss = distinct_class_penalty(all_logits)
        zero_logits_penalty = non_zero_logits_penalty(all_logits)
        
        # Combine the losses, scaling the penalty term
        total_loss = entropy_loss + 0.1 * distinct_class_penalty_loss

        return total_loss


    @tf.function
    def call(self, inputs):
        # Inputs shape: (bs, 1, h, w, oc)

        '''Update General Object States'''
        # get all current class object representations (class object hidden states) as input
        # reshaped to (nc, bs, h, w, oc) -> (bs, 1, h, w, nc*oc) for processing in ConvLSTM
        all_class_object_representations = self.class_states_h # (nc, bs, h, w, oc)
        all_class_object_representations = tf.expand_dims(tf.concat(tf.unstack(all_class_object_representations, axis=0), axis=-1), axis=1) # (bs, 1, h, w, nc*oc)

        # get current general object states
        current_general_states = [self.general_states_h[0], self.general_states_c[0]] # (bs, h, w, oc) x 2

        # apply the general ConvLSTM layer to get the updated general states
        _, new_general_state_h, new_general_state_c = self.conv_lstm_general(all_class_object_representations, initial_state=current_general_states) # (bs, h, w, oc) x 3

        # update the general states
        general_update_h = tf.tensor_scatter_nd_update(self.general_states_h, [[0]], [new_general_state_h])
        general_update_c = tf.tensor_scatter_nd_update(self.general_states_c, [[0]], [new_general_state_c])

        self.general_states_h.assign(general_update_h)
        self.general_states_c.assign(general_update_c)
        
        '''Update Class States and Obtain Outputs'''
        # Note that inputs are nc*3 channels, where nc is the number of classes and thus we process each 3-channel block separately
        
        output_class_tensors = []
        all_class_logits = []
        self.predicted_class_IDs = []
        for i in range(self.num_classes):

            frame = inputs[..., i*3:(i+1)*3] # (bs, 1, h, w, 3)

            # get updated general object representation (hidden general object state)
            new_general_object_representation = tf.expand_dims(new_general_state_h, axis=1) # (bs, 1, h, w, oc)

            # concatenate the general object representation with the input
            augmented_inputs = tf.concat([frame, new_general_object_representation], axis=-1) # (bs, 1, h, w, 3+oc)
            
            # classify the input frame to get the class logits predictions
            class_logits = self.classifier(tf.squeeze(frame, axis=1)) # (bs, nc)
            assert (class_logits.shape == (self.batch_size, self.num_classes) or class_logits.shape == (None, self.num_classes))

            # get current class states
            current_class_states = [self.diff_gather(self.class_states_h, class_logits), self.diff_gather(self.class_states_c, class_logits)] # (bs, h, w, oc)

            # apply the shared class ConvLSTM layer to get the updated class states
            class_output, new_class_state_h, new_class_state_c = self.conv_lstm_class(augmented_inputs, initial_state=current_class_states) # (bs, h, w, oc) x 3

            # update the class states
            class_update_h = self.diff_scatter_nd_update(self.class_states_h, new_class_state_h, class_logits)
            class_update_c = self.diff_scatter_nd_update(self.class_states_c, new_class_state_c, class_logits)
            
            self.class_states_h.assign(class_update_h)
            self.class_states_c.assign(class_update_c)

            # append the class output to the list of output_class_tensors
            output_class_tensors.append(class_output)
            # print("Class output shape:", class_output.shape)
            assert (class_output.shape == (self.batch_size, self.im_height, self.im_width, self.output_channels) or class_output.shape == (None, self.im_height, self.im_width, self.output_channels))

            # append the class logits to the list of all_class_logits
            all_class_logits.append(class_logits)
            
            """Debugging 1/2 - check if classifier is working"""
            val = tf.argmax(tf.nn.softmax(class_logits*1e10), axis=-1) # (bs,)
            try:
                # val = tf.squeeze(tf.convert_to_tensor(val))
                self.predicted_class_IDs.append(val)
            except Exception:
                # Skip symbolic tensors
                pass

        # stack the class outputs to get the final output
        output = tf.concat(output_class_tensors, axis=-1)
        # print("Output shape:", output.shape)
        assert (output.shape == (self.batch_size, self.im_height, self.im_width, self.num_classes*self.output_channels) or output.shape == (None, self.im_height, self.im_width, self.num_classes*self.output_channels))

        # calculate the classification diversity loss
        all_class_logits = tf.stack(all_class_logits) # (np, bs, nc)
        assert (all_class_logits.shape == (self.num_classes, self.batch_size, self.num_classes) or all_class_logits.shape == (self.num_classes, None, self.num_classes))
        classification_diversity_loss = self.calculate_classification_diversity_loss(all_class_logits)
        
        """Debugging 2/2 - check if classifier is working"""
        vals = tf.stack(self.predicted_class_IDs, axis=-1) # (bs, nc)
        tf.print("vals:", vals, output_stream=sys.stdout)
        # tf.print("shape:", vals.shape, output_stream=sys.stdout)

        # return the class-specific object representations (hidden class states)
        return output, classification_diversity_loss


class SceneDecomposer:
    def __init__(self, n_colors=4):
        self.n_colors = n_colors
        self.last_colors = None
        self.last_masks = None

    def report_state(self):
        print("Last colors: ", self.last_colors)
        print("Last masks: ", self.last_masks)

    def quantize_image(self, image, num_colors=4):
        return image.convert('RGBA').quantize(colors=num_colors, method=Image.FASTOCTREE)

    def process_single_image(self, image):
        '''
        Process a single image and return a list of masks, one for each color in the image.
        Expected input: PIL Image or numpy array with shape (H, W, 3), float32, range [0, 1]
        '''
        if type(image) is np.ndarray:
            # print(type(image))
            # print("Processing single image")
            image = Image.fromarray((image * 255).astype(np.uint8))
        quantized_image = self.quantize_image(image, self.n_colors)
        quantized_image = quantized_image.convert('RGBA')
        data = quantized_image.load()

        # Find unique colors in the quantized image
        unique_colors = set()
        for y in range(quantized_image.size[1]):
            for x in range(quantized_image.size[0]):
                unique_colors.add(data[x, y])

        unique_colors = list(unique_colors)

        # Ensure we have at least n_colors unique colors
        while len(unique_colors) < self.n_colors:
            # print(f"Colors: {len(unique_colors)}, {self.n_colors}")
            unique_colors.append((0, 0, 0, 255))

        # if len(unique_colors) < self.n_colors:
        #     # input("Press Enter to continue...")
        #     raise ValueError("Not enough unique colors")

        unique_colors = np.array(unique_colors)

        # if self.stage == 1:
        #     # Random backgrounds
        #     masks = [np.random.randint(0, 256, (quantized_image.size[1], quantized_image.size[0], 4), dtype=np.uint8) for _ in range(self.n_colors)]
        # elif self.stage == 2:
        #     # Black backgrounds
        #     masks = [np.full((quantized_image.size[1], quantized_image.size[0], 4), (0, 0, 0, 255), dtype=np.uint8) for _ in range(self.n_colors)]

            # Black backgrounds
        masks = [np.full((quantized_image.size[1], quantized_image.size[0], 4), (0, 0, 0, 255), dtype=np.uint8) for _ in range(self.n_colors)]
        # masks = [Image.new('RGBA', quantized_image.size, (0, 0, 0, 255)) for _ in range(self.n_colors)]
        # mask_data = [mask.load() for mask in masks]

        color_to_index = {tuple(color): index for index, color in enumerate(unique_colors)}

        for y in range(quantized_image.size[1]):
            for x in range(quantized_image.size[0]):
                pixel = data[x, y]
                masks[color_to_index[tuple(pixel)]][y, x] = pixel

        new_masks = self.align_masks_by_color(unique_colors, masks)

        # print(f"{[np.array(mask).shape for mask in new_masks]}")
        # print(f"{len(unique_colors)} unique colors")
        # print(f"{[col for col in unique_colors]}")

        # new_masks = [np.array(mask)[..., :3] for mask in new_masks]
        new_masks = [mask[..., :3] for mask in new_masks]
        new_masks = np.concatenate(new_masks, axis=-1)

        # input("Press Enter to continue...")

        return new_masks

    def align_masks_by_color(self, new_colors, new_masks):
        if self.last_colors is None:
            self.last_colors = new_colors
            self.last_masks = new_masks
            return new_masks

        # Ensure both color arrays are 2-dimensional
        new_colors = np.array(new_colors).reshape(-1, 4)
        self.last_colors = np.array(self.last_colors).reshape(-1, 4)

        distance_matrix = cdist(self.last_colors, new_colors, metric='euclidean')
        row_indices, col_indices = linear_sum_assignment(distance_matrix)
        # print(f"Row indices: {row_indices}, Col indices: {col_indices}")

        reordered_masks = [None] * len(new_masks)
        for i, j in zip(row_indices, col_indices):
            reordered_masks[i] = new_masks[j]
        
        # Also reorder the new colors
        new_colors = new_colors[col_indices]

        self.last_colors = new_colors
        self.last_masks = reordered_masks

        return reordered_masks

    def process_sequence(self, sequence, stage=2):
        """
        Process a sequence of images with shape (T, 64, 64, 3) and return masks with shape (T, 64, 64, n_colors*3)
        """
        T, H, W, C = sequence.shape
        masks_sequence = np.zeros((T, H, W, self.n_colors*C), dtype=np.float32)

        self.clear_state()
        for t in range(T):
            new_masks = self.process_single_image(sequence[t])
            masks_sequence[t] = new_masks

        if stage == 1:
            for t in range(T):
                mask_stack = masks_sequence[t]

                for i in range(self.n_colors):
                    mask = mask_stack[..., i*3:(i+1)*3]

                    # Set black pixels to random colors
                    black_pixels_mask = (mask == [0, 0, 0]).all(axis=-1)
                    random_colors = np.random.randint(0, 256, (black_pixels_mask.sum(), 3), dtype=np.uint8)
                    mask[black_pixels_mask] = random_colors

        # finally, convert sequence to float between 0 and 1
        masks_sequence = masks_sequence / 255.0

        return masks_sequence

    def process_batch(self, batch):
        """
        Process a batch of images with shape (BS, T, 64, 64, 3) and return masks with shape (BS, T, 64, 64, n_colors*3)
        """
        BS, T, H, W, _ = batch.shape
        masks_batch = np.zeros((BS, T, H, W, self.n_colors*3), dtype=np.float32)

        for b in range(BS):
            self.clear_state()
            for t in range(T):
                image = Image.fromarray(batch[b, t])
                new_masks = self.process_single_image(image)

                for i, mask in enumerate(new_masks):
                    mask_array = np.array(mask)
                    masks_batch[b, t, ..., i*3:(i+1)*3] = mask_array[..., :3] # Discard the alpha channel

        return masks_batch

    def process_list(self, image_list):
        '''Expecting a sequence of images in a list'''
        print("Processing list of images")
        decomposed_images = [self.process_single_image(im) for im in image_list]
        self.clear_state()
        # print("Decomposed images: ", len(decomposed_images))
        return decomposed_images

    def clear_state(self):
        self.last_colors = None
        self.last_masks = None