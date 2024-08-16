import os
import warnings
# Suppress warnings
warnings.filterwarnings("ignore")
# or '2' to filter out INFO messages too
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
from keras import layers
from keras import backend as K
import keras
import numpy as np
import sys
from PN_models.PN_Common import Target, Prediction, Error, Representation
from PN_models.LinearPNet import LinearPNet
from keras.layers import Dense, GlobalAveragePooling2D, ConvLSTM2D, BatchNormalization, Dropout, Conv2D, MaxPooling2D, Flatten, Input, UpSampling2D, Concatenate, Add, Activation, Multiply
from keras.models import Model
from keras.applications import MobileNetV2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from PN_models.ObjectRepresentations import ObjectRepresentations, ObjectRepDecoder

import pdb



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
        self.num_classes = training_args['num_classes']
        self.output_channels = output_channels
        self.bottom_layer = bottom_layer
        self.top_layer = top_layer
        # R = Representation, P = Prediction, T = Target, E = Error, and P == A_hat and T == A
        self.states = {"R": None, "P": None, "PM": None, "T": None, "E": None, "TD_inp": None, "L_inp": None}
        
        self.representation = Representation(output_channels, layer_num=self.layer_num, name=f"Representation_Layer{self.layer_num}")
        # if self.bottom_layer and self.training_args['object_representations']:
        #     self.prediction = Prediction(output_channels // self.num_classes, layer_num=self.layer_num, name=f"Prediction_Layer{self.layer_num}")
        # else:
        self.prediction = Prediction(output_channels, layer_num=self.layer_num, name=f"Prediction_Layer{self.layer_num}")
        
        if not self.bottom_layer:
            self.target = Target(output_channels, layer_num=self.layer_num, name=f"Target_Layer{self.layer_num}")
        
        if self.bottom_layer and self.training_args['object_representations']:
            # self.object_representations = ObjectRepresentation(training_args, num_classes=self.num_classes, batch_size=self.batch_size, im_height=self.im_height, im_width=self.im_width, output_channels=3, name=f"ObjectRepresentation_Layer{self.layer_num}")
            self.object_representations = ObjectRepresentations(training_args, latent_dim=32, num_im_in_seq=2, seq_vae_convlstm_channels=16, name="Object_Representations_Layer0")
            # self.object_representations = ObjectRepresentation_ConvVAE_LatentLPN(training_args, num_classes=self.num_classes, batch_size=self.batch_size, im_height=self.im_height, im_width=self.im_width, output_channels=1, name=f"ObjectRepresentation_Layer{self.layer_num}")
        
        self.error = Error(layer_num=self.layer_num, name=f"Error_Layer{self.layer_num}")
        
        self.upsample = layers.UpSampling2D((2, 2), name=f"Upsample_Layer{self.layer_num}")

        if self.training_args['object_representations']:
            self.or_decoder = ObjectRepDecoder(training_args=self.training_args, layer_num=self.layer_num, im_height=self.im_height, im_width=self.im_width, latent_dim=32, num_slv_keep=20, num_im_in_seq=2, num_classes=self.num_classes, name=f"Object_Rep_Decoder_Layer_{self.layer_num}")

        
        # if self.training_args['top_down_attention'] and not self.top_layer:
        #     self.attention = unet_attention((self.im_height, self.im_width, self.training_args["output_channels"][layer_num+1]), self.output_channels)
        # else:
        #     self.attention = None
        
        self.last_frame = None

    def initialize_states(self, batch_size):
        # Initialize internal layer states
        self.states["R"] = tf.zeros((batch_size, self.im_height, self.im_width, self.output_channels))
        self.states["P_M"] = tf.zeros((batch_size, self.im_height, self.im_width, self.output_channels))
        self.states["P"] = tf.zeros((batch_size, self.im_height, self.im_width, self.output_channels))
        self.states["T"] = tf.zeros((batch_size, self.im_height, self.im_width, self.output_channels))
        # if self.training_args['top_down_attention'] and not self.top_layer:
        #     self.states["E"] = tf.zeros((batch_size, self.im_height, self.im_width, 4 * self.output_channels)) # double for the pos/neg concatenated error
        # else:
        self.states["E"] = tf.zeros((batch_size, self.im_height, self.im_width, 2 * self.output_channels)) # double for the pos/neg concatenated error
        self.states["TD_inp"] = None
        self.states["lstm"] = None
        self.states["ORs"] = tf.zeros((batch_size, self.im_height, self.im_width, self.num_classes * 1)) # Object representations are single channel

        self.last_frame = tf.zeros((batch_size, self.im_height, self.im_width, self.training_args['output_channels'][0]))

        if self.bottom_layer and self.training_args['object_representations']:
            self.object_representations.initialize_states()

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
        self.states["ORs"] = None

        self.clear_last_frame()

        if self.bottom_layer and self.training_args['object_representations']:
            self.object_representations.clear_states()

    def set_last_frame(self, last_frame):
        for _ in range(self.layer_num):
            last_frame = keras.layers.MaxPool2D((2, 2))(last_frame)
        self.last_frame = last_frame

    def clear_last_frame(self):
        self.last_frame = None

    def update_object_representations(self, meta_latent_vectors, latest_sequence_latent_vectors, recent_frame_sequences):
        self.states["ORs"] = self.or_decoder([meta_latent_vectors, latest_sequence_latent_vectors, recent_frame_sequences])

    def get_meta_latent_vectors_and_recent_frame_sequences(self):
        assert self.bottom_layer == True, "Meta and sequence latent vectors are only created by the bottom PredLayer."
        meta_latent_vectors, latest_sequence_latent_vectors, recent_frame_sequences = self.object_representations(self.last_frame)
        return meta_latent_vectors, latest_sequence_latent_vectors, recent_frame_sequences


    def call(self, inputs=None, direction="top_down", paddings=None):
        # PredLayer should update internal states when called with new TD and BU inputs, inputs[0] = BU, inputs[1] = TD

        if direction == "top_down":
            # UPDATE REPRESENTATION
            if self.top_layer:
                R_inp = keras.layers.Concatenate()([self.states["E"], self.states["R"]])
            else:
                self.states["TD_inp"] = self.upsample(inputs[1])
                self.states["TD_inp"] = keras.layers.ZeroPadding2D(paddings)(self.states["TD_inp"])
                # if self.training_args['top_down_attention']:
                #     self.states["E"] = Concatenate()([self.states["E"] * self.attention(self.states["TD_inp"]), self.states["E"]])
                # if self.attention:
                #     R_inp = keras.layers.Concatenate()([self.states["E"], self.states["R"], self.states["R"] * self.attention(self.states["TD_inp"]), self.states["TD_inp"]])
                # else:
                R_inp = keras.layers.Concatenate()([self.states["E"], self.states["R"], self.states["TD_inp"]])
            
            if self.training_args['object_representations']:
                # object_representations, classification_diversity_loss = self.object_representations(tf.expand_dims(self.last_frame, axis=1)) # Inputs shape: (bs, 1, h, w, ic), Outputs shape: (bs, h, w, nc*oc)
                # object_representations = self.object_representations(self.last_frame) # Inputs shape: (bs, 1, h, w, ic), Outputs shape: (bs, h, w, nc*oc)
                R_inp = keras.layers.Concatenate()([self.states["ORs"], R_inp])

            if self.states["lstm"] is None:
                self.states["R"], self.states["lstm"] = self.representation(R_inp)
            else:
                self.states["R"], self.states["lstm"] = self.representation(R_inp, initial_states=self.states["lstm"])

            # if self.training_args['top_down_attention'] and not self.top_layer:
            #     self.states["R"] = Concatenate()([R_out * self.attention(self.states["TD_inp"]), R_out])
            # else:
            # self.states["R"] = R_out

            # FORM PREDICTION(S)
            # if self.bottom_layer and self.training_args['object_representations']:
            #     preds = []
            #     orc = object_representations.shape[-1] // self.num_classes
            #     for i in range(self.states["R"].shape[-1] // 3):
            #         P_inp = Concatenate()([self.states["R"][..., i*3:(i+1)*3], object_representations[..., i*orc:(i+1)*orc]])
            #         P_out = K.minimum(self.prediction(P_inp), self.pixel_max) if self.bottom_layer else self.prediction(P_inp)
            #         preds.append(P_out)
            #     self.states["P"] = Concatenate()(preds)
            # else:
            self.states["P"] = K.minimum(self.prediction(self.states["R"]), self.pixel_max) if self.bottom_layer else self.prediction(self.states["R"])

            # return classification_diversity_loss if self.bottom_layer and self.training_args['object_representations'] else tf.constant(0.0, shape=(self.batch_size, 1))
            # return maintained_vectors_loss if self.bottom_layer and self.training_args['object_representations'] else 0.0 #{"lpn_loss":0, "vae_loss":0}

        elif direction == "bottom_up":
            # RETRIEVE TARGET(S) (bottom-up input) ~ (batch_size, im_height, im_width, output_channels)
            target = inputs[0]

            if self.bottom_layer: 
                self.set_last_frame(target)

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
            # set dummy last frame for first layer
            if l == 0:
                self.predlayers[-1].set_last_frame(tf.random.uniform((self.batch_size, self.im_height, self.im_width, self.layer_input_channels[0])))
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

            if self.training_args['object_representations']:
                # Get updated meta latent vectors from object representations unit in the bottom PredLayer
                meta_latent_vectors, latest_sequence_latent_vectors, recent_frame_sequences = self.predlayers[0].get_meta_latent_vectors_and_recent_frame_sequences()

            for l, layer in reversed(list(enumerate(self.predlayers))):
                # BU_inp = bottom-up input, TD_inp = top-down input

                if self.training_args['object_representations']:
                    # Each layer creates its own object representations based on the class-specific meta latent vectors and recent frame sequences for each class
                    layer.update_object_representations(meta_latent_vectors, latest_sequence_latent_vectors, recent_frame_sequences)

                # Top layer
                if l == self.num_layers - 1:
                    BU_inp = None
                    TD_inp = None
                    layer([BU_inp, TD_inp], direction="top_down", paddings=self.paddings[l])

                # Middle layers
                elif l < self.num_layers - 1 and l > 0:
                    BU_inp = None
                    TD_inp = self.predlayers[l + 1].states["R"]
                    layer([BU_inp, TD_inp], direction="top_down", paddings=self.paddings[l])

                # Bottom layer
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
                # Note: reconstruction error doesn't work with the randomized backgrounds in the input images for first stage
                predictions = self.predlayers[0].states["P"]
                targets = self.predlayers[0].states["T"]

                # If necessary, isolate the masks and prepare for reconstruction
                if self.training_args["include_frame"]:
                    pred_masks = predictions[..., :-3]
                    pred_frames = predictions[..., -3:]
                    target_masks = targets[..., :-3]
                    target_frames = targets[..., -3:]
                    bs, h, w, n_c = pred_masks.shape
                    n = n_c // 3  # Number of masks
                else:
                    pred_masks = predictions
                    pred_frames = None
                    target_masks = targets
                    target_frames = None
                    bs, h, w, n_c = pred_masks.shape
                    n = n_c // 3  # Number of masks
                
                # Form reconstructions
                pred_masks_reshaped = tf.reshape(pred_masks, (bs, h, w, n, 3))
                target_masks_reshaped = tf.reshape(target_masks, (bs, h, w, n, 3))
                reconstructed_image = K.minimum(1.0, tf.reduce_max(pred_masks_reshaped, axis=2))
                original_image = K.minimum(1.0, tf.reduce_max(target_masks_reshaped, axis=2))
                
                # Calculate reconstruction error
                recon_e_down = keras.backend.relu(original_image - reconstructed_image)
                recon_e_up = keras.backend.relu(reconstructed_image - original_image)
                recon_error = keras.layers.Concatenate(axis=-1)([recon_e_down, recon_e_up])
                recon_error = K.mean(K.batch_flatten(recon_error), axis=-1, keepdims=True)
                total_prediction_errors = 0.1*recon_error + all_frame_errors # (batch_size, 1), average of frame and reconstruction errors

            elif self.output_mode == "Error" and (not self.training_args["decompose_images"] or not self.training_args["second_stage"]):
                total_prediction_errors = all_frame_errors

            # save outputs over time
            if self.output_mode == "Error":
                all_error = total_prediction_errors # Calculate total error, shape: (batch_size, 1)
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

# Define U-Net for generating attention weights
def unet_attention(input_shape, output_channels):
    inputs = Input(input_shape)
    
    # Encoding path
    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D((2, 2))(conv1)
    
    conv2 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D((2, 2))(conv2)
    
    # Bottleneck
    conv3 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv3)
    
    # Decoding path
    up1 = UpSampling2D((2, 2))(conv3)
    concat1 = Concatenate()([up1, conv2])
    conv4 = Conv2D(128, (3, 3), activation='relu', padding='same')(concat1)
    conv4 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv4)
    
    up2 = UpSampling2D((2, 2))(conv4)
    concat2 = Concatenate()([up2, conv1])
    conv5 = Conv2D(64, (3, 3), activation='relu', padding='same')(concat2)
    conv5 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv5)
    
    # Output layer for attention weights, matching the shape of R
    output = Conv2D(output_channels, (1, 1), activation='sigmoid', padding='same')(conv5)
    
    return Model(inputs, output)


class CustomCNN(keras.layers.Layer):
    def __init__(self, num_classes, num_conv_layers=3, trainable=True, *args, **kwargs):
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

        # Set trainability
        # self.set_trainable(trainable)

    def convert_to_binary(self, input_tensor):
        # Assume input_tensor is your input of shape (BS, h, w, c)
        # Create a mask where non-black pixels are set to 1 and black pixels are set to 0
        mask = tf.reduce_sum(input_tensor, axis=-1, keepdims=True) > 0
        
        # Convert the mask to float
        binary_image = tf.cast(mask, tf.float32)
        
        # Return the binary image with the same number of channels
        return tf.tile(binary_image, [1, 1, 1, input_tensor.shape[-1]])
    
    def call(self, inputs):
        # Convert inputs (BS, 64, 64, 3) to x (BS, 64, 64, 3) with binary 0.0 or 1.0 values
        x = self.convert_to_binary(inputs)
        for layer in self.conv_layers:
            x = layer(x)
        x = self.flatten(x)
        x = self.dense1024(x)
        pre_out = self.predictions(x)
        out = pre_out + tf.random.uniform(tf.shape(pre_out), 0, 1e-6) # Add noise to prevent zero logits
        return out

    def set_trainable(self, trainable):
        for layer in self.conv_layers:
            layer.trainable = trainable
        self.dense1024.trainable = trainable
        self.predictions.trainable = trainable
        print(f"Classifier trainability set to: {trainable}")

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
        # self.batch_norm = BatchNormalization()
        # self.dropout = Dropout(0.5)
        self.predictions = Dense(self.num_classes, activation=None)
    
    def compute_output_shape(self, input_shape):
        return (self.num_classes,)

    def call(self, inputs):
        # Convert RGB to Grayscale
        x = tf.image.rgb_to_grayscale(inputs)
        # Convert grayscale back to 3 channels by duplicating the single channel three times
        x = tf.image.grayscale_to_rgb(x)
        x = self.base_model(inputs)
        x = self.global_avg_pool(x)
        x = self.dense_1024(x)
        # x = self.batch_norm(x)
        # x = self.dropout(x)
        pre_out = self.predictions(x)
        out = pre_out + tf.random.uniform(tf.shape(pre_out), 0, 1e-6) # Add noise to prevent zero logits
        return out

class CustomConvLSTM(keras.layers.Layer):
    def __init__(self, output_channels, layer_num=0, *args, **kwargs):
        super(CustomConvLSTM, self).__init__(*args, **kwargs)
        self.output_channels = output_channels
        self.layer_num = layer_num
        self.conv_i = Conv2D(output_channels, (3, 3), padding="same", activation="hard_sigmoid", name=f"Conv_i_Layer{layer_num}")
        self.conv_f = Conv2D(output_channels, (3, 3), padding="same", activation="hard_sigmoid", name=f"Conv_f_Layer{layer_num}")
        self.conv_o = Conv2D(output_channels, (3, 3), padding="same", activation="hard_sigmoid", name=f"Conv_o_Layer{layer_num}")
        self.conv_c = Conv2D(output_channels, (3, 3), padding="same", activation="tanh", name=f"Conv_c_Layer{layer_num}")

    def call(self, inputs, initial_state=None):
        inputs = tf.squeeze(inputs, axis=1)
        i = self.conv_i(inputs)
        f = self.conv_f(inputs)
        o = self.conv_o(inputs)
        h, c = initial_state if initial_state is not None else 2 * [tf.zeros(f.shape, dtype=tf.float32)]
        c = f * c + i * self.conv_c(inputs)
        h = o * keras.activations.tanh(c)
        output = h
        states = [h, c]
        return output, h, c

from keras.layers import Conv2D, Layer, Concatenate, Add, Activation, Multiply
from keras import Sequential
import tensorflow as tf

class InceptionModule(Layer):
    def __init__(self, output_channels, layer_num, gate, *args, **kwargs):
        super(InceptionModule, self).__init__(*args, **kwargs)
        initializer = tf.keras.initializers.GlorotUniform()
        self.conv1 = Conv2D(output_channels, (1, 1), padding="same", kernel_initializer=initializer, name=f"Conv_{gate}_Layer{layer_num}_1x1")
        self.conv3 = Conv2D(output_channels, (3, 3), padding="same", kernel_initializer=initializer, name=f"Conv_{gate}_Layer{layer_num}_3x3")
        self.conv5 = Conv2D(output_channels, (5, 5), padding="same", kernel_initializer=initializer, name=f"Conv_{gate}_Layer{layer_num}_5x5")
        self.conv5 = Conv2D(output_channels, (7, 7), padding="same", kernel_initializer=initializer, name=f"Conv_{gate}_Layer{layer_num}_5x5")
        self.align_channels = Conv2D(output_channels, (1, 1), padding="same", kernel_initializer=initializer, name=f"Align_{gate}_Layer{layer_num}_1x1")
    
    def call(self, inputs):
        conv1_out = self.conv1(inputs)
        conv3_out = self.conv3(inputs)
        conv5_out = self.conv5(inputs)
        concat_out = Concatenate()([conv1_out, conv3_out, conv5_out])
        aligned_out = self.align_channels(concat_out)
        return aligned_out

class DilatedConv(Layer):
    def __init__(self, output_channels, layer_num, gate, *args, **kwargs):
        super(DilatedConv, self).__init__(*args, **kwargs)
        initializer = tf.keras.initializers.GlorotUniform()
        self.dilated_conv = Conv2D(output_channels, (3, 3), padding="same", dilation_rate=2, kernel_initializer=initializer, name=f"Conv_{gate}_Layer{layer_num}_dilated")
    
    def call(self, inputs):
        return self.dilated_conv(inputs)

class AdvCustomConvLSTM(Layer):
    def __init__(self, output_channels, input_channels, layer_num=0, *args, **kwargs):
        super(AdvCustomConvLSTM, self).__init__(*args, **kwargs)
        self.output_channels = output_channels
        self.input_channels = input_channels
        self.layer_num = layer_num
        initializer = tf.keras.initializers.GlorotUniform()
        
        # Inception and dilated convolution modules for each gate
        self.inception_i = InceptionModule(self.output_channels, layer_num, 'i')
        self.inception_f = InceptionModule(self.output_channels, layer_num, 'f')
        self.inception_o = InceptionModule(self.output_channels, layer_num, 'o')
        self.inception_c = InceptionModule(self.output_channels, layer_num, 'c')
        
        self.dilated_i = DilatedConv(self.output_channels, layer_num, 'i')
        self.dilated_f = DilatedConv(self.output_channels, layer_num, 'f')
        self.dilated_o = DilatedConv(self.output_channels, layer_num, 'o')
        self.dilated_c = DilatedConv(self.output_channels, layer_num, 'c')
        
        # Align input channels with output channels for residual connections
        self.align_input_i = Conv2D(self.output_channels, (1, 1), padding="same", kernel_initializer=initializer, name=f"Align_input_i_Layer{layer_num}")
        self.align_input_f = Conv2D(self.output_channels, (1, 1), padding="same", kernel_initializer=initializer, name=f"Align_input_f_Layer{layer_num}")
        self.align_input_o = Conv2D(self.output_channels, (1, 1), padding="same", kernel_initializer=initializer, name=f"Align_input_o_Layer{layer_num}")
        self.align_input_c = Conv2D(self.output_channels, (1, 1), padding="same", kernel_initializer=initializer, name=f"Align_input_c_Layer{layer_num}")
        
        # Attention mechanism
        self.attention = Conv2D(output_channels, (1, 1), padding="same", activation="sigmoid", kernel_initializer=initializer, name=f"Attention_Layer{layer_num}")
        
        # Batch normalization layers
        self.bn_i = BatchNormalization(name=f"BN_i_Layer{layer_num}")
        self.bn_f = BatchNormalization(name=f"BN_f_Layer{layer_num}")
        self.bn_o = BatchNormalization(name=f"BN_o_Layer{layer_num}")
        self.bn_c = BatchNormalization(name=f"BN_c_Layer{layer_num}")

    def call(self, inputs, initial_state=None):
        inputs = tf.squeeze(inputs, axis=1)
        
        # Align input channels for residual connections
        aligned_input_i = self.align_input_i(inputs)
        aligned_input_f = self.align_input_f(inputs)
        aligned_input_o = self.align_input_o(inputs)
        aligned_input_c = self.align_input_c(inputs)
        
        # Process inputs through inception modules and dilated convolutions
        inception_i_out = self.inception_i(inputs)
        inception_f_out = self.inception_f(inputs)
        inception_o_out = self.inception_o(inputs)
        inception_c_out = self.inception_c(inputs)
        
        dilated_i_out = self.dilated_i(inception_i_out)
        dilated_f_out = self.dilated_f(inception_f_out)
        dilated_o_out = self.dilated_o(inception_o_out)
        dilated_c_out = self.dilated_c(inception_c_out)
        
        # Add residual connections
        i = Add()([aligned_input_i, dilated_i_out])
        f = Add()([aligned_input_f, dilated_f_out])
        o = Add()([aligned_input_o, dilated_o_out])
        c_tilde = Add()([aligned_input_c, dilated_c_out])
        
        # Apply batch normalization
        # i = self.bn_i(i)
        # f = self.bn_f(f)
        # o = self.bn_o(o)
        # c_tilde = self.bn_c(c_tilde)

        def check_nan(tensor, name):
            if tf.is_symbolic_tensor(tensor):
                return False
            elif tf.reduce_any(tf.math.is_nan(tensor)):
                tf.print(f"NaN detected in {name}")
                return True
            return False

        # Add checks in the call method
        if check_nan(i, "i") or check_nan(f, "f") or check_nan(o, "o") or check_nan(c_tilde, "c_tilde"):
            raise ValueError("NaN detected")


        h, c = initial_state if initial_state is not None else 2 * [tf.zeros_like(i, dtype=tf.float32)]
        c = f * c + i * c_tilde
        h = o * tf.keras.activations.tanh(c)
        
        # Apply attention mechanism
        attention_weights = self.attention(h)
        h = Multiply()([h, attention_weights])
        
        output = h
        states = [h, c]
        return output, h, c

class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

class CustomConvLSTMCell(layers.Layer):
    def __init__(self, output_channels=12, layer_num=0, *args, **kwargs):
        super(CustomConvLSTMCell, self).__init__(*args, **kwargs)
        self.output_channels = output_channels
        self.layer_num = layer_num
        self.conv_i = layers.Conv2D(output_channels, (3, 3), padding="same", activation="hard_sigmoid", name=f"Conv_i_Layer{layer_num}")
        self.conv_f = layers.Conv2D(output_channels, (3, 3), padding="same", activation="hard_sigmoid", name=f"Conv_f_Layer{layer_num}")
        self.conv_o = layers.Conv2D(output_channels, (3, 3), padding="same", activation="hard_sigmoid", name=f"Conv_o_Layer{layer_num}")
        self.conv_c = layers.Conv2D(output_channels, (3, 3), padding="same", activation="tanh", name=f"Conv_c_Layer{layer_num}")

    def call(self, inputs, states):
        h, c = states
        i = self.conv_i(inputs)
        f = self.conv_f(inputs)
        o = self.conv_o(inputs)
        c = f * c + i * self.conv_c(inputs)
        h = o * tf.keras.activations.tanh(c)
        return h, [h, c]

class CustomLSTM(layers.Layer):
    def __init__(self, output_units, layer_num=0, *args, **kwargs):
        super(CustomLSTM, self).__init__(*args, **kwargs)
        self.output_units = output_units
        self.layer_num = layer_num
        self.lstm_cell = layers.LSTM(output_units, return_state=True, return_sequences=False, name=f"LSTM_Layer{layer_num}")

    def call(self, inputs, initial_state=None):
        inputs = tf.expand_dims(inputs, axis=1)
        # Assuming inputs shape is (batch_size, time_steps, features)
        if initial_state is not None:
            h, c = initial_state
        else:
            h = tf.zeros((tf.shape(inputs)[0], self.output_units), dtype=tf.float32)
            c = tf.zeros((tf.shape(inputs)[0], self.output_units), dtype=tf.float32)
        
        output, h, c = self.lstm_cell(inputs, initial_state=[h, c])
        return output, h, c

class ConvLSTMVAE(keras.models.Model):
    def __init__(self, latent_dim, image_shape, output_channels=12, *args, **kwargs):
        super(ConvLSTMVAE, self).__init__(*args, **kwargs)
        self.latent_dim = latent_dim
        self.image_shape = image_shape
        self.output_channels = output_channels
        
        # Encoder
        self.encoder_conv = tf.keras.Sequential([
            # layers.InputLayer(input_shape=(*image_shape,)),
            layers.Conv2D(16, (3, 3), activation='relu', strides=(2, 2), padding='same'),
            layers.Conv2D(32, (3, 3), activation='relu', strides=(2, 2), padding='same')
        ])
        self.encoder_lstm_cell = CustomConvLSTMCell(64)
        self.encoder_dense = layers.Dense(latent_dim + latent_dim)  # z_mean and z_log_var

        # Latent space
        self.sampling = Sampling()
        
        # Decoder
        self.decoder_dense = layers.Dense(8 * 8 * self.output_channels, activation='relu')
        self.decoder_lstm_cell = CustomConvLSTMCell(self.output_channels)
        self.decoder_conv = tf.keras.Sequential([
            # layers.InputLayer(input_shape=(8, 8, self.output_channels)),
            layers.Conv2DTranspose(32, (3, 3), activation='relu', strides=(2, 2), padding='same'),
            layers.Conv2DTranspose(16, (3, 3), activation='relu', strides=(2, 2), padding='same'),
            layers.Conv2DTranspose(8, (3, 3), activation='relu', strides=(2, 2), padding='same'),
            layers.Conv2DTranspose(image_shape[-1], (3, 3), activation='sigmoid', padding='same')
        ])

        # Latent Rep to Object Rep
        self.object_rep_dense = layers.Dense(8 * 8 * self.output_channels, activation='relu')
        self.object_rep_lstm_cell = CustomConvLSTMCell(self.output_channels)
        self.object_rep_conv = tf.keras.Sequential([
            # layers.InputLayer(input_shape=(8, 8, self.output_channels)),
            layers.Conv2DTranspose(32, (3, 3), activation='relu', strides=(2, 2), padding='same'),
            layers.Conv2DTranspose(16, (3, 3), activation='relu', strides=(2, 2), padding='same'),
            layers.Conv2DTranspose(8, (3, 3), activation='relu', strides=(2, 2), padding='same'),
            layers.Conv2DTranspose(self.output_channels, (3, 3), activation='sigmoid', padding='same')
        ])
    
    def encode(self, x, states=None):
        # x shape: (batch_size, 64, 64, 3)
        x = self.encoder_conv(x) # (batch_size, 16, 16, 64)
        if states is None:
            h, c = [tf.zeros_like(x) for _ in range(2)]
        else:
            h, c = states
        x = tf.concat((x, h), axis=-1) # Add a time dimension, (batch_size, 16, 16, 128)
        h, [h, c] = self.encoder_lstm_cell(x, [h, c])
        h_flat = tf.reshape(h, [tf.shape(h)[0], -1])  # Flatten the spatial dimensions for the dense layer
        z_mean, z_log_var = tf.split(self.encoder_dense(h_flat), num_or_size_splits=2, axis=1)
        return z_mean, z_log_var, [h, c]

    def decode(self, z, states=None):
        x = self.decoder_dense(z)
        x = tf.reshape(x, (-1, 8, 8, self.output_channels))
        if states is None:
            h, c = [tf.zeros_like(x) for _ in range(2)]
        else:
            h, c = states
        x = tf.concat((x, h), axis=-1) # Add a time dimension, (batch_size, 8, 8, 2*output_channels)
        h, [h, c] = self.decoder_lstm_cell(x, [h, c])
        reconstructed = self.decoder_conv(h)
        return reconstructed, [h, c]
    
    def reparameterize(self, z_mean, z_log_var):
        return self.sampling((z_mean, z_log_var))
    
    def call(self, inputs, initial_states=None):
        # inputs shape: (batch_size, 64, 64, 3)
        if initial_states is None:
            encoder_states, decoder_states, object_rep_states = None, None, None
        else:
            encoder_states, decoder_states, object_rep_states = initial_states
        z_mean, z_log_var, encoder_states = self.encode(inputs, encoder_states)
        z = self.reparameterize(z_mean, z_log_var)
        x_logit, decoder_states = self.decode(z, decoder_states)
        object_rep, object_rep_states = self.get_object_representation(z, object_rep_states)

        cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=inputs)
        logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
        logpz = -0.5 * tf.reduce_sum(z**2, axis=1)
        logqz_x = -0.5 * tf.reduce_sum(z_log_var + tf.exp(z_log_var) + (z_mean - z)**2 / tf.exp(z_log_var), axis=1)

        loss = -tf.reduce_mean(logpx_z + logpz - logqz_x)

        return object_rep, loss, [encoder_states, decoder_states, object_rep_states]

    def get_latent_representation(self, inputs, states=None):
        z_mean, z_log_var, states = self.encode(inputs, states)
        z = self.reparameterize(z_mean, z_log_var)
        return z, states

    def get_object_representation(self, z, states=None):
        x = self.object_rep_dense(z)
        x = tf.reshape(x, (-1, 8, 8, self.output_channels))
        if states is None:
            h, c = [tf.zeros_like(x) for _ in range(2)]
        else:
            h, c = states
        x = tf.concat((x, h), axis=-1) # Add a time dimension, (batch_size, 8, 8, 2*output_channels)
        h, [h, c] = self.object_rep_lstm_cell(x, [h, c])
        object_rep = self.object_rep_conv(h)
        return object_rep, [h, c]

    def compute_loss(self, x, encoder_states=None, decoder_states=None):
        z_mean, z_log_var, encoder_states = self.encode(x, encoder_states)
        z = self.reparameterize(z_mean, z_log_var)
        x_logit, decoder_states = self.decode(z, decoder_states)

        cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
        logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
        logpz = -0.5 * tf.reduce_sum(z**2, axis=1)
        logqz_x = -0.5 * tf.reduce_sum(z_log_var + tf.exp(z_log_var) + (z_mean - z)**2 / tf.exp(z_log_var), axis=1)

        return -tf.reduce_mean(logpx_z + logpz - logqz_x)

class ConvVAE(keras.models.Model):
    def __init__(self, latent_dim, image_shape, output_channels=12, *args, **kwargs):
        super(ConvVAE, self).__init__(*args, **kwargs)
        self.latent_dim = latent_dim
        self.image_shape = image_shape
        self.output_channels = output_channels
        
        # Encoder
        self.encoder_conv = tf.keras.Sequential([
            # layers.InputLayer(input_shape=(*image_shape,)), # (batch_size, 64, 64, 3)
            layers.Conv2D(16, (3, 3), activation='relu', strides=(2, 2), padding='same'), # (batch_size, 32, 32, 16)
            layers.Conv2D(32, (3, 3), activation='relu', strides=(2, 2), padding='same') # (batch_size, 16, 16, 32)
        ])
        self.encoder_dense = layers.Dense(latent_dim + latent_dim)  # z_mean and z_log_var
        
        # Decoder
        self.decoder_dense = layers.Dense(8 * 8 * self.output_channels, activation='relu')
        self.decoder_conv = tf.keras.Sequential([
            # layers.InputLayer(input_shape=(8, 8, self.output_channels)),
            layers.Conv2DTranspose(32, (3, 3), activation='relu', strides=(2, 2), padding='same'),
            layers.Conv2DTranspose(16, (3, 3), activation='relu', strides=(2, 2), padding='same'),
            layers.Conv2DTranspose(8, (3, 3), activation='relu', strides=(2, 2), padding='same'),
            layers.Conv2DTranspose(image_shape[-1], (3, 3), activation='sigmoid', padding='same')
        ])

        # Latent Rep to Object Rep
        self.object_rep_dense = layers.Dense(8 * 8 * self.output_channels, activation='relu')
        self.object_rep_conv = tf.keras.Sequential([
            # layers.InputLayer(input_shape=(8, 8, self.output_channels)),
            layers.Conv2DTranspose(32, (3, 3), activation='relu', strides=(2, 2), padding='same'),
            layers.Conv2DTranspose(16, (3, 3), activation='relu', strides=(2, 2), padding='same'),
            layers.Conv2DTranspose(8, (3, 3), activation='relu', strides=(2, 2), padding='same'),
            layers.Conv2DTranspose(self.output_channels, (3, 3), activation='sigmoid', padding='same')
        ])
    
    def encode(self, x):
        # x shape: (batch_size, 64, 64, 3)
        x = self.encoder_conv(x) # (batch_size, 16, 16, 64)
        x_flat = tf.reshape(x, [tf.shape(x)[0], -1])  # Flatten the spatial dimensions for the dense layer
        z_mean, z_log_var = tf.split(self.encoder_dense(x_flat), num_or_size_splits=2, axis=1)
        return z_mean, z_log_var

    def decode(self, z):
        x = self.decoder_dense(z)
        x = tf.reshape(x, (-1, 8, 8, self.output_channels))
        reconstructed = self.decoder_conv(x)
        return reconstructed

    def get_object_representation(self, z):
        x = self.object_rep_dense(z)
        x = tf.reshape(x, (-1, 8, 8, self.output_channels))
        object_rep = self.object_rep_conv(x)
        return object_rep
    
    def reparameterize(self, z_mean, z_log_var):
        return self.sample((z_mean, z_log_var))
        
    def sample(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon
    
    def call(self, inputs):
        # inputs shape: (batch_size, 64, 64, 3)
        z_mean, z_log_var = self.encode(inputs)
        z = self.reparameterize(z_mean, z_log_var)
        x_logit = self.decode(z)
        object_rep = self.get_object_representation(z)

        # Calculate VAE loss as the sum of reconstruction loss and KL divergence
        cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=inputs)
        logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
        logpz = -0.5 * tf.reduce_sum(z**2, axis=1)
        logqz_x = -0.5 * tf.reduce_sum(z_log_var + tf.exp(z_log_var) + (z_mean - z)**2 / tf.exp(z_log_var), axis=1)

        vae_loss = -tf.reduce_mean(logpx_z + logpz - logqz_x)

        return object_rep, vae_loss

class KerasSampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # self.seed_generator = keras.random.SeedGenerator(1337)

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean + tf.math.exp(0.5 * z_log_var) * epsilon

class KerasVAE(keras.Model):
    def __init__(self, latent_dim, output_channels, num_classes, **kwargs):
        super().__init__(**kwargs)

        ###### Start ConvVAE ######
        latent_dim = latent_dim
        input_shape = (64, 64, output_channels)

        encoder_inputs1 = keras.Input(shape=input_shape)
        label_inputs1 = keras.Input(shape=(num_classes,))
        x = layers.Conv2D(32, 7, activation="relu", strides=2, padding="same")(encoder_inputs1)
        x = layers.Conv2D(32, 5, activation="relu", strides=2, padding="same")(x)
        x = layers.Conv2D(64, 7, activation="relu", strides=2, padding="same")(x)
        x = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)
        x = layers.Flatten()(x)
        x = layers.Concatenate()([x, label_inputs1])
        x = layers.Dense(128, activation="relu")(x)
        x = layers.Dense(latent_dim, activation="relu")(x)
        z_mean = layers.Dense(latent_dim, name="z_mean")(x)
        z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
        z = KerasSampling()([z_mean, z_log_var])
        
        encoder = keras.Model([encoder_inputs1, label_inputs1], [z_mean, z_log_var, z], name="encoder")

        latent_inputs2 = keras.Input(shape=(latent_dim,))
        label_inputs2 = keras.Input(shape=(num_classes,))
        x = layers.Concatenate()([latent_inputs2, label_inputs2])
        x = layers.Dense(8 * 8 * 128, activation="relu")(x)
        x = layers.Reshape((8, 8, 128))(x)
        x = layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
        x = layers.Conv2DTranspose(48, 3, activation="relu", strides=2, padding="same")(x)
        x = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)
        decoder_outputs = layers.Conv2DTranspose(input_shape[-1], 3, activation="sigmoid", padding="same")(x)
        decoder = keras.Model([latent_inputs2, label_inputs2], decoder_outputs, name="decoder")
        
        
        latent_inputs3 = keras.Input(shape=(latent_dim,))
        label_inputs3 = keras.Input(shape=(num_classes,))
        x = layers.Concatenate()([latent_inputs3, label_inputs3])
        x = layers.Dense(8 * 8 * 128, activation="relu")(x)
        x = layers.Reshape((8, 8, 128))(x)
        x = layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
        x = layers.Conv2DTranspose(48, 3, activation="relu", strides=2, padding="same")(x)
        x = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)
        object_rep_decoder_outputs = layers.Conv2DTranspose(input_shape[-1], 3, activation="sigmoid", padding="same")(x)
        object_rep_decoder = keras.Model([latent_inputs3, label_inputs3], object_rep_decoder_outputs, name="object_rep_decoder")

        self.encoder = encoder
        self.decoder = decoder
        self.object_rep_decoder = object_rep_decoder

    def call(self, data, training=True):
        images, labels = data
        binary_masks = tf.expand_dims(tf.cast(tf.reduce_any(tf.not_equal(images, 0), axis=-1), tf.float32), axis=-1)
        z_mean, z_log_var, z = self.encoder([binary_masks, labels], training=training)
        recon_masks = self.decoder([z, labels], training=training)
        
        return binary_masks, recon_masks, z_mean, z_log_var

    def encode(self, data, training=True):
        images, labels = data
        binary_masks = tf.expand_dims(tf.cast(tf.reduce_any(tf.not_equal(images, 0), axis=-1), tf.float32), axis=-1)
        z_mean, z_log_var, z = self.encoder([binary_masks, labels], training=training)

        return z

    def decode(self, data, training=True):
        z, labels = data
        recon_masks = self.decoder([z, labels], training=training)

        return recon_masks

    def decode_object_rep(self, data, training=True):
        z, labels = data
        object_rep = self.object_rep_decoder([z, labels], training=training)

        return object_rep

    def compute_loss(self, masks, recon_masks, z_mean, z_log_var):
        # tf.debugging.check_numerics(masks, "NaN or Inf found in masks")
        # tf.debugging.check_numerics(recon_masks, "NaN or Inf found in recon_masks")
        # tf.debugging.check_numerics(z_mean, "NaN or Inf found in z_mean")
        # tf.debugging.check_numerics(z_log_var, "NaN or Inf found in z_log_var")

        mask_reconstruction_loss = tf.reduce_mean(
            tf.reduce_sum(
                keras.losses.binary_crossentropy(masks, recon_masks),
                axis=(1, 2),
            )
        ) * 64 * 64 * 1
        
        kl_loss = -0.5 * (1 + z_log_var - tf.math.square(z_mean) - tf.math.exp(z_log_var))
        kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
        
        total_loss = mask_reconstruction_loss + 0.25*kl_loss
        
        return total_loss, mask_reconstruction_loss, kl_loss

class BroadcastDecoderNet(layers.Layer):
    def __init__(self, w_broadcast, h_broadcast, net, **kwargs):
        super(BroadcastDecoderNet, self).__init__(**kwargs)
        self.w_broadcast = w_broadcast
        self.h_broadcast = h_broadcast
        self.net = net

        # Create the constant coordinate map
        ys = tf.linspace(-1.0, 1.0, h_broadcast)
        xs = tf.linspace(-1.0, 1.0, w_broadcast)
        xs, ys = tf.meshgrid(xs, ys)
        self.coord_map = tf.stack([ys, xs], axis=-1)
        self.coord_map = tf.expand_dims(self.coord_map, axis=0)  # Shape: (1, h_broadcast, w_broadcast, 2)

    def call(self, data, training=False):
        inputs, pseudo_labels = data
        inputs = tf.concat([inputs, pseudo_labels], axis=-1)
        # Broadcast the latent vector. For inputs shaped (batch_size, latent_size), the output will be shaped (batch_size, h_broadcast, w_broadcast, latent_size)
        z_tiled = tf.tile(tf.expand_dims(tf.expand_dims(inputs, 1), 1), [1, self.h_broadcast, self.w_broadcast, 1])
        # Concatenate the coordinate map
        combined = tf.concat([z_tiled, self.coord_map * tf.ones_like(z_tiled[:, :, :, :1])], axis=-1)
        result = self.net(combined)
        return result

def make_sequential_cnn(input_channels, channels, kernels, paddings, activations, batchnorms):
    model = keras.Sequential()
    for i, (ch, kernel, padding, activation, bn) in enumerate(zip(channels, kernels, paddings, activations, batchnorms)):
        if i == 0:  # Only the first layer receives the input_shape parameter
            layer = layers.Conv2D(ch, kernel, padding='same',
                                  activation=activation, input_shape=(None, None, input_channels))
        else:
            layer = layers.Conv2D(ch, kernel, padding='same',
                                  activation=activation)

        model.add(layer)
        if i != len(channels) - 1:  # Do not add dropout after the last layer
            model.add(layers.Dropout(0.2)) 

        if bn:
            model.add(layers.BatchNormalization())

    return model

class ObjectRepresentation(layers.Layer):
    '''
    Convert images of object masks to class IDs, then update and extract the corresponding object representations
    '''
    def __init__(self, training_args, num_classes, batch_size, im_height, im_width, output_channels, **kwargs):
        super(ObjectRepresentation, self).__init__(**kwargs)
        self.training_args = training_args
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.im_height = im_height
        self.im_width = im_width
        self.output_channels = output_channels
        # self.conv_lstm_general = ConvLSTM2D(filters=output_channels, kernel_size=(3, 3), padding='same', return_sequences=False, return_state=True, stateful=True, name='conv_lstm_general')
        # self.conv_lstm_class = ConvLSTM2D(filters=output_channels, kernel_size=(3, 3), padding='same', return_sequences=False, return_state=True, stateful=True, name='conv_lstm_class')
        self.conv_lstm_general = CustomConvLSTM(output_channels=output_channels, input_channels=3, name='conv_lstm_general')
        self.conv_lstm_class = CustomConvLSTM(output_channels=output_channels, input_channels=3, name='conv_lstm_class')
        # self.classifier = CustomMobileNetV2(num_classes=4, input_shape=(self.im_height, self.im_width, 3), name='classifier')
        self.classifier = CustomCNN(num_classes=4, num_conv_layers=3, trainable=False, name='classifier')

        # Initialize states
        self.general_states_h = self.add_weight(shape=(1, im_height, im_width, output_channels), initializer='zeros', trainable=False, name='general_state_h')
        self.general_states_c = self.add_weight(shape=(1, im_height, im_width, output_channels), initializer='zeros', trainable=False, name='general_state_c')
        self.class_states_h = self.add_weight(shape=(num_classes, im_height, im_width, output_channels), initializer='zeros', trainable=False, name='class_state_h')
        self.class_states_c = self.add_weight(shape=(num_classes, im_height, im_width, output_channels), initializer='zeros', trainable=False, name='class_state_c')

        self.predicted_class_IDs = []
        self.plot_num = 0


    def diff_gather(self, params, logits, beta=3):
        '''
        Differentiable gather operation.
        Params shape: (num_classes, h, w, c)
        Logits shape: (batch_size, num_classes)
        '''
        # expand params to include batch_dim
        expanded_params = tf.expand_dims(params, axis=1) # (num_classes, 1, h, w, c)
        weights = tf.transpose(tf.nn.softmax(logits * beta), [1, 0]) # (num_classes, batch_size)
        current_weights_shape = weights.shape
        reshaped_weights = weights
        for _ in range(len(expanded_params.shape) - len(current_weights_shape)):
            reshaped_weights = tf.expand_dims(reshaped_weights, axis=-1) # (num_classes, batch_size, 1, 1, 1)

        weighted_params = reshaped_weights * expanded_params # broadcasting to shape (num_classes, batch_size, ...)
        weighted_sum = tf.reduce_sum(weighted_params, axis=0) # (batch_size, h, w, c)
        return weighted_sum


    def diff_scatter_nd_update(self, A, B, logits, beta=1e10):
        """
        Update tensor A with values from tensor B based on highest indices indicated by a logits matrix.
        Like tf.tensor_scatter_nd_update, but differentiable, in the sense that integer class indices are not required.

        Args:
        A (tf.Tensor): A tensor of shape (nc, h, w, oc).
        B (tf.Tensor): A tensor of shape (bs, h, w, oc).
        logits (tf.Tensor): A logits matrix of shape (bs, nc).

        Returns:
        tf.Tensor: Updated tensor A.
        """
        # Convert logits to one-hot
        one_hot = tf.nn.softmax(logits * beta) # (bs, nc)

        # Check dimensions
        if len(A.shape) != 4 or len(B.shape) != 4 or len(one_hot.shape) != 2:
            raise ValueError("Input tensors must be of the shape (nc, h, w, oc), (bs, h, w, oc), and (bs, nc) respectively.")
        
        # Check dimension matching
        nc, h, w, oc = A.shape
        if (B.shape[1:] != (h, w, oc)) or (one_hot.shape[1:] != (nc)):
            raise ValueError("Dimension mismatch among inputs.")

        # Expand A to match B's batch dimension
        A_expanded = tf.expand_dims(A, 1) # (nc, 1, h, w, oc)

        # Expand B to broadcast over the nc dimension
        B_expanded = tf.expand_dims(B, 0) # (1, bs, h, w, oc)

        # Expand the one-hot matrix to match A's dimensions
        mask = tf.expand_dims(tf.expand_dims(tf.expand_dims(one_hot, -1), -1), -1) # (bs, nc, 1, 1, 1)
        mask = tf.transpose(mask, [1, 0, 2, 3, 4])  # Reshape to (nc, bs, 1, 1, 1)

        # Multiply A by (1 - mask) to zero out the update positions
        A_masked = A_expanded * (1 - mask)

        # Multiply B by mask to align updates
        B_masked = B_expanded * mask

        # Combine the two components
        A_updated = A_masked + B_masked

        # # Sum across axis 1 (batch size dimension)
        # summed_reps = tf.reduce_sum(A_updated, axis=1)  # Resulting shape: (nc, h, w, c)

        # # Normalize each (h, w, c) representation independently
        # # Find the min and max values for each class's representation
        # min_vals = tf.reduce_min(summed_reps, axis=(1, 2, 3), keepdims=True)  # Shape: (nc, 1, 1, 1)
        # max_vals = tf.reduce_max(summed_reps, axis=(1, 2, 3), keepdims=True)  # Shape: (nc, 1, 1, 1)

        # # Normalize to range [0, 1]
        # normalized_reps = (summed_reps - min_vals) / (max_vals - min_vals + 1e-8)  # Shape: (nc, h, w, c)

        # A_updated = normalized_reps

        # Reduce_max over the batch dimension to create single updated class state
        A_updated = tf.reduce_max(A_updated, axis=1, keepdims=False) # (nc, h, w, oc)

        return A_updated


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
            preds = tf.nn.softmax(logits*1e10, axis=-1) # (batch_size, num_predictions, num_classes)
            predicted_class_counts = tf.reduce_sum(preds, axis=-2) # (batch_size, num_classes)
            ideal_class_counts = tf.ones_like(predicted_class_counts) # (batch_size, num_classes)
            penalty = tf.reduce_mean(tf.square(ideal_class_counts - predicted_class_counts), axis=-1)
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
        total_loss = entropy_loss + distinct_class_penalty_loss

        return total_loss


    def plot_images_with_labels(self, images, labels, save_path='images_with_labels'):
        """images shaped: (nc, h, w, 3) and labels shaped (nc, )"""
        # Convert tensors to numpy arrays
        images_np = images.numpy()
        labels_np = labels.numpy()
        
        nc = images_np.shape[0]
        
        # Create a plot
        plt.figure(figsize=(15, 15))
        
        for i in range(nc):
            plt.subplot(1, nc, i + 1)
            plt.imshow(images_np[i])
            plt.title(labels_np[i])
            plt.axis('off')
        
        # Save the plot to disk
        plt.savefig(save_path+f'/{self.plot_num}.png')
        self.plot_num += 1
        plt.close()

    
    def process_valid_frame(self, frame, general_states_h):
        # Only processing the frame if it is not empty (valid)
        
        # classify the input frame to get the class logits predictions
        class_logits = self.classifier(tf.squeeze(frame, axis=1)) # (bs, nc)
        assert (class_logits.shape == (self.batch_size, self.num_classes) or class_logits.shape == (None, self.num_classes))

        # get current class states
        current_class_states = [self.diff_gather(self.class_states_h, class_logits), self.diff_gather(self.class_states_c, class_logits)] # (bs, h, w, oc) via broadcasting
        
        # get updated general object representation (hidden general object state)
        new_general_object_representation = general_states_h # (1, h, w, oc)

        # tile the general object representation to match the batch size
        new_general_object_representation = tf.tile(tf.expand_dims(new_general_object_representation, axis=0), [self.batch_size, 1, 1, 1, 1]) # (bs, 1, h, w, oc)

        # concatenate the general object representation with the input and previous class hidden states (expanded for time dimension)
        augmented_inputs = tf.concat([frame, new_general_object_representation, tf.expand_dims(current_class_states[0], axis=1)], axis=-1) # (bs, 1, h, w, 3+oc+oc)

        # apply the shared class ConvLSTM layer to get the updated class states
        class_output, new_class_state_h, new_class_state_c = self.conv_lstm_class(augmented_inputs, initial_state=current_class_states) # (bs, h, w, oc) x 3

        # update the class states
        class_update_h = self.diff_scatter_nd_update(self.class_states_h, new_class_state_h, class_logits) # (nc, h, w, oc)
        class_update_c = self.diff_scatter_nd_update(self.class_states_c, new_class_state_c, class_logits) # (nc, h, w, oc)
        # self.LSTM_states_class = [class_update_h, class_update_c]
        
        # Also update the permanent copies.
        self.class_states_h.assign(class_update_h)
        self.class_states_c.assign(class_update_c)

        # print("Class output shape:", class_output.shape)
        assert (class_output.shape == (self.batch_size, self.im_height, self.im_width, self.output_channels) or class_output.shape == (None, self.im_height, self.im_width, self.output_channels))

        return class_output, class_logits


    def process_null_frame(self,):
        # If the input frame is empty, set the class output to zero
        class_output = tf.zeros((self.batch_size, self.im_height, self.im_width, self.output_channels), dtype=tf.float32)
        class_logits = tf.zeros((self.batch_size, self.num_classes), dtype=tf.float32)
        return class_output, class_logits
    

    
    def call(self, inputs):
        # Inputs shape: (bs, 1, h, w, oc)

        '''Update General Object States'''
        # get all current class object representations (class object hidden states) as input
        # reshaped to (nc, h, w, oc) -> (1, h, w, nc*oc) to include batch_dim and time_dim (for processing in ConvLSTM)
        all_class_object_representations = self.class_states_h # (nc, h, w, oc)
        all_class_object_representations = tf.expand_dims(tf.expand_dims(tf.concat(tf.unstack(all_class_object_representations, axis=0), axis=-1), axis=0), axis=0) # (1, 1, h, w, nc*oc)

        # tile the class object representations to match the batch size
        all_class_object_representations = tf.tile(all_class_object_representations, [self.batch_size, 1, 1, 1, 1]) # (bs, 1, h, w, nc*oc)

        # get current general object states
        current_general_states = [self.general_states_h, self.general_states_c] # (1, h, w, oc) x 2

        # tile states to match the batch size
        current_general_states = [tf.tile(current_general_states[0], [self.batch_size, 1, 1, 1]), tf.tile(current_general_states[1], [self.batch_size, 1, 1, 1])] # (bs, h, w, oc) x 2

        # concantenate the class object representations with the current general object hidden states (expanded for time dimension)
        augmented_inputs = tf.concat([all_class_object_representations, tf.expand_dims(current_general_states[0], axis=1)], axis=-1) # (bs, 1, h, w, nc*oc+oc)

        # apply the general ConvLSTM layer to get the updated general states
        _, new_general_state_h, new_general_state_c = self.conv_lstm_general(augmented_inputs, initial_state=current_general_states) # (bs, h, w, oc) x 3

        # reduce_mean new general states across the batch dimension to create single updated general state
        new_general_state_h = tf.reduce_mean(new_general_state_h, axis=0, keepdims=False) # (h, w, oc)
        new_general_state_c = tf.reduce_mean(new_general_state_c, axis=0, keepdims=False) # (h, w, oc)

        # update the general states
        general_update_h = tf.tensor_scatter_nd_update(self.general_states_h, [[0]], [new_general_state_h]) # (1, h, w, oc)
        general_update_c = tf.tensor_scatter_nd_update(self.general_states_c, [[0]], [new_general_state_c]) # (1, h, w, oc)
        # self.LSTM_states_general = [general_update_h, general_update_c]

        # Also update the permanent copies.
        self.general_states_h.assign(general_update_h)
        self.general_states_c.assign(general_update_c)
        
        '''Update Class States and Obtain Outputs'''
        # Note that inputs are nc*3 channels, where nc is the number of classes and thus we process each 3-channel block separately
        
        output_class_tensors = []
        all_class_logits = []
        self.predicted_class_IDs = []
        self.frames = tf.stack([inputs[0, 0, ..., i*3:(i+1)*3] for i in range(self.num_classes)]) # (nc, h, w, 3)
        for i in range(self.num_classes):

            frame = inputs[..., i*3:(i+1)*3] # (bs, 1, h, w, 3)
            class_output, class_logits = None, None

            # Process the frame
            class_output, class_logits = tf.cond(tf.reduce_sum(frame) > 0, lambda: self.process_valid_frame(frame, general_update_h), lambda: self.process_null_frame())
            output_class_tensors.append(class_output)
            all_class_logits.append(class_logits)
            
            """Debugging 1/2 - check if classifier is working"""
            if self.training_args["debug_model"]:
                try:
                    val = tf.argmax(tf.nn.softmax(class_logits*1e10), axis=-1) if class_logits is not None else tf.constant(-1, tf.int32)
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
        classification_diversity_loss = self.calculate_classification_diversity_loss(all_class_logits) if False else tf.constant(0.0, shape=(self.batch_size, 1))
        
        """Debugging 2/2 - check if classifier is working"""
        if self.training_args["debug_model"]:
            try:
                vals = tf.stack(self.predicted_class_IDs, axis=-1) # (bs, nc)
                vals = tf.squeeze(vals) # (nc,)
                # self.plot_images_with_labels(self.frames, vals)
                tf.print("vals:", vals, output_stream=sys.stdout)
            except Exception:
                # Skip symbolic tensors
                pass

        # return the class-specific object representations (hidden class states)
        return output, classification_diversity_loss

class ObjectRepresentation_ConvLSTMVAE(layers.Layer):
    '''
    Convert images of object masks to class IDs, then update and extract the corresponding object representations
    '''
    def __init__(self, training_args, num_classes, batch_size, im_height, im_width, output_channels, **kwargs):
        super(ObjectRepresentation_ConvLSTMVAE, self).__init__(**kwargs)
        self.training_args = training_args
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.im_height = im_height
        self.im_width = im_width
        self.output_channels = output_channels
        # self.conv_lstm_vae_general = ConvLSTMVAE(latent_dim=128, image_shape=(im_height, im_width, 3), output_channels=output_channels, name='conv_lstm_vae_general')
        self.conv_lstm_vae_class = ConvLSTMVAE(latent_dim=64, image_shape=(im_height, im_width, 3), output_channels=output_channels, name='conv_lstm_vae_class') # call returns: object_rep, loss, [encoder_states, decoder_states, object_rep_states]
        self.classifier = CustomCNN(num_classes=4, num_conv_layers=3, trainable=False, name='classifier')

        # Initialize states
        # self.general_encoder_states = [self.add_weight(shape=(num_classes, im_height, im_width, output_channels), initializer='zeros', trainable=False, name='general_encoder_state_h'), self.add_weight(shape=(num_classes, im_height, im_width, output_channels), initializer='zeros', trainable=False, name='general_encoder_state_c')]
        # self.general_decoder_states = [self.add_weight(shape=(num_classes, im_height, im_width, output_channels), initializer='zeros', trainable=False, name='general_decoder_state_h'), self.add_weight(shape=(num_classes, im_height, im_width, output_channels), initializer='zeros', trainable=False, name='general_decoder_state_c')]
        # self.general_object_rep_states = [self.add_weight(shape=(num_classes, im_height, im_width, output_channels), initializer='zeros', trainable=False, name='general_object_rep_state_h'), self.add_weight(shape=(num_classes, im_height, im_width, output_channels), initializer='zeros', trainable=False, name='general_eobject_rep_state_c')]
        self.class_encoder_states = [self.add_weight(shape=(num_classes, 16, 16, 64), initializer='zeros', trainable=False, name='class_encoder_state_h'), self.add_weight(shape=(num_classes, 16, 16, 64), initializer='zeros', trainable=False, name='class_encoder_state_c')]
        self.class_decoder_states = [self.add_weight(shape=(num_classes, 8, 8, output_channels), initializer='zeros', trainable=False, name='class_decoder_state_h'), self.add_weight(shape=(num_classes, 8, 8, output_channels), initializer='zeros', trainable=False, name='class_decoder_state_c')]
        self.class_object_rep_states = [self.add_weight(shape=(num_classes, 8, 8, output_channels), initializer='zeros', trainable=False, name='class_object_rep_state_h'), self.add_weight(shape=(num_classes, 8, 8, output_channels), initializer='zeros', trainable=False, name='class_eobject_rep_state_c')]
        # self.general_states = [self.general_encoder_states, self.general_decoder_states, self.general_object_rep_states]
        self.class_states = [self.class_encoder_states, self.class_decoder_states, self.class_object_rep_states]


        self.predicted_class_IDs = []
        self.plot_num = 0


    def diff_gather(self, params, logits, beta=3):
        '''
        Differentiable gather operation.
        Params shape: (num_classes, h, w, c)
        Logits shape: (batch_size, num_classes)
        '''
        # expand params to include batch_dim
        expanded_params = tf.expand_dims(params, axis=1) # (num_classes, 1, h, w, c)
        weights = tf.transpose(tf.nn.softmax(logits * beta), [1, 0]) # (num_classes, batch_size)
        current_weights_shape = weights.shape
        reshaped_weights = weights
        for _ in range(len(expanded_params.shape) - len(current_weights_shape)):
            reshaped_weights = tf.expand_dims(reshaped_weights, axis=-1) # (num_classes, batch_size, 1, 1, 1)

        weighted_params = reshaped_weights * expanded_params # broadcasting to shape (num_classes, batch_size, ...)
        weighted_sum = tf.reduce_sum(weighted_params, axis=0) # (batch_size, h, w, c)
        return weighted_sum


    def diff_scatter_nd_update(self, A, B, logits, beta=1e10):
        """
        Update tensor A with values from tensor B based on highest indices indicated by a logits matrix.
        Like tf.tensor_scatter_nd_update, but differentiable, in the sense that integer class indices are not required.

        Args:
        A (tf.Tensor): A tensor of shape (nc, h, w, oc).
        B (tf.Tensor): A tensor of shape (bs, h, w, oc).
        logits (tf.Tensor): A logits matrix of shape (bs, nc).

        Returns:
        tf.Tensor: Updated tensor A.
        """
        # Convert logits to one-hot
        one_hot = tf.nn.softmax(logits * beta) # (bs, nc)

        # Check dimensions
        if len(A.shape) != 4 or len(B.shape) != 4 or len(one_hot.shape) != 2:
            raise ValueError("Input tensors must be of the shape (nc, h, w, oc), (bs, h, w, oc), and (bs, nc) respectively.")
        
        # Check dimension matching
        nc, h, w, oc = A.shape
        if (B.shape[1:] != (h, w, oc)) or (one_hot.shape[1:] != (nc)):
            raise ValueError("Dimension mismatch among inputs.")

        # Expand A to match B's batch dimension
        A_expanded = tf.expand_dims(A, 1) # (nc, 1, h, w, oc)

        # Expand B to broadcast over the nc dimension
        B_expanded = tf.expand_dims(B, 0) # (1, bs, h, w, oc)

        # Expand the one-hot matrix to match A's dimensions
        mask = tf.expand_dims(tf.expand_dims(tf.expand_dims(one_hot, -1), -1), -1) # (bs, nc, 1, 1, 1)
        mask = tf.transpose(mask, [1, 0, 2, 3, 4])  # Reshape to (nc, bs, 1, 1, 1)

        # Multiply A by (1 - mask) to zero out the update positions
        A_masked = A_expanded * (1 - mask)

        # Multiply B by mask to align updates
        B_masked = B_expanded * mask

        # Combine the two components
        A_updated = A_masked + B_masked

        # # Sum across axis 1 (batch size dimension)
        # summed_reps = tf.reduce_sum(A_updated, axis=1)  # Resulting shape: (nc, h, w, c)

        # # Normalize each (h, w, c) representation independently
        # # Find the min and max values for each class's representation
        # min_vals = tf.reduce_min(summed_reps, axis=(1, 2, 3), keepdims=True)  # Shape: (nc, 1, 1, 1)
        # max_vals = tf.reduce_max(summed_reps, axis=(1, 2, 3), keepdims=True)  # Shape: (nc, 1, 1, 1)

        # # Normalize to range [0, 1]
        # normalized_reps = (summed_reps - min_vals) / (max_vals - min_vals + 1e-8)  # Shape: (nc, h, w, c)

        # A_updated = normalized_reps

        # Reduce_max over the batch dimension to create single updated class state
        A_updated = tf.reduce_max(A_updated, axis=1, keepdims=False) # (nc, h, w, oc)

        return A_updated


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
            preds = tf.nn.softmax(logits*1e10, axis=-1) # (batch_size, num_predictions, num_classes)
            predicted_class_counts = tf.reduce_sum(preds, axis=-2) # (batch_size, num_classes)
            ideal_class_counts = tf.ones_like(predicted_class_counts) # (batch_size, num_classes)
            penalty = tf.reduce_mean(tf.square(ideal_class_counts - predicted_class_counts), axis=-1)
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
        total_loss = entropy_loss + distinct_class_penalty_loss

        return total_loss


    def plot_images_with_labels(self, images, labels, save_path='images_with_labels'):
        """images shaped: (nc, h, w, 3) and labels shaped (nc, )"""
        # Convert tensors to numpy arrays
        images_np = images.numpy()
        labels_np = labels.numpy()
        
        nc = images_np.shape[0]
        
        # Create a plot
        plt.figure(figsize=(15, 15))
        
        for i in range(nc):
            plt.subplot(1, nc, i + 1)
            plt.imshow(images_np[i])
            plt.title(labels_np[i])
            plt.axis('off')
        
        # Save the plot to disk
        plt.savefig(save_path+f'/{self.plot_num}.png')
        self.plot_num += 1
        plt.close()


    def process_valid_frame(self, frame, skip):
        # Only processing the frame if it is not empty (valid)
        
        # classify the input frame to get the class logits predictions
        class_logits = self.classifier(frame) # (bs, nc)
        assert (class_logits.shape == (self.batch_size, self.num_classes) or class_logits.shape == (None, self.num_classes))

        # get current class states
        current_class_states = [[self.diff_gather(self.class_states[i][0], class_logits), self.diff_gather(self.class_states[i][1], class_logits)] for i in range(3)] # (bs, h, w, oc) via broadcasting

        # # tile the general object representation to match the batch size
        # new_general_object_representation = tf.tile(tf.expand_dims(new_general_object_representation, axis=0), [self.batch_size, 1, 1, 1, 1]) # (bs, 1, h, w, oc)

        # concatenate the general object representation with the input and previous class hidden states (expanded for time dimension)
        # augmented_inputs = tf.concat([frame, tf.expand_dims(tf.concat([current_class_states[i][0] for i in range(3)], axis=-1), axis=1)], axis=-1) # (bs, 1, h, w, 3+oc+oc)

        # apply the shared class ConvLSTM layer to get the updated class states
        class_object_rep, loss, new_class_states = self.conv_lstm_vae_class(frame, initial_states=current_class_states)

        # update the class states
        class_state_updates = [[self.diff_scatter_nd_update(self.class_states[i][0], new_class_states[i][0], class_logits), self.diff_scatter_nd_update(self.class_states[i][1], new_class_states[i][1], class_logits)] for i in range(3)] # (bs, h, w, oc) via broadcasting
        
        # Also update the permanent copies.
        [[self.class_states[i][0].assign(class_state_updates[i][0]), self.class_states[i][1].assign(class_state_updates[i][1])] for i in range(3)]

        # print("Class output shape:", class_output.shape)
        # assert (class_output.shape == (self.batch_size, self.im_height, self.im_width, self.output_channels) or class_output.shape == (None, self.im_height, self.im_width, self.output_channels))

        return class_object_rep, loss, class_logits


    def process_null_frame(self,):
        # If the input frame is empty, set the class output to zero
        class_object_rep = tf.zeros((self.batch_size, self.im_height, self.im_width, self.output_channels), dtype=tf.float32)
        loss = tf.constant(0.0, shape=())
        class_logits = tf.zeros((self.batch_size, self.num_classes), dtype=tf.float32)

        return class_object_rep, loss, class_logits
    

    
    def call(self, inputs):
        # Inputs shape: (bs, 1, h, w, oc)

        # '''Update General Object States'''
        # # get all current class object representations (class object hidden states) as input
        # # reshaped to (nc, h, w, oc) -> (1, h, w, nc*oc) to include batch_dim and time_dim (for processing in ConvLSTM)
        # all_class_object_representations = self.class_states_h # (nc, h, w, oc)
        # all_class_object_representations = tf.expand_dims(tf.expand_dims(tf.concat(tf.unstack(all_class_object_representations, axis=0), axis=-1), axis=0), axis=0) # (1, 1, h, w, nc*oc)

        # # tile the class object representations to match the batch size
        # all_class_object_representations = tf.tile(all_class_object_representations, [self.batch_size, 1, 1, 1, 1]) # (bs, 1, h, w, nc*oc)

        # # get current general object states
        # current_general_states = [self.general_states_h, self.general_states_c] # (1, h, w, oc) x 2

        # # tile states to match the batch size
        # current_general_states = [tf.tile(current_general_states[0], [self.batch_size, 1, 1, 1]), tf.tile(current_general_states[1], [self.batch_size, 1, 1, 1])] # (bs, h, w, oc) x 2

        # # concantenate the class object representations with the current general object hidden states (expanded for time dimension)
        # augmented_inputs = tf.concat([all_class_object_representations, tf.expand_dims(current_general_states[0], axis=1)], axis=-1) # (bs, 1, h, w, nc*oc+oc)

        # # apply the general ConvLSTM layer to get the updated general states
        # _, new_general_state_h, new_general_state_c = self.conv_lstm_general(augmented_inputs, initial_state=current_general_states) # (bs, h, w, oc) x 3

        # # reduce_mean new general states across the batch dimension to create single updated general state
        # new_general_state_h = tf.reduce_mean(new_general_state_h, axis=0, keepdims=False) # (h, w, oc)
        # new_general_state_c = tf.reduce_mean(new_general_state_c, axis=0, keepdims=False) # (h, w, oc)

        # # update the general states
        # general_update_h = tf.tensor_scatter_nd_update(self.general_states_h, [[0]], [new_general_state_h]) # (1, h, w, oc)
        # general_update_c = tf.tensor_scatter_nd_update(self.general_states_c, [[0]], [new_general_state_c]) # (1, h, w, oc)
        # # self.LSTM_states_general = [general_update_h, general_update_c]

        # # Also update the permanent copies.
        # self.general_states_h.assign(general_update_h)
        # self.general_states_c.assign(general_update_c)
        
        '''Update Class States and Obtain Outputs'''
        # Note that inputs are nc*3 channels, where nc is the number of classes and thus we process each 3-channel block separately
        
        output_class_tensors = []
        all_losses = []
        all_class_logits = []
        self.predicted_class_IDs = []
        self.frames = tf.stack([inputs[0, 0, ..., i*3:(i+1)*3] for i in range(self.num_classes)]) # (nc, h, w, 3)
        for i in range(self.num_classes):

            frame = inputs[..., i*3:(i+1)*3] # (bs, 1, h, w, 3)
            # class_object_rep, class_logits = None, None

            # Process the frame
            class_object_rep, loss, class_logits = tf.cond(tf.reduce_sum(frame) > 0, lambda: self.process_valid_frame(frame, 0), lambda: self.process_null_frame())
            output_class_tensors.append(class_object_rep)
            all_losses.append(loss)
            all_class_logits.append(class_logits)
            
            """Debugging 1/2 - check if classifier is working"""
            if self.training_args["debug_model"]:
                try:
                    val = tf.argmax(tf.nn.softmax(class_logits*1e10), axis=-1) if class_logits is not None else tf.constant(-1, tf.int32)
                    self.predicted_class_IDs.append(val)
                except Exception:
                    # Skip symbolic tensors
                    pass

        # stack the class outputs to get the final output
        output = tf.concat(output_class_tensors, axis=-1)
        # print("Output shape:", output.shape)
        assert (output.shape == (self.batch_size, self.im_height, self.im_width, self.num_classes*self.output_channels) or output.shape == (None, self.im_height, self.im_width, self.num_classes*self.output_channels))

        # # calculate the classification diversity loss
        # all_class_logits = tf.stack(all_class_logits) # (np, bs, nc)
        # assert (all_class_logits.shape == (self.num_classes, self.batch_size, self.num_classes) or all_class_logits.shape == (self.num_classes, None, self.num_classes))
        # classification_diversity_loss = self.calculate_classification_diversity_loss(all_class_logits) if False else tf.constant(0.0, shape=(self.batch_size, 1))
        
        """Debugging 2/2 - check if classifier is working"""
        if self.training_args["debug_model"]:
            try:
                vals = tf.stack(self.predicted_class_IDs, axis=-1) # (bs, nc)
                vals = tf.squeeze(vals) # (nc,)
                # self.plot_images_with_labels(self.frames, vals)
                tf.print("vals:", vals, output_stream=sys.stdout)
            except Exception:
                # Skip symbolic tensors
                pass

        # return the class-specific object representations (hidden class states)
        return output, tf.reduce_sum(all_losses)

class ObjectRepresentation_ConvVAE(layers.Layer):
    '''
    Convert images of object masks to class IDs, then update and extract the corresponding object representations
    '''
    def __init__(self, training_args, num_classes, batch_size, im_height, im_width, output_channels, **kwargs):
        super(ObjectRepresentation_ConvVAE, self).__init__(**kwargs)
        self.training_args = training_args
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.im_height = im_height
        self.im_width = im_width
        self.output_channels = output_channels
        # self.conv_vae_class = ConvVAE(latent_dim=64, image_shape=(im_height, im_width, 3), output_channels=output_channels, name='conv_lstm_vae_class') # call returns: object_rep, loss, [encoder_states, decoder_states, object_rep_states]
        self.conv_vae_class = KerasVAE(output_channels=output_channels, num_classes=num_classes, name='keras_conv_vae_class') # call returns: object_rep, loss, [encoder_states, decoder_states, object_rep_states]
        self.classifier = CustomCNN(num_classes=4, num_conv_layers=3, trainable=False, name='classifier')

        self.predicted_class_IDs = []
        self.plot_num = 0

        self.temperature = self.add_weight(
            name='temperature',
            shape=(),
            initializer=keras.initializers.Constant(1),
            trainable=True,
            constraint=keras.constraints.NonNeg()  # Ensures temperature is non-negative
        )

    def process_valid_frame(self, frame, pseudo_labels):
        class_object_rep, vae_loss = self.conv_vae_class([frame, pseudo_labels], training=False)

        return class_object_rep, vae_loss


    def process_null_frame(self,):
        # If the input frame is empty, set the class output to zero
        class_object_rep = tf.zeros((self.batch_size, self.im_height, self.im_width, self.output_channels), dtype=tf.float32)
        loss = tf.constant(0.0, shape=())

        return class_object_rep, loss
    
    
    def call(self, inputs):
        # Inputs shape: (bs, 1, h, w, nc*3)
        # Note that inputs are nc*3 channels, where nc is the number of classes and thus we process each 3-channel block separately
        
        output_class_tensors = []
        all_vae_losses = []
        self.predicted_class_IDs = []
        self.frames = tf.stack([inputs[0, 0, ..., i*3:(i+1)*3] for i in range(self.num_classes)]) # (nc, h, w, 3)
        for i in range(self.num_classes):
            frame = inputs[..., i*3:(i+1)*3] # (bs, 1, h, w, 3)

            # classify the input frame to get the class logits predictions
            class_logits = self.classifier(frame) # (bs, nc)
            assert (class_logits.shape == (self.batch_size, self.num_classes) or class_logits.shape == (None, self.num_classes))

            # Create pseudo-labels
            pseudo_labels = tf.nn.softmax(class_logits * self.temperature, axis=-1)  # Shape: (bs, nc)

            # Process the frame
            class_object_rep, vae_loss = tf.cond(tf.reduce_sum(frame) > 0, lambda: self.process_valid_frame(frame, pseudo_labels), lambda: self.process_null_frame())
            output_class_tensors.append(class_object_rep)
            all_vae_losses.append(vae_loss)

        # stack the class outputs to get the final output
        output = tf.concat(output_class_tensors, axis=-1)
        # print("Output shape:", output.shape)
        # if not tf.is_symbolic_tensor(inputs):
            # assert (output.shape == (self.batch_size, self.im_height, self.im_width, self.num_classes*self.output_channels) or output.shape == (None, self.im_height, self.im_width, self.num_classes*self.output_channels))

        return output, tf.reduce_mean(all_vae_losses)

class ObjectRepresentation_ConvVAE_LatentLSTM(layers.Layer):
    '''
    Convert images of object masks to class IDs, then update and extract the corresponding object representations
    '''
    def __init__(self, training_args, num_classes, batch_size, im_height, im_width, output_channels, **kwargs):
        super(ObjectRepresentation_ConvVAE_LatentLSTM, self).__init__(**kwargs)
        self.training_args = training_args
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.im_height = im_height
        self.im_width = im_width
        self.output_channels = output_channels
        self.latent_dim = 64
        # self.conv_lstm_vae_general = ConvLSTMVAE(latent_dim=128, image_shape=(im_height, im_width, 3), output_channels=output_channels, name='conv_lstm_vae_general')
        self.conv_vae_class = KerasVAE(latent_dim=self.latent_dim, output_channels=output_channels, num_classes=num_classes, name='conv_vae_class') # call returns: object_rep, loss, [encoder_states, decoder_states, object_rep_states]
        self.classifier = CustomCNN(num_classes=num_classes, num_conv_layers=3, name='classifier')
        # self.latent_LSTM = CustomLSTM(output_units=self.latent_dim, layer_num=0, name='latent_LSTM')
        self.latent_LSTM = layers.LSTM(units=self.latent_dim, return_state=True, return_sequences=False, name=f"latent_LSTM")

        # Initialize states
        self.class_object_rep_states = [tf.zeros(shape=(self.num_classes, self.latent_dim)), tf.zeros(shape=(self.num_classes, self.latent_dim))]
        # self.class_object_rep_states = [tf.Variable(tf.zeros(shape=(self.num_classes, self.latent_dim))), tf.Variable(tf.zeros(shape=(self.num_classes, self.latent_dim)))]

        self.predicted_class_IDs = []
        self.plot_num = 0

    def initialize_states(self):
        self.clear_states()

    def clear_states(self):
        self.class_object_rep_states = [tf.zeros(shape=(self.num_classes, self.latent_dim)), tf.zeros(shape=(self.num_classes, self.latent_dim))]
        # self.class_object_rep_states = [self.add_weight(shape=(self.num_classes, self.latent_dim), initializer='zeros', trainable=False, name='class_object_rep_state_h'), self.add_weight(shape=(self.num_classes, self.latent_dim), initializer='zeros', trainable=False, name='class_eobject_rep_state_c')]
        # self.class_object_rep_states = [tf.Variable(tf.zeros(shape=(self.num_classes, self.latent_dim))), tf.Variable(tf.zeros(shape=(self.num_classes, self.latent_dim)))]

    def diff_gather(self, params, logits, beta=1e10):
        '''
        Differentiable gather operation.
        Params shape: (num_classes, c)
        Logits shape: (batch_size, num_classes)
        '''
        # expand params to include batch_dim
        expanded_params = tf.expand_dims(params, axis=1) # (num_classes, 1, c)
        weights = tf.transpose(tf.nn.softmax(logits * beta), [1, 0]) # (num_classes, batch_size)
        current_weights_shape = weights.shape
        reshaped_weights = weights
        for _ in range(len(expanded_params.shape) - len(current_weights_shape)):
            reshaped_weights = tf.expand_dims(reshaped_weights, axis=-1) # (num_classes, batch_size, 1)

        weighted_params = reshaped_weights * expanded_params # broadcasting to shape (num_classes, batch_size, ...)
        weighted_sum = tf.reduce_sum(weighted_params, axis=0) # (batch_size, c)
        return weighted_sum


    def diff_scatter_nd_update(self, A, B, logits, beta=1e10):
        """
        Update tensor A with values from tensor B based on highest indices indicated by a logits matrix.
        Like tf.tensor_scatter_nd_update, but differentiable, in the sense that integer class indices are not required.

        Args:
        A (tf.Tensor): A tensor of shape (nc, oc).
        B (tf.Tensor): A tensor of shape (bs, oc).
        logits (tf.Tensor): A logits matrix of shape (bs, nc).

        Returns:
        tf.Tensor: Updated tensor A.
        """
        # Convert logits to one-hot
        one_hot = tf.nn.softmax(logits * beta) # (bs, nc)

        # Check dimensions
        if len(A.shape) != 2 or len(B.shape) != 2 or len(one_hot.shape) != 2:
            raise ValueError("Input tensors must be of the shape (nc, oc), (bs, oc), and (bs, nc) respectively.")
        
        # Check dimension matching
        nc, oc = A.shape
        if (B.shape[1:] != (oc)) or (one_hot.shape[1:] != (nc)):
            raise ValueError("Dimension mismatch among inputs.")

        # Expand A to match B's batch dimension
        A_expanded = tf.expand_dims(A, 1) # (nc, 1, oc)

        # Expand B to broadcast over the nc dimension
        B_expanded = tf.expand_dims(B, 0) # (1, bs, oc)

        # Expand the one-hot matrix to match A's dimensions
        mask = tf.expand_dims(one_hot, -1) # (bs, nc, 1)
        mask = tf.transpose(mask, [1, 0, 2])  # Reshape to (nc, bs, 1)

        # Multiply A by (1 - mask) to zero out the update positions
        A_masked = A_expanded * (1 - mask)

        # Multiply B by mask to align updates
        B_masked = B_expanded * mask

        # Combine the two components
        A_updated = A_masked + B_masked

        # Reduce_max over the batch dimension to create single updated class state
        A_updated = tf.reduce_max(A_updated, axis=1, keepdims=False) # (nc, oc)

        return A_updated


    def process_valid_frame(self, frame):
        # Only processing the frame if it is not empty (valid)
        
        # classify the input frame to get the class logits predictions
        class_logits = self.classifier(frame) # (bs, nc)
        assert (class_logits.shape == (self.batch_size, self.num_classes) or class_logits.shape == (None, self.num_classes))

        # get current class states
        current_class_states = [self.diff_gather(self.class_object_rep_states[0], class_logits), self.diff_gather(self.class_object_rep_states[1], class_logits)] # (bs, h, w, oc) via broadcasting

        # VAE trained with hard label distributions, so apply softmax to get pseudo-labels
        labels = tf.math.softmax(class_logits*1000)

        # encode frame and class logits into latent vector
        z = self.conv_vae_class.encode([frame, labels])

        # Compute the loss from last prediction
        lstm_loss = tf.reduce_mean(tf.square(z - current_class_states[0]))
        # lstm_loss = tf.cond(tf.reduce_sum(current_class_states[0]) == 0, lambda: tf.constant(0.0, shape=()), lambda: tf.reduce_mean(tf.square(z - current_class_states[0])))

        # Apply latent LSTM to get the predicted next-frame latent states
        _, new_latent_state_h, new_latent_state_c = self.latent_LSTM(tf.expand_dims(z, axis=1), initial_state=current_class_states) # (bs, latent_dim) x 3

        # update the class states
        class_state_updates = [self.diff_scatter_nd_update(self.class_object_rep_states[0], new_latent_state_h, class_logits), self.diff_scatter_nd_update(self.class_object_rep_states[1], new_latent_state_c, class_logits)] # (bs, h, w, oc) via broadcasting
        
        # Also update the permanent copies.
        self.class_object_rep_states[0] = class_state_updates[0]
        self.class_object_rep_states[1] = class_state_updates[1]

        # Decode predicted latent vector into class object representation
        class_object_rep = self.conv_vae_class.decoder([new_latent_state_h, labels], training=False)

        return class_object_rep, lstm_loss


    def process_null_frame(self,):
        # If the input frame is empty, set the class output to zero
        class_object_rep = tf.zeros((self.batch_size, self.im_height, self.im_width, self.output_channels), dtype=tf.float32)
        loss = tf.constant(0.0, shape=())

        return class_object_rep, loss
    

    
    def call(self, inputs):
        # Inputs shape: (bs, 1, h, w, oc)
        
        '''Update Class States and Obtain Outputs'''
        # Note that inputs are nc*3 channels, where nc is the number of classes and thus we process each 3-channel block separately
        
        output_class_tensors = []
        all_losses = []
        all_class_logits = []
        self.predicted_class_IDs = []
        self.frames = tf.stack([inputs[0, 0, ..., i*3:(i+1)*3] for i in range(self.num_classes)]) # (nc, h, w, 3)
        for i in range(self.num_classes):

            frame = inputs[..., i*3:(i+1)*3] # (bs, 1, h, w, 3)

            # Process the frame
            # class_object_rep, loss = tf.cond(tf.reduce_sum(frame) > 0, lambda: self.process_valid_frame(frame), lambda: self.process_null_frame())
            class_object_rep, loss = self.process_valid_frame(frame)
            output_class_tensors.append(class_object_rep)
            all_losses.append(loss)
            

        # stack the class outputs to get the final output
        output = tf.concat(output_class_tensors, axis=-1)
        # print("Output shape:", output.shape)
        assert (output.shape == (self.batch_size, self.im_height, self.im_width, self.num_classes*self.output_channels) or output.shape == (None, self.im_height, self.im_width, self.num_classes*self.output_channels))

        # return the class-specific object representations (hidden class states)
        return output, tf.reduce_mean(all_losses)

class ObjectRepresentation_ConvVAE_LatentLPN(layers.Layer):
    '''
    Convert images of object masks to class IDs, then update and extract the corresponding object representations
    '''
    def __init__(self, training_args, num_classes, batch_size, im_height, im_width, output_channels, **kwargs):
        super(ObjectRepresentation_ConvVAE_LatentLPN, self).__init__(**kwargs)
        self.training_args = training_args
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.im_height = im_height
        self.im_width = im_width
        self.output_channels = output_channels
        self.latent_dim = 64
        self.conv_vae_class = KerasVAE(latent_dim=self.latent_dim, output_channels=output_channels, num_classes=self.num_classes, name='conv_vae_class') # num_classes-1 because the null class is not included in the pre-trained conv_vae
        self.classifier = CustomCNN(num_classes=self.num_classes, num_conv_layers=3, trainable=False, name='classifier') # num_classes-1 because the null class is not included in the pre-trained classifier
        self.latent_LPN = LinearPNet(batch_size=self.batch_size, latent_dim=self.latent_dim, num_classes=self.num_classes+1, name='latent_LPN') # num_classes + 1 for the null class

    def initialize_states(self):
        self.latent_LPN.init_layer_states()

    def clear_states(self):
        self.latent_LPN.clear_layer_states()

    def process_frame(self, frame):
        # Only processing the frame if it is not empty (valid)
        
        # classify the input frame to get the class logits predictions
        class_logits = self.classifier(frame) # (bs, nc)

        # VAE trained with hard label distributions, so apply softmax to get pseudo-labels
        labels = tf.math.softmax(class_logits*1e10) # (bs, nc)

        # For the latent_LPN, swap labels pertaining to null frames to null class
        labels_with_null = tf.concat([labels, tf.zeros((self.batch_size, 1), dtype=tf.float32)], axis=-1) # (bs, nc+1)
        cond = tf.reduce_sum(frame, axis=[1, 2, 3]) > 0
        labels_with_null = tf.where(tf.reshape(cond, [-1,1]), labels_with_null, tf.one_hot(tf.fill([self.batch_size], self.num_classes), self.num_classes+1)) # (bs, nc+1)

        # encode frame and class logits into latent vector
        z = self.conv_vae_class.encode([frame, labels])

        # Apply latent LSTM to get the predicted next-frame latent states, using the bloated labels
        pred_next_z, lpn_loss = self.latent_LPN([z, labels_with_null]) # (bs, latent_dim), (bs, 1)

        # Decode predicted latent vector into class object representation
        class_object_rep = self.conv_vae_class.decode_object_rep([pred_next_z, labels])

        # Compute VAE reconstruction loss
        vae_loss, mask_reconstruction_loss, kl_loss = self.conv_vae_class.compute_loss(*self.conv_vae_class([frame, labels]))

        # Nullify predictions and losses associated with null frames
        class_object_rep = tf.where(tf.reshape(cond, [-1,1,1,1]), class_object_rep, tf.zeros_like(class_object_rep))
        lpn_loss = tf.where(tf.reshape(cond, [-1,1]), lpn_loss, tf.zeros_like(lpn_loss))
        vae_loss = tf.where(tf.reshape(cond, [-1,1]), vae_loss, tf.zeros_like(vae_loss))

        return class_object_rep, lpn_loss#+vae_loss # {"lpn_loss": lpn_loss, "vae_loss": mask_reconstruction_loss}
    
    def call(self, inputs):
        # Inputs shape: (bs, 1, h, w, oc)
        
        '''Update Class States and Obtain Outputs'''
        # Note that inputs are nc*3 channels, where nc is the number of classes and thus we process each 3-channel block separately
        
        output_class_tensors = []
        all_losses = [] # {"lpn_loss":0, "vae_loss":0}
        for i in range(self.num_classes):

            frame = inputs[..., i*3:(i+1)*3] # (bs, h, w, 3)

            # Process the frame
            class_object_rep, OR_losses = self.process_frame(frame)
            assert (class_object_rep.shape == (self.batch_size, self.im_height, self.im_width, self.output_channels) or class_object_rep.shape == (None, self.im_height, self.im_width, self.output_channels)), f"class_object_rep shape: {class_object_rep.shape}"
            output_class_tensors.append(class_object_rep)
            all_losses.append(OR_losses) # = {key: all_losses[key] + OR_losses[key] for key in all_losses}


        # stack the class outputs to get the final output
        output = tf.concat(output_class_tensors, axis=-1)
        all_losses = tf.reduce_mean(all_losses) # = {key: tf.reduce_mean(all_losses[key]) for key in all_losses}

        return output, all_losses

class SceneDecomposer:
    def __init__(self, n_colors=4, include_frame=False):
        self.n_colors = n_colors
        self.last_colors = None
        self.last_masks = None
        self.include_frame = include_frame

    def report_state(self):
        print("Last colors: ", self.last_colors)
        print("Last masks: ", self.last_masks)

    def quantize_image(self, image, num_colors=4):
        return image.convert('RGBA').quantize(colors=num_colors, method=Image.FASTOCTREE)

    def process_single_image(self, image):
        '''
        Process a single image and return a list of masks, one for each color in the image.
        Expected input: PIL Image or numpy array with shape (H, W, 3), float32, range [0, 1]
        Returns: List of masks, each with shape (H, W, 3), uint8, range [0, 255]
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
        Optionally include original frame in the output
        """
        T, H, W, C = sequence.shape
        masks_sequence = np.zeros((T, H, W, self.n_colors*C), dtype=np.float32) if not self.include_frame else np.zeros((T, H, W, (self.n_colors+1)*C), dtype=np.float32)

        self.clear_state()
        for t in range(T):
            new_masks = self.process_single_image(sequence[t])
            if self.include_frame:
                new_masks = np.concatenate([new_masks, (sequence[t] * 255).astype(np.uint8)], axis=-1)
                assert new_masks.shape[-1] == (self.n_colors+1)*C
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
