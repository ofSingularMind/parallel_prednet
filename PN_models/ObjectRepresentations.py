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
from keras.layers import Dense, GlobalAveragePooling2D, ConvLSTM2D, BatchNormalization, Dropout, Conv2D, MaxPooling2D, Flatten, Input, UpSampling2D, Concatenate, Add, Activation, Multiply
from keras.models import Model
from keras.applications import MobileNetV2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from keras.callbacks import LearningRateScheduler, ModelCheckpoint, TensorBoard

import pdb

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

class SequenceVAE(keras.Model):
    def __init__(self, training_args, latent_dim, meta_latent_dim, seq_vae_convlstm_channels, num_im_in_seq, num_classes, **kwargs):
        super().__init__(**kwargs)

        self.training_args = training_args
        self.batch_size = self.training_args["batch_size"]
        self.im_height = self.training_args["SSM_im_shape"][0]
        self.im_width = self.training_args["SSM_im_shape"][1]

        ###### Start ConvVAE ######
        self.latent_dim = latent_dim
        self.meta_latent_dim = meta_latent_dim
        self.conv_lstm_channels = seq_vae_convlstm_channels
        input_shape = (num_im_in_seq, self.im_height, self.im_width, 1)  # binary masks: (BS, num_im_in_seq, H, W, 1)

        # Encoder
        encoder_inputs1 = keras.Input(shape=input_shape, batch_size=self.batch_size, name="seq_encoder_input")
        label_inputs1 = keras.Input(shape=(num_classes,), batch_size=self.batch_size, name="seq_encoder_label_input")
        x = ConvLSTM2D(self.conv_lstm_channels, (3, 3), padding='same', return_sequences=False, return_state=False, name="seq_encoder_convlstm")(encoder_inputs1)
        x = layers.Conv2D(32, 7, activation="relu", strides=2, padding="same", name="seq_encoder_conv2d_1")(x)
        x = layers.Conv2D(32, 5, activation="relu", strides=2, padding="same", name="seq_encoder_conv2d_2")(x)
        x = layers.Conv2D(64, 7, activation="relu", strides=2, padding="same", name="seq_encoder_conv2d_3")(x)
        x = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same", name="seq_encoder_conv2d_4")(x)
        x = layers.Flatten(name="seq_encoder_flatten")(x)
        x = layers.Concatenate(name="seq_encoder_concat")([x, label_inputs1])
        x = layers.Dense(128, activation="relu", name="seq_encoder_dense_1")(x)
        x = layers.Dense(latent_dim, activation="relu", name="seq_encoder_dense_2")(x)
        z_mean = layers.Dense(latent_dim, name="seq_z_mean")(x)
        z_log_var = layers.Dense(latent_dim, name="seq_z_log_var")(x)
        z = KerasSampling(name="seq_encoder_sampling")([z_mean, z_log_var])
        encoder = keras.Model([encoder_inputs1, label_inputs1], [z_mean, z_log_var, z], name="seq_encoder")

        # Decoder
        latent_inputs2 = keras.Input(shape=(latent_dim,), batch_size=self.batch_size, name="seq_decoder_latent_input")
        label_inputs2 = keras.Input(shape=(num_classes,), batch_size=self.batch_size, name="seq_decoder_label_input")
        x = layers.Concatenate(name="seq_decoder_concat_1")([latent_inputs2, label_inputs2])
        x = layers.Dense(8 * 8 * 128, activation="relu", name="seq_decoder_dense_1")(x)
        x = layers.Reshape((8, 8, 128), name="seq_decoder_reshape")(x)
        x = layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same", name="seq_decoder_conv2d_transpose_1")(x)
        x = layers.Conv2DTranspose(48, 3, activation="relu", strides=2, padding="same", name="seq_decoder_conv2d_transpose_2")(x)
        x = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same", name="seq_decoder_conv2d_transpose_3")(x)
        x = layers.Conv2DTranspose(num_im_in_seq, 3, activation="sigmoid", padding="same", name="seq_decoder_conv2d_transpose_4")(x)
        decoder_outputs = layers.Reshape((num_im_in_seq, self.im_height, self.im_width, input_shape[-1]), name="seq_decoder_output_reshape")(x)
        decoder = keras.Model([latent_inputs2, label_inputs2], decoder_outputs, name="seq_decoder")

        # # Object Representation Decoder
        # def unstack_and_concat(x):
        #     # Unstack along axis 1 (sequence dimension)
        #     unstacked = tf.unstack(x, axis=1)
        #     # Concatenate along the channel axis (axis=-1)
        #     concatenated = tf.concat(unstacked, axis=-1)
        #     return concatenated

        # encoder_inputs3 = keras.Input(shape=input_shape, batch_size=self.batch_size, name="object_rep_encoder_input")
        # meta_latent_inputs3 = keras.Input(shape=(meta_latent_dim,), batch_size=self.batch_size, name="object_rep_meta_latent_input")
        # label_inputs3 = keras.Input(shape=(num_classes,), batch_size=self.batch_size, name="object_rep_label_input")
        # x = layers.Concatenate(name="object_rep_concat_1")([meta_latent_inputs3, label_inputs3])
        # x = layers.Dense(8 * 8 * 128, activation="relu", name="object_rep_dense_1")(x)
        # x = layers.Reshape((8, 8, 128), name="object_rep_reshape")(x)
        # x = layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same", name="object_rep_conv2d_transpose_1")(x)
        # x = layers.Conv2DTranspose(48, 3, activation="relu", strides=2, padding="same", name="object_rep_conv2d_transpose_2")(x)
        # x = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same", name="object_rep_conv2d_transpose_3")(x)
        # x = layers.Conv2DTranspose(64, 3, activation="sigmoid", padding="same", name="object_rep_conv2d_transpose_4")(x)
        # # Unstack and concatenate the encoder inputs
        # encoder_inputs3_stacked = layers.Lambda(unstack_and_concat, name="object_rep_unstack_and_concat")(encoder_inputs3)
        # # Concatenate the decoder output with the stacked encoder inputs
        # x = layers.Concatenate(name="object_rep_concat_2")([x, encoder_inputs3_stacked])
        # x1 = layers.Conv2D(64, 3, activation="relu", padding="same", name="object_rep_conv2d_1")(x)
        # x2 = layers.Conv2D(32, 3, activation="relu", padding="same", name="object_rep_conv2d_2")(x1)
        # x3 = layers.Conv2D(16, 3, activation="relu", padding="same", name="object_rep_conv2d_3")(x2)
        # x = layers.Concatenate(name="object_rep_concat_3")([x3, x2, x1])
        # object_rep_decoder_outputs = layers.Conv2D(input_shape[-1], 3, activation="relu", padding="same", name="object_rep_output_conv2d")(x)
        # object_rep_decoder = keras.Model([meta_latent_inputs3, label_inputs3, encoder_inputs3], object_rep_decoder_outputs, name="object_rep_decoder")

        self.encoder = encoder
        self.decoder = decoder
        # self.object_rep_decoder = object_rep_decoder
    
    def call(self, data, training=True):
        images, labels = data
        binary_masks = self.images_to_masks(images)
        z_mean, z_log_var, z = self.encoder([binary_masks, labels], training=training)
        recon_masks = self.decoder([z, labels], training=training)
        return binary_masks, recon_masks, z_mean, z_log_var

    def images_to_masks(self, images):
        # images = (BS, num_im_in_seq, H, W, 3)
        binary_masks = tf.expand_dims(tf.cast(tf.reduce_any(tf.not_equal(images, 0), axis=-1), tf.float32), axis=-1)
        return binary_masks

    def encode(self, data, training=True):
        # images = (BS, num_im_in_seq, H, W, 3), labels = (BS, num_classes)
        images, labels = data
        binary_masks = self.images_to_masks(images)
        z_mean, z_log_var, z = self.encoder([binary_masks, labels], training=training)
        return z

    def decode(self, data, training=True):
        z, labels = data
        recon_masks = self.decoder([z, labels], training=training)
        return recon_masks

    def compute_loss(self, masks, recon_masks, z_mean, z_log_var):
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

class MetaLatentVAE(keras.Model):
    def __init__(self, training_args, latent_dim, meta_latent_dim, num_slv_keep, num_im_in_seq, num_classes, **kwargs):
        super().__init__(**kwargs)

        self.training_args = training_args
        self.batch_size = self.training_args["batch_size"]
        self.im_height = self.training_args["SSM_im_shape"][0]
        self.im_width = self.training_args["SSM_im_shape"][1]

        ###### Start ConvVAE ######
        self.num_slv_keep = num_slv_keep
        self.latent_dim = latent_dim
        self.meta_latent_dim = meta_latent_dim
        # self.conv_lstm_channels = seq_vae_convlstm_channels
        stored_latent_vectors_input_shape = (num_slv_keep, latent_dim) # a set of stored sequence latent vectors, batch dimension is added in call
        class_sequence_input_shape = (num_im_in_seq, self.im_height, self.im_width, 1)  # binary masks: (BS, num_im_in_seq, H, W, 1)

        # Encoder
        encoder_inputs1 = keras.Input(shape=stored_latent_vectors_input_shape, batch_size=1, name="ml_encoder_input")
        label_inputs1 = keras.Input(shape=(num_classes,), batch_size=1, name="ml_encoder_label_input")
        x = layers.Flatten(name="ml_encoder_flatten")(encoder_inputs1)
        x = layers.Concatenate(name="ml_encoder_concat")([x, label_inputs1])
        x = layers.Dense(512, activation="relu", name="ml_encoder_dense_1")(x)
        x = layers.Dense(256, activation="relu", name="ml_encoder_dense_2")(x)
        x = layers.Dense(128, activation="relu", name="ml_encoder_dense_3")(x)
        x = layers.Dense(meta_latent_dim, activation="relu", name="ml_encoder_dense_4")(x)
        z_mean = layers.Dense(meta_latent_dim, name="ml_z_mean")(x)
        z_log_var = layers.Dense(meta_latent_dim, name="ml_z_log_var")(x)
        z = KerasSampling(name="ml_encoder_sampling")([z_mean, z_log_var])
        ml_encoder = keras.Model([encoder_inputs1, label_inputs1], [z_mean, z_log_var, z], name="ml_encoder")

        # Decoder
        meta_latent_inputs2 = keras.Input(shape=(meta_latent_dim,), batch_size=1, name="ml_decoder_latent_input")
        label_inputs2 = keras.Input(shape=(num_classes,), batch_size=1, name="ml_decoder_label_input")
        x = layers.Concatenate(name="ml_decoder_concat_1")([meta_latent_inputs2, label_inputs2])
        x = layers.Dense(128, activation="relu", name="ml_decoder_dense_1")(x)
        x = layers.Dense(256, activation="relu", name="ml_decoder_dense_2")(x)
        x = layers.Dense(512, activation="relu", name="ml_decoder_dense_3")(x)
        ml_decoder_outputs = layers.Dense(num_slv_keep*latent_dim, activation="relu", name="ml_decoder_dense_4")(x)
        # ml_decoder_outputs = layers.Reshape((stored_latent_vectors_input_shape), name="ml_decoder_output_reshape")(x)
        ml_decoder = keras.Model([meta_latent_inputs2, label_inputs2], ml_decoder_outputs, name="ml_decoder")

        # Object Representation Decoder
        def unstack_and_concat(x):
            # Unstack along axis 1 (sequence dimension)
            unstacked = tf.unstack(x, axis=1)
            # Concatenate along the channel axis (axis=-1)
            concatenated = tf.concat(unstacked, axis=-1)
            return concatenated

        encoder_inputs3 = keras.Input(shape=class_sequence_input_shape, batch_size=self.batch_size, name="object_rep_encoder_input")
        meta_latent_inputs3 = keras.Input(shape=(meta_latent_dim,), batch_size=self.batch_size, name="object_rep_meta_latent_input")
        label_inputs3 = keras.Input(shape=(num_classes,), batch_size=self.batch_size, name="object_rep_label_input")
        x = layers.Concatenate(name="object_rep_concat_1")([meta_latent_inputs3, label_inputs3])
        x = layers.Dense(8 * 8 * 128, activation="relu", name="object_rep_dense_1")(x)
        x = layers.Reshape((8, 8, 128), name="object_rep_reshape")(x)
        x = layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same", name="object_rep_conv2d_transpose_1")(x)
        x = layers.Conv2DTranspose(48, 3, activation="relu", strides=2, padding="same", name="object_rep_conv2d_transpose_2")(x)
        x = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same", name="object_rep_conv2d_transpose_3")(x)
        x = layers.Conv2DTranspose(64, 3, activation="sigmoid", padding="same", name="object_rep_conv2d_transpose_4")(x)
        # Unstack and concatenate the encoder inputs
        encoder_inputs3_stacked = layers.Lambda(unstack_and_concat, name="object_rep_unstack_and_concat")(encoder_inputs3)
        # Concatenate the decoder output with the stacked encoder inputs
        x = layers.Concatenate(name="object_rep_concat_2")([x, encoder_inputs3_stacked])
        x1 = layers.Conv2D(64, 3, activation="relu", padding="same", name="object_rep_conv2d_1")(x)
        x2 = layers.Conv2D(32, 3, activation="relu", padding="same", name="object_rep_conv2d_2")(x1)
        x3 = layers.Conv2D(16, 3, activation="relu", padding="same", name="object_rep_conv2d_3")(x2)
        x = layers.Concatenate(name="object_rep_concat_3")([x3, x2, x1])
        object_rep_decoder_outputs = layers.Conv2D(class_sequence_input_shape[-1], 3, activation="relu", padding="same", name="object_rep_output_conv2d")(x)
        object_rep_decoder = keras.Model([meta_latent_inputs3, label_inputs3, encoder_inputs3], object_rep_decoder_outputs, name="object_rep_decoder")

        self.ml_encoder = ml_encoder
        self.ml_decoder = ml_decoder
        self.object_rep_decoder = object_rep_decoder
    
    def call(self, data, training=True):
        sequence_latent_vectors, labels = data # sequence_latent_vectors = (num_slv_keep, latent_dim), labels = (BS, num_classes)
        labels = tf.expand_dims(labels[0], axis=0) # Add batch dimension: (1, num_classes)
        sequence_latent_vectors = tf.expand_dims(sequence_latent_vectors, axis=0) # Add batch dimension: (1, num_slv_keep, latent_dim)

        z_mean, z_log_var, meta_z = self.ml_encoder([sequence_latent_vectors, labels], training=training)
        recon_sequence_latent_vectors = self.ml_decoder([meta_z, labels], training=training)
        return sequence_latent_vectors, recon_sequence_latent_vectors, z_mean, z_log_var

    def images_to_masks(self, images):
        # images = (BS, num_im_in_seq, H, W, 3)
        binary_masks = tf.expand_dims(tf.cast(tf.reduce_any(tf.not_equal(images, 0), axis=-1), tf.float32), axis=-1)
        return binary_masks

    def encode(self, data, training=True):
        sequence_latent_vectors, labels = data # sequence_latent_vectors = (num_slv_keep, latent_dim), labels = (BS, num_classes)
        labels = tf.expand_dims(labels[0], axis=0) # Add batch dimension: (1, num_classes)
        sequence_latent_vectors = tf.expand_dims(sequence_latent_vectors, axis=0) # Add batch dimension: (1, num_slv_keep, latent_dim)

        z_mean, z_log_var, meta_z = self.ml_encoder([sequence_latent_vectors, labels], training=training)
        return meta_z

    def decode(self, data, training=True):
        meta_z, labels = data
        recon_sequence_latent_vectors = self.ml_decoder([meta_z, labels], training=training)
        return recon_sequence_latent_vectors

    def decode_object_rep(self, data, training=True):
        # z = (1, meta_latent_dim), images = (BS, num_im_in_seq, H, W, 3), labels = (BS, num_classes)
        meta_z, labels, images = data
        meta_z = tf.tile(meta_z, [images.shape[0], 1]) # Tile z to match batch size
        binary_masks = self.images_to_masks(images)
        object_rep = self.object_rep_decoder([meta_z, labels, binary_masks], training=training)
        return object_rep

    def compute_loss(self, sequence_latent_vectors, recon_sequence_latent_vectors, z_mean, z_log_var):
        sequence_latent_vectors = K.batch_flatten(sequence_latent_vectors)
        sequence_latent_vector_reconstruction_loss = tf.reduce_mean(
            keras.losses.mean_squared_error(sequence_latent_vectors, recon_sequence_latent_vectors) + 
            keras.losses.mean_absolute_error(sequence_latent_vectors, recon_sequence_latent_vectors)
        ) * self.num_slv_keep * self.latent_dim
        
        kl_loss = -0.5 * (1 + z_log_var - tf.math.square(z_mean) - tf.math.exp(z_log_var))
        kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
        total_loss = sequence_latent_vector_reconstruction_loss + 0.25*kl_loss
        return total_loss, sequence_latent_vector_reconstruction_loss, kl_loss

class ObjectRepDecoder(keras.Model):
    def __init__(self, training_args, layer_num, im_height, im_width, latent_dim, num_slv_keep, num_im_in_seq, num_classes, **kwargs):
        super().__init__(**kwargs)

        self.training_args = training_args
        self.batch_size = self.training_args["batch_size"]
        self.im_height = im_height
        self.im_width = im_width
        self.num_classes = num_classes
        self.target_im_height = self.training_args["SSM_im_shape"][0]
        self.target_im_width = self.training_args["SSM_im_shape"][1]
        self.layer_num = layer_num

        ###### Start ConvVAE ######
        self.num_slv_keep = num_slv_keep
        self.latent_dim = latent_dim
        class_sequence_input_shape = (num_im_in_seq, self.target_im_height, self.target_im_width, 1)  # binary masks: (BS, num_im_in_seq, H, W, 1)

        # Object Representation Decoder
        def unstack_and_concat(x):
            # Unstack along axis 1 (sequence dimension)
            unstacked = tf.unstack(x, axis=1)
            # Concatenate along the channel axis (axis=-1)
            concatenated = tf.concat(unstacked, axis=-1)
            return concatenated

        def max_pooling(x):
            pooled = x
            for _ in range(self.layer_num):
                pooled = layers.MaxPooling2D((2, 2))(pooled)
            return pooled

        encoder_inputs3 = keras.Input(shape=class_sequence_input_shape, batch_size=self.batch_size, name=f"object_rep_encoder_input_Layer_{self.layer_num}")
        meta_latent_inputs3 = keras.Input(shape=(latent_dim,), batch_size=self.batch_size, name=f"object_rep_meta_latent_input_Layer_{self.layer_num}")
        label_inputs3 = keras.Input(shape=(num_classes,), batch_size=self.batch_size, name=f"object_rep_label_input_Layer_{self.layer_num}")
        x = layers.Concatenate(name=f"object_rep_concat_1_Layer_{self.layer_num}")([meta_latent_inputs3, label_inputs3])
        x = layers.Dense(8 * 8 * 64, activation="relu", name=f"object_rep_dense_1_Layer_{self.layer_num}")(x)
        x = layers.Reshape((8, 8, 64), name=f"object_rep_reshape_Layer_{self.layer_num}")(x)
        x = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same", name=f"object_rep_conv2d_transpose_1_Layer_{self.layer_num}")(x)
        x = layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same", name=f"object_rep_conv2d_transpose_2_Layer_{self.layer_num}")(x)
        x = layers.Conv2DTranspose(16, 3, activation="relu", strides=2, padding="same", name=f"object_rep_conv2d_transpose_3_Layer_{self.layer_num}")(x)
        x = layers.Conv2DTranspose(8, 3, activation="sigmoid", padding="same", name=f"object_rep_conv2d_transpose_4_Layer_{self.layer_num}")(x)
        # Unstack and concatenate the encoder inputs
        encoder_inputs3_unstacked = layers.Lambda(unstack_and_concat, name=f"object_rep_unstack_and_concat_Layer_{self.layer_num}")(encoder_inputs3)
        # Concatenate the decoder output with the stacked encoder inputs
        x = layers.Concatenate(name=f"object_rep_concat_2_Layer_{self.layer_num}")([x, encoder_inputs3_unstacked])
        x1 = layers.Conv2D(32, 3, activation="relu", padding="same", name=f"object_rep_conv2d_1_Layer_{self.layer_num}")(x)
        x2 = layers.Conv2D(16, 3, activation="relu", padding="same", name=f"object_rep_conv2d_2_Layer_{self.layer_num}")(x1)
        x3 = layers.Conv2D(8, 3, activation="relu", padding="same", name=f"object_rep_conv2d_3_Layer_{self.layer_num}")(x2)
        x = layers.Concatenate(name=f"object_rep_concat_3_Layer_{self.layer_num}")([x3, x2, x1])
        object_rep_decoder_outputs_unpooled = layers.Conv2D(class_sequence_input_shape[-1], 3, activation="relu", padding="same", name=f"object_rep_output_conv2d_Layer_{self.layer_num}")(x)
        object_rep_decoder_outputs_pooled = layers.Lambda(max_pooling, name=f"object_rep_final_output_Layer_{self.layer_num}")(object_rep_decoder_outputs_unpooled)
        object_rep_decoder = keras.Model([meta_latent_inputs3, label_inputs3, encoder_inputs3], object_rep_decoder_outputs_pooled, name=f"object_rep_decoder_Layer_{self.layer_num}")

        self.object_rep_decoder = object_rep_decoder
    
    def call(self, data, training=True):
        meta_latent_vectors, recent_frame_sequences = data # meta_latent_vectors = (num_classes, BS, latent_dim), recent_frame_sequences = (num_classes, BS, num_im_in_seq, SSM_H, SSM_W, 3)
        object_representations = []
        for i in range(self.num_classes):
            class_sequence_binary_masks = self.images_to_masks(recent_frame_sequences[i])
            labels = tf.one_hot(tf.fill((self.batch_size,), i), self.num_classes) # (BS, num_classes)
            object_rep = self.object_rep_decoder([meta_latent_vectors[i], labels, class_sequence_binary_masks], training=training) # (BS, H, W, 1)
            object_representations.append(object_rep)
        object_representations = tf.concat(object_representations, axis=-1) # (BS, H, W, num_classes)
        return object_representations

    def images_to_masks(self, images):
        # images = (BS, num_im_in_seq, H, W, 3)
        binary_masks = tf.expand_dims(tf.cast(tf.reduce_any(tf.not_equal(images, 0), axis=-1), tf.float32), axis=-1)
        return binary_masks

class SequenceLatentMaintainer(layers.Layer):
    def __init__(self, training_args, n, k, m, latent_dim, num_classes, num_slv_keep, **kwargs):
        super(SequenceLatentMaintainer, self).__init__(**kwargs)
        self.training_args = training_args
        self.num_slv_keep = num_slv_keep
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.batch_size = self.training_args["batch_size"]
        self.n = self.batch_size + self.num_slv_keep  # Number of input vectors
        self.output_layer = layers.Dense(n*self.num_slv_keep)  # Output layer to produce (num_slv_keep, n) matrix

        # These latent vectors are maintained to be the most diverse set of "sequence observations" for each class
        # From this set, an overall understanding of the spatio-temporal dynamics of the class can be inferred, hopefully
        # self.historic_sequence_latent_vectors = [None for _ in range(self.num_classes)]
        self.historic_sequence_latent_vectors_stored = [
            self.add_weight(name=f"historic_SLVs_for_class_{i}", shape=(self.num_slv_keep, latent_dim), dtype=tf.float32, initializer='zeros', trainable=False)
            for i in range(self.num_classes)
        ]

        # # OPTION 1 (FUNCTIONAL) MATRIX Logits_Maker for the maintained latent vectors selection
        # logits_input = keras.Input(shape=self.n*self.latent_dim, batch_size=1, name="logits_input")
        # x = layers.Dense(512, activation="relu", name="logits_dense_1")(logits_input)
        # x = layers.Dense(256, activation="relu", name="logits_dense_2")(x)
        # x = layers.Dense(self.n*self.num_slv_keep, activation="relu", name="logits_dense_3")(x)
        # logits_output = layers.Reshape((self.num_slv_keep, self.n), name="logits_output")(x)
        # self.logits_layer = keras.Model(logits_input, logits_output, name="logits_layer")

        # # OPTION 2 (NON-FUNCTIONAL) SCALAR Logits_Maker for the maintained latent vectors selection
        # logits_input = keras.Input(shape=self.n*self.latent_dim, batch_size=1, name="logits_input")
        # x = layers.Dense(512, activation="relu", name="logits_dense_1")(logits_input)
        # x = layers.Dense(256, activation="relu", name="logits_dense_2")(x)
        # logits_output = layers.Dense(self.n, activation="relu", name="logits_output")(x)
        # self.logits_layer = keras.Model(logits_input, logits_output, name="logits_layer")

    def call(self, new_vectors, class_label):

        # trained_logits_layer_weights = np.load("/home/evalexii/Documents/Thesis/code/parallel_prednet/model_weights/SSM/multiShape/logits_layer_weights.npz", allow_pickle=True)
        # trained_logits_layer_weights = [trained_logits_layer_weights[key] for key in trained_logits_layer_weights.keys()]
        # un_trained_logits_layer = self.logits_layer
        # un_trained_logits_layer.set_weights(trained_logits_layer_weights)
        # print("Pre-trained logits layer weights loaded successfully")

        # Combine maintained vectors with new input vectors
        combined_vectors = tf.concat([self.historic_sequence_latent_vectors_stored[class_label], new_vectors], axis=0)  # Shape: (n, latent_dim)
        
        # # OPTION 1: Use the output layer to produce the (num_slv_keep, n) matrix
        # # Produce the (num_slv_keep, n) matrix
        # x = tf.expand_dims(K.flatten(combined_vectors), axis=0) # Shape: (1, n*latent_dim)
        # logits = tf.squeeze(self.logits_layer(x), axis=0) # Shape: (num_slv_keep, n)
        # logits = logits + tf.random.uniform(tf.shape(logits), 0, 1e-6)  # Add noise to prevent zero logits
        # # Apply softmax to get the soft selection matrix
        # softmax_output = tf.nn.softmax(logits * 1e10, axis=-1) # Shape: (num_slv_keep, n)
        # # Compute the product with the combined vectors
        # selected_vectors = tf.matmul(softmax_output, combined_vectors)# Shape: (num_slv_keep, latent_dim)
        
        # # OPTION 2: Use the logits layer to produce the (n, 1) logits and perform "hard" selection of num_slv_keep vectors
        # # Pass through the dense layers
        # x = tf.expand_dims(K.flatten(combined_vectors), axis=0)
        # x = self.logits_layer(x)
        # logits = tf.transpose(x, [1,0]) # Shape: (n, 1)
        # selected_vectors = self.hard_select_vectors(combined_vectors, logits) # Shape: (num_slv_keep, latent_dim)

        # # OPTION 3: Just take the last num_slv_keep vectors...
        selected_vectors = combined_vectors[-self.num_slv_keep:]
        
        # Store the selected vectors back to the maintained state
        self.historic_sequence_latent_vectors_stored[class_label].assign(selected_vectors)

        # Calulate the maintained_vectors_loss as inverse volume spanned and sum of pairwise distances
        maintained_vectors_loss = self.combined_loss(selected_vectors)

        
        return selected_vectors, maintained_vectors_loss

    def calculate_volume_spanned_loss(self, vectors, epsilon=1e-6):
        # Note: vectors is of shape (num_slv_keep, latent_dim)
        # Step 1: Compute the Gram matrix (G = V * V^T)
        gram_matrix = tf.matmul(vectors, vectors, transpose_b=True)

        # Step 2: Add a small regularization term to the diagonal of the Gram matrix
        gram_matrix += tf.eye(tf.shape(gram_matrix)[0]) * epsilon

        # Step 3: Compute the determinant of the Gram matrix as a proxy for the volume spanned by the vectors
        volume = tf.linalg.logdet(gram_matrix)

        # Step 4: Compute the loss as the inverse of the volume
        loss = 1 / (volume + epsilon) # Inverse of the volume, to be minimized to maximize the volume

        return loss

    def calculate_sum_pairwise_distances(self, vectors, epsilon=1e-6):
        # Step 1: Compute the pairwise distance matrix
        # Use broadcasting to compute pairwise differences and then square them
        pairwise_diff = tf.expand_dims(vectors, axis=1) - tf.expand_dims(vectors, axis=0)
        
        # Step 2: Compute the squared Euclidean distances
        pairwise_distances_squared = tf.reduce_sum(tf.square(pairwise_diff), axis=-1)
        
        # Step 3: Compute the sum of pairwise distances by taking the square root
        pairwise_distances = tf.sqrt(pairwise_distances_squared)
        
        # Step 4: Sum all the pairwise distances
        sum_distances = tf.reduce_sum(pairwise_distances)

        # Step 5: Compute the loss as the inverse of the sum of pairwise distances
        loss = 1 / (sum_distances + epsilon)

        return loss

    def combined_loss(self, vectors, epsilon=1e-3):
        # Volume loss
        gram_matrix = tf.matmul(vectors, vectors, transpose_b=True)
        gram_matrix += tf.eye(tf.shape(gram_matrix)[0]) * epsilon
        log_volume = tf.linalg.logdet(gram_matrix)
        volume_loss = -log_volume

        # Pairwise distance loss
        pairwise_diff = tf.expand_dims(vectors, axis=1) - tf.expand_dims(vectors, axis=0)
        pairwise_distances_squared = tf.reduce_sum(tf.square(pairwise_diff), axis=-1)
        pairwise_distances = tf.sqrt(pairwise_distances_squared)
        sum_distances = tf.reduce_sum(pairwise_distances)
        distance_loss = -sum_distances

        # Combined loss with a weighting factor
        combined_loss = volume_loss + 0.1 * distance_loss

        return combined_loss

    def hard_select_vectors(self, vectors, logits):
        """
        Selects the top k vectors in a ~hard manner from the input vectors based on the logits.
        Args:
            vectors: The input vectors of shape (n, latent_dim)
            logits: The logits of shape (n, 1)
        Returns:
            selected_vectors: The top num_slv_keep selected vectors of shape (num_slv_keep, latent_dim)
        """
        # Scale logits
        scaled_logits = tf.squeeze(logits * 1000, axis=-1)
        # Add scaled logits to vectors
        modified_vectors = vectors + tf.expand_dims(scaled_logits, axis=-1)
        # Transpose so that the axis of interest becomes the last dimension
        transposed_vectors = tf.transpose(modified_vectors, [1, 0])
        # Select top k transposed_vectors
        top_k_transposed_vectors, _ = tf.nn.top_k(transposed_vectors, k=self.num_slv_keep)
        # Transpose back to original shape
        top_k_modified_vectors = tf.transpose(top_k_transposed_vectors, [1, 0])
        # Get top k scaled logits for correct subtraction
        top_k_scaled_logits, _ = tf.nn.top_k(scaled_logits, k=self.num_slv_keep)
        # Subtract out scaled logits to isolate original vectors
        selected_vectors = top_k_modified_vectors - tf.expand_dims(top_k_scaled_logits, axis=-1)
        return selected_vectors
       
    def reset_stored_sequence_latent_vectors(self):
        [self.historic_sequence_latent_vectors_stored[i].assign(tf.keras.initializers.GlorotUniform()((self.num_slv_keep, self.latent_dim))) for i in range(self.num_classes)]


    def pad_to_num_slv_keep(self, tensor):
        # Ensure that the tensor has the correct shape by padding with zeros if necessary
        num_vectors = tf.shape(tensor)[0]
        padding_needed = self.num_slv_keep - num_vectors
        
        # If padding is needed, pad the tensor; otherwise, return the tensor as is
        if padding_needed > 0:
            padding = tf.zeros((padding_needed, self.latent_dim))
            tensor = tf.concat([tensor, padding], axis=0)
        
        return tensor

class ObjectRepresentations(keras.Model):
    def __init__(self, training_args, latent_dim, num_im_in_seq, seq_vae_convlstm_channels, **kwargs):
        super().__init__(**kwargs)

        self.training_args = training_args
        self.pretraining_step = self.training_args.get("pretraining_step", 0)
        self.nt = self.training_args["nt"] # for pretraining sequence VAE only
        self.batch_size = self.training_args["batch_size"]
        self.im_height = self.training_args["SSM_im_shape"][0]
        self.im_width = self.training_args["SSM_im_shape"][1]
        self.num_classes = self.training_args["num_classes"]
        self.shape_of_input = (self.batch_size, 1, self.im_height, self.im_width, self.num_classes * 3) # (BS, 1, H, W, num_classes * 3)
        self.latent_dim = latent_dim
        self.meta_latent_dim = 3 * latent_dim
        self.num_im_in_seq = num_im_in_seq
        self.num_slv_keep = 20

        self.sVAE = SequenceVAE(training_args, latent_dim, self.meta_latent_dim, seq_vae_convlstm_channels, num_im_in_seq, self.num_classes, name="Sequence_VAE")
        # self.mlVAE = MetaLatentVAE(training_args, latent_dim, self.meta_latent_dim, self.num_slv_keep, num_im_in_seq, self.num_classes, name="Meta_Latent_VAE")
        self.classifier = CustomCNN(num_classes=self.num_classes, num_conv_layers=3, trainable=False, name='classifier')
        self.seq_latent_maintainer = SequenceLatentMaintainer(training_args, self.batch_size, self.num_im_in_seq, self.num_classes, self.latent_dim, self.num_classes, self.num_slv_keep, name="Sequence_Latent_Maintainer")
        # self.or_decoder = ObjectRepDecoder(training_args, im_height, im_width, latent_dim, self.num_slv_keep, self.num_im_in_seq, self.num_classes, name="Object_Rep_Decoder")
        # self.meta_latent_former = tf.keras.Sequential([
        #     tf.keras.layers.Dense(512, activation='relu'),
        #     tf.keras.layers.Dense(256, activation='relu'),
        #     tf.keras.layers.Dense(self.meta_latent_dim)
        # ], name="Meta_Latent_Former")

        # These frame sequences are num_im_in_seq consecutive frames of the same class. Each prednet batch operates over a sequence of nt frames
        # This model will be called on each frame sequentially, and this recent_frame_sequences will be updated with each new frame
        # It is just a tensor of the last num_im_in_seq frames observed for each class
        self.recent_frame_sequences = tf.zeros((self.num_classes, self.batch_size, self.num_im_in_seq, self.im_height, self.im_width, 3))

        # These meta latent vectors are formed from the most diverse set of "sequence observations" for each class.
        # They serve as the input to creation of a 3D object representation tensor that the PredNet
        # bottom-layer Representation unit can interpret and utilize to inform their predictions
        # Actually, we don't store them.
        # self.meta_latent_vectors = tf.zeros((self.num_classes, 64))

    def initialize_states(self):
        self.recent_frame_sequences = tf.zeros((self.num_classes, self.batch_size, self.num_im_in_seq, self.im_height, self.im_width, 3))

    def clear_states(self):
        self.recent_frame_sequences = None

    def call(self, frame):
        # Assume frame is a single decomposed image (BS, 1, H, W, num_classes * 3)
        # frame = tf.squeeze(frame, axis=1)  # Remove the time dimension
        self.update_historic_frames(frame)

        # output_object_representations = []
        # Process each classes' historic-frame sequence separately
        for i in range(self.num_classes):
            class_sequence = self.recent_frame_sequences[i] # (BS, num_im_in_seq, H, W, 3)
            labels = tf.one_hot(tf.fill((self.batch_size,), i), self.num_classes) # (BS, num_classes)
            new_class_sequence_latent_vectors = self.sVAE.encode([class_sequence, labels])  # (BS, latent_dim)
            _, _ = self.seq_latent_maintainer(new_class_sequence_latent_vectors, i) # Update stored sequence latent vectors
        meta_latent_vectors = self.create_meta_latent_vectors() # List of tensors of shape (BS, latent_dim)
        # for i in range(self.num_classes):
        #     object_rep = self.or_decoder((meta_latent_vectors[i], labels, class_sequence))  # (BS, H, W, 1)
        #     output_object_representations.append(object_rep)

        # return tf.concat(output_object_representations, axis=-1) # (BS, H, W, num_classes * 1), (BS,)

        return meta_latent_vectors, self.recent_frame_sequences
    
    def create_meta_latent_vectors(self):
        # Assume sequence_latent_vectors is a tensor of shape (num_slv_keep, latent_dim)
        # Due to the structure encoded into the sequence_latent_vectors by the Sequence VAE, we can simply take the mean
        sequence_latent_vectors = tf.stack(self.seq_latent_maintainer.historic_sequence_latent_vectors_stored)  # (num_classes, num_slv_keep, latent_dim)
        meta_latent_vectors = tf.reduce_mean(sequence_latent_vectors, axis=-2)  # (num_classes, latent_dim)
        meta_latent_vectors = tf.expand_dims(meta_latent_vectors, axis=1)  # Add batch dimension, shape: (num_classes, 1, latent_dim)
        meta_latent_vectors = tf.tile(meta_latent_vectors, [1, self.batch_size, 1])  # Tile to match batch size, shape: (num_classes, BS, latent_dim)
        meta_latent_vectors = tf.unstack(meta_latent_vectors)  # List of tensors of shape (BS, latent_dim)
        return meta_latent_vectors

    def get_weights_considering_empty_frames(self, separated_images, logits_list):
        # separated_images = List of tensors of shape (batch_size, im_height, im_width, 3)
        # Stack the images along a new axis, if necessary
        stacked_images = tf.stack(separated_images) # Shape: (num_classes, batch_size, im_height, im_width, 3)
        stacked_logits = tf.stack(logits_list) # Shape: (num_classes, batch_size, num_classes)

        unstacked_nullified_logits = []
        for i in range(self.batch_size):
            stacked_images_batch = stacked_images[:, i, :, :, :] # Shape: (num_classes, im_height, im_width, 3)

            logits = stacked_logits[:, i, :] # Shape: (num_classes, num_classes)

            # Compute the sum across the desired axes
            sum_images = tf.reduce_sum(stacked_images_batch, axis=[1, 2, 3])
            sum_images = tf.expand_dims(sum_images, axis=-1) # Shape: (num_classes, 1)

            nullified_logits = tf.where(sum_images != 0.0, logits, tf.zeros_like(logits)) + tf.random.uniform(tf.shape(logits), 0, 1e-6) # to break ties

            unstacked_nullified_logits.append(nullified_logits)

        stacked_logits = tf.stack(unstacked_nullified_logits, axis=1) # Shape: (num_classes, batch_size, num_classes)

        weights = tf.nn.softmax(stacked_logits*1e6, axis=-1) # lock in predictions for non-empty frames, shape: (num_classes, batch_size, num_classes)
        weights = tf.nn.softmax(weights*1e6, axis=0) # distribute the remaining predictions for empty frames to un-predicted classes. ties are broken by adding noise, shape: (num_classes, batch_size, num_classes)
        weights_list = tf.unstack(weights) # List of tensors of shape (batch_size, num_classes)
        weights_list = [tf.expand_dims(tf.expand_dims(tf.expand_dims(weights, axis=1), axis=1), axis=1) for weights in weights_list] # Each of shape (batch_size, 1, 1, 1, num_classes)

        return weights_list
 
    def update_historic_frames(self, frame):
        # Assume frame is a single decomposed image (BS, H, W, num_classes * 3)
        # Separate the stacked images by splitting along the last axis
        separated_images = tf.split(frame, self.num_classes, axis=-1)  # List of tensors of shape (batch_size, im_height, im_width, 3)
        # Stack separated images along a new axis
        stacked_images = tf.stack(separated_images, axis=-1)  # Shape: (batch_size, im_height, im_width, 3, num_classes)

        logits_list = [self.classifier(img) for img in separated_images]  # Each of shape (batch_size, num_classes)
        weights_list = self.get_weights_considering_empty_frames(separated_images, logits_list) # Each of shape (batch_size, 1, 1, 1, num_classes)
        
        reordered_images = tf.stack([tf.reduce_sum(weights * stacked_images, axis=-1) for weights in weights_list], axis=0)
        most_recent_frame_sequences = self.recent_frame_sequences[:, :, 1:, :, :, :]  # Removing the oldest length-one sequence
        new_sequence = tf.expand_dims(reordered_images, axis=2)  # Add a time dimension to the reordered images
        self.recent_frame_sequences = tf.concat([most_recent_frame_sequences, new_sequence], axis=2)  # Add the new sequence

    def pretrain_sequence_vae(self, data):
        self.initialize_states()
        frames, y = data
        # Assume frames is a sequence of nt decomposed images (BS, nt, H, W, num_classes * 3)
        seq_total_loss = 0
        seq_mask_loss = 0
        seq_kl_loss = 0
        for i in range(self.nt):
            frame = frames[:, i, :, :, :]
            # self.update_historic_frames(frame)
            _, _ = self(frame)
            # Process each classes' historic-frame sequence separately
            total_loss = 0
            mask_loss = 0
            kl_loss = 0
            for c in range(self.num_classes):
                class_sequence = self.recent_frame_sequences[c] # (BS, num_im_in_seq, H, W, 3)
                labels = tf.one_hot(tf.fill((self.batch_size,), c), self.num_classes) # (BS, num_classes)
                binary_masks, recon_masks, z_mean, z_log_var = self.sVAE([class_sequence, labels])
                total_loss_part, mask_loss_part, kl_loss_part = self.sVAE.compute_loss(binary_masks, recon_masks, z_mean, z_log_var)
                
                total_loss += total_loss_part
                mask_loss += mask_loss_part
                kl_loss += kl_loss_part
                
            total_loss /= self.num_classes
            mask_loss /= self.num_classes
            kl_loss /= self.num_classes

            seq_total_loss += total_loss
            seq_mask_loss += mask_loss
            seq_kl_loss += kl_loss

        seq_total_loss /= self.nt
        seq_mask_loss /= self.nt
        seq_kl_loss /= self.nt
        self.clear_states()
        self.seq_latent_maintainer.reset_stored_sequence_latent_vectors()
        
        return seq_total_loss, seq_mask_loss, seq_kl_loss

    def pretrain_sequence_latent_maintainer(self, data):
        # if np.random.rand() < 0.3:
        #     self.seq_latent_maintainer.reset_stored_sequence_latent_vectors()
        self.initialize_states()
        frames, y = data
        # Assume frames is a sequence of nt decomposed images (BS, nt, H, W, num_classes * 3)
        all_maintained_vectors_loss = 0
        for i in range(self.nt):
            frame = frames[:, i, :, :, :]
            self.update_historic_frames(frame)
            # _, _ = self(frame)
            # Process each classes' historic-frame sequence separately
            maintained_vectors_loss = 0
            for c in range(self.num_classes):
                class_sequence = self.recent_frame_sequences[c] # (BS, num_im_in_seq, H, W, 3)
                labels = tf.one_hot(tf.fill((self.batch_size,), c), self.num_classes) # (BS, num_classes)
                new_class_sequence_latent_vectors = self.sVAE.encode([class_sequence, labels]) # (BS, latent_dim)
                _, maintained_vectors_loss_part = self.seq_latent_maintainer(new_class_sequence_latent_vectors, c) # Shape: (num_slv_keep, latent_dim)
                maintained_vectors_loss += maintained_vectors_loss_part
                
            maintained_vectors_loss /= self.num_classes

            all_maintained_vectors_loss += maintained_vectors_loss

        all_maintained_vectors_loss /= self.nt

        final_loss = all_maintained_vectors_loss

        self.clear_states()
        
        return final_loss, all_maintained_vectors_loss

    def pretrain_meta_latent_vae(self, data):
        self.initialize_states()
        frames, y = data
        # Assume frames is a sequence of nt decomposed images (BS, nt, H, W, num_classes * 3)
        all_total_loss = 0
        seq_mask_loss = 0
        all_seq_lv_loss = 0
        all_kl_loss = 0
        for i in range(self.nt):
            frame = frames[:, i, :, :, :]
            self.update_historic_frames(frame)
            # _, _ = self(frame)
            # Process each classes' historic-frame sequence separately
            total_loss = 0
            mask_loss = 0
            seq_lv_recon_loss = 0
            kl_loss = 0
            for c in range(self.num_classes):
                class_sequence = self.recent_frame_sequences[c] # (BS, num_im_in_seq, H, W, 3)
                labels = tf.one_hot(tf.fill((self.batch_size,), c), self.num_classes) # (BS, num_classes)
                binary_masks, recon_masks, z_mean, z_log_var = self.sVAE([class_sequence, labels])
                total_loss_part_seq, mask_loss_part, kl_loss_part_seq = self.sVAE.compute_loss(binary_masks, recon_masks, z_mean, z_log_var)
                new_class_sequence_latent_vectors = self.sVAE.encode([class_sequence, labels]) # (BS, latent_dim)
                self.seq_latent_maintainer.historic_sequence_latent_vectors_stored[c].assign(
                    tf.concat(
                        (self.seq_latent_maintainer.historic_sequence_latent_vectors_stored[c][1:], new_class_sequence_latent_vectors)
                        , axis=0)) # Update the maintained vectors
                # consolidated_vectors, _ = self.seq_latent_maintainer(new_class_sequence_latent_vectors, c) # Shape: (num_slv_keep, latent_dim)
                sequence_latent_vectors, recon_sequence_latent_vectors, z_mean, z_log_var = self.mlVAE([self.seq_latent_maintainer.historic_sequence_latent_vectors_stored[c], labels])
                total_loss_part_ml, seq_lv_recon_loss_part, kl_loss_part_ml = self.mlVAE.compute_loss(sequence_latent_vectors, recon_sequence_latent_vectors, z_mean, z_log_var)
                
                total_loss += total_loss_part_seq + total_loss_part_ml
                mask_loss += mask_loss_part
                seq_lv_recon_loss += seq_lv_recon_loss_part
                kl_loss += kl_loss_part_seq + kl_loss_part_ml
                
            total_loss /= self.num_classes
            mask_loss /= self.num_classes
            seq_lv_recon_loss /= self.num_classes
            kl_loss /= self.num_classes

            all_total_loss += total_loss
            seq_mask_loss += mask_loss
            all_seq_lv_loss += seq_lv_recon_loss
            all_kl_loss += kl_loss

        all_total_loss /= self.nt
        seq_mask_loss /= self.nt
        all_seq_lv_loss /= self.nt
        all_kl_loss /= self.nt

        final_loss = all_seq_lv_loss + seq_mask_loss + 0.25*all_kl_loss

        self.clear_states()
        
        return final_loss, all_seq_lv_loss, seq_mask_loss, all_kl_loss

    def train_step(self, data):
        pretraining_step = self.training_args.get("pretraining_step", 0)
        # data is a sequence of nt decomposed images (BS, nt, H, W, num_classes * 3)
        # Pre-training the sequence VAE
        if pretraining_step == 0:
            with tf.GradientTape() as tape:
                total_loss, mask_loss, kl_loss = self.pretrain_sequence_vae(data)
                grads = tape.gradient(total_loss, self.sVAE.trainable_weights)
                self.optimizer.apply_gradients(zip(grads, self.sVAE.trainable_weights))
            return {"loss": total_loss, "L_mask": mask_loss, "KL": kl_loss}
        # Pre-training the sequence latent maintainer
        elif pretraining_step == 1:
            with tf.GradientTape() as tape:
                total_loss, maintained_vectors_loss = self.pretrain_sequence_latent_maintainer(data)
                # grads = tape.gradient(total_loss, self.mlVAE.trainable_weights)
                # self.optimizer.apply_gradients(zip(grads, self.mlVAE.trainable_weights))
            return {"loss": total_loss, "MV_loss": maintained_vectors_loss}
        # Pre-training the meta-latent VAE
        elif pretraining_step == 2:
            with tf.GradientTape() as tape:
                total_loss, seq_lv_loss, mask_loss, kl_loss = self.pretrain_meta_latent_vae(data)
                grads = tape.gradient(total_loss, self.mlVAE.trainable_weights)
                self.optimizer.apply_gradients(zip(grads, self.mlVAE.trainable_weights))
            return {"loss": total_loss, "L_seq_lv": seq_lv_loss, "L_mask": mask_loss, "KL": kl_loss}

    def test_step(self, data):
        pretraining_step = self.training_args.get("pretraining_step", 0)
        # data is a sequence of nt decomposed images (BS, nt, H, W, num_classes * 3)
        # Pre-training the sequence VAE
        if pretraining_step == 0:
            total_loss, mask_loss, kl_loss = self.pretrain_sequence_vae(data)
            return {"loss": total_loss, "L_mask": mask_loss, "KL": kl_loss}
        # Pre-training the sequence latent maintainer
        elif pretraining_step == 1:
            total_loss, maintained_vectors_loss = self.pretrain_sequence_latent_maintainer(data)
            return {"loss": total_loss, "MV_loss": maintained_vectors_loss}
        # Pre-training the meta-latent VAE
        elif pretraining_step == 2:
            total_loss, seq_lv_loss, mask_loss, kl_loss = self.pretrain_meta_latent_vae(data)
            return {"loss": total_loss, "L_seq_lv": seq_lv_loss, "L_mask": mask_loss, "KL": kl_loss}

# ###### Pre-train Sequence or Meta-Latent VAEs ######

# # TRAINING STAGES
# # 0. Pre-train the Sequence VAE
# # 1. (Nevermind, train with PredNet) Pre-train the Sequence Latent Maintainer
# # 2. (Nevermind, we don't use this) Pre-train the Meta-Latent VAE

# training_args = {
#     "decompose_images": True, 
#     "second_stage": True,
#     "nt": 10,
#     "batch_size": 1,
#     "SSM_im_shape": [64, 64],
#     "num_classes": 4,
#     "include_frame": False
# }

# latent_dim=32
# num_im_in_seq = 2
# seq_vae_convlstm_channels = 16

# seq_VAE_weights_file = f"/home/evalexii/Documents/Thesis/code/parallel_prednet/model_weights/SSM/multiShape/seq_vae_weights_{latent_dim}_{num_im_in_seq}_{seq_vae_convlstm_channels}.h5"
# seq_maintainer_weights_file = f"/home/evalexii/Documents/Thesis/code/parallel_prednet/model_weights/SSM/multiShape/seq_maint_weights.h5"
# or_decoder_weights_file = f"/home/evalexii/Documents/Thesis/code/parallel_prednet/model_weights/SSM/multiShape/OCPN_wOR_OR_weights_final.h5"

# ### SET PRETRAINING STEP HERE ###
# training_args["pretraining_step"] = 0
# # if training_args["pretraining_step"] == 2: training_args["batch_size"] = 5
# weights_files = [seq_VAE_weights_file, seq_maintainer_weights_file, or_decoder_weights_file]
# load_weights_file = weights_files[training_args["pretraining_step"]-1] if training_args["pretraining_step"] > 0 else weights_files[training_args["pretraining_step"]]
# save_weights_file = weights_files[training_args["pretraining_step"]]

# nt = training_args["nt"]
# batch_size = training_args["batch_size"]
# im_shape = training_args["SSM_im_shape"] + [12] # 3 channels per class, 4 classes

# from data_utils import SequenceDataLoader
# train_dataset, train_size = SequenceDataLoader(training_args, "/home/evalexii/Documents/Thesis/code/parallel_prednet/data/animations/multi_gen_shape_strafing/frames/multi_gen_shape_2nd_stage_train", nt, batch_size, im_shape[0], im_shape[1], im_shape[2], True, training_args["include_frame"]).create_tf_dataset()
# val_dataset, val_size = SequenceDataLoader(training_args, "/home/evalexii/Documents/Thesis/code/parallel_prednet/data/animations/multi_gen_shape_strafing/frames/multi_gen_shape_2nd_stage_val", nt, batch_size, im_shape[0], im_shape[1], im_shape[2], True, training_args["include_frame"]).create_tf_dataset()

# OR_generator = ObjectRepresentations(training_args, latent_dim=latent_dim, num_im_in_seq=num_im_in_seq, seq_vae_convlstm_channels=seq_vae_convlstm_channels, name="Object_Representations")
# OR_generator.compile(optimizer='adam')

# # "Build" the model
# data_sample = next(iter(train_dataset))
# _ = OR_generator(data_sample[0][:,0,...])
# OR_generator.compile(optimizer='adam')

# # Set pre-weight-loading trainability based on pretraining step
# OR_generator.classifier.trainable = False
# if training_args["pretraining_step"] == 0:
#     OR_generator.sVAE.trainable = False
#     OR_generator.seq_latent_maintainer.trainable = True
#     # OR_generator.or_decoder.trainable = True
# elif training_args["pretraining_step"] == 1:
#     OR_generator.sVAE.trainable = True
#     OR_generator.seq_latent_maintainer.trainable = False
#     # OR_generator.or_decoder.trainable = False
# elif training_args["pretraining_step"] == 2:
#     OR_generator.sVAE.trainable = False
#     OR_generator.seq_latent_maintainer.trainable = True
#     # OR_generator.or_decoder.trainable = False
# OR_generator.compile(optimizer='adam')

# # Load model weights
# try: 
#     OR_generator.load_weights(load_weights_file, by_name=True, skip_mismatch=True) 
#     print("Successfully loaded model weights.")
# except Exception as e: 
#     print("Could not load model weights. Starting from scratch but loading classifier weights.")
#     print(e)

#     # Apply classifier weights
#     classifier_weights_file = "/home/evalexii/Documents/Thesis/code/parallel_prednet/model_weights/SSM/multiShape/OCPN_wOR_Classifier_weights.npz"
#     trained_classifier_weights = np.load(os.path.join(classifier_weights_file), allow_pickle=True)
#     trained_classifier_weights = [trained_classifier_weights[key] for key in trained_classifier_weights.keys()]
#     OR_generator.classifier.set_weights(trained_classifier_weights)
#     print("Successfully loaded classifier weights.")

# # Set post-weight-loading trainability based on pretraining step
# OR_generator.classifier.trainable = False
# if training_args["pretraining_step"] == 0:
#     OR_generator.sVAE.trainable = False
#     OR_generator.seq_latent_maintainer.trainable = True
#     # OR_generator.or_decoder.trainable = True
# elif training_args["pretraining_step"] == 1:
#     OR_generator.sVAE.trainable = False
#     OR_generator.seq_latent_maintainer.trainable = True
#     # OR_generator.or_decoder.trainable = False
# elif training_args["pretraining_step"] == 2:
#     OR_generator.sVAE.trainable = False
#     OR_generator.seq_latent_maintainer.trainable = True
#     # OR_generator.or_decoder.trainable = True
# OR_generator.compile(optimizer='adam')

# # # Debug the model
# # tf.config.run_functions_eagerly(True)
# # data_sample = next(iter(train_dataset))
# # _ = OR_generator.train_step(data_sample)

# # Training setup
# epochs = 50
# def lr_schedule(epoch):
#     """
#     Returns a custom learning rate that decreases as epochs progress.
#     """
#     # if epoch < (epochs // 3):
#     #     learning_rate = 0.0005
#     # elif epoch < 2*(epochs // 3):
#     #     learning_rate = 0.0001
#     # else:
#     #     learning_rate = 0.00005
#     # return learning_rate
#     return 0.00001

# # Create checkpoints
# callbacks = [LearningRateScheduler(lr_schedule)]
# callbacks.append(ModelCheckpoint(filepath=save_weights_file, monitor="val_loss", save_best_only=True, save_weights_only=True, verbose=1))

# # Train sequence VAE
# OR_generator.fit(train_dataset, epochs=epochs, validation_data=val_dataset, steps_per_epoch=3, validation_steps=1, callbacks=callbacks)

# ### Scoreboard ###
# # 31-2-16: val_loss = 19019, speed = 230ms/step
# # 64-2-16: val_loss = 19274, speed = 231ms/step