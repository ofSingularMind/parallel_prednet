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
from data_utils import SequenceDataLoader

import pdb


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
    def __init__(self, latent_dim, num_im_in_seq, num_classes, **kwargs):
        super().__init__(**kwargs)

        ###### Start ConvVAE ######
        latent_dim = latent_dim
        input_shape = (64, 64, num_im_in_seq * 3)

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
        binary_masks = self.images_to_masks(images)
        z_mean, z_log_var, z = self.encoder([binary_masks, labels], training=training)
        recon_masks = self.decoder([z, labels], training=training)
        
        return binary_masks, recon_masks, z_mean, z_log_var

    def images_to_masks(self, images):
        # Assume images_concat is your concatenated tensor with shape (h, w, n*c)
        h, w, nc = images.shape
        n = nc // 3  # Calculate n from the shape

        # Reshape the concatenated images to (h, w, n, c)
        images_reshaped = tf.reshape(images, (h, w, n, 3))

        # Compute binary masks for each of the n images
        binary_masks = tf.expand_dims(tf.cast(tf.reduce_any(tf.not_equal(images_reshaped, 0), axis=-1), tf.float32), axis=-1)

        # Reshape the binary masks to (h, w, n*1)
        binary_masks = tf.reshape(binary_masks, (h, w, n))

        return binary_masks

    def encode(self, data, training=True):
        images, labels = data
        binary_masks = self.images_to_masks(images)
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
    
    def train_step(self, data):
        (images, labels), targets = data
        total_loss = 0
        mask_loss = 0
        kl_loss = 0
        with tf.GradientTape() as tape:
            for i in range(4):
                image = images[..., i*3:(i+1)*3]
                label = labels[:, i]
                label = tf.one_hot(tf.cast(label, tf.int32), 4)
                
                binary_mask, recon_mask, z_mean, z_log_var = self([image, label], training=True)
                # binary_mask = tf.expand_dims(tf.cast(tf.reduce_any(tf.not_equal(image, 0), axis=-1), tf.float32), axis=-1)
                total_loss_part, mask_loss, kl_loss_part = self.compute_loss(binary_mask, recon_mask, z_mean, z_log_var)
                
                total_loss += total_loss_part
                mask_loss += mask_loss
                kl_loss += kl_loss_part

            total_loss /= 4
            mask_loss /= 4
            kl_loss /= 4

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return {"L": total_loss, "L_mask": mask_loss, "KL": kl_loss}

    def test_step(self, data):
        (images, labels), targets = data
        total_loss = 0
        mask_loss = 0
        kl_loss = 0
        with tf.GradientTape() as tape:
            for i in range(4):
                image = images[..., i*3:(i+1)*3]
                label = labels[:, i]
                label = tf.one_hot(tf.cast(label, tf.int32), 4)
                
                binary_mask, recon_mask, z_mean, z_log_var = self([image, label], training=True)
                # binary_mask = tf.expand_dims(tf.cast(tf.reduce_any(tf.not_equal(image, 0), axis=-1), tf.float32), axis=-1)
                total_loss_part, mask_loss, kl_loss_part = self.compute_loss(binary_mask, recon_mask, z_mean, z_log_var)
                
                total_loss += total_loss_part
                mask_loss += mask_loss
                kl_loss += kl_loss_part

            total_loss /= 4
            mask_loss /= 4
            kl_loss /= 4

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        
        return {"L": total_loss, "L_mask": mask_loss, "KL": kl_loss}


class SceneDecomposer_pretrainC:
    def __init__(self, n_colors=4, include_frame=False):
        self.n_colors = n_colors
        self.include_frame = include_frame

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
            unique_colors.append((0, 0, 0, 255))

        unique_colors = np.array(unique_colors)

        # Black backgrounds
        masks = [np.full((quantized_image.size[1], quantized_image.size[0], 4), (0, 0, 0, 255), dtype=np.uint8) for _ in range(self.n_colors)]

        color_to_index = {tuple(color): index for index, color in enumerate(unique_colors)}

        for y in range(quantized_image.size[1]):
            for x in range(quantized_image.size[0]):
                pixel = data[x, y]
                masks[color_to_index[tuple(pixel)]][y, x] = pixel

        masks = [mask[..., :3] for mask in masks]

        # randomly order the masks
        np.random.shuffle(masks)

        masks = np.concatenate(masks, axis=-1)

        return masks / 255.0

    def label_masks(self, masks):
        """
        Assign class labels to mask images based on the colors present in the image
        ASSUME: crosses are red, ellipses are green, occlusions are blue, and background is white
        Args:
            masks: np.array with shape (H, W, n_colors*3), range [0, 1]
        Returns:
            np.array with shape (n_colors,), containing the ordered class labels for each mask
        """
        # Extract the colors from the masks
        colors = masks.reshape(-1, self.n_colors, 3)
        colors = np.round(colors * 255).astype(np.uint8)

        # Compute the max color of each mask
        mean_colors = np.max(colors, axis=0)

        # Compute the distance between each mean color and the target colors
        target_colors = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 255]])
        distances = cdist(mean_colors, target_colors)

        # Use the Hungarian algorithm to assign the labels
        row_ind, col_ind = linear_sum_assignment(distances)
        labels = col_ind

        return labels

    def process_and_label_batch(self, batch):
        """
        Process a batch of images with shape (B, 64, 64, 3) and return masks and labels with shapes (B, 64, 64, n_colors*3) and (B, n_colors), respectively
        """
        B, H, W, C = batch.shape
        masks_batch = np.zeros((B, H, W, self.n_colors*C), dtype=np.float32)
        labels_batch = np.zeros((B, self.n_colors), dtype=np.int32)

        for b in range(B):
            masks = self.process_single_image(batch[b])
            labels = self.label_masks(masks)
            masks_batch[b] = masks
            labels_batch[b] = labels
        
        return masks_batch, labels_batch


class BatchDataLoader_pretrainC:
    def __init__(self, training_args, folder_path, batch_size, img_height, img_width, processed_img_channels, shuffle=True):
        self.training_args = training_args
        self.folder_path = folder_path
        self.batch_size = batch_size
        self.img_height = img_height
        self.img_width = img_width
        self.processed_img_channels = processed_img_channels
        self.shuffle = shuffle
        self.img_filenames = sorted([f for f in os.listdir(folder_path) if f.endswith('.png')])
        self.num_images = len(self.img_filenames)
        if self.training_args["decompose_images"]:
            self.sceneDecomposer = SceneDecomposer_pretrainC(n_colors=4)

    def load_image(self, file_path):
        img = Image.open(file_path)
        img_array = np.array(img, dtype=np.float32) / 255.0
        return img_array

    def generate_batch(self):
        all_indices = np.arange(self.num_images)
        if self.shuffle:
            np.random.shuffle(all_indices)
        
        for i in range(0, len(all_indices), self.batch_size):
            batch_images = []
            for j in range(self.batch_size):
                if i + j < len(all_indices):
                    img_path = os.path.join(self.folder_path, self.img_filenames[all_indices[i + j]])
                    img_array = self.load_image(img_path)
                    batch_images.append(img_array)
            if batch_images:
                batch_images = np.stack(batch_images, axis=0)
                if self.training_args["decompose_images"]:
                    decomposed_images, labels = self.sceneDecomposer.process_and_label_batch(batch_images)
                    yield (decomposed_images, labels), decomposed_images

    def create_tf_dataset(self):
        dataset = tf.data.Dataset.from_generator(
            self.generate_batch,
            output_signature=(
                (tf.TensorSpec(shape=(None, self.img_height, self.img_width, self.processed_img_channels * 4), dtype=tf.float32),
                tf.TensorSpec(shape=(None, 4), dtype=tf.float32)),
                tf.TensorSpec(shape=(None, self.img_height, self.img_width, self.processed_img_channels * 4), dtype=tf.float32)
            )
        )
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        dataset = dataset.repeat()
        return dataset, self.num_images


# Example usage
training_args = {"decompose_images": True}
train_path = '/home/evalexii/Documents/Thesis/code/parallel_prednet/data/animations/multi_gen_shape_strafing_pretrain_classifier/frames/multi_gen_shape_2nd_stage_train'
val_path = '/home/evalexii/Documents/Thesis/code/parallel_prednet/data/animations/multi_gen_shape_strafing_pretrain_classifier/frames/multi_gen_shape_2nd_stage_val'
weights_path = '/home/evalexii/Documents/Thesis/code/parallel_prednet/model_weights/SSM/multiShape/OCPN_wOR_ConvVAE_weights_64.npz'
batch_size = 30
img_height = 64
img_width = 64
processed_img_channels = 3

epochs=6
steps_per_epoch=1000

train_dataset, train_size = BatchDataLoader_pretrainC(training_args, train_path, batch_size, img_height, img_width, processed_img_channels).create_tf_dataset()
val_dataset, val_size = BatchDataLoader_pretrainC(training_args, val_path, batch_size, img_height, img_width, processed_img_channels).create_tf_dataset()

train_dataset, train_size = SequenceDataLoader({"decompose_images": True, "second_stage": True}, DATA_DIR + f"multi_gen_shape_strafing/frames/multi_gen_shape_{stage}_train", nt, batch_size, im_shape[0], im_shape[1], im_shape[2], True, args["include_frame"]).create_tf_dataset()
val_dataset, val_size = SequenceDataLoader({"decompose_images": True, "second_stage": True}, DATA_DIR + f"multi_gen_shape_strafing/frames/multi_gen_shape_{stage}_val", nt, batch_size, im_shape[0], im_shape[1], im_shape[2], True, args["include_frame"]).create_tf_dataset()
        

# Instantiate the model
vae = SequenceVAE(latent_dim=64, num_im_in_seq=3, num_classes=4)
vae.compile(optimizer=keras.optimizers.Adam())

if os.path.exists(weights_path):
    try:
        # custom_objects = {'dummy_model': dummy_model}
        # trained_classifier_weights = keras.models.load_model(os.path.join(WEIGHTS_DIR, classifier_weights_file_name), custom_objects=custom_objects).layers[1].layers[0].classifier.get_weights()
        trained_vae_weights = np.load(weights_path, allow_pickle=True)
        trained_vae_weights = [trained_vae_weights[key] for key in trained_vae_weights.keys()]
        vae.set_weights(trained_vae_weights)
        print("Pre-trained VAE weights loaded successfully")
    except Exception as e:
        print("Error loading pre-trained VAE weights: ", e)
        print("Training VAE from scratch")

# if True:
#     # Save the model weights
#     np.savez('/home/evalexii/Documents/Thesis/code/parallel_prednet/model_weights/SSM/multiShape/OCPN_wOR_ConvVAE_weights.npz', *model.weights)
#     exit()

examine = False  
###################################
##########DEBUG MODE###############
###################################
if examine:
    tf.config.run_functions_eagerly(True)
    # Create a TensorFlow function for optimized execution
    @tf.function
    def debug_step(x, y):
        loss, mask_loss, kl_loss = vae.train_step((x, y))
        print("Loss: ", loss, "Mask Loss: ", mask_loss, "KL Loss: ", kl_loss)

    # Iterate over the dataset
    for x, y in train_dataset:
        debug_step(x, y)
###################################
##########END DEBUG MODE###########
###################################

else:

    def lr_schedule(epoch):
        """
        Returns a custom learning rate that decreases as epochs progress.
        """
        if epoch < (epochs // 3):
            learning_rate = 0.001
        elif epoch < 2*(epochs // 3):
            learning_rate = 0.0005
        else:
            learning_rate = 0.0001
        return learning_rate

    # Train the model
    callbacks = [LearningRateScheduler(lr_schedule)]
    # callbacks.append(ModelCheckpoint(filepath=weights_path, monitor="val_loss", save_best_only=True, save_weights_only=False, verbose=1))
    vae.fit(train_dataset, batch_size=batch_size, epochs=epochs, steps_per_epoch=steps_per_epoch, validation_data=val_dataset, validation_steps=50, callbacks=callbacks)

    np.savez(weights_path, *vae.weights)
    print("Saved VAE weights")