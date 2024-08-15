import os
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")
# or '2' to filter out INFO messages too
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
import matplotlib.pyplot as plt
from keras import layers
from keras import backend as K
import keras
import numpy as np
import numpy as np
from PIL import Image
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

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

        encoder_inputs = keras.Input(shape=input_shape)
        label_inputs = keras.Input(shape=(num_classes,))
        x = layers.Conv2D(32, 7, activation="relu", strides=2, padding="same")(encoder_inputs)
        # x = layers.Dropout(0.2)(x)
        x = layers.Conv2D(32, 5, activation="relu", strides=2, padding="same")(x)
        # x = layers.Dropout(0.2)(x)
        x = layers.Conv2D(64, 7, activation="relu", strides=2, padding="same")(x)
        # x = layers.Dropout(0.2)(x)
        x = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)
        # x = layers.Dropout(0.2)(x)

        x = layers.Flatten()(x)
        x = layers.Concatenate()([x, label_inputs])
        x = layers.Dense(128, activation="relu")(x)
        x = layers.Dense(latent_dim, activation="relu")(x)
        
        z_mean = layers.Dense(latent_dim, name="z_mean")(x)
        z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
        z = KerasSampling()([z_mean, z_log_var])
        encoder = keras.Model([encoder_inputs, label_inputs], [z_mean, z_log_var, z], name="encoder")

        latent_inputs = keras.Input(shape=(latent_dim,))
        x = layers.Concatenate()([latent_inputs, label_inputs])
        x = layers.Dense(8 * 8 * 128, activation="relu")(x)
        x = layers.Reshape((8, 8, 128))(x)
        x = layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
        # x = layers.Dropout(0.2)(x)
        # x = layers.Concatenate()([layers.UpSampling2D()(x), y])
        x = layers.Conv2DTranspose(48, 3, activation="relu", strides=2, padding="same")(x)
        # x = layers.Dropout(0.2)(x)
        # x = layers.Concatenate()([layers.UpSampling2D()(x), y])
        x = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)
        # x = layers.Dropout(0.2)(x)
        # x = layers.Concatenate()([layers.UpSampling2D()(x), y])
        decoder_outputs = layers.Conv2DTranspose(input_shape[-1], 3, activation="sigmoid", padding="same")(x)
        # threshold at 0.5
        # decoder_outputs = tf.cast(decoder_outputs > 0.5, tf.float32)
        decoder = keras.Model([latent_inputs, label_inputs], decoder_outputs, name="decoder")

        x = layers.Concatenate()([latent_inputs, label_inputs])
        x = layers.Dense(8 * 8 * 64, activation="relu")(x)
        x = layers.Reshape((8, 8, 64))(x)
        y = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)
        y = layers.Dropout(0.2)(y)
        x = layers.Concatenate()([layers.UpSampling2D()(x), y])
        y = layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
        y = layers.Dropout(0.2)(y)
        x = layers.Concatenate()([layers.UpSampling2D()(x), y])
        y = layers.Conv2DTranspose(16, 3, activation="relu", strides=2, padding="same")(x)
        y = layers.Dropout(0.2)(y)
        x = layers.Concatenate()([layers.UpSampling2D()(x), y])
        object_rep_outputs = layers.Conv2DTranspose(input_shape[-1], 3, activation="sigmoid", padding="same")(y)
        object_rep = keras.Model([latent_inputs, label_inputs], object_rep_outputs, name="object_rep")
        ###### End ConvVAE ######

        # decoder_net = make_sequential_cnn(
        #     input_channels=latent_dim + 2 + num_classes,  # Assuming latent size + 2 for coordinate channels and num_classes for class ID
        #     channels=[32, 32, 64, 64, input_shape[-1]],
        #     kernels=[3, 3, 3, 3, 1],
        #     paddings=[0, 0, 0, 0, 0],
        #     activations=['relu', 'relu', 'relu', 'relu', 'sigmoid'],
        #     batchnorms=[True, True, True, True, False]
        # )

        # decoder_net = make_sequential_cnn(
        #     input_channels=latent_dim + 2 + num_classes,  # Assuming latent size + 2 for coordinate channels and num_classes for class ID
        #     channels=[128, 64, 32, 16, input_shape[-1]],
        #     kernels=[3, 3, 3, 3, 1],
        #     paddings=[0, 0, 0, 0, 0],
        #     activations=['relu', 'relu', 'relu', 'relu', 'sigmoid'],
        #     batchnorms=[True, True, True, True, False]
        # )

        # object_rep_net = make_sequential_cnn(
        #     input_channels=latent_dim + 2 + num_classes,  # Assuming latent size + 2 for coordinate channels and num_classes for class ID
        #     channels=[32, 32, 64, 64, output_channels],
        #     kernels=[3, 3, 3, 3, 1],
        #     paddings=[0, 0, 0, 0, 0],
        #     activations=['relu', 'relu', 'relu', 'relu', 'sigmoid'],
        #     batchnorms=[True, True, True, True, False]
        # )

        # decoder = BroadcastDecoderNet(w_broadcast=64, h_broadcast=64, net=decoder_net, name="decoder")
        # object_rep = BroadcastDecoderNet(w_broadcast=64, h_broadcast=64, net=object_rep_net)

        self.encoder = encoder
        self.decoder = decoder
        self.object_rep = object_rep

    # def call(self, data, training=False):
    #     # For reconstruction and mask together
    #     images, labels = data
    #     binary_masks = tf.expand_dims(tf.cast(tf.reduce_any(tf.not_equal(images, 0), axis=-1), tf.float32), axis=-1)
    #     images_and_masks = tf.concat([images, binary_masks], axis=-1)
    #     z_mean, z_log_var, z = self.encoder([images_and_masks, labels], training=training)
    #     reconstructed_images_and_masks = self.decoder([z, labels], training=training)
    #     object_rep = self.object_rep([z, labels], training=training)
    #     recon_im, recon_mask = tf.split(reconstructed_images_and_masks, [3, 1], axis=-1)
    #     reconstruction = recon_im * recon_mask

    #     if training:
    #         return reconstruction, recon_mask, z_mean, z_log_var
    #     else:
    #         return object_rep

    def call(self, data, training=False):
        images, labels = data
        binary_masks = tf.expand_dims(tf.cast(tf.reduce_any(tf.not_equal(images, 0), axis=-1), tf.float32), axis=-1)
        # add a bit of noise to the binary masks
        # binary_masks = tf.clip_by_value(binary_masks + tf.random.normal(tf.shape(binary_masks), mean=0.0, stddev=0.1), 0.0, 1.0)
        z_mean, z_log_var, z = self.encoder([binary_masks, labels], training=training)

        masks = self.decoder([z, labels], training=training)
        object_rep = self.object_rep([z, labels], training=training)
        if True:
            return masks, z_mean, z_log_var
        else:
            return object_rep

    def encode(self, data):
        images, labels = data
        binary_masks = tf.expand_dims(tf.cast(tf.reduce_any(tf.not_equal(images, 0), axis=-1), tf.float32), axis=-1)
        z_mean, z_log_var, z = self.encoder([binary_masks, labels], training=False)
        return z

    def compute_loss(self, images, masks, reconstruction, recon_masks, z_mean, z_log_var):
        tf.debugging.check_numerics(masks, "NaN or Inf found in masks")
        tf.debugging.check_numerics(recon_masks, "NaN or Inf found in recon_masks")
        tf.debugging.check_numerics(z_mean, "NaN or Inf found in z_mean")
        tf.debugging.check_numerics(z_log_var, "NaN or Inf found in z_log_var")
        # image_reconstruction_loss = tf.reduce_mean(
        #     tf.reduce_sum(
        #         keras.losses.binary_crossentropy(images, reconstruction),
        #         axis=(1, 2),
        #     )
        # ) * 64 * 64 * 3

        mask_reconstruction_loss = tf.reduce_mean(
            tf.reduce_sum(
                keras.losses.binary_crossentropy(masks, recon_masks),
                axis=(1, 2),
            )
        ) * 64 * 64 * 1
        # reconstruction_loss = 0.1*image_reconstruction_loss + mask_reconstruction_loss
        
        kl_loss = -0.5 * (1 + z_log_var - tf.math.square(z_mean) - tf.math.exp(z_log_var))
        kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
        
        total_loss = mask_reconstruction_loss + 0.25*kl_loss
        
        return total_loss, mask_reconstruction_loss, kl_loss

    def train_step(self, data):
        (images, labels), targets = data
        total_loss = 0
        # reconstruction_loss = 0
        mask_loss = 0
        kl_loss = 0
        with tf.GradientTape() as tape:
            for i in range(4):
                image = images[..., i*3:(i+1)*3]
                label = labels[:, i]
                label = tf.one_hot(tf.cast(label, tf.int32), 4)
                
                recon_mask, z_mean, z_log_var = self([image, label], training=True)
                binary_mask = tf.expand_dims(tf.cast(tf.reduce_any(tf.not_equal(image, 0), axis=-1), tf.float32), axis=-1)
                total_loss_part, mask_loss, kl_loss_part = self.compute_loss(None, binary_mask, None, recon_mask, z_mean, z_log_var)
                
                total_loss += total_loss_part
                # reconstruction_loss += rec_loss
                mask_loss += mask_loss
                kl_loss += kl_loss_part

            total_loss /= 4
            # reconstruction_loss /= 4
            mask_loss /= 4
            kl_loss /= 4

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return {"L": total_loss, "L_mask": mask_loss, "KL": kl_loss}


    def test_step(self, data):
        (images, labels), targets = data
        total_loss = 0
        # reconstruction_loss = 0
        mask_loss = 0
        kl_loss = 0
        with tf.GradientTape() as tape:
            for i in range(4):
                image = images[..., i*3:(i+1)*3]
                label = labels[:, i]
                label = tf.one_hot(tf.cast(label, tf.int32), 4)
                
                recon_mask, z_mean, z_log_var = self([image, label], training=True)
                binary_mask = tf.expand_dims(tf.cast(tf.reduce_any(tf.not_equal(image, 0), axis=-1), tf.float32), axis=-1)
                total_loss_part, mask_loss, kl_loss_part = self.compute_loss(None, binary_mask, None, recon_mask, z_mean, z_log_var)
                
                total_loss += total_loss_part
                # reconstruction_loss += rec_loss
                mask_loss += mask_loss
                kl_loss += kl_loss_part

            total_loss /= 4
            # reconstruction_loss /= 4
            mask_loss /= 4
            kl_loss /= 4

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return {"L": total_loss, "L_mask": mask_loss, "KL": kl_loss}

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
        self.set_trainable(trainable)

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

class CustomLSTM(layers.Layer):
    def __init__(self, output_units, layer_num=0, *args, **kwargs):
        super(CustomLSTM, self).__init__(*args, **kwargs)
        self.output_units = output_units
        self.layer_num = layer_num
        self.lstm_cell = layers.LSTM(output_units, return_state=True, return_sequences=True, name=f"LSTM_Layer{layer_num}")

    def call(self, inputs, initial_state=None):
        # Assuming inputs shape is (batch_size, time_steps, features)
        if initial_state is not None:
            h, c = initial_state
        else:
            h = tf.zeros((inputs.shape[0], self.output_units), dtype=tf.float32)
            c = tf.zeros((inputs.shape[0], self.output_units), dtype=tf.float32)
        
        output, h, c = self.lstm_cell(inputs, initial_state=[h, c])
        return output, h, c

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
        self.classifier = CustomCNN(num_classes=num_classes, num_conv_layers=3, trainable=False, name='classifier')
        self.latent_LSTM = CustomLSTM(output_units=self.latent_dim, layer_num=0, name='latent_LSTM')

        # Initialize states
        self.class_object_rep_states = [self.add_weight(shape=(num_classes, self.latent_dim), initializer='zeros', trainable=False, name='class_object_rep_state_h'), self.add_weight(shape=(num_classes, self.latent_dim), initializer='zeros', trainable=False, name='class_eobject_rep_state_c')]

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
        current_class_states = [self.diff_gather(self.class_object_rep_states[0], class_logits), self.diff_gather(self.class_object_rep_states[1], class_logits)] # (bs, h, w, oc) via broadcasting

        # encode frame and class logits into latent vector
        z = self.conv_vae_class.encode([frame, class_logits])

        # Compute the loss from last prediction
        lstm_loss = tf.reduce_mean(tf.square(z - current_class_states[0])) if tf.reduce_sum(current_class_states[0]) > 0 else tf.constant(0.0, shape=()) 

        # Apply latent LSTM to get the predicted next-frame latent states
        _, new_latent_state_h, new_latent_state_c = self.latent_LSTM(z, initial_state=current_class_states) # (bs, latent_dim) x 3

        # update the class states
        class_state_updates = [self.diff_scatter_nd_update(self.class_object_rep_states[0], new_latent_state_h, class_logits), self.diff_scatter_nd_update(self.class_object_rep_states[1], new_latent_state_c, class_logits)] # (bs, h, w, oc) via broadcasting
        
        # Also update the permanent copies.
        [self.class_object_rep_states[0].assign(class_state_updates[0]), self.class_object_rep_states[1].assign(class_state_updates[1])]

        # Decode predicted latent vector into class object representation
        class_object_rep = self.conv_vae_class.decoder([new_latent_state_h, class_logits], training=False)

        return class_object_rep, lstm_loss


    def process_null_frame(self,):
        # If the input frame is empty, set the class output to zero
        class_object_rep = tf.zeros((self.batch_size, self.im_height, self.im_width, self.output_channels), dtype=tf.float32)
        loss = tf.constant(0.0, shape=())

        return class_object_rep, loss
    

    @tf.function
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
            class_object_rep, loss = tf.cond(tf.reduce_sum(frame) > 0, lambda: self.process_valid_frame(frame, 0), lambda: self.process_null_frame())
            output_class_tensors.append(class_object_rep)
            all_losses.append(loss)
            

        # stack the class outputs to get the final output
        output = tf.concat(output_class_tensors, axis=-1)
        # print("Output shape:", output.shape)
        assert (output.shape == (self.batch_size, self.im_height, self.im_width, self.num_classes*self.output_channels) or output.shape == (None, self.im_height, self.im_width, self.num_classes*self.output_channels))

        # return the class-specific object representations (hidden class states)
        return output, tf.reduce_mean(all_losses)

# Example usage
training_args = {"decompose_images": True}
train_path = '/home/evalexii/Documents/Thesis/code/parallel_prednet/data/animations/multi_gen_shape_strafing_pretrain_classifier/frames/multi_gen_shape_2nd_stage_train'
val_path = '/home/evalexii/Documents/Thesis/code/parallel_prednet/data/animations/multi_gen_shape_strafing_pretrain_classifier/frames/multi_gen_shape_2nd_stage_val'
weights_path = '/home/evalexii/Documents/Thesis/code/parallel_prednet/model_weights/SSM/multiShape/OCPN_wOR_ConvVAE_weights.npz'
batch_size = 30
img_height = 64
img_width = 64
processed_img_channels = 3

epochs=6
steps_per_epoch=1000

train_dataset, train_size = BatchDataLoader_pretrainC(training_args, train_path, batch_size, img_height, img_width, processed_img_channels).create_tf_dataset()
val_dataset, val_size = BatchDataLoader_pretrainC(training_args, val_path, batch_size, img_height, img_width, processed_img_channels).create_tf_dataset()

# Instantiate the model
vae = KerasVAE(output_channels=12, num_classes=4)
vae.compile(optimizer=keras.optimizers.Adam())

if os.path.exists(weights_path):
    try:
        # custom_objects = {'dummy_model': dummy_model}
        # trained_classifier_weights = keras.models.load_model(os.path.join(WEIGHTS_DIR, classifier_weights_file_name), custom_objects=custom_objects).layers[1].layers[0].classifier.get_weights()
        trained_vae_weights = np.load(weights_path, allow_pickle=True)
        trained_vae_weights = [trained_vae_weights[key] for key in trained_vae_weights.keys()]
        un_trained_classifier = vae
        un_trained_classifier.set_weights(trained_vae_weights)
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
            learning_rate = 0.0001
        elif epoch < 2*(epochs // 3):
            learning_rate = 0.0001
        else:
            learning_rate = 0.0001
        return learning_rate

    # Train the model
    callbacks = [LearningRateScheduler(lr_schedule)]
    # callbacks.append(ModelCheckpoint(filepath=weights_path, monitor="val_loss", save_best_only=True, save_weights_only=False, verbose=1))
    vae.fit(train_dataset, batch_size=batch_size, epochs=epochs, steps_per_epoch=steps_per_epoch, validation_data=val_dataset, validation_steps=50, callbacks=callbacks)

    np.savez('/home/evalexii/Documents/Thesis/code/parallel_prednet/model_weights/SSM/multiShape/OCPN_wOR_ConvVAE_weights.npz', *vae.weights)
    print("Saved VAE weights")