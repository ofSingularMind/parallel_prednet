import numpy as np
import os
import warnings
import matplotlib.pyplot as plt

import numpy as np
from PIL import Image
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from keras.layers import Dense, GlobalAveragePooling2D, ConvLSTM2D, BatchNormalization, Dropout, Conv2D, MaxPooling2D, Flatten
from keras.applications import MobileNetV2
from keras.callbacks import LearningRateScheduler, ModelCheckpoint, TensorBoard


import pdb

# Suppress warnings
warnings.filterwarnings("ignore")
# or '2' to filter out INFO messages too
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
# from tf.keras.saving import register_keras_serializable

# Enable eager execution mode before creating the dataset
tf.config.run_functions_eagerly(True)


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
                    yield decomposed_images, labels

    def create_tf_dataset(self):
        dataset = tf.data.Dataset.from_generator(
            self.generate_batch,
            output_signature=(
                tf.TensorSpec(shape=(None, self.img_height, self.img_width, self.processed_img_channels * 4), dtype=tf.float32),
                tf.TensorSpec(shape=(None, 4), dtype=tf.int32)
            )
        )
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        dataset = dataset.repeat()
        return dataset, self.num_images

class CustomCNN(tf.keras.layers.Layer):
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

class dummy_layer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(dummy_layer, self).__init__(**kwargs)
        # self.classifier = CustomMobileNetV2(num_classes=4, input_shape=(64, 64, 3), name='classifier')
        self.classifier = CustomCNN(num_classes=4, name='classifier')
    
    def call(self, inputs):
        outs = []
        for i in range(4):
            out = self.classifier(inputs[..., i*3:(i+1)*3]) # Shape: (B, 4)
            outs.append(out)
        return tf.stack(outs, axis=1)

# @register_keras_serializable()
class dummy_model(tf.keras.Model):
    def __init__(self, **kwargs):
        super(dummy_model, self).__init__(**kwargs)
        self.dummy_layer = dummy_layer(name="ObjectRepresentation_Layer0")
    
    def call(self, inputs):
        return self.dummy_layer(inputs)



# Example usage
training_args = {"decompose_images": True}
train_path = '/home/evalexii/Documents/Thesis/code/parallel_prednet/data/animations/multi_gen_shape_strafing_pretrain_classifier/frames/multi_gen_shape_2nd_stage_train'
val_path = '/home/evalexii/Documents/Thesis/code/parallel_prednet/data/animations/multi_gen_shape_strafing_pretrain_classifier/frames/multi_gen_shape_2nd_stage_val'
weights_path = '/home/evalexii/Documents/Thesis/code/parallel_prednet/model_weights/SSM/multiShape/OCPN_wOR_Classifier_weights.hdf5'
batch_size = 30
img_height = 64
img_width = 64
processed_img_channels = 3

train_dataset, train_size = BatchDataLoader_pretrainC(training_args, train_path, batch_size, img_height, img_width, processed_img_channels).create_tf_dataset()
val_dataset, val_size = BatchDataLoader_pretrainC(training_args, val_path, batch_size, img_height, img_width, processed_img_channels).create_tf_dataset()

# Create the model
input = tf.keras.Input(shape=(img_height, img_width, processed_img_channels * 4))
output = dummy_model(name="PredLayer0")(input)

model = tf.keras.Model(inputs=input, outputs=output)
model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

if os.path.exists(weights_path):
    try:
        model.load_weights(weights_path)
        print("Loaded classifier weights")
    except:
        print("Failed to load classifier weights")

# if True:
#     # Save the model weights
#     np.savez('/home/evalexii/Documents/Thesis/code/parallel_prednet/model_weights/SSM/multiShape/OCPN_wOR_Classifier_weights.npz', *model.weights)
#     exit()

examine = False  
###################################
##########DEBUG MODE###############
###################################
if examine:
    # Create a TensorFlow function for optimized execution
    @tf.function
    def debug_step(x, y):
        with tf.GradientTape() as tape:
            predictions = model(x, training=False)  # Get model predictions
            print(tf.argmax(predictions, axis=-1))
            print(y)
            # input("Press Enter to continue...")

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
        learning_rate = 0.01
        if epoch > 2:
            learning_rate = 0.005
        if epoch > 4:
            learning_rate = 0.001
        if epoch > 6:
            learning_rate = 0.0005
        return learning_rate

    # Train the model
    callbacks = [LearningRateScheduler(lr_schedule)]
    callbacks.append(ModelCheckpoint(filepath=weights_path, monitor="val_loss", save_best_only=True, save_weights_only=False, verbose=1))
    model.fit(train_dataset, batch_size=batch_size, epochs=10, steps_per_epoch=60, validation_data=val_dataset, validation_steps=20, callbacks=callbacks)

    np.savez('/home/evalexii/Documents/Thesis/code/parallel_prednet/model_weights/SSM/multiShape/OCPN_wOR_Classifier_weights.npz', *model.weights)
    print("Saved classifier weights")