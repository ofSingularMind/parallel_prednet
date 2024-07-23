from keras import layers
from keras import backend as K
import keras
import keras_cv
import tensorflow as tf
import numpy as np
import os
import warnings
import matplotlib.pyplot as plt

from keras.applications import MobileNetV2
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
from keras.layers import ConvLSTM2D

# Suppress warnings
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

class CustomConvLSTM2D(keras.layers.Layer):
    def __init__(self, output_channels, layer_num, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Add ConvLSTM, being sure to pass previous states in OR use stateful=True
        conv_i = layers.Conv2D(output_channels, (3, 3), padding="same", activation="hard_sigmoid", name=f"CustomConvLSTM2D_Conv_i_Layer{layer_num}")
        conv_f = layers.Conv2D(output_channels, (3, 3), padding="same", activation="hard_sigmoid", name=f"CustomConvLSTM2D_Conv_f_Layer{layer_num}")
        conv_o = layers.Conv2D(output_channels, (3, 3), padding="same", activation="hard_sigmoid", name=f"CustomConvLSTM2D_Conv_o_Layer{layer_num}")
        conv_c = layers.Conv2D(output_channels, (3, 3), padding="same", activation="tanh", name=f"CustomConvLSTM2D_Conv_c_Layer{layer_num}")
        self.convs = {"conv_i": conv_i, "conv_f": conv_f, "conv_o": conv_o, "conv_c": conv_c}

    def call(self, inputs, initial_states=None):
        i = self.convs["conv_i"](inputs)
        f = self.convs["conv_f"](inputs)
        o = self.convs["conv_o"](inputs)
        h, c = initial_states if initial_states is not None else 2 * [tf.zeros(f.shape, dtype=tf.float32)]
        c = f * c + i * self.convs["conv_c"](inputs)
        h = o * keras.activations.tanh(c)
        output = h
        states = [h, c]

        return output, *states

class MultiClassStatefulConvLSTM2D(tf.keras.layers.Layer):
    def __init__(self, output_channels=32, num_classes=4, **kwargs):
        super().__init__(**kwargs)
        self.output_channels = output_channels
        self.num_classes = num_classes
        # self.conv_lstm = ConvLSTM2D(filters=output_channels, kernel_size=(3, 3), padding='same', return_sequences=False, return_state=True)
        self.conv_lstm = CustomConvLSTM2D(output_channels=output_channels, layer_num=0)

    def build(self, input_shape):
        state_shape = (input_shape[0], input_shape[1], input_shape[2], self.output_channels)
        self.states_h = self.add_weight(shape=(self.num_classes, *state_shape), initializer='zeros', trainable=False, name='states_h')
        self.states_c = self.add_weight(shape=(self.num_classes, *state_shape), initializer='zeros', trainable=False, name='states_c')

    def call(self, inputs, class_ID=0):
        current_states = [tf.gather(self.states_h, class_ID), tf.gather(self.states_c, class_ID)]

        # Debugging: Print shapes to verify
        # print("Shape of current_state_h:", current_states[0].shape)
        # print("Shape of current_state_c:", current_states[1].shape)

        # outputs, state_h, state_c = self.conv_lstm(inputs, initial_state=current_states)
        outputs, state_h, state_c = self.conv_lstm(inputs, initial_states=current_states)

        # Debugging: Print shapes to verify
        # print("Shape of output_state_h:", state_h.shape)
        # print("Shape of output_state_c:", state_c.shape)
        self.states_h = tf.tensor_scatter_nd_update(self.states_h, [[class_ID]], [state_h])
        self.states_c = tf.tensor_scatter_nd_update(self.states_c, [[class_ID]], [state_c])

        return outputs

    def reset_states(self):
        self.states_h.assign(tf.zeros_like(self.states_h))
        self.states_c.assign(tf.zeros_like(self.states_c))

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
        self.predictions = Dense(self.num_classes, activation='softmax')
    
    def compute_output_shape(self, input_shape):
        return (self.num_classes,)

    def call(self, inputs):
        x = self.base_model(inputs)
        x = self.global_avg_pool(x)
        x = self.dense_1024(x)
        return self.predictions(x)

class ObjectRepresentation(layers.Layer):
    '''
    Convert images of object masks to class IDs, then update and extract the corresponding object representations
    '''
    def __init__(self, training_args, num_classes, layer_num, im_height, im_width, **kwargs):
        super(ObjectRepresentation, self).__init__(**kwargs)
        self.training_args = training_args
        self.layer_num = layer_num
        self.im_height = im_height
        self.im_width = im_width
        self.num_classes = num_classes
        self.batch_size = training_args['batch_size']
        self.frame_channels = training_args['output_channels'][0]
        self.classifier = CustomMobileNetV2(num_classes=4, input_shape=(self.im_height, self.im_width, 3))

        self.general_object_tensor = tf.random.normal((1, self.im_height, self.im_width, self.frame_channels))
        self.class_tensors = tf.stack([tf.random.normal((1, self.im_height, self.im_width, self.frame_channels)) for _ in range(num_classes)])

        self.conv_lstm_general = MultiClassStatefulConvLSTM2D(output_channels=self.frame_channels, num_classes=1, name=f"ObjectRepresentation_ConvLSTM_General_Layer{self.layer_num}")
        self.conv_lstm_class = MultiClassStatefulConvLSTM2D(output_channels=self.frame_channels, num_classes=4, name=f"ObjectRepresentation_ConvLSTM_Class_Layer{self.layer_num}")
        self.conv_lstm_general.build((self.batch_size, self.im_height, self.im_width, self.num_classes * self.frame_channels))
        self.conv_lstm_class.build((self.batch_size, self.im_height, self.im_width, self.frame_channels + 3))


    def compute_output_shape(self, input_shape):
        return (self.im_height, self.im_width, self.frame_channels * self.num_classes)
    
    def call(self, inputs):
        output_tensors = []

        for i in range(0, self.frame_channels, 3):
            frame = inputs[..., i:i+3]
            class_probs = self.classifier(frame)
            class_label = tf.math.argmax(class_probs, axis=-1)
            class_label = tf.squeeze(class_label, axis=None) # Ensure class_label is a scalar if possible

            # Update class tensor using ConvLSTM over general object tensor and new frame
            # concatenated_input = tf.expand_dims(tf.concat([self.general_object_tensor, frame], axis=-1), axis=1)
            concatenated_input = tf.concat([self.general_object_tensor, frame], axis=-1)
            updated_class_tensor = self.conv_lstm_class(inputs=concatenated_input, class_ID=class_label)
            # updated_class_tensor = tf.squeeze(updated_class_tensor, axis=1)
            tf.tensor_scatter_nd_update(self.class_tensors, [[class_label]], [updated_class_tensor])
            
            output_tensors.append(updated_class_tensor)
            # TODO: If the same class is predicted twice in the same frame, it will be upated twice. Is this OK? Maybe they are similar enough.

        all_class_tensors = tf.reshape(self.class_tensors, (1, self.im_height, self.im_width, self.num_classes * self.frame_channels))
        # concatenated_classes = tf.expand_dims(all_class_tensors, axis=1)
        concatenated_classes = all_class_tensors
        updated_general_object_tensor = self.conv_lstm_general(inputs=concatenated_classes, class_ID=0)
        # self.general_object_tensor.assign(tf.squeeze(updated_general_object_tensor, axis=1))
        # tf.tensor_scatter_nd_update(self.general_object_tensor, [], [updated_general_object_tensor])
        self.general_object_tensor = updated_general_object_tensor

        out = tf.concat(output_tensors, axis=-1)
        shape = (None, self.im_height, self.im_width, self.training_args['output_channels'][0]*self.num_classes)
        out.set_shape(shape)
        
        return out



# Create input layer
input_layer = layers.Input(shape=(64, 64, 12))

# Create ObjectRepresentation layer
object_representation = ObjectRepresentation(training_args={'output_channels': [12], 'batch_size': 1}, num_classes=4, layer_num=1, im_height=64, im_width=64)

output = object_representation(input_layer)

model = Model(inputs=input_layer, outputs=output)

model.summary()

# Compile the model
model.compile(optimizer=keras.optimizers.Adam(), loss=keras.losses.mean_squared_error, metrics=["accuracy"])


# Create random test dataset for keras.model.fit()
num_samples = 100
input_shape = (64, 64, 12)
num_classes = 4

# Generate random inputs
x_train = np.random.random((num_samples, *input_shape)).astype(np.float32)

# Generate random outputs (fake target)
y_train = np.random.random((num_samples, 64, 64, num_classes * 12)).astype(np.float32)

from tensorflow.keras.callbacks import Callback

class PrintBatchNumberCallback(Callback):
    def on_train_batch_end(self, batch, logs=None):
        print(f"End of batch {batch}, Loss: {logs['loss']:.4f}")

class PrintEpochNumberCallback(Callback):
    def on_epoch_begin(self, epoch, logs=None):
        print(f"Starting Epoch {epoch+1}")

    def on_epoch_end(self, epoch, logs=None):
        print(f"End of Epoch {epoch+1}")

callbacks = [PrintBatchNumberCallback(), PrintEpochNumberCallback()]

# Train the model using the fake dataset
model.fit(x_train, y_train, epochs=5, batch_size=1, callbacks=callbacks)