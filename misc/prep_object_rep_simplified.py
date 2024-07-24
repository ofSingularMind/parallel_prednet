import os
import warnings
# Suppress warnings
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from keras import layers
import keras
import tensorflow as tf
import numpy as np
from keras.models import Model
from keras.callbacks import ModelCheckpoint

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
        self.conv_lstm_general = layers.ConvLSTM2D(filters=output_channels, kernel_size=(3, 3), padding='same', return_sequences=False, return_state=True, stateful=False, name='conv_lstm_general')
        self.conv_lstm_class = layers.ConvLSTM2D(filters=output_channels, kernel_size=(3, 3), padding='same', return_sequences=False, return_state=True, stateful=False, name='conv_lstm_class')

        self.class_states_h = self.add_weight(shape=(num_classes, batch_size, im_height, im_width, output_channels), initializer='zeros', trainable=False, name='class_state_h')
        self.class_states_c = self.add_weight(shape=(num_classes, batch_size, im_height, im_width, output_channels), initializer='zeros', trainable=False, name='class_state_c')
        self.general_states_h = self.add_weight(shape=(1, batch_size, im_height, im_width, output_channels), initializer='zeros', trainable=False, name='general_state_h')
        self.general_states_c = self.add_weight(shape=(1, batch_size, im_height, im_width, output_channels), initializer='zeros', trainable=False, name='general_state_c')

    @tf.function
    def call(self, inputs):
        # Inputs shape: (bs, 1, h, w, oc)

        '''Update General Object States'''
        # get current general object states
        current_general_states = [self.general_states_h[0], self.general_states_c[0]] # (bs, h, w, oc) x 2

        # get all current class object representations (class object hidden states)
        all_class_object_representations = self.class_states_h # (nc, bs, h, w, oc)

        # reshape the class object representations to (nc, bs, h, w, oc) -> (bs, 1, h, w, nc*oc) for processing in ConvLSTM
        all_class_object_representations = tf.reshape(all_class_object_representations, (self.batch_size, 1, self.im_height, self.im_width, self.num_classes*self.output_channels))

        # apply the general ConvLSTM layer to get the updated general states
        _, new_general_state_h, new_general_state_c = self.conv_lstm_general(all_class_object_representations, initial_state=current_general_states) # (bs, h, w, oc) x 3

        # update the general states
        general_update_h = tf.tensor_scatter_nd_update(self.general_states_h, [[0]], [new_general_state_h])
        general_update_c = tf.tensor_scatter_nd_update(self.general_states_c, [[0]], [new_general_state_c])

        self.general_states_h.assign(general_update_h)
        self.general_states_c.assign(general_update_c)
        
        '''Update Class States and Obtain Output'''
        # get updated general object representation (hidden general object state)
        new_general_object_representation = tf.expand_dims(new_general_state_h, axis=1) # (bs, 1, h, w, oc)

        # concatenate the general object representation with the input
        augmented_inputs = tf.concat([inputs, new_general_object_representation], axis=-1) # (bs, 1, h, w, 2*oc)
        
        # No classification happening yet, so create a dummy class ID
        class_ID = np.random.randint(self.num_classes)

        # get current class states
        current_class_states = [tf.gather(self.class_states_h, class_ID), tf.gather(self.class_states_c, class_ID)] # (bs, h, w, oc)

        # apply the shared class ConvLSTM layer to get the updated class states
        output, new_class_state_h, new_class_state_c = self.conv_lstm_class(augmented_inputs, initial_state=current_class_states) # (bs, h, w, oc) x 3

        # update the class states
        class_update_h = tf.tensor_scatter_nd_update(self.class_states_h, [[class_ID]], [new_class_state_h])
        class_update_c = tf.tensor_scatter_nd_update(self.class_states_c, [[class_ID]], [new_class_state_c])
        
        self.class_states_h.assign(class_update_h)
        self.class_states_c.assign(class_update_c)

        # return the class-specific object representation (hidden class state)
        return output

class OuterLayer(layers.Layer):
    def __init__(self, num_classes, batch_size, im_height, im_width, output_channels, **kwargs):
        super(OuterLayer, self).__init__(**kwargs)
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.im_height = im_height
        self.im_width = im_width
        self.output_channels = output_channels
        self.object_representation = ObjectRepresentation(num_classes, batch_size, im_height, im_width, output_channels, name='object_representation')

    def call(self, inputs):
        nt = inputs.shape[1]
        all_outputs = []
        for t in range(nt):
            decomposed_frame = tf.expand_dims(inputs[:, t, ...], axis=1) # (bs, 1, h, w, oc)
            out = self.object_representation(decomposed_frame) # (bs, h, w, oc)
            out = layers.Flatten()(out)
            all_outputs.append(out)
        return tf.reduce_sum(tf.stack(all_outputs), axis=0)

"""Build model"""
nc = 4
nt = 10
oc = 12
bs = 1
h = 64
w = 64
assert bs == 1

input_layer = layers.Input(shape=(nt, h, w, oc), batch_size=bs)
outer = OuterLayer(num_classes=nc, batch_size=bs, im_height=h, im_width=w, output_channels=oc, name='outer_layer')
output = outer(input_layer)
model = Model(inputs=input_layer, outputs=output)

model.summary()
# Compile the model
model.compile(optimizer=keras.optimizers.Adam(), loss=keras.losses.mean_squared_error)

# Define paths for weights and state weights
weights_path = "model_weights.h5"

# Load weights if they exist
if os.path.exists(weights_path):
    try:
        model.load_weights(weights_path)
    except Exception as e:
        print("Error loading weights: ", e, "Removing the weights file and starting from scratch")
        os.remove(weights_path)
    print("Weights loaded successfully")
else:
    print("No weights found")

# Callback to save the model weights and state weights
checkpoint_callback = ModelCheckpoint(filepath=weights_path, save_weights_only=True, save_best_only=True, monitor='val_loss', mode='min', verbose=1)

# Train the model using the fake dataset
num_samples = 10
x_train = np.random.random((num_samples, nt, h, w, oc)).astype(np.float32)
y_train = np.random.random((num_samples,)).astype(np.float32)
x_val = np.random.random((num_samples // 10, nt, h, w, oc)).astype(np.float32)
y_val = np.random.random((num_samples // 10,)).astype(np.float32)

model.fit(x_train, y_train, epochs=5, batch_size=bs, validation_data=(x_val, y_val), callbacks=[checkpoint_callback])
