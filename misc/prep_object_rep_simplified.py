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

from keras.applications import MobileNetV2
from keras.layers import Dense, GlobalAveragePooling2D, Flatten, ConvLSTM2D
from keras.models import Model

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
        self.predictions = Dense(self.num_classes, activation=None)
    
    def compute_output_shape(self, input_shape):
        return (self.num_classes,)

    def call(self, inputs):
        x = self.base_model(inputs)
        x = self.global_avg_pool(x)
        x = self.dense_1024(x)
        out = self.predictions(x)
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

        self.class_states_h = self.add_weight(shape=(num_classes, batch_size, im_height, im_width, output_channels), initializer='zeros', trainable=False, name='class_state_h')
        self.class_states_c = self.add_weight(shape=(num_classes, batch_size, im_height, im_width, output_channels), initializer='zeros', trainable=False, name='class_state_c')
        self.general_states_h = self.add_weight(shape=(1, batch_size, im_height, im_width, output_channels), initializer='zeros', trainable=False, name='general_state_h')
        self.general_states_c = self.add_weight(shape=(1, batch_size, im_height, im_width, output_channels), initializer='zeros', trainable=False, name='general_state_c')

    def diff_gather(self, params, logits, beta=1e10):
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
        if (B.shape != (bs, h, w, oc)) or (one_hot.shape != (bs, nc)):
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
        for i in range(self.num_classes):

            frame = inputs[..., i*3:(i+1)*3] # (bs, 1, h, w, 3)

            # get updated general object representation (hidden general object state)
            new_general_object_representation = tf.expand_dims(new_general_state_h, axis=1) # (bs, 1, h, w, oc)

            # concatenate the general object representation with the input
            augmented_inputs = tf.concat([frame, new_general_object_representation], axis=-1) # (bs, 1, h, w, 3+oc)
            
            # classify the input frame to get the class logits predictions
            class_logits = self.classifier(tf.squeeze(frame, axis=1)) # (bs, nc)

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
            assert class_output.shape == (self.batch_size, self.im_height, self.im_width, self.output_channels)

        # stack the class outputs to get the final output
        output = tf.concat(output_class_tensors, axis=-1)
        # print("Output shape:", output.shape)
        assert output.shape == (self.batch_size, self.im_height, self.im_width, self.num_classes*self.output_channels)

        # return the class-specific object representations (hidden class states)
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
            output_class_tensors = self.object_representation(decomposed_frame) # (bs, h, w, nc*oc)
            out = Flatten()(output_class_tensors)
            all_outputs.append(out)
        return tf.reduce_sum(tf.stack(all_outputs), axis=0)

"""Build model"""
nc = 4
nt = 10
ic = 12
oc = 3
bs = 1
h = 64
w = 64
assert bs == 1

input_layer = layers.Input(shape=(nt, h, w, ic), batch_size=bs)
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
x_train = np.random.random((num_samples, nt, h, w, ic)).astype(np.float32)
y_train = np.random.random((num_samples,)).astype(np.float32)
x_val = np.random.random((num_samples // 10, nt, h, w, ic)).astype(np.float32)
y_val = np.random.random((num_samples // 10,)).astype(np.float32)

model.fit(x_train, y_train, epochs=5, batch_size=bs, validation_data=(x_val, y_val), callbacks=[checkpoint_callback])
