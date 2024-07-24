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

def sharpened_softmax(logits):
    # print("Raw logits:", logits.numpy())

    # Subtract 0.99 * max of the logits from each logit
    max_logit = tf.reduce_max(logits)
    adjusted_logits = logits - 0.99 * max_logit
    # print("Adjusted logits:", adjusted_logits.numpy())
    
    # Apply ReLU
    relu_logits = tf.nn.relu(adjusted_logits)
    # print("ReLU applied:", relu_logits.numpy())

    # Scale the relu_logits
    scaled_relu_logits = tf.multiply(relu_logits, 1e9)
    # print("Scaled RuLU:", scaled_relu_logits.numpy())
    
    # Apply softmax
    sharpened_softmax_probs = tf.nn.softmax(scaled_relu_logits)
    # print("Softmax probabilities:", sharpened_softmax_probs.numpy())
    
    return sharpened_softmax_probs

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
        new_weights_shape = current_weights_shape + (1,) * len(params.shape[2:])
        reshaped_weights = tf.reshape(weights, new_weights_shape)

        weighted_params = reshaped_weights * params
        weighted_sum = tf.reduce_sum(weighted_params, axis=0)
        return weighted_sum

    def diff_scatter_nd_update(self, A, B, logits, beta=1e10):
        """
        Update tensor A with values from tensor B based on highest indices indicated by a logits matrix.
        Like tf.tensor_scatter_nd_update, but differentiable.

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
        
        '''Update Class States and Obtain Outputs'''
        # Note that inputs are nc*3 channels, where nc is the number of classes and thus we process each 3-channel block separately
        
        output_class_tensors = []
        for i in range(self.num_classes):

            frame = inputs[..., i*3:(i+1)*3] # (bs, 1, h, w, 3)

            # get updated general object representation (hidden general object state)
            new_general_object_representation = tf.expand_dims(new_general_state_h, axis=1) # (bs, 1, h, w, oc)

            # concatenate the general object representation with the input
            augmented_inputs = tf.concat([frame, new_general_object_representation], axis=-1) # (bs, 1, h, w, 3+oc)
            
            # No classification happening yet, so create a dummy class ID
            class_logits = self.classifier(tf.squeeze(frame, axis=1)) # (bs, h, w, 3)
            # class_label = self.softargmax(class_probs)
            # class_ID = tf.squeeze(class_label, axis=None) # Ensure class_label is a scalar if possible
            # print("Class ID: ", class_ID)
            # class_ID = i # np.random.randint(self.num_classes)

            # get current class states
            current_class_states = [self.diff_gather(self.class_states_h, class_logits), self.diff_gather(self.class_states_c, class_logits)] # (bs, h, w, oc)

            # apply the shared class ConvLSTM layer to get the updated class states
            class_output, new_class_state_h, new_class_state_c = self.conv_lstm_class(augmented_inputs, initial_state=current_class_states) # (bs, h, w, oc) x 3

            # update the class states
            # class_update_h = tf.tensor_scatter_nd_update(self.class_states_h, [[class_ID]], [new_class_state_h])
            # class_update_c = tf.tensor_scatter_nd_update(self.class_states_c, [[class_ID]], [new_class_state_c])
            class_update_h = self.diff_scatter_nd_update(self.class_states_h, new_class_state_h, class_logits)
            class_update_c = self.diff_scatter_nd_update(self.class_states_c, new_class_state_c, class_logits)
            
            self.class_states_h.assign(class_update_h)
            self.class_states_c.assign(class_update_c)

            # append the class output to the list of output_class_tensors
            output_class_tensors.append(class_output)

        # stack the class outputs to get the final output
        output = tf.stack(output_class_tensors, axis=-1)

        # return the class-specific object representation (hidden class state)
        return output

class NewObjectRepresentation(layers.Layer):
    def __init__(self, num_classes, batch_size, im_height, im_width, output_channels, **kwargs):
        super(NewObjectRepresentation, self).__init__(**kwargs)
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.im_height = im_height
        self.im_width = im_width
        self.output_channels = output_channels
        self.conv_lstm_general = ConvLSTM2D(filters=output_channels, kernel_size=(3, 3), padding='same', return_sequences=False, return_state=True, stateful=False, name='conv_lstm_general')
        self.conv_lstm_class = ConvLSTM2D(filters=output_channels, kernel_size=(3, 3), padding='same', return_sequences=False, return_state=True, stateful=False, name='conv_lstm_class')
        self.classifier = CustomMobileNetV2(num_classes=num_classes, input_shape=(self.im_height, self.im_width, 3))

        self.class_states_h = self.add_weight(shape=(num_classes, batch_size, im_height, im_width, output_channels), initializer='zeros', trainable=False, name='class_state_h')
        self.class_states_c = self.add_weight(shape=(num_classes, batch_size, im_height, im_width, output_channels), initializer='zeros', trainable=False, name='class_state_c')
        self.general_states_h = self.add_weight(shape=(1, batch_size, im_height, im_width, output_channels), initializer='zeros', trainable=False, name='general_state_h')
        self.general_states_c = self.add_weight(shape=(1, batch_size, im_height, im_width, output_channels), initializer='zeros', trainable=False, name='general_state_c')

    @tf.function
    def call(self, inputs):
        current_general_states = [self.general_states_h[0], self.general_states_c[0]] # (bs, h, w, oc) x 2
        all_class_object_representations = self.class_states_h # (nc, bs, h, w, oc)
        all_class_object_representations = tf.reshape(all_class_object_representations, (self.batch_size, 1, self.im_height, self.im_width, self.num_classes * self.output_channels)) # (bs, 1, h, w, nc*oc)

        _, new_general_state_h, new_general_state_c = self.conv_lstm_general(all_class_object_representations, initial_state=current_general_states) # (bs, h, w, oc) x 3
        general_update_h = tf.tensor_scatter_nd_update(self.general_states_h, [[0]], [new_general_state_h])
        general_update_c = tf.tensor_scatter_nd_update(self.general_states_c, [[0]], [new_general_state_c])
        self.general_states_h.assign(general_update_h)
        self.general_states_c.assign(general_update_c)

        output_class_tensors = []
        for i in range(self.num_classes):
            frame = inputs[..., i*3:(i+1)*3] # (bs, 1, h, w, 3)
            new_general_object_representation = tf.expand_dims(new_general_state_h, axis=1) # (bs, 1, h, w, oc)
            augmented_inputs = tf.concat([frame, new_general_object_representation], axis=-1) # (bs, 1, h, w, 3+oc)

            class_logits = self.classifier(tf.squeeze(frame, axis=1)) # (bs, nc)
            class_probs = tf.expand_dims(sharpened_softmax(class_logits), axis=1) # (bs, nc)

            # Compute weighted sum of class states
            current_class_states_h = tf.einsum('bij,bjhwc->bhwc', class_probs, self.class_states_h) # (nc)
            current_class_states_c = tf.einsum('bij,bjhwc->bhwc', class_probs, self.class_states_c)

            class_output, new_class_state_h, new_class_state_c = self.conv_lstm_class(augmented_inputs, initial_state=[current_class_states_h, current_class_states_c])

            # Update all class states using soft assignments
            new_class_state_h = tf.einsum('bij,bhwc->bjhwc', class_probs, new_class_state_h)
            new_class_state_c = tf.einsum('bij,bhwc->bjhwc', class_probs, new_class_state_c)

            self.class_states_h.assign(tf.reduce_sum(new_class_state_h, axis=0))
            self.class_states_c.assign(tf.reduce_sum(new_class_state_c, axis=0))

            output_class_tensors.append(class_output)

        output = tf.stack(output_class_tensors, axis=-1)
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
