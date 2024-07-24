from keras import layers
import keras
import tensorflow as tf
import os
import warnings
from keras.applications import MobileNetV2
from keras.layers import Dense, GlobalAveragePooling2D

# Suppress warnings
warnings.filterwarnings("ignore")
# or '2' to filter out INFO messages too
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

class Target(keras.layers.Layer):
    def __init__(self, output_channels, layer_num, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.output_channels = output_channels
        # Add Conv
        self.conv = layers.Conv2D(self.output_channels, (3, 3), padding="same", activation="relu", name=f"Target_Conv_Layer{layer_num}")
        # Add Pool
        self.pool = layers.MaxPooling2D((2, 2), padding="valid", name=f"Target_Pool_Layer{layer_num}")

    def call(self, inputs):
        x = self.conv(inputs)
        return self.pool(x)


class Prediction(keras.layers.Layer):
    def __init__(self, output_channels, layer_num, activation='relu', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.output_channels = output_channels
        self.layer_num = layer_num
        self.conv_layer = layers.Conv2D(self.output_channels, (3, 3), padding="same", activation=activation, name=f"Prediction_Conv_Layer{layer_num}")

    def call(self, inputs):
        return self.conv_layer(inputs)


class Error(keras.layers.Layer):
    def __init__(self, layer_num, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.layer_num = layer_num
        # Add Subtract
        # Add ReLU

    def call(self, predictions, targets):
        # compute errors
        e_down = keras.backend.relu(targets - predictions)
        e_up = keras.backend.relu(predictions - targets)
        return keras.layers.Concatenate(axis=-1)([e_down, e_up])


class Representation(keras.layers.Layer):
    def __init__(self, output_channels, layer_num, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Add ConvLSTM, being sure to pass previous states in OR use stateful=True
        conv_i = layers.Conv2D(output_channels, (3, 3), padding="same", activation="hard_sigmoid", name=f"Representation_Conv_i_Layer{layer_num}")
        conv_f = layers.Conv2D(output_channels, (3, 3), padding="same", activation="hard_sigmoid", name=f"Representation_Conv_f_Layer{layer_num}")
        conv_o = layers.Conv2D(output_channels, (3, 3), padding="same", activation="hard_sigmoid", name=f"Representation_Conv_o_Layer{layer_num}")
        conv_c = layers.Conv2D(output_channels, (3, 3), padding="same", activation="tanh", name=f"Representation_Conv_c_Layer{layer_num}")
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

        return output, states


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
            print("Class logits:", class_logits)

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

        # stack the class outputs to get the final output
        output = tf.stack(output_class_tensors, axis=-1)

        # return the class-specific object representation (hidden class state)
        return output


