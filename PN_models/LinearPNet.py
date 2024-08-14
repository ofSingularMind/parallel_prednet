
import os
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")
# or '2' to filter out INFO messages too
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from keras import layers
import keras
from keras import backend as K
from keras.layers import Layer, Flatten
from keras.models import Model
import tensorflow as tf
import numpy as np

class Target(Layer):
    def __init__(self, units, layer_num, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dense = layers.Dense(units, activation="relu")

    def call(self, inputs):
        return self.dense(inputs)


class Prediction(Layer):
    def __init__(self, units, layer_num, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dense = layers.Dense(units, activation='relu')

    def call(self, inputs):
        return self.dense(inputs)


class Error(Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def call(self, predictions, targets):
        # compute errors
        e_down = keras.backend.relu(targets - predictions)
        e_up = keras.backend.relu(predictions - targets)
        return keras.layers.Concatenate(axis=-1)([e_down, e_up])


class Representation(Layer):
    def __init__(self, units, num_classes, *args, **kwargs):
        super(Representation, self).__init__(*args, **kwargs)
        self.units = units
        self.num_classes = num_classes
        self.lstm_cell = layers.LSTM(units, return_state=True, return_sequences=False)
        self.states = {"lstm": [None, None]}

    def call(self, inputs, class_logits):
        current_states = self.get_current_states(class_logits)
        
        inputs = tf.expand_dims(inputs, axis=1) if len(tf.shape(inputs)) == 2 else inputs
        output, h, c = self.lstm_cell(inputs, initial_state=current_states)

        self.update_states(class_logits, [h, c])

        return output

    def get_current_states(self, class_logits):
        return [self.diff_gather(self.states["lstm"][0], class_logits), self.diff_gather(self.states["lstm"][1], class_logits)]

    def update_states(self, class_logits, updated_class_states):
        # update the class states
        class_state_updates = [self.diff_scatter_nd_update(self.states["lstm"][0], updated_class_states[0], class_logits), self.diff_scatter_nd_update(self.states["lstm"][1], updated_class_states[1], class_logits)] # (bs, h, w, oc) via broadcasting
        self.states["lstm"][0] = class_state_updates[0]
        self.states["lstm"][1] = class_state_updates[1]
    
    def initialize_states(self):
        # Initialize internal lstm states
        self.states["lstm"][0] = tf.keras.initializers.GlorotUniform()(shape=(self.num_classes, self.units))
        self.states["lstm"][1] = tf.keras.initializers.GlorotUniform()(shape=(self.num_classes, self.units))

    def clear_states(self):
        # Clear internal lstm states
        self.states["lstm"][0] = tf.keras.initializers.GlorotUniform()(shape=(self.num_classes, self.units))
        self.states["lstm"][1] = tf.keras.initializers.GlorotUniform()(shape=(self.num_classes, self.units))
    
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

class LinearPNet_Layer(Model):
    def __init__(self, pn_args, layer_num, *args, **kwargs):
        super(LinearPNet_Layer, self).__init__(*args, **kwargs)
        self.layer_num = layer_num
        self.pn_args = pn_args
        self.batch_size = pn_args["batch_size"]
        self.units = pn_args["layer_units"][self.layer_num]
        self.num_classes = pn_args["num_classes"]
        self.representation_multiplier = pn_args["representation_multiplier"]
        
        self.top_layer = self.layer_num == len(pn_args["layer_units"]) - 1
        self.bottom_layer = self.layer_num == 0

        self.target = Target(self.units, self.layer_num, name=f"LPN_Target_Layer{layer_num}")
        self.prediction = Prediction(self.units, self.layer_num, name=f"LPN_Prediction_Layer{layer_num}")
        self.error = Error(name=f"LPN_Error_Layer{self.layer_num}")
        self.representation = Representation(self.representation_multiplier*self.units, self.num_classes, name=f"Representation_Layer{layer_num}")

        self.states = {"R": None, "P": None, "T": None, "E": None, "TD_inp": None}
        self.initialize_states()

    def initialize_states(self):
        # Initialize internal layer states
        self.states["R"] = tf.zeros((self.batch_size, self.representation_multiplier*self.units))
        self.states["P"] = tf.zeros((self.batch_size, self.units))
        self.states["T"] = tf.zeros((self.batch_size, self.units))
        self.states["E"] = tf.zeros((self.batch_size, 2 * self.units)) # double for the pos/neg concatenated error
        if not self.top_layer:
            self.states["TD_inp"] = tf.zeros((self.batch_size, self.representation_multiplier*self.pn_args["layer_units"][self.layer_num+1]))

        self.representation.initialize_states()

    def clear_states(self):
        # Clear internal layer states
        self.states["R"] = tf.zeros((self.batch_size, self.representation_multiplier*self.units))
        self.states["P"] = tf.zeros((self.batch_size, self.units))
        self.states["T"] = tf.zeros((self.batch_size, self.units))
        self.states["E"] = tf.zeros((self.batch_size, 2 * self.units))
        if not self.top_layer:
            self.states["TD_inp"] = tf.zeros((self.batch_size, self.representation_multiplier*self.pn_args["layer_units"][self.layer_num+1]))
        
        self.representation.clear_states()

    def get_state(self, state_name):
        return self.states[state_name]

    def top_down(self, inputs=None, class_logits=None):
        # UPDATE REPRESENTATION
        if self.top_layer:
            R_inp = keras.layers.Concatenate()([self.states["E"], self.states["R"]])
        else:
            self.states["TD_inp"] = inputs
            R_inp = keras.layers.Concatenate()([self.states["E"], self.states["R"], self.states["TD_inp"]])

        # extract class_states and form representation
        self.states["R"] = self.representation(R_inp, class_logits)

        # FORM PREDICTION(S)
        self.states["P"] = self.prediction(self.states["R"])

        return self.states["P"]

    def bottom_up(self, inputs=None):
        # RETRIEVE TARGET(S) (bottom-up input) ~ (batch_size, output_channels)
        self.states["T"] = inputs if self.bottom_layer else self.target(inputs)

        # COMPUTE TARGET ERROR
        self.states["E"] = self.error(self.states["P"], self.states["T"])

        return self.states["E"]

class LinearPNet(Model):
    def __init__(self, batch_size, latent_dim, num_classes, *args, **kwargs):
        super(LinearPNet, self).__init__(*args, **kwargs)
        self.pn_args = {}
        self.pn_args["batch_size"] = batch_size
        self.pn_args["layer_units"] = [latent_dim, latent_dim, latent_dim]
        self.pn_args["num_classes"] = num_classes
        self.pn_args["representation_multiplier"] = 2
        self.predlayers = [LinearPNet_Layer(self.pn_args, i, name=f"LPN_Layer_{i}") for i in range(len(self.pn_args["layer_units"]))]

        self.layer_weights = [1] + [0.1] * (len(self.pn_args["layer_units"]) - 1)

    def init_layer_states(self):
        for layer in self.predlayers:
            layer.initialize_states()

    def clear_layer_states(self):
        for layer in self.predlayers:
            layer.clear_states()

    def bottom_up(self, inputs):
        """ Perform bottom-up pass, starting from the bottom layer """
        for l, layer in enumerate(self.predlayers):
            
            error = layer.bottom_up(inputs if layer.bottom_layer else self.predlayers[layer.layer_num-1].get_state("E"))
                
            # Update error in bottom-up pass
            layer_error = self.layer_weights[l] * tf.reduce_mean(error, axis=-1, keepdims=True)  # (batch_size, 1)
            all_errors = layer_error if layer.bottom_layer else tf.add(all_errors, layer_error)  # (batch_size, 1)
        
        return all_errors

    def top_down(self, class_logits):
        """Perform top-down pass, starting from the top layer"""
        for layer in reversed(self.predlayers):
            if layer.bottom_layer:
                prediction = layer.top_down(self.predlayers[layer.layer_num+1].get_state("R"), class_logits)
            elif not layer.top_layer:
                _ = layer.top_down(self.predlayers[layer.layer_num+1].get_state("R"), class_logits)
            else:
                _ = layer.top_down(None, class_logits)

        return prediction

    def call(self, data):
        # inputs will be a batch of vectors and class logits
        inputs, class_logits = data

        all_errors = self.bottom_up(inputs) # initial prediction is all zeros

        bottom_layer_prediction_for_next_inputs = self.top_down(class_logits)

        return bottom_layer_prediction_for_next_inputs, all_errors


#     ####### FOR TESTING #######
#     @tf.function
#     def train_step(self, data):
#         # Unpack the data
#         x, logits, zero = data

#         for i in range(tf.shape(x)[1] // 10):

#             x_i = x[:, i*10:(i+1)*10, :]
#             logits_i = logits[:, i*10:(i+1)*10, :]

#             self.init_layer_states()

#             total_loss = 0.0
#             with tf.GradientTape() as tape:

#                 for j in range(tf.shape(x_i)[1]):
#                     x_i_i = x[:, j, :]
#                     logits_i_i = logits[:, j, :]
#                     # Forward pass
#                     pred, error = self((x_i_i, logits_i_i), training=True)
#                     # Compute the loss value
#                     loss = self.compiled_loss(zero, error)
#                     total_loss += loss
#                     # print("Loss: ", loss)
#                 print(f"***** Iteration {i}, Total Loss: {total_loss}")

#             grads = tape.gradient(total_loss, self.trainable_weights)
#             self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

#             self.clear_layer_states()

#         return {"Loss:": total_loss}

# bs = 1
# nt = 3000
# ld = 64
# nc = 4

# tf.config.run_functions_eagerly(True)

# data_inputs = keras.Input(shape=(ld))
# logits_inputs = keras.Input(shape=(nc))
# LPN = LinearPNet(batch_size=bs, latent_dim=ld, num_classes=nc)
# # LPN = keras.Model(inputs=[data_inputs, logits_inputs], outputs=LPN([data_inputs, logits_inputs]), name="LinearPNet")
# LPN.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.005), loss='mean_squared_error')

# # test_data = np.random.rand(bs, nt, ld)
# # test_logits = tf.one_hot(np.random.randint(nc, size=(bs, nt)), depth=nc)

# # Generate learnable test data: linear sequences
# test_data = np.array([[[i + j for j in range(ld)] for i in range(nt)] for _ in range(bs)], dtype=np.float32)

# # Generate test logits: constant class
# test_logits = np.zeros((bs, nt, nc), dtype=np.float32)
# test_logits[:, :, 0] = 1.0  # Assume class '0' is the constant class

# LPN((test_data[:,0,:], test_logits[:,0,:]))
# LPN.summary()
# LPN.train_step((test_data, test_logits, tf.constant(0.0, shape=(bs, 1))))

# print("good")