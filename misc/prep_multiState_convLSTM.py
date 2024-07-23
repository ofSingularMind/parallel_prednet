import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import ConvLSTM2D
from tensorflow.keras import Model
import os

# Define the custom ConvLSTM layer
# class CustomConvLSTM(tf.keras.layers.Layer):
#     def __init__(self, output_channels, **kwargs):
#         super().__init__(**kwargs)
#         self.conv_lstm = ConvLSTM2D(filters=output_channels, kernel_size=(3, 3), padding='same', return_sequences=False, return_state=True)
#         self.init = True
#         self.state_h = None
#         self.state_c = None

#     def call(self, inputs, states=None):
#         if states is None:
#             outputs, state_h, state_c = self.conv_lstm(inputs)
#         else:
#             outputs, state_h, state_c = self.conv_lstm(inputs, initial_state=states)

#         if self.init:
#             self.state_h = tf.Variable(state_h, trainable=False)
#             self.state_c = tf.Variable(state_c, trainable=False)
#             self.init = False
#         else:
#             self.state_h.assign(state_h)
#             self.state_c.assign(state_c)
        
#         return outputs, [self.state_h, self.state_c]

#     def reset_states(self):
#         self.state_h = None
#         self.state_c = None

class CustomConvLSTM(tf.keras.layers.Layer):
    def __init__(self, output_channels, num_classes, **kwargs):
        super().__init__(**kwargs)
        self.output_channels = output_channels
        self.conv_lstm = ConvLSTM2D(filters=output_channels, kernel_size=(3, 3), padding='same', return_sequences=False, return_state=True)

    def build(self, input_shape):
        # Initialize state variables with the appropriate shape
        state_shape = (input_shape[0], input_shape[2], input_shape[3], self.output_channels) # (batch_size, height, width, channels)
        self.state_h = self.add_weight(shape=state_shape, initializer='zeros', trainable=False, name='state_h')
        self.state_c = self.add_weight(shape=state_shape, initializer='zeros', trainable=False, name='state_c')

    def call(self, inputs, states=None):
        if states is None:
            states = [self.state_h, self.state_c]

        outputs, state_h, state_c = self.conv_lstm(inputs, initial_state=states)
        self.state_h.assign(state_h)
        self.state_c.assign(state_c)

        return outputs

    def reset_states(self):
        self.state_h.assign(tf.zeros_like(self.state_h))
        self.state_c.assign(tf.zeros_like(self.state_c))


# Define the synthetic dataset
def generate_synthetic_data(num_samples=1000, sequence_length=5, image_height=64, image_width=64):
    # Generate random sequences of grayscale images
    x = np.random.rand(num_samples, sequence_length, image_height, image_width, 3).astype(np.float32)
    y = np.random.rand(num_samples, image_height, image_width, 32).astype(np.float32)  # Target can be the next frame or any other related task
    return x, y

x_train, y_train = generate_synthetic_data()

# Prepare the dataset for training
batch_size = 1
dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
dataset = dataset.batch(batch_size)

# Training utilities
conv_lstm_layer = CustomConvLSTM(output_channels=32)
# conv_lstm_layer.build(input_shape=(None, 5, 64, 64, 3)) 
optimizer = tf.keras.optimizers.Adam()
loss_fn = tf.keras.losses.MeanSquaredError()

# If weights exist, load them
if os.path.exists('conv_lstm_weights.h5'):
    conv_lstm_layer.load_weights('conv_lstm_weights.h5')

@tf.function
def train_step(x, y, states=None):
    with tf.GradientTape() as tape:
        predictions = conv_lstm_layer(x)
        loss = loss_fn(y, predictions)
    grads = tape.gradient(loss, conv_lstm_layer.trainable_variables)
    optimizer.apply_gradients(zip(grads, conv_lstm_layer.trainable_variables))
    return loss

# Training loop
num_epochs = 3
for epoch in range(num_epochs):
    for x_batch, y_batch in dataset:
        loss = train_step(x_batch, y_batch)
        # [state.detach() for state in states]
    print(f'Epoch {epoch+1}, Loss: {loss.numpy()}')
    conv_lstm_layer.reset_states()  # Optionally reset states at the end of each epoch or as required by the training dynamics

# Save the model weights
conv_lstm_layer.save_weights('conv_lstm_weights.h5')
