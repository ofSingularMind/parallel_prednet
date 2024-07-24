import os
import warnings
# Suppress warnings
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from keras import layers
from keras import backend as K
import keras
import keras_cv
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from keras.applications import MobileNetV2
from keras.layers import Dense, GlobalAveragePooling2D, Flatten
from keras.models import Model
from keras.layers import ConvLSTM2D
from keras.callbacks import Callback, ModelCheckpoint

class MultiClassStatefulConvLSTM2D(tf.keras.layers.Layer):
    def __init__(self, output_channels=32, num_classes=4, **kwargs):
        super().__init__(**kwargs)
        self.output_channels = output_channels
        self.num_classes = num_classes
        self.conv_lstms = [ConvLSTM2D(filters=output_channels, kernel_size=(3, 3), padding='same', return_sequences=False, return_state=True, stateful=True) for _ in range(num_classes)]
        # self.conv_lstm = CustomConvLSTM2D(output_channels=output_channels, layer_num=0)

    # def build(self, input_shape):
    #     super().build(input_shape)
    #     state_shape = (input_shape[0], input_shape[1], input_shape[2], self.output_channels)
    #     self.states_h = self.add_weight(shape=(self.num_classes, *state_shape), initializer='zeros', trainable=False, name='states_h')
    #     self.states_c = self.add_weight(shape=(self.num_classes, *state_shape), initializer='zeros', trainable=False, name='states_c')
    #     for i in range(self.num_classes):
    #         dummy_input = tf.expand_dims(tf.zeros(input_shape), axis=1)
    #         output, state_h, state_c = self.conv_lstms[i](dummy_input, initial_state=[self.states_h[i], self.states_c[i]])

    def call(self, inputs, class_ID=0):
        # current_states = [tf.gather(self.states_h, class_ID), tf.gather(self.states_c, class_ID)]

        outputs, state_h, state_c = self.conv_lstms[class_ID](inputs)

        self.states_h = tf.tensor_scatter_nd_update(self.states_h, [[class_ID]], [state_h])
        self.states_c = tf.tensor_scatter_nd_update(self.states_c, [[class_ID]], [state_c])

        return outputs

    def get_hidden_states(self):
        h_states = tf.stack([conv_lstm.states[0] for conv_lstm in self.conv_lstms])
        return h_states
        # return self.states_h if self.num_classes > 1 else self.states_h[0]

    def reset_states(self, states=None):
        # self.states_h = tf.zeros_like(self.states_h)
        # self.states_c = tf.zeros_like(self.states_c)
        if states is None:
            states = [self.states_h, self.states_c] 
        for i in range(self.num_classes):
            self.conv_lstms[i].reset_states(states=states[i])
    
    def detach_states(self):
        # self.states_h = tf.stop_gradient(self.states_h)
        # self.states_c = tf.stop_gradient(self.states_c)
        # self.states_h = self.states_h.numpy()
        # self.states_c = self.states_c.numpy()
        self.states_h = detach(self.states_h)
        self.states_c = detach(self.states_c)

class oldMCSCL(tf.keras.layers.Layer):
    def __init__(self, output_channels=32, num_classes=4, **kwargs):
        super().__init__(**kwargs)
        self.output_channels = output_channels
        self.num_classes = num_classes
        self.conv_lstm = ConvLSTM2D(filters=output_channels, kernel_size=(3, 3), padding='same', return_sequences=False, return_state=True, stateful=False)
        # self.conv_lstm = CustomConvLSTM2D(output_channels=output_channels, layer_num=0)

    def build(self, input_shape):
        super().build(input_shape)
        state_shape = (input_shape[0], input_shape[1], input_shape[2], self.output_channels)
        self.states_h = self.add_weight(shape=(self.num_classes, *state_shape), initializer='zeros', trainable=False, name='states_h')
        self.states_c = self.add_weight(shape=(self.num_classes, *state_shape), initializer='zeros', trainable=False, name='states_c')

    def call(self, inputs, class_ID=0):
        current_states = [tf.gather(self.states_h, class_ID), tf.gather(self.states_c, class_ID)]

        outputs, state_h, state_c = self.conv_lstm(inputs, initial_state=current_states)

        self.states_h = tf.tensor_scatter_nd_update(self.states_h, [[class_ID]], [state_h])
        self.states_c = tf.tensor_scatter_nd_update(self.states_c, [[class_ID]], [state_c])

        return outputs

    def get_hidden_states(self):
        return self.states_h if self.num_classes > 1 else self.states_h[0]

    def reset_states(self):
        # self.states_h = tf.zeros_like(self.states_h)
        # self.states_c = tf.zeros_like(self.states_c)
        self.conv_lstm.reset_states()
    
    def detach_states(self):
        # self.states_h = tf.stop_gradient(self.states_h)
        # self.states_c = tf.stop_gradient(self.states_c)
        # self.states_h = self.states_h.numpy()
        # self.states_c = self.states_c.numpy()
        self.states_h = detach(self.states_h)
        self.states_c = detach(self.states_c)

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
        assert self.batch_size == 1, "Only working for batch_size 1"
        self.frame_channels = training_args['output_channels'][0]
        self.classifier = CustomMobileNetV2(num_classes=4, input_shape=(self.im_height, self.im_width, 3))

        # self.general_object_tensor = tf.random.normal((1, self.im_height, self.im_width, self.frame_channels))
        # self.class_tensors = tf.stack([tf.random.normal((1, self.im_height, self.im_width, self.frame_channels)) for _ in range(num_classes)])

        self.multi_CLSTM_general = MultiClassStatefulConvLSTM2D(output_channels=self.frame_channels, num_classes=1, name=f"ObjectRepresentation_ConvLSTM_General_Layer{self.layer_num}")
        self.multi_CLSTM_class = MultiClassStatefulConvLSTM2D(output_channels=self.frame_channels, num_classes=4, name=f"ObjectRepresentation_ConvLSTM_Class_Layer{self.layer_num}")
        # self.multi_CLSTM_general.build((self.batch_size, self.im_height, self.im_width, self.num_classes * self.frame_channels))
        # self.multi_CLSTM_class.build((self.batch_size, self.im_height, self.im_width, self.frame_channels + 3))

    def call(self, inputs):
        output_tensors = []

        # if t == 0:
        #     self.multi_CLSTM_general.detach_states()
        #     self.multi_CLSTM_class.detach_states()

        for i in range(0, self.frame_channels, 3):
            frame = inputs[..., i:i+3]
            class_probs = self.classifier(frame)
            class_label = tf.math.argmax(class_probs, axis=-1)
            class_label = tf.squeeze(class_label, axis=None) # Ensure class_label is a scalar if possible

            # Update class tensor using ConvLSTM over general object tensor and new frame
            general_object_tensor = self.multi_CLSTM_general.get_hidden_states() # (nc, bs, 64, 64, 12)
            concatenated_input = tf.expand_dims(tf.concat([general_object_tensor, frame], axis=-1), axis=1)
            updated_class_tensor = self.multi_CLSTM_class(inputs=concatenated_input, class_ID=class_label)
            # tf.tensor_scatter_nd_update(self.class_tensors, [[class_label]], [updated_class_tensor])
            
            output_tensors.append(updated_class_tensor)
            # TODO: If the same class is predicted twice in the same frame, it will be upated twice. Is this OK? Maybe they are similar enough.

        all_class_tensors = self.multi_CLSTM_class.get_hidden_states() # (nc, bs, 64, 64, 12)
        all_class_tensors = tf.reshape(all_class_tensors, (self.batch_size, 1, self.im_height, self.im_width, self.num_classes * self.frame_channels))
        # all_class_tensors = tf.expand_dims(all_class_tensors, axis=1)
        _ = self.multi_CLSTM_general(inputs=all_class_tensors, class_ID=0)
        # updated_general_object_tensor = tf.squeeze(updated_general_object_tensor)
        # tf.tensor_scatter_nd_update(self.general_object_tensor, [0], updated_general_object_tensor)
        # tf.compat.v1.assign(self.general_object_tensor, updated_general_object_tensor)

        out = tf.concat(output_tensors, axis=-1)
        shape = (self.batch_size, self.im_height, self.im_width, self.training_args['output_channels'][0]*self.num_classes)
        out.set_shape(shape)

        return out

class dummyLayer(layers.Layer):
    def __init__(self, training_args, num_classes, layer_num, im_height, im_width, **kwargs):
        super(dummyLayer, self).__init__(**kwargs)
        self.object_representation = ObjectRepresentation(training_args=training_args, num_classes=num_classes, layer_num=layer_num, im_height=im_height, im_width=im_width)
        self.nt = training_args["nt"]

    def call(self, inputs):
        # inputs: (bs, nt, 64, 64, oc)
        sequence_output_tensors = []
        for t in range(self.nt):
            decomposed_frame = inputs[:, t, ...] # (bs, 64, 64, oc)
            sequence_output_tensors.append(self.object_representation(decomposed_frame)) # (bs, 64, 64, oc*num_classes)

        return tf.stack(sequence_output_tensors)
        
            

"""Build model"""
# Create input layer
nt = 10
oc = 12
bs = 1
assert bs == 1
input_layer = layers.Input(shape=(nt, 64, 64, oc), batch_size=bs)

# Create dummyLayer layer
dummy = dummyLayer(training_args={'output_channels': [oc], 'batch_size': bs, 'nt': nt}, num_classes=4, layer_num=1, im_height=64, im_width=64)

output = dummy(input_layer)

model = Model(inputs=input_layer, outputs=output)

model.summary()

# Compile the model
model.compile(optimizer=keras.optimizers.Adam(), loss=keras.losses.mean_squared_error, metrics=["accuracy"])

# If weights exist, load them
weights_file = 'dummyLayer_weights.hdf5'
if os.path.exists(weights_file):
    model.load_weights(weights_file)

# Reset / Set initial states of ConvLSTM layers
dummy.object_representation.multi_CLSTM_general.reset_states()
dummy.object_representation.multi_CLSTM_class.reset_states()


"""Train model"""
# Create random test dataset for keras.model.fit()
input_shape = (nt, 64, 64, oc)
num_classes = 4

class PrintBatchNumberCallback(Callback):
    def on_train_batch_end(self, batch, logs=None):
        print(f"End of batch {batch}, Loss: {logs['loss']:.4f}")

class PrintEpochNumberCallback(Callback):
    def on_epoch_begin(self, epoch, logs=None):
        print(f"Starting Epoch {epoch+1}")

    def on_epoch_end(self, epoch, logs=None):
        print(f"End of Epoch {epoch+1}")

callbacks = [PrintBatchNumberCallback(), PrintEpochNumberCallback()]
callbacks.append(ModelCheckpoint(filepath=weights_file, monitor="val_loss", save_best_only=True, save_weights_only=True))

"""TRAIN OPTION 1"""

# Generate random inputs
x_train = np.random.random((bs, *input_shape)).astype(np.float32)

# Generate random outputs (fake target)
y_train = np.random.random((bs, nt, 64, 64, num_classes * oc)).astype(np.float32)

# Train the model using the fake dataset
model.fit(x_train, y_train, epochs=5, batch_size=bs, callbacks=callbacks)


"""TRAIN OPTION 2"""

# # Define the synthetic dataset
# def generate_synthetic_data(num_samples=1000, sequence_length=5, image_height=64, image_width=64):
#     # Generate random sequences of grayscale images
#     x = np.random.random((num_samples, *input_shape)).astype(np.float32)
#     y = np.random.random((num_samples, 64, 64, num_classes * oc)).astype(np.float32)  # Target can be the next frame or any other related task
#     return x, y

# x_train, y_train = generate_synthetic_data()

# # Prepare the dataset for training
# dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
# dataset = dataset.batch(bs)

# optimizer = tf.keras.optimizers.Adam()
# loss_fn = tf.keras.losses.MeanSquaredError()

# # If weights exist, load them
# if os.path.exists('conv_lstm_weights.h5'):
#     model.load_weights('conv_lstm_weights.h5')

# @tf.function
# def train_step(x, y, states=None):
#     with tf.GradientTape() as tape:
#         predictions = model(x)
#         loss = loss_fn(y, predictions)
#     grads = tape.gradient(loss, model.trainable_variables)
#     optimizer.apply_gradients(zip(grads, model.trainable_variables))
#     return loss

# # Training loop
# num_epochs = 3
# print("Start training...")
# for epoch in range(num_epochs):
#     for x_batch, y_batch in dataset:
#         loss = train_step(x_batch, y_batch)
#         # [state.detach() for state in states]
#     print(f'Epoch {epoch+1}, Loss: {loss.numpy()}')
#     model.reset_states()  # Optionally reset states at the end of each epoch or as required by the training dynamics

# # Save the model weights
# model.save_weights('conv_lstm_weights.h5')

"""HOLDING"""


def detach(tensor):
    copied_tensor = tf.identity(tensor)
    detached_tensor_values = tf.keras.backend.eval(copied_tensor)
    return detached_tensor_values

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
