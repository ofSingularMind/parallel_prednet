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
        print(h.shape, c.shape)
        c = f * c + i * self.convs["conv_c"](inputs)
        h = o * keras.activations.tanh(c)
        output = h
        states = [h, c]

        return output, states


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
        self.frame_channels = training_args['output_channels'][0]
        self.classifier = CustomMobileNetV2(num_classes=4, input_shape=(self.im_height, self.im_width, 3))
        self.general_object_tensor = tf.Variable(tf.random.normal((1, self.im_height, self.im_width, self.frame_channels)), trainable=True)
        self.general_object_LSTM_states = [
                tf.Variable(tf.random.normal((1, self.im_height, self.im_width, self.frame_channels)), trainable=True),
                tf.Variable(tf.random.normal((1, self.im_height, self.im_width, self.frame_channels)), trainable=True)
            ]        
        self.class_tensors = tf.Variable(
            tf.stack([tf.random.normal((1, self.im_height, self.im_width, self.frame_channels)) for _ in range(num_classes)]),
            trainable=True
        )
        self.class_LSTM_states = [
            [
                tf.Variable(tf.random.normal((1, self.im_height, self.im_width, self.frame_channels)), trainable=True),
                tf.Variable(tf.random.normal((1, self.im_height, self.im_width, self.frame_channels)), trainable=True)
            ] for _ in range(num_classes)
        ]
        self.conv_lstm_general = CustomConvLSTM2D(output_channels=self.frame_channels, layer_num=layer_num)
        self.conv_lstm_class = CustomConvLSTM2D(output_channels=self.frame_channels, layer_num=layer_num)

    def call(self, inputs):
        output_tensors = []
        batch_size = tf.shape(inputs)[0]

        for i in range(0, self.frame_channels, 3):
            frame = inputs[..., i:i+3]
            class_probs = self.classifier(frame)
            class_label = tf.math.argmax(class_probs, axis=-1)
            
            # Ensure class_label is a scalar if possible
            class_label = tf.squeeze(class_label, axis=None)

            # Update class tensor using ConvLSTM over general object tensor and new frame
            concatenated_input = tf.expand_dims(tf.concat([self.general_object_tensor, frame], axis=-1), axis=1)
            
            state_h = tf.gather([state[0] for state in self.class_LSTM_states], class_label)
            state_c = tf.gather([state[1] for state in self.class_LSTM_states], class_label)
            initial_states = [state_h, state_c]
            
            updated_class_tensor, new_states = self.conv_lstm_class(concatenated_input, initial_states=initial_states)
            
            updated_class_tensor = tf.squeeze(updated_class_tensor, axis=1)
            new_states = [tf.squeeze(new_states[i], axis=1) for i in range(len(new_states))]

            # Update the tensors using tf.tensor_scatter_nd_update
            tf.tensor_scatter_nd_update(self.class_tensors, [[class_label]], [updated_class_tensor])
            tf.tensor_scatter_nd_update(self.class_LSTM_states[0], [[class_label]], [new_states[0]])
            tf.tensor_scatter_nd_update(self.class_LSTM_states[1], [[class_label]], [new_states[1]])
            
            output_tensors.append(updated_class_tensor)

        # all_class_tensors = self.class_tensors.reshape((1, self.im_height, self.im_width, self.frame_channels, -1))
        all_class_tensors = tf.reshape(self.class_tensors, (1, self.im_height, self.im_width, self.num_classes * self.frame_channels))
        concatenated_classes = tf.expand_dims(all_class_tensors, axis=1)
        if self.general_object_LSTM_states is None:
            updated_general_object_tensor, self.general_object_LSTM_states = self.conv_lstm_general(concatenated_classes)
        else:
            updated_general_object_tensor, self.general_object_LSTM_states = self.conv_lstm_general(concatenated_classes, initial_states=self.general_object_LSTM_states)
        self.general_object_tensor.assign(tf.squeeze(updated_general_object_tensor, axis=1))

        out = tf.concat(output_tensors, axis=-1)
        shape = (None, self.im_height, self.im_width, self.training_args['output_channels'][0]*self.num_classes)
        out.set_shape(shape)
        
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
        self.predictions = Dense(self.num_classes, activation='softmax')
    
    def call(self, inputs):
        x = self.base_model(inputs)
        x = self.global_avg_pool(x)
        x = self.dense_1024(x)
        return self.predictions(x)

