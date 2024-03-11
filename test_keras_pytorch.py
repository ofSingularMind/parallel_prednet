import keras
from keras.layers import Input, Dense, Layer
import torch
from torch import nn
import numpy as np


class pytorch_dense(Layer):
    def __init__(self, input_dim, output_dim):
        super(pytorch_dense, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def call(self, inputs):
        inputs = np.array(inputs)
        inputs = torch.from_numpy(inputs)
        return self.linear(inputs)

inputs = Input(shape=(784,))
x = Dense(64, activation='relu')(inputs)
x = Dense(64, activation='relu')(x)
x = pytorch_dense(64, 64)(x)
predictions = Dense(10, activation='softmax')(x)

model = keras.Model(inputs=inputs, outputs=predictions)

# This builds the model for the first time:
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Create some mock data
data = np.random.random((1000, 784))
labels = np.random.random((1000, 10))

# Train the model, iterating on the data in batches of 32 samples
model.fit(data, labels, epochs=10, batch_size=32)
    