# import keras
# from keras.layers import Input, Dense, Layer
# import torch
# from torch import nn
# import numpy as np
# from keras.layers import TorchModuleWrapper


# class pytorch_dense(keras.Model):
#     def __init__(self, input_dim, output_dim):
#         super(pytorch_dense, self).__init__()
#         self.linear = TorchModuleWrapper(nn.Linear(input_dim, output_dim))
#         self.d1 = Dense(64, activation='relu')
#         self.d2 = Dense(64, activation='relu')
#         self.p = Dense(10, activation='softmax')

#     def call(self, inputs):
#         # inputs = np.array(inputs)
#         # inputs = torch.from_numpy(inputs)
#         x = self.d1(inputs)
#         x = self.d2(x)
#         x = self.linear(x)
#         return self.p(x)

# # inputs = Input(shape=(784,))
# # x = Dense(64, activation='relu')(inputs)
# # x = Dense(64, activation='relu')(x)
# # x = pytorch_dense(64, 64)(x)
# # predictions = Dense(10, activation='softmax')(x)

# # model = keras.Model(inputs=inputs, outputs=predictions)

# # This builds the model for the first time:
# model = pytorch_dense(784, 10)
# model.build(input_shape=(None, 784))
# model.compile(optimizer='rmsprop',
#               loss='categorical_crossentropy',
#               metrics=['accuracy'])

# # Create some mock data
# data = np.random.random((1000, 784))
# labels = np.random.random((1000, 10))

# # Train the model, iterating on the data in batches of 32 samples
# model.fit(data, labels, epochs=10, batch_size=32)
    
# # keras=2.15.0
import torch
import torch.nn as nn
import torch.nn.functional as F

import keras
from keras.layers import TorchModuleWrapper

class Classifier(keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Wrap `torch.nn.Module`s with `TorchModuleWrapper`
        # if they contain parameters
        self.inputs = keras.Input(shape=(28, 28, 1))
        self.conv1 = TorchModuleWrapper(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3, 3))
        )
        self.conv2 = TorchModuleWrapper(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3))
        )
        self.pool = nn.MaxPool2d(kernel_size=(2, 2))
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(p=0.5)
        self.fc = TorchModuleWrapper(nn.Linear(1600, 10))

    def call(self, inputs):
        x = F.relu(self.conv1(inputs))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.fc(x)
        return F.softmax(x, dim=1)


model = Classifier()
model.build((1, 28, 28))
# if torch.cuda.is_available():
#     device = torch.device("cuda:0")
#     data = torch.ones(1, 1, 28, 28).to(device)
# else:
# device = torch.device("cpu")
# data = torch.ones(1, 1, 28, 28).to(device)
import tensorflow as tf
import numpy as np
data = np.ones((10, 28, 28, 1))
print("# Output shape", model(data).shape)

model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer="adam",
    metrics=["accuracy"]
)
# model.fit(train_loader, epochs=5)