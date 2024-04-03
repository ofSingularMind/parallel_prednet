import os
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")
# or '2' to filter out INFO messages too
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import keras
import tensorflow as tf
import numpy as np

l = 5
r = 20
c = 20
oc = 200

# Create a large random tensor (representation) of size 200 x 200 x 3
x = tf.ones((r, c, oc), dtype=np.float32)





class full_radial_convolution(keras.layers.Layer):
    def __init__(self, r, c, oc, downscale_factor=1, num_patches=10):
        super().__init__()
        
        self.downscale_factor = downscale_factor
        self.num_patches = num_patches # number of patches in each dimension

        assert r % self.num_patches == 0, "Image patch factor must divide spatial dimensions evenly"
        assert c % self.num_patches == 0, "Image patch factor must divide spatial dimensions evenly"
        self.r, self.c = r, c
        
        # Create a conv2D with filter size equal to spatial dimensions and stride equal to patch dimensions
        self.create_spatial_guide()
        self.create_radial_filters_tensor()
        self.conv = keras.layers.Conv2D(oc, (r,c), strides=(r,c), padding="valid")

    def call(self, inputs):

        # Need to create a mega-representation of the input and radial filters
        # It is a tensor of shape (r^2, c^2, oc), where each r x c x oc block is the 
        # input multiplied by that block's (element's) radial filter. Then, we can
        # apply a 2D convolution to the entire tensor, with kernel size (r,c) and stride (r,c).
        # This will give us the radially-weighted convolution of the full input for each pixel.
        # r = inputs.shape._dims[1]
        # c = inputs.shape._dims[2]

        num_rows_per_patch = self.r//self.num_patches
        num_cols_per_patch = self.c//self.num_patches

        inputs = tf.tile(inputs, [1, num_rows_per_patch, num_cols_per_patch, 1])
        patch_outputs = (self.num_patches)*[None]
        for i in range(self.num_patches):
            patch_outputs_row = (self.num_patches)*[None]
            for j in range(self.num_patches):
                # patch_of_radial_filters = self.create_patch_radial_filters_tensor(i*num_rows_per_patch, (i+1)*num_rows_per_patch, j*num_cols_per_patch, (j+1)*num_cols_per_patch)
                
                patch_of_radial_filters = tf.expand_dims(self.radial_filters[i*(self.r*num_rows_per_patch):(i+1)*(self.r*num_rows_per_patch), j*(self.c*num_cols_per_patch):(j+1)*(self.c*num_cols_per_patch), :], axis=0)
                
                # patch_of_radial_filters_list_rows = self.radial_filters[i*num_rows_per_patch:(i+1)*num_rows_per_patch]
                # patch_of_radial_filters_list = [patch_of_radial_filters_list_rows[ii][j*num_cols_per_patch:(j+1)*num_cols_per_patch] for ii in range(num_rows_per_patch)]
                # patch_of_radial_filters_rows = [tf.concat(row, axis=1) for row in patch_of_radial_filters_list]
                # patch_of_radial_filters = tf.expand_dims(tf.concat(patch_of_radial_filters_rows, axis=0), axis=0)
                patch_of_weighted_input = tf.multiply(inputs, patch_of_radial_filters)
                patch_of_output = self.conv(patch_of_weighted_input)
                patch_outputs_row[j] = patch_of_output
            patch_outputs[i] = tf.concat(patch_outputs_row, axis=2) # axes shift because of batch dimension
        patch_outputs = tf.concat(patch_outputs, axis=1)
        return patch_outputs

    def create_spatial_guide(self):
        # Create master radial pattern
        ra, ca = np.linspace(start=0, stop=1, num=self.r), np.linspace(start=0, stop=1, num=self.c)
        rb, cb = np.linspace(start=1, stop=0, num=self.r)[1:], np.linspace(start=1, stop=0, num=self.c)[1:]
        row_guide, col_guide = np.concatenate([ra, rb]), np.concatenate([ca, cb])
        spatial_guide = np.zeros((2*self.r-1, 2*self.c-1))
        for row in range(2*self.r-1):
            for col in range(2*self.c-1):
                spatial_guide[row,col] = np.minimum(row_guide[row], col_guide[col])
        self.spatial_guide = tf.convert_to_tensor(spatial_guide, dtype=tf.float32)

    def create_radial_filters_list(self):
        # Create radial filter, one per pixel
        # Max intensity at pixel location, decreasing towards surrounding edges
        radial_filters = r*[c*[None]]
        for row in range(r):
            for col in range(c):
                radial_filter = self.spatial_guide[r - row - 1:2 * r - row - 1, c - col - 1:2 * c - col - 1]
                radial_filter = tf.expand_dims(radial_filter, axis=-1)
                radial_filter = tf.tile(radial_filter, [1, 1, oc])
                radial_filters[row][col] = radial_filter
        self.radial_filters = radial_filters

    def create_radial_filters_tensor(self):
        # Create radial filter, one per pixel
        # Max intensity at pixel location, decreasing towards surrounding edges
        radial_filters = r*[None]
        for row in range(r):
            radial_filters_row = c*[None]
            for col in range(c):
                radial_filter = self.spatial_guide[r - row - 1:2 * r - row - 1, c - col - 1:2 * c - col - 1]
                radial_filter = tf.expand_dims(radial_filter, axis=-1)
                radial_filter = tf.tile(radial_filter, [1, 1, oc])
                radial_filters_row[col] = radial_filter
            radial_filters[row] = tf.concat(radial_filters_row, axis=1)
        radial_filters = tf.concat(radial_filters, axis=0)
        self.radial_filters = radial_filters

    def create_patch_radial_filters_tensor(self, rs, re, cs, ce):
        # Create patch of radial filters, one per pixel
        # Max intensity at pixel location, decreasing towards surrounding edges
        radial_filters = (re-rs)*[None]
        for i, row in enumerate(range(rs, re)):
            radial_filters_row = (ce-cs)*[None]
            for j, col in enumerate(range(cs, ce)):
                radial_filter = self.spatial_guide[self.r - row - 1:2 * self.r - row - 1, self.c - col - 1:2 * self.c - col - 1]
                radial_filter = tf.expand_dims(radial_filter, axis=-1)
                radial_filter = tf.tile(radial_filter, [1, 1, oc])
                radial_filters_row[j] = radial_filter
            radial_filters[i] = tf.concat(radial_filters_row, axis=1)
        radial_filters = tf.concat(radial_filters, axis=0)
        return radial_filters

# # Print out the results (probs garb)
# inputs = keras.layers.Input((r, c, oc))
# outputs = full_radial_convolution(r,c,oc,num_patches=5)(inputs)

# model = keras.models.Model(inputs=inputs, outputs=outputs)

# model.compile(optimizer="adam", loss="mean_squared_error")
# model.build(x.shape)
# model.fit(tf.expand_dims(x, axis=0), tf.expand_dims(2*x, axis=0), epochs=100)

# y = model(tf.expand_dims(x, axis=0))

import time
mod = full_radial_convolution(r,c,oc,num_patches=5)
start = time.time()
y = mod(tf.expand_dims(x, axis=0))
print(f"Time taken for convolution: {time.time() - start} seconds")