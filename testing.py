# from data_utils import test_dataset

# test_dataset()

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Generate a random image
image = np.random.randn(10, 10)  # Example with negative and positive values

# Generate positive and negative error arrays of the same shape as the image
positive_errors = np.random.rand(10, 10)  # Example positive errors
negative_errors = -np.random.rand(10, 10)  # Example negative errors

# Add errors to the image
noisy_image = positive_errors + negative_errors

# Define the custom colormap
colors = [(0, 'red'), (0.5, 'black'), (1, 'green')]  # Red to LightGray to Green
custom_cmap = LinearSegmentedColormap.from_list('custom_cmap', colors)

# Plot the original image with error bars
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.colorbar()

plt.subplot(1, 2, 2)
plt.imshow(noisy_image, cmap=custom_cmap)
plt.title('Image with Positive and Negative Errors')
plt.colorbar()

plt.show()
