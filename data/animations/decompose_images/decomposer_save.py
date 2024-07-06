import os
import numpy as np
from PIL import Image
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

class SceneDecomposer:
    def __init__(self, input_folder=None, output_folder=None, n_colors=4, mode='save'):
        # mode: 'return' or 'save'
        self.mode = mode
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.n_colors = n_colors
        self.last_colors = None
        self.last_masks = None
        self.num_images_processed = 0

    def quantize_image(self, image, num_colors=4):
        return image.convert('RGBA').quantize(colors=num_colors, method=Image.FASTOCTREE)

    def process_image(self, image_path):
        image = Image.open(image_path).convert('RGBA')
        quantized_image = self.quantize_image(image, self.n_colors)
        quantized_image = quantized_image.convert('RGBA')
        data = quantized_image.load()

        # Find unique colors in the quantized image
        unique_colors = set()
        for y in range(quantized_image.size[1]):
            for x in range(quantized_image.size[0]):
                unique_colors.add(data[x, y])

        # Ensure we have at least n_colors unique colors
        while len(unique_colors) < self.n_colors:
            unique_colors.add((0, 0, 0, 255))

        unique_colors = np.array(list(unique_colors))

        masks = [Image.new('RGBA', image.size, (0, 0, 0, 255)) for _ in range(len(unique_colors))]
        mask_data = [mask.load() for mask in masks]

        color_to_index = {tuple(color): index for index, color in enumerate(unique_colors)}

        for y in range(quantized_image.size[1]):
            for x in range(quantized_image.size[0]):
                pixel = data[x, y]
                mask_data[color_to_index[tuple(pixel)]][x, y] = pixel

        new_masks = self.align_masks_by_color(unique_colors, masks)

        if self.mode == 'save':
            self.save_masks(new_masks)
        elif self.mode == 'return':
            return new_masks
        else:
            raise ValueError("Invalid mode")

    def align_masks_by_color(self, new_colors, new_masks):
        if self.last_colors is None:
            self.last_colors = new_colors
            self.last_masks = new_masks
            return new_masks

        # Ensure both color arrays are 2-dimensional
        new_colors = np.array(new_colors).reshape(-1, 4) # 4 colors x RGBA
        self.last_colors = np.array(self.last_colors).reshape(-1, 4)

        distance_matrix = cdist(self.last_colors, new_colors, metric='euclidean')
        row_indices, col_indices = linear_sum_assignment(distance_matrix)

        reordered_masks = [None] * len(new_masks)
        for i, j in zip(row_indices, col_indices):
            reordered_masks[i] = new_masks[j]
        # Also reorder the new colors
        new_colors = new_colors[col_indices]

        self.last_colors = new_colors
        self.last_masks = reordered_masks

        return reordered_masks

    def save_masks(self, masks):
        base_filename = os.path.splitext(os.path.basename(self.current_image_path))[0]
        for i, mask in enumerate(masks):
            mask.save(os.path.join(self.output_folder, f"{base_filename}_mask_{i}.png"))

    def process_images(self):
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

        files = sorted([f for f in os.listdir(self.input_folder) if f.endswith(".png")])
        for n, filename in enumerate(files):
            if n >= 30:
                break
            self.current_image_path = os.path.join(self.input_folder, filename)
            self.process_image(self.current_image_path)
            self.num_images_processed = n + 1

# Configuration
input_folder = '/home/evalexii/Documents/Thesis/code/parallel_prednet/data/animations/multi_gen_shape_strafing/frames/multi_gen_shape_2nd_stage'  # Folder with source images
output_folder = '/home/evalexii/Documents/Thesis/code/parallel_prednet/data/animations/decompose_images/output'  # Folder to save the processed images

if os.path.exists(output_folder):
    for filename in os.listdir(output_folder):
        os.remove(os.path.join(output_folder, filename))

# Instantiate and process images
processor = SceneDecomposer(input_folder, output_folder, n_colors=4, mode='save')
processor.process_images()

# Plot 10 sets of masks, 40 masks in total in a 10x4 grid
import matplotlib.pyplot as plt
from matplotlib import gridspec

fig = plt.figure(figsize=(5, 40))
gs = gridspec.GridSpec(30, 4)

files = sorted([f for f in os.listdir(output_folder) if f.endswith(".png")])

for i in range(30):
    for j in range(4):
        ax = plt.subplot(gs[i, j])
        mask = Image.open(os.path.join(output_folder, files[i * 4 + j]))
        ax.imshow(mask)
        ax.axis('off')
plt.show()
