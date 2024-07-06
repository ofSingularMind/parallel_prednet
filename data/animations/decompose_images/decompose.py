import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
# import cv2

from sklearn.cluster import KMeans

def quantize_image(image, num_colors=4):
    return image.convert('RGB').quantize(colors=num_colors, method=Image.FASTOCTREE)

def process_image(image_path, output_folder, num_masks=4):
    # Open and quantize the image
    image = Image.open(image_path).convert('RGB')
    quantized_image = quantize_image(image)
    quantized_image = quantized_image.convert('RGB')
    data = quantized_image.load()

    # Find unique colors in the quantized image
    unique_colors = set()
    for y in range(quantized_image.size[1]):
        for x in range(quantized_image.size[0]):
            unique_colors.add(data[x, y])

    # Create masks for each color
    masks = [Image.new('RGB', image.size, 0) for _ in range(num_masks)]
    mask_data = [mask.load() for mask in masks]

    color_to_index = {color: index for index, color in enumerate(unique_colors)}

    for y in range(quantized_image.size[1]):
        for x in range(quantized_image.size[0]):
            pixel = data[x, y]
            mask_data[color_to_index[pixel]][x, y] = pixel

    # Save the masks
    base_filename = os.path.splitext(os.path.basename(image_path))[0]
    for i, mask in enumerate(masks):
        mask.save(os.path.join(output_folder, f"{base_filename}_mask_{i}.png"))

def process_images(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Get all .png files and sort them
    files = sorted([f for f in os.listdir(input_folder) if f.endswith(".png")])

    for n, filename in enumerate(files):
        if n >= 10:
            break
        image_path = os.path.join(input_folder, filename)
        process_image(image_path, output_folder)


#  def process_images(input_folder, output_folder):
#     if not os.path.exists(output_folder):
#         os.makedirs(output_folder)
#     for n, filename in enumerate(os.listdir(input_folder)):
#         if filename.endswith(".png"):
#             image_path = os.path.join(input_folder, filename)
#             process_image(image_path, output_folder)
#         if n == 10:
#             break

# Configuration
input_folder = '/home/evalexii/Documents/Thesis/code/parallel_prednet/data/animations/multi_gen_shape_strafing/frames/multi_gen_shape_2nd_stage'  # Folder with source images
output_folder = '/home/evalexii/Documents/Thesis/code/parallel_prednet/data/animations/decompose_images/output'  # Folder to save the processed images

if os.path.exists(output_folder):
    for filename in os.listdir(output_folder):
        os.remove(os.path.join(output_folder, filename))



# Process images
process_images(input_folder, output_folder)
