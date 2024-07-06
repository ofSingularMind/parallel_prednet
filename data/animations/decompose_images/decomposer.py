import numpy as np
from PIL import Image
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
import time
import cv2  # OpenCV is useful for some image operations
import os

class SceneDecomposer:
    def __init__(self, n_colors=4):
        self.n_colors = n_colors
        self.last_colors = None
        self.last_masks = None

    def report_state(self):
        print("Last colors: ", self.last_colors)
        print("Last masks: ", self.last_masks)

    def quantize_image(self, image, num_colors=4):
        return image.convert('RGBA').quantize(colors=num_colors, method=Image.FASTOCTREE)

    def process_single_image(self, image):
        '''
        Process a single image and return a list of masks, one for each color in the image.
        Expected input: PIL Image or numpy array with shape (H, W, 3), float32, range [0, 1]
        '''
        if type(image) is np.ndarray:
            # print(type(image))
            # print("Processing single image")
            image = Image.fromarray((image * 255).astype(np.uint8))
        quantized_image = self.quantize_image(image, self.n_colors)
        quantized_image = quantized_image.convert('RGBA')
        data = quantized_image.load()

        # Find unique colors in the quantized image
        unique_colors = set()
        for y in range(quantized_image.size[1]):
            for x in range(quantized_image.size[0]):
                unique_colors.add(data[x, y])

        unique_colors = list(unique_colors)

        # Ensure we have at least n_colors unique colors
        while len(unique_colors) < self.n_colors:
            # print(f"Colors: {len(unique_colors)}, {self.n_colors}")
            unique_colors.append((0, 0, 0, 255))

        # if len(unique_colors) < self.n_colors:
        #     # input("Press Enter to continue...")
        #     raise ValueError("Not enough unique colors")

        unique_colors = np.array(unique_colors)

        masks = [Image.new('RGBA', quantized_image.size, (0, 0, 0, 255)) for _ in range(self.n_colors)]
        mask_data = [mask.load() for mask in masks]

        color_to_index = {tuple(color): index for index, color in enumerate(unique_colors)}

        for y in range(quantized_image.size[1]):
            for x in range(quantized_image.size[0]):
                pixel = data[x, y]
                mask_data[color_to_index[tuple(pixel)]][x, y] = pixel

        new_masks = self.align_masks_by_color(unique_colors, masks)

        # print(f"{[np.array(mask).shape for mask in new_masks]}")
        # print(f"{len(unique_colors)} unique colors")
        # print(f"{[col for col in unique_colors]}")

        new_masks = [np.array(mask)[..., :3] for mask in new_masks]

        new_masks = np.concatenate(new_masks, axis=-1) / 255.0

        # input("Press Enter to continue...")

        return new_masks

    def align_masks_by_color(self, new_colors, new_masks):
        if self.last_colors is None:
            self.last_colors = new_colors
            self.last_masks = new_masks
            return new_masks

        # Ensure both color arrays are 2-dimensional
        new_colors = np.array(new_colors).reshape(-1, 4)
        self.last_colors = np.array(self.last_colors).reshape(-1, 4)

        distance_matrix = cdist(self.last_colors, new_colors, metric='euclidean')
        row_indices, col_indices = linear_sum_assignment(distance_matrix)
        # print(f"Row indices: {row_indices}, Col indices: {col_indices}")

        reordered_masks = [None] * len(new_masks)
        for i, j in zip(row_indices, col_indices):
            reordered_masks[i] = new_masks[j]
        
        # Also reorder the new colors
        new_colors = new_colors[col_indices]

        self.last_colors = new_colors
        self.last_masks = reordered_masks

        return reordered_masks

    def process_dataset(self, dataset):
        """
        Process a batch of images with shape (BS, T, 64, 64, 3) and return masks with shape (BS, T, 64, 64, n_colors*3)
        """
        T, H, W, C = dataset.shape
        masks_batch = np.zeros((T, H, W, self.n_colors*C), dtype=np.float32)

        self.clear_state()
        for t in range(T):
            # image = Image.fromarray((dataset[t] * 255).astype(np.uint8))
            new_masks = self.process_single_image(dataset[t])
            masks_batch[t] = new_masks

            # for i in range(self.n_colors):
            #     mask_array = np.array(mask)
            #     masks_batch[t, ..., i*C:(i+1)*C] = mask_array[..., :3] # Discard the alpha channel

        return masks_batch

    def process_batch(self, batch):
        """
        Process a batch of images with shape (BS, T, 64, 64, 3) and return masks with shape (BS, T, 64, 64, n_colors*3)
        """
        BS, T, H, W, _ = batch.shape
        masks_batch = np.zeros((BS, T, H, W, self.n_colors*3), dtype=np.float32)

        for b in range(BS):
            self.clear_state()
            for t in range(T):
                image = Image.fromarray(batch[b, t])
                new_masks = self.process_single_image(image)

                for i, mask in enumerate(new_masks):
                    mask_array = np.array(mask)
                    masks_batch[b, t, ..., i*3:(i+1)*3] = mask_array[..., :3] # Discard the alpha channel

        return masks_batch

    def process_list(self, image_list):
        '''Expecting a sequence of images in a list'''
        print("Processing list of images")
        decomposed_images = [self.process_single_image(im) for im in image_list]
        self.clear_state()
        # print("Decomposed images: ", len(decomposed_images))
        return decomposed_images

    def clear_state(self):
        self.last_colors = None
        self.last_masks = None



class SceneDecomposer_CV:
    def __init__(self, input_folder=None, output_folder=None, n_colors=4, mode='save'):
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.n_colors = n_colors
        self.mode = mode
        self.last_colors = None
        self.last_masks = None

    def save_masks(self, masks):
        base_filename = os.path.splitext(os.path.basename(self.current_image_path))[0]
        for i, mask in enumerate(masks):
            # mask.save(os.path.join(self.output_folder, f"{base_filename}_mask_{i}.png"))
            cv2.imwrite(os.path.join(self.output_folder, f"{base_filename}_mask_{i}.png"), mask)

    def process_images(self):
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

        files = sorted([f for f in os.listdir(self.input_folder) if f.endswith(".png")])
        for n, filename in enumerate(files):
            if n >= 30:
                break
            self.current_image_path = os.path.join(self.input_folder, filename)
            image = cv2.imread(self.current_image_path) / 255.0
            self.process_single_image(image)
            self.num_images_processed = n + 1

    def quantize_image(self, image, num_colors=4):
        # Use k-means clustering to quantize the image to a fixed number of colors
        Z = image.reshape((-1, 3))
        Z = np.float32(Z)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        ret, label, center = cv2.kmeans(Z, num_colors, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        center = np.uint8(center)
        res = center[label.flatten()]
        quantized_image = res.reshape((image.shape))
        return quantized_image, center

    def process_single_image(self, image):
        """
        Process a single image and return a list of masks, one for each color in the image.
        Expected input: numpy array with shape (H, W, 3), float32, range [0, 1]
        """
        quantized_image, unique_colors = self.quantize_image(image, self.n_colors)
        
        # Create masks for each unique color
        masks = np.zeros((self.n_colors, image.shape[0], image.shape[1], 3), dtype=np.float32)
        for i, color in enumerate(unique_colors):
            masks[i] = np.all(quantized_image == color, axis=-1, keepdims=True) * color

        new_masks = self.align_masks_by_color(unique_colors, masks)

        # Combine all masks into a single array with shape (H, W, n_colors * 3)
        new_masks = np.concatenate(new_masks, axis=-1) / 255.0

        if self.mode == 'save':
            self.save_masks(new_masks)
        elif self.mode == 'return':
            return new_masks
        else:
            raise ValueError("Invalid mode")

        # return new_masks

    def align_masks_by_color(self, new_colors, new_masks):
        if self.last_colors is None:
            self.last_colors = new_colors
            self.last_masks = new_masks
            return new_masks

        distance_matrix = cdist(self.last_colors, new_colors, metric='euclidean')
        row_indices, col_indices = linear_sum_assignment(distance_matrix)

        reordered_masks = [None] * len(new_masks)
        for i, j in zip(row_indices, col_indices):
            reordered_masks[i] = new_masks[j]

        self.last_colors = new_colors[col_indices]
        self.last_masks = reordered_masks

        return reordered_masks

    def process_batch(self, batch):
        """
        Process a batch of images with shape (BS, T, H, W, 3) and return masks with shape (BS, T, H, W, n_colors * 3)
        """
        BS, T, H, W, _ = batch.shape
        masks_batch = np.zeros((BS, T, H, W, self.n_colors * 3), dtype=np.float32)

        for b in range(BS):
            self.clear_state()
            for t in range(T):
                new_masks = self.process_single_image(batch[b, t])
                masks_batch[b, t, ...] = new_masks

        return masks_batch

    def clear_state(self):
        self.last_colors = None
        self.last_masks = None




# # Configuration
# input_folder = '/home/evalexii/Documents/Thesis/code/parallel_prednet/data/animations/multi_gen_shape_strafing/frames/multi_gen_shape_2nd_stage'  # Folder with source images
# output_folder = '/home/evalexii/Documents/Thesis/code/parallel_prednet/data/animations/decompose_images/output'  # Folder to save the processed images

# if os.path.exists(output_folder):
#     for filename in os.listdir(output_folder):
#         os.remove(os.path.join(output_folder, filename))

# # Instantiate and process images
# processor = SceneDecomposer2(input_folder, output_folder, n_colors=4, mode='save')
# processor.process_images()

# # Plot 10 sets of masks, 40 masks in total in a 10x4 grid
# import matplotlib.pyplot as plt
# from matplotlib import gridspec

# fig = plt.figure(figsize=(5, 40))
# gs = gridspec.GridSpec(30, 4)

# files = sorted([f for f in os.listdir(output_folder) if f.endswith(".png")])

# for i in range(30):
#     for j in range(4):
#         ax = plt.subplot(gs[i, j])
#         mask = Image.open(os.path.join(output_folder, files[i * 4 + j]))
#         ax.imshow(mask)
#         ax.axis('off')
# plt.show()

# # Example usage
# import numpy as np

# # Configuration
# BS = 2  # Batch size
# T = 3  # Sequence length
# H = 64  # Height
# W = 64  # Width

# # Create a batch of images (BS x T x 64 x 64 x 3)
# im = np.random.rand(H, W, 3)
# files = sorted([f for f in os.listdir(input_folder) if f.endswith(".png")])
# im = Image.open(os.path.join(input_folder, files[0]))

# # Instantiate and process the batch of images
# decomposer1 = SceneDecomposer(n_colors=4)
# masks = decomposer1.process_single_image(im)
# # decomposer2 = SceneDecomposer2(n_colors=4)
# # start_time = time.time()
# # for i in range(2000):
# #     masks = decomposer1.process_single_image(im)
# # print("Decomposer1: --- %s seconds ---" % (time.time() - start_time))
# # start_time = time.time()
# # for i in range(2000):
# #     masks = decomposer2.process_single_image(im)
# # print("Decomposer2: --- %s seconds ---" % (time.time() - start_time))

# # Print the shape of the returned masks batch
# print(masks.shape)  # Expected shape: (BS, T, 12, 64, 64)

# # Clear the internal state
# decomposer1.clear_state()
