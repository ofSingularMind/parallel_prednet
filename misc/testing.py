import os
import cv2
import numpy as np

# shape_color_prediction_network will feed into convLSTM input, as well as feed optical flow calculations for next frame prediction
# this is effective because

minB,  minG, minR, maxB, maxG, maxR = shape_color_prediction_network()

def preprocess_frame(frame):
    # Assuming the moving shape has a distinct color range, use color thresholding
    # Adjust the thresholds according to your shape's color
    mask = cv2.inRange(frame, (minB, minG, minR), (maxB, maxG, maxR))
    return mask

def compute_optical_flow(prev_frame, current_frame):
    prev_mask = preprocess_frame(prev_frame)
    current_mask = preprocess_frame(current_frame)
    
    # Compute optical flow only on the masked areas (where the shape is likely to be)
    flow = cv2.calcOpticalFlowFarneback(prev_mask, current_mask, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    return flow

def visualize_optical_flow(flow):
    # Calculate the magnitude and angle of the flow vectors
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    
    # Normalize the magnitude to a range of 0 to 1
    max_mag = np.max(magnitude)
    magnitude = magnitude / max_mag if max_mag != 0 else magnitude  # Avoid division by zero

    # Scale the angle between 0 and 180 for HSV color mapping
    angle = angle * 180 / np.pi / 2

    # Create an HSV image where:
    # Hue represents the direction
    # Value represents the magnitude
    # Saturation is set to maximum for visibility
    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
    hsv[..., 0] = angle.astype(np.uint8)
    hsv[..., 1] = 255
    hsv[..., 2] = (magnitude * 255).astype(np.uint8)

    # Convert HSV to BGR for display or saving
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return rgb

def save_optical_flow_images(flow, frame_number, base_dir):
    # Calculate magnitude and angle from flow vectors
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    
    # Normalize magnitude and angle
    normalized_mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    normalized_ang = (ang * 180 / np.pi / 2).astype(np.uint8)  # Convert to degrees and scale to [0, 255]
    
    # Normalize flow vectors
    flow_min = np.min(flow, axis=(0, 1), keepdims=True)
    flow_max = np.max(flow, axis=(0, 1), keepdims=True)
    normalized_flow = ((flow - flow_min) / (flow_max - flow_min) * 255).astype(np.uint8)
    
    # Create 3-channel images
    optical_flow_raw = np.zeros((*flow.shape[:2], 3), dtype=np.uint8)
    optical_flow_polar = np.zeros((*flow.shape[:2], 3), dtype=np.uint8)
    
    # Assign channels for raw format (u, v, dummy)
    optical_flow_raw[..., 0] = normalized_flow[..., 0]
    optical_flow_raw[..., 1] = normalized_flow[..., 1]
    optical_flow_raw[..., 2] = np.zeros_like(normalized_flow[..., 0])  # Placeholder channel
    
    # Assign channels for polar format (magnitude, angle, dummy)
    optical_flow_polar[..., 0] = normalized_mag
    optical_flow_polar[..., 1] = normalized_ang
    optical_flow_polar[..., 2] = np.zeros_like(normalized_mag)  # Placeholder channel

    # Save to folders
    raw_dir = os.path.join(base_dir, 'optical_flow_raw')
    polar_dir = os.path.join(base_dir, 'optical_flow_polar')
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(polar_dir, exist_ok=True)

    # Write images
    cv2.imwrite(os.path.join(raw_dir, f"{frame_number:03d}.png"), optical_flow_raw)
    cv2.imwrite(os.path.join(polar_dir, f"{frame_number:03d}.png"), optical_flow_polar)


# Load the sequence of 2D images
# Assuming images is a list containing the sequence of images
# base_dir = "/home/evalexii/Documents/Thesis/code/parallel_prednet/data/animations/general_ellipse_vertical/"
base_dir = "/home/evalexii/Documents/Thesis/code/parallel_prednet/data/animations/rolling_circle/"
images = []
num_images = 100  # Replace 10 with the actual number of images
for i in range(num_images):
    img = cv2.imread(base_dir + f"frames/single_rolling_circle/{i+1:03d}.png")
    images.append(img)

# Initialize previous frame
prev_frame = cv2.cvtColor(images[0], cv2.COLOR_BGR2GRAY)

# Optical flow parameters
lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Loop through the images to calculate optical flow
for i in range(1, len(images)):
    # Convert current frame to grayscale
    current_frame = cv2.cvtColor(images[i], cv2.COLOR_BGR2GRAY)

    # Calculate optical flow
    flow = cv2.calcOpticalFlowFarneback(prev_frame, current_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    # Optionally, visualize the optical flow
    rgb_image = visualize_optical_flow(flow)
    resized_image = cv2.resize(rgb_image, (800, 800))  # Resize the image to 800x600
    cv2.imshow('Optical Flow Visualization', resized_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Save the optical flow image
    # save_optical_flow_images(flow, i, base_dir)

    # Update previous frame
    prev_frame = current_frame



# Example usage:
# Assuming 'flow' is the optical flow result and 'i' is the frame index
# save_optical_flow_images(flow, i, '/path/to/your/base_directory')


# import matplotlib.pyplot as plt
# import numpy as np

# import os
# import warnings

# # Suppress warnings
# warnings.filterwarnings("ignore")
# # or '2' to filter out INFO messages too
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# import tensorflow as tf

# # x = np.linspace(0, 6*np.pi, 100)
# # y = np.sin(x)

# # # You probably won't need this if you're embedding things in a tkinter plot...
# # plt.ion()

# # fig = plt.figure()
# # ax = fig.add_subplot(111)
# # line1, = ax.plot(x, y, 'r-') # Returns a tuple of line objects, thus the comma

# # for phase in np.linspace(0, 10*np.pi, 500):
# #     line1.set_ydata(np.sin(x + phase))
# #     fig.canvas.draw()
# #     fig.canvas.flush_events()

# a = tf.data.Dataset.range(1, 4)  # ==> [ 1, 2, 3 ]
# b = tf.data.Dataset.range(4, 8)  # ==> [ 4, 5, 6, 7 ]
# ds = a.concatenate(b).shuffle(10)
# print(a)
# print(b)
# print(ds)
# print(list(ds.as_numpy_iterator()))

# # The input dataset and dataset to be concatenated should have
# # compatible element specs.
# # c = tf.data.Dataset.zip((a, b))
# # a.concatenate(c)



# # d = tf.data.Dataset.from_tensor_slices(["a", "b", "c"])
# # a.concatenate(d)