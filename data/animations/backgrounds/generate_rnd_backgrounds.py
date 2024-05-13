import os
from PIL import Image
import numpy as np
import cv2

h = 50
w = 50
alpha = 0.3 # percentage of noise

for i in range(1000):

    noise = np.random.uniform(0, 1, (h, w, 3)) * 255
    grey = np.ones((h, w, 3)) * 128
    img = grey * (1-alpha) + noise * alpha

    # img = Image.fromarray(img, 'RGB')
    
    # white = np.ones((h, w, 3)) * 255
    # noise = np.random.randint(0, 256, (h, w, 3))
    # # img = grey * (1-alpha) + noise * alpha # Image.blend(Image.fromarray(white, 'RGB'), Image.fromarray(noise, 'RGB'), 0.01)    
    # img = white + alpha * (noise - white)

    # img = Image.fromarray(noise, 'RGB')

    if not os.path.exists(f"/home/evalexii/Documents/Thesis/code/parallel_prednet/data/animations/backgrounds/{h}x{w}/"):
        os.makedirs(f"/home/evalexii/Documents/Thesis/code/parallel_prednet/data/animations/backgrounds/{h}x{w}/")
    # img.save(f"/home/evalexii/Documents/Thesis/code/parallel_prednet/data/animations/backgrounds/{h}x{w}/{i:03}.png")
    cv2.imwrite(f"/home/evalexii/Documents/Thesis/code/parallel_prednet/data/animations/backgrounds/{h}x{w}/{i:03}.png", img.astype(np.uint8))

