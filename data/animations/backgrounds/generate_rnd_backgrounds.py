import os
from PIL import Image
import numpy as np

h = 50
w = 50

for i in range(1000):
    noise = np.random.normal(0, 1, (h, w, 3))

    img = Image.fromarray(noise, 'RGB')

    if not os.path.exists(f"/home/evalexii/Documents/Thesis/code/parallel_prednet/data/animations/backgrounds/{h}x{w}/"):
        os.makedirs(f"/home/evalexii/Documents/Thesis/code/parallel_prednet/data/animations/backgrounds/{h}x{w}/")
    img.save(f"/home/evalexii/Documents/Thesis/code/parallel_prednet/data/animations/backgrounds/{h}x{w}/{i:03}.png")

