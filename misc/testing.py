from PIL import Image
import numpy as np

for i in range(10):
    noise = np.random.normal(0, 1, (50, 100, 3))

    img = Image.fromarray(noise, 'RGB')

    img.save(f"/home/evalexii/Documents/Thesis/animations/backgrounds/50x100/{i:03}.png")

# a = 4
# print(f'{a:03}')