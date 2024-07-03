import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import UnsupervisedImageDataset
# from monet_pytorch import Monet
from model import Monet
import torch
import matplotlib.pyplot as plt
import numpy as np
import time
import h5py

# Do init stuff
from math import prod
from model import Monet
import omegaconf
omegaconf.OmegaConf.register_new_resolver('prod', lambda *numbers: int(prod(float(x) for x in numbers)))
omegaconf.OmegaConf.register_new_resolver('sum', lambda *numbers: int(sum(float(x) for x in numbers)))
__all__ = ['Monet']

# Model evaluation script
loadModel = True
delftblue = False
second_stage = True
stage = "2nd" if second_stage else "1st"
if delftblue:
    WEIGHTS_PATH = "/home/aledbetter/parallel_prednet/monet_pytorch_GH_michedev/model_weights/delftblue/"
    DATASET_PATH = "/scratch/aledbetter/multi_gen_shape_2nd_stage_for_objects/"
else:
    WEIGHTS_PATH = "/home/evalexii/Documents/Thesis/code/parallel_prednet/monet_pytorch_GH_michedev/model_weights/laptop/"
    DATASET_PATH = f"/home/evalexii/Documents/Thesis/code/parallel_prednet/data/animations/multi_gen_shape_strafing/frames/multi_gen_shape_{stage}_stage/"

# Load datset and create dataloader
transform = transforms.Compose([
    # transforms.Resize((64, 64)),
    transforms.ToTensor()
])

# Create dataset and dataloader
dataset = UnsupervisedImageDataset(root_dir=DATASET_PATH, transform=transform)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

# Initialize model, loss function, and optimizer. Load saved weights if resuming training.
monet = Monet.from_config(model='monet-SSM', dataset=None, scene_max_objects=4, dataset_width=64, dataset_height=64).to('cpu')
monet.load_state_dict(torch.load(WEIGHTS_PATH + 'monet_weights.pth', map_location=torch.device('cpu')))
monet.eval()
print('Successfully loaded model weights...')

# Iterate over the DataLoader
decomposed_images = []
for batch_idx, images in enumerate(dataloader):
    # Forward pass
    outputs = monet(images)
    masks = outputs['mask'][:,1:3,...]
    masks = torch.unsqueeze(masks, 2)
    slots = outputs['slot'][:,1:3,...]
    masked_recons = masks * slots
    decomposed_images.append(masked_recons.detach().cpu().numpy())

    if batch_idx % 10 == 0:
        print(f"Batch {batch_idx} processed")

    # if batch_idx == 50:
    #     break

decomposed_images = np.concatenate(decomposed_images, axis=0)

# Save decomposed images to HDF5 file
hdf5_file = 'decomposed_images.h5'
with h5py.File(hdf5_file, 'w') as f:
    f.create_dataset('images', data=decomposed_images, compression="gzip")

print("Decomposed images saved to HDF5 file successfully.")

class PairedDataset:
    def __init__(self, hdf5_file):
        self.hdf5_file = hdf5_file
        self.hdf5_data = h5py.File(hdf5_file, 'r')
        self.length = len(self.hdf5_data['images'])

    def __len__(self):
        return self.length

    def get_images(self, idx):
        with h5py.File(self.hdf5_file, 'r') as f:
            img1 = f['images'][idx, 0]
            img2 = f['images'][idx, 1]f.create_dataset('images', data=decomposed_images, compression="gzip")
        return img1, img2

# Example usage
paired_dataset = PairedDataset(hdf5_file)

for idx in range(10):

    img1, img2 = paired_dataset.get_images(idx)
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(img1.transpose(1, 2, 0))
    ax[1].imshow(img2.transpose(1, 2, 0))
    plt.show()

