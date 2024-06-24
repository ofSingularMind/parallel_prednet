import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import UnsupervisedImageDataset
from monet_pytorch import Monet
import torch
import matplotlib.pyplot as plt
import numpy as np

# Model evaluation script
loadModel = True
delftblue = True
if delftblue:
    WEIGHTS_PATH = "/home/aledbetter/parallel_prednet/monet_pytorch_GH@michedev/model_weights/delftblue/"
    DATASET_PATH = "/scratch/aledbetter/multi_gen_shape_2nd_stage_for_objects/"
else:
    WEIGHTS_PATH = "/home/evalexii/Documents/Thesis/code/parallel_prednet/monet_pytorch_GH@michedev/alex/model_weights/laptop/"
    DATASET_PATH = "/home/evalexii/Documents/Thesis/code/parallel_prednet/data/animations/multi_gen_shape_strafing/frames/multi_gen_shape_2nd_stage_for_objects/"

# Load datset and create dataloader
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

# Create dataset and dataloader
dataset = UnsupervisedImageDataset(root_dir=DATASET_PATH, transform=transform)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# Initialize model, loss function, and optimizer. Load saved weights if resuming training.
monet = Monet.from_config(model='monet', dataset_width=64, dataset_height=64)
monet.load_state_dict(torch.load(WEIGHTS_PATH + 'monet_weights.pth'))
monet.eval()
print('Successfully loaded model weights...')

# Record loss value
with open(WEIGHTS_PATH + 'loss.txt', 'r') as f:
    best_loss = float(f.read())

# Iterate over the DataLoader
for batch_idx, images in enumerate(dataloader):
    # Forward pass
    outputs = monet(images)
    loss = outputs['loss']

    # Display outputs['mask'][0] and outputs['slot'][0] images in matplotlib.pyplot, each in their own row of a grid
    fig, axs = plt.subplots(3, outputs['mask'].shape[1])
    for i in range(outputs['mask'].shape[1]):
        mask = outputs['mask'][0, i].detach().numpy()
        recon = np.moveaxis(outputs['slot'][0, i].detach().numpy(), 0, -1)
        masked_recon = np.expand_dims(mask, -1) * recon
        axs[0, i].imshow(mask, cmap=plt.colormaps['Greys'])
        axs[1, i].imshow(recon)
        axs[2, i].imshow(masked_recon)
    plt.show()
    input('Press Enter to continue...')

# print('Finished Plotting')
