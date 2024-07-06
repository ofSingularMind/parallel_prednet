# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader
# from torchvision import transforms
# from dataset import UnsupervisedImageDataset
# # from monet_pytorch import Monet
# from model import Monet
# import torch
# import matplotlib.pyplot as plt
# import numpy as np
# import time

# # Do init stuff
# from math import prod
# from model import Monet
# import omegaconf
# omegaconf.OmegaConf.register_new_resolver('prod', lambda *numbers: int(prod(float(x) for x in numbers)))
# omegaconf.OmegaConf.register_new_resolver('sum', lambda *numbers: int(sum(float(x) for x in numbers)))
# __all__ = ['Monet']

# # Model evaluation script
# loadModel = True
# delftblue = False
# second_stage = True
# stage = "2nd" if second_stage else "1st"
# if delftblue:
#     WEIGHTS_PATH = "/home/aledbetter/parallel_prednet/monet_pytorch_GH_michedev/model_weights/delftblue/"
#     DATASET_PATH = "/scratch/aledbetter/multi_gen_shape_2nd_stage_for_objects/"
# else:
#     WEIGHTS_PATH = "/home/evalexii/Documents/Thesis/code/parallel_prednet/monet_pytorch_GH_michedev/model_weights/laptop/"
#     DATASET_PATH = f"/home/evalexii/Documents/Thesis/code/parallel_prednet/data/animations/multi_gen_shape_strafing/frames/multi_gen_shape_{stage}_stage_for_objects/"

# # Load datset and create dataloader
# transform = transforms.Compose([
#     transforms.Resize((64, 64)),
#     transforms.ToTensor()
# ])

# # Create dataset and dataloader
# dataset = UnsupervisedImageDataset(root_dir=DATASET_PATH, transform=transform)
# dataloader = DataLoader(dataset, batch_size=20, shuffle=False)

# # Initialize model, loss function, and optimizer. Load saved weights if resuming training.
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# monet = Monet.from_config(model='monet-SSM', dataset=None, scene_max_objects=4, dataset_width=64, dataset_height=64).to(device)
# monet.load_state_dict(torch.load(WEIGHTS_PATH + 'monet_weights.pth', map_location=torch.device(device)))
# monet.eval()
# print('Successfully loaded model weights.... Model on device: ', device)

# # Record loss value
# with open(WEIGHTS_PATH + 'loss.txt', 'r') as f:
#     best_loss = float(f.read())

# # plt.tick_params(
# #     axis='both',          # changes apply to the x-axis
# #     which='both',      # both major and minor ticks are affected
# #     bottom=False,      # ticks along the bottom edge are off
# #     top=False,         # ticks along the top edge are off
# #     labelbottom=False) # labels along the bottom edge are off

# # Iterate over the DataLoader
# for batch_idx, images in enumerate(dataloader):
#     # Forward pass
#     outputs = monet(images.to(device))
#     loss = outputs['loss']
#     likelihood = outputs['neg_log_p_x']
#     kl_mask = outputs['kl_mask']
#     kl_latent = outputs['kl_latent']
#     print(f'Loss: [{loss:.4f}], Neg Log P(x): [{likelihood:.4f}], KL Mask: [{kl_mask:.4f}], KL Latent: [{kl_latent:.4f}], 100K*MSE Loss: [{outputs["mse_loss"]:.4f}]')

#     # Display outputs['mask'][0] and outputs['slot'][0] images in matplotlib.pyplot, each in their own row of a grid
#     fig, axs = plt.subplots(5, outputs['mask'].shape[1], figsize=(12, 12))
    
#     # Turn off axes for all subplots
#     labels = ['Ground Truth', 'Mask', 'Reconstruction', 'Masked Reconstruction', 'Composite']
#     # for ax in axs.flatten():
#     #     ax.axis('off')
    
#     # Set labels for left-most subplots
#     for i, ax in enumerate(axs[:, 0]):  # Loop through the first column only
#         ax.set_ylabel(labels[i])
    
#     masked_recons = []
#     for i in range(outputs['mask'].shape[1]):
#         mask = outputs['mask'][0, i].detach().cpu().numpy()
#         recon = np.moveaxis(outputs['slot'][0, i].detach().cpu().numpy(), 0, -1)
#         masked_recon = np.expand_dims(mask, -1) * recon
#         masked_recons.append(masked_recon)
#         gt = np.moveaxis(images[0].detach().cpu().numpy(), 0, -1)
#         axs[0, 0].imshow(gt)
#         axs[1, i].imshow(mask, cmap=plt.colormaps['Greys'])
#         axs[2, i].imshow(recon, cmap=None)
#         axs[3, i].imshow(masked_recon, cmap=None)
#     axs[4,0].imshow(np.sum(masked_recons, axis=0))
#     plt.show()
#     # time.sleep(0.5)

#     # Show composite masked reconstruction
#     # fig, ax = plt.subplots(1, 1, figsize=(12, 12))


    
#     # input('Press Enter to continue...')
#     # if batch_idx == 0:
#     #     break

# # print('Finished Plotting')


import matplotlib.pyplot as plt
import numpy as np
import os
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import UnsupervisedImageDataset
# from monet_pytorch import Monet
# from model import Monet
import torch
import matplotlib.pyplot as plt
import numpy as np
# from multi_object_datasets import multi_dsprites

# Do init stuff
from math import prod
from model import Monet
import omegaconf
omegaconf.OmegaConf.register_new_resolver('prod', lambda *numbers: int(prod(float(x) for x in numbers)))
omegaconf.OmegaConf.register_new_resolver('sum', lambda *numbers: int(sum(float(x) for x in numbers)))
__all__ = ['Monet']


# Set paths
loadModel = True
delftblue = False
second_stage = True
stage = "2nd" if second_stage else "1st"
if delftblue:
    WEIGHTS_PATH = "/home/aledbetter/parallel_prednet/monet_pytorch_GH_michedev/model_weights/delftblue/"
    DATASET_PATH = "/scratch/aledbetter/multi_gen_shape_2nd_stage_for_objects/"
else:
    WEIGHTS_PATH = "/home/evalexii/Documents/Thesis/code/parallel_prednet/monet_pytorch_GH_michedev/model_weights/laptop/"
    DATASET_PATH = f"/home/evalexii/Documents/Thesis/code/parallel_prednet/data/animations/multi_gen_shape_strafing/frames/multi_gen_shape_{stage}_stage_for_objects/"
if not os.path.exists(WEIGHTS_PATH):
    os.makedirs(WEIGHTS_PATH)
    loadModel = False
    print('Model weights directory not found, beginning training from scratch...')

# set device
device = torch.device("cpu")
print(f"Using device: {device}")


# Load datset and create dataloader
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

# Create dataset and dataloader

batch_size = 1
dataset = UnsupervisedImageDataset(DATASET_PATH, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

# Initialize model, loss function, and optimizer. Load saved weights if resuming training.
monet = Monet.from_config(model='monet-SSM', dataset=None, scene_max_objects=4, dataset_width=64, dataset_height=64).to(device)

# Load weights if resuming training
try:
    monet.load_state_dict(torch.load(WEIGHTS_PATH + 'monet_weights.pth', map_location=torch.device(device)))
    print('Successfully loaded model weights, resuming training...')
    with open(WEIGHTS_PATH + 'loss.txt', 'r') as f:
        best_loss = float(f.read())
    with open(WEIGHTS_PATH + 'log.txt', 'a+') as f:
        f.write(f'Resuming training with best loss: {best_loss}\n')
except FileNotFoundError:
    print('Model weights not found, beginning training from scratch...')
    best_loss = float('inf')


monet.eval()


# Training loop
num_epochs = 10
save_interval = 1

print("Begin eval:")
print(f'Epoch [{1}/{num_epochs}], Step [{1}/{len(dataloader)}], Loss: {best_loss:.4f}')
for epoch in range(num_epochs):
    running_loss = 0.0
    running_likelihood = 0.0
    running_kl_mask = 0.0
    running_kl_latent = 0.0
    steps = 0
    # Iterate over the DataLoader
    for batch_idx, images in enumerate(dataloader):
        images = images.to(device)

        # # Zero the parameter gradients
        # optimizer.zero_grad()
        
        # Forward pass
        outputs = monet(images)
        loss = outputs['loss']
        
        # # Backward pass and optimize
        # loss.backward()
        # optimizer.step()
        
        running_loss += loss.item()
        running_likelihood += outputs['neg_log_p_x'].item()
        running_kl_mask += outputs['kl_mask'].item()
        running_kl_latent += outputs['kl_latent'].item()
        steps += 1
        if steps == save_interval:  # Print every 10 mini-batches
            avg_loss = running_loss / save_interval
            avg_likelihood = running_likelihood / save_interval
            avg_kl_mask = running_kl_mask / save_interval
            avg_kl_latent = running_kl_latent / save_interval
            print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{batch_idx + 1}/{len(dataloader)}], Loss: [{avg_loss:.4f}], Neg Log P(x): [{avg_likelihood:.4f}], KL Mask: [{avg_kl_mask:.4f}], KL Latent: [{avg_kl_latent:.4f}], 100K*MSE Loss: [{outputs["mse_loss"]:.4f}]')
            running_loss = 0.0
            running_likelihood = 0.0
            running_kl_mask = 0.0
            running_kl_latent = 0.0
            steps = 0
            # Display outputs['mask'][0] and outputs['slot'][0] images in matplotlib.pyplot, each in their own row of a grid
            fig, axs = plt.subplots(5, outputs['mask'].shape[1], figsize=(12, 12))
            
            # Turn off axes for all subplots
            labels = ['Ground Truth', 'Mask', 'Reconstruction', 'Masked Reconstruction', 'Composite']
            # for ax in axs.flatten():
            #     ax.axis('off')
            
            # Set labels for left-most subplots
            for i, ax in enumerate(axs[:, 0]):  # Loop through the first column only
                ax.set_ylabel(labels[i])
            
            masked_recons = []
            for i in range(outputs['mask'].shape[1]):
                mask = outputs['mask'][0, i].detach().numpy()
                recon = np.moveaxis(outputs['slot'][0, i].detach().numpy(), 0, -1)
                masked_recon = np.expand_dims(mask, -1) * recon
                masked_recons.append(masked_recon)
                gt = np.moveaxis(images[0].detach().numpy(), 0, -1)
                axs[0, 0].imshow(gt)
                axs[1, i].imshow(mask, cmap=plt.colormaps['Greys'])
                axs[2, i].imshow(recon, cmap=None)
                axs[3, i].imshow(masked_recon, cmap=None)
            axs[4,0].imshow(np.sum(masked_recons, axis=0))
            plt.show()

print('Finished Training')
