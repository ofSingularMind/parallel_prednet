import os
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import UnsupervisedImageDataset
# from monet_pytorch import Monet
from model import Monet
import torch
# from multi_object_datasets import multi_dsprites

# Do init stuff
from math import prod
from model import Monet
import omegaconf
omegaconf.OmegaConf.register_new_resolver('prod', lambda *numbers: int(prod(float(x) for x in numbers)))
omegaconf.OmegaConf.register_new_resolver('sum', lambda *numbers: int(sum(float(x) for x in numbers)))
__all__ = ['Monet']


loadModel = True
delftblue = True
if delftblue:
    WEIGHTS_PATH = "/home/aledbetter/parallel_prednet/monet_pytorch_GH@michedev/model_weights/delftblue/"
    DATASET_PATH = "/scratch/aledbetter/multi_gen_shape_2nd_stage_for_objects/"
else:
    WEIGHTS_PATH = "/home/evalexii/Documents/Thesis/code/parallel_prednet/monet_pytorch_GH@michedev/model_weights/laptop/"
    DATASET_PATH = "/home/evalexii/Documents/Thesis/code/parallel_prednet/data/animations/multi_gen_shape_strafing/frames/multi_gen_shape_2nd_stage_for_objects/"
if not os.path.exists(WEIGHTS_PATH):
    os.makedirs(WEIGHTS_PATH)
    loadModel = False
    print('Model weights directory not found, beginning training from scratch...')


# Load datset and create dataloader
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

# Create dataset and dataloader
dataset = UnsupervisedImageDataset(root_dir=DATASET_PATH, transform=transform)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# Initialize model, loss function, and optimizer. Load saved weights if resuming training.
monet = Monet.from_config(model='monet', dataset_width=64, dataset_height=64)
if loadModel:
    monet.load_state_dict(torch.load(WEIGHTS_PATH + 'monet_weights.pth'))
    print('Successfully loaded model weights, resuming training...')
    monet.eval()
    with open(WEIGHTS_PATH + 'loss.txt', 'r') as f:
        best_loss = float(f.read())
else:
    best_loss = float('inf')

# criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(monet.parameters(), lr=0.0001)

# Training loop
num_epochs = 10
save_interval = 10

print("Begin training:")
print(f'Epoch [{1}/{num_epochs}], Step [{1}/{len(dataloader)}], Loss: {best_loss:.4f}')
for epoch in range(num_epochs):
    running_loss = 0.0
    steps = 0
    # Iterate over the DataLoader
    for batch_idx, images in enumerate(dataloader):
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = monet(images)
        loss = outputs['loss']
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        steps += 1
        if steps == save_interval:  # Print every 10 mini-batches
            avg_loss = running_loss / save_interval
            print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{batch_idx + 1}/{len(dataloader)}], Loss: {avg_loss:.4f}')
            running_loss = 0.0
            steps = 0
            if avg_loss < best_loss:
                best_loss = avg_loss
                print('Loss went down; Saving model...')
                torch.save(monet.state_dict(), WEIGHTS_PATH + 'monet_weights.pth')
                # Save loss value to file
                with open(WEIGHTS_PATH + 'loss.txt', 'w') as f:
                    f.write(str(avg_loss))

print('Finished Training')
