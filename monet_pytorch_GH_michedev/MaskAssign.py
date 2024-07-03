import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

import numpy as np
from scipy.optimize import linear_sum_assignment

import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import matplotlib.pyplot as plt
from model import Monet

from math import prod
import omegaconf
omegaconf.OmegaConf.register_new_resolver('prod', lambda *numbers: int(prod(float(x) for x in numbers)))
omegaconf.OmegaConf.register_new_resolver('sum', lambda *numbers: int(sum(float(x) for x in numbers)))
__all__ = ['Monet']


class RandomTranslation:
    def __init__(self, max_translate):
        self.max_translate = max_translate

    def __call__(self, img):
        # Convert PIL image to numpy array
        img = np.array(img)

        # Get image dimensions
        h, w, c = img.shape

        # Randomly choose translation values for x and y within the max_translate range
        tx = np.random.randint(-self.max_translate, self.max_translate)
        ty = np.random.randint(-self.max_translate, self.max_translate)

        # Define the crop region
        left = max(0, -tx)
        top = max(0, -ty)
        right = min(w, w - tx)
        bottom = min(h, h - ty)

        # Crop the image
        img_cropped = img[top:bottom, left:right]

        # Pad the image back to the original size
        img_padded = np.pad(img_cropped, ((max(0, ty), max(0, -ty)), (max(0, tx), max(0, -tx)), (0, 0)), mode='constant')

        # Convert numpy array back to PIL image
        img_padded = Image.fromarray(img_padded)
        return img_padded

# Define data augmentation transformations
data_augmentation = transforms.Compose([
    # transforms.RandomHorizontalFlip(),
    # RandomTranslation(max_translate=10),  # Randomly translate the image by up to 10 pixels
    # transforms.RandomRotation(10),
    # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    # transforms.RandomGrayscale(p=0.1),
    transforms.ToTensor()
])

basic_transform = transforms.Compose([
    transforms.ToTensor()
])

# Define a custom CNN
class ImageEncoder(nn.Module):
    def __init__(self, output_dim):
        super(ImageEncoder, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(128 * 8 * 8, output_dim)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 128 * 8 * 8)
        x = self.fc1(x)
        return x

class AttentionMechanism(nn.Module):
    def __init__(self, input_dim, d_k):
        super(AttentionMechanism, self).__init__()
        self.W_Q = nn.Linear(input_dim, d_k, bias=False)
        self.W_K = nn.Linear(input_dim, d_k, bias=False)
    
    def forward(self, X, Y):
        Q = self.W_Q(X)
        K = self.W_K(Y)
        attention_scores = (Q @ K.T) / torch.sqrt(torch.tensor(Q.size(-1), dtype=torch.float32))
        similarity_matrix = F.softmax(attention_scores, dim=-1)
        return similarity_matrix

class SimilarityMechanism(nn.Module):
    def __init__(self):
        super(SimilarityMechanism, self).__init__()

    # def batched_cosine_similarity(self, u, v):
    #     dot_product = torch.matmul(u, v.T)
    #     norm_u = torch.norm(u, dim=1, keepdim=True)
    #     norm_v = torch.norm(v, dim=1, keepdim=True)
    #     return dot_product / (norm_u * norm_v.T)

    def batched_cosine_similarity(self, u, v):
        u = F.normalize(u, p=2, dim=1)  # Normalize u
        v = F.normalize(v, p=2, dim=1)  # Normalize v
        return torch.matmul(u, v.T)

    def calc_similarity_matrix(self, input_vectors, existing_vectors):
        return self.batched_cosine_similarity(input_vectors, existing_vectors)
    
    def forward(self, X, Y):
        similarity_matrix = self.calc_similarity_matrix(X, Y)
        return similarity_matrix
        

def solve_assignment(similarity_matrix, threshold):
    # Convert similarity matrix to numpy array
    similarity_matrix_np = similarity_matrix.detach().cpu().numpy()
    
    # Apply threshold to similarity matrix
    similarity_matrix_np[similarity_matrix_np < threshold] = -100
    
    # Solve assignment problem
    row_ind, col_ind = linear_sum_assignment(-similarity_matrix_np)
    
    # Create assignment results with "no match" (-1) for similarities below threshold
    assignments = [-1] * similarity_matrix_np.shape[0]
    for i, j in zip(row_ind, col_ind):
        if similarity_matrix_np[i, j] >= threshold:
            assignments[i] = j
    
    return assignments

class MaskAssignPipeline(nn.Module):
    def __init__(self, feature_extractor, attention_mechanism, threshold):
        super(MaskAssignPipeline, self).__init__()
        self.feature_extractor = feature_extractor
        self.attention_mechanism = attention_mechanism
        self.threshold = threshold
    
    def forward(self, input_images, existing_images):
        input_vectors = self.feature_extractor(input_images)
        existing_vectors = self.feature_extractor(existing_images)
        similarity_matrix = self.attention_mechanism(input_vectors, existing_vectors)
        assignments = solve_assignment(similarity_matrix, self.threshold)
        return input_vectors, similarity_matrix, assignments

# Example function to create existing images and indices
def create_existing_images(input_images):
    # input images is a tensor of shape (batch_size, num_objects, 3, H, W)
    existing_images = []
    indices = list(range(len(input_images)))
    for img in input_images:
        augmented_img = data_augmentation(transforms.ToPILImage()(img))
        existing_images.append(augmented_img)
    
    # Shuffle existing images and indices together
    combined = list(zip(existing_images, indices))
    # random.shuffle(combined)
    existing_images[:], indices[:] = zip(*combined)
    
    return torch.stack(existing_images), indices

# Define a loss function (customize based on your application)
# def custom_loss(similarity_matrix, target_labels):
#     loss = 0
#     for i, label in enumerate(target_labels):
#         if label == -1:
#             loss += (1 - similarity_matrix[i].max())  # Penalize for no match
#         else:
#             loss += (1 - similarity_matrix[i, label])  # Penalize for incorrect match
#     return loss / len(target_labels)

def custom_loss(similarity_matrix, target_indices):
    # Create a one-hot encoded target matrix
    batch_size = len(target_indices)
    target_matrix = torch.zeros_like(similarity_matrix)
    for i, index in enumerate(target_indices):
        if index != -1:
            target_matrix[i, index] = 1.0
    
    # Compute the cross-entropy loss
    log_probs = F.log_softmax(similarity_matrix, dim=-1)
    loss = -torch.sum(target_matrix * log_probs) / batch_size
    
    return loss

class ImageDataset(Dataset):
    def __init__(self, root_dir, transform=None, device='cpu'):
        self.root_dir = root_dir
        self.transform = transform
        self.device = device
        self.image_paths = []

        for root, _, files in os.walk(root_dir):
            files.sort()  # Sort the files to ensure consistent order
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                    self.image_paths.append(os.path.join(root, file))

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
        
        return image


'''-----------------------------------------------------------------------------------------------------------------'''

# Setup
loadModel = True
batch_size = 2
input_dim = 512
d_k = 128
threshold = 0#0.1 / (batch_size/10) # Adjust threshold based on batch size
num_epochs = 100
total_loss = 0
save_interval = 100
learning_rate = 0.0005

# Load your images
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
image_root_dir = "/home/evalexii/Documents/Thesis/code/parallel_prednet/data/animations/multi_gen_shape_strafing/frames/multi_gen_shape_2nd_stage"
WEIGHTS_PATH = "/home/evalexii/Documents/Thesis/code/parallel_prednet/monet_pytorch_GH_michedev/MaskAssign/model_weights/"
dataset = ImageDataset(image_root_dir, transform=basic_transform, device=device)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

image_encoder = ImageEncoder(output_dim=input_dim)
# attention_mechanism = AttentionMechanism(input_dim=input_dim, d_k=d_k)
attention_mechanism = SimilarityMechanism()
pipeline = MaskAssignPipeline(image_encoder, attention_mechanism, threshold)
# Load weights if resuming training
if loadModel:
    try:
        pipeline.load_state_dict(torch.load(WEIGHTS_PATH + 'MaskAssign_weights.pth'))
        print('Successfully loaded model weights, resuming training...')
        with open(WEIGHTS_PATH + 'loss.txt', 'r') as f:
            best_loss = float(f.read())
        with open(WEIGHTS_PATH + 'log.txt', 'a+') as f:
            f.write(f'Resuming training with best loss: {best_loss}\n')
    except:
        print("Model weights not found or don't fit, beginning training from scratch...")
        best_loss = float('inf')
else:
    best_loss = float('inf')
    if os.path.exists(WEIGHTS_PATH + 'log.txt'):
        os.remove(WEIGHTS_PATH + 'log.txt')
    if os.path.exists(WEIGHTS_PATH + 'loss.txt'):
        os.remove(WEIGHTS_PATH + 'loss.txt')
    if os.path.exists(WEIGHTS_PATH + 'MaskAssign_weights.pth'):
        os.remove(WEIGHTS_PATH + 'MaskAssign_weights.pth')

# Example training loop
pipeline.to(device)

optimizer = torch.optim.Adam(pipeline.parameters(), lr=learning_rate)

# Initialize model, loss function, and optimizer. Load saved weights if resuming training.
monet = Monet.from_config(model='monet-SSM', dataset=None, scene_max_objects= 4, dataset_width=64, dataset_height=64).to(device)
monet_WEIGHTS_PATH = "/home/evalexii/Documents/Thesis/code/parallel_prednet/monet_pytorch_GH_michedev/model_weights/laptop/"
monet.load_state_dict(torch.load(monet_WEIGHTS_PATH + 'monet_weights.pth', map_location=torch.device(device)))
monet.eval()


print('Starting training with best loss:', best_loss)

for epoch in range(num_epochs):
    steps = 0
    for batch_idx, input_images in enumerate(data_loader):
        input_images = input_images.to(device)

        outputs = monet(input_images)
        masks = outputs['mask'][:,1:3,...]
        masks = torch.unsqueeze(masks, 2)
        slots = outputs['slot'][:,1:3,...]
        masked_recons = masks * slots
        input_images = masked_recons[0]
        
        # Create existing images and shuffled indices
        existing_images, target_indices = create_existing_images(masked_recons[1])
        existing_images = existing_images.to(device)

        # # Based on batch size, plot input images in one row, and existing images in another row
        # input_images_np = input_images.detach().cpu().numpy()
        # existing_images_np = existing_images.detach().cpu().numpy()
        # fig, ax = plt.subplots(2, batch_size*1, figsize=(15, 5))
        # for i in range(input_images_np.shape[0]):
        #     ax[0, i].imshow(np.transpose(input_images_np[i], (1, 2, 0)))
        #     ax[0, i].axis('off')
        #     ax[1, i].imshow(np.transpose(existing_images_np[i], (1, 2, 0)))
        #     ax[1, i].axis('off')
        # plt.show()

        
        # Forward pass
        input_vectors, similarity_matrix, assignments = pipeline(input_images, existing_images)
        
        # Compute loss (customize based on your specific application)
        loss = custom_loss(similarity_matrix, target_indices)
        
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        steps += 1
        if steps == save_interval:
            avg_loss = total_loss / (save_interval*batch_size)
            print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{batch_idx + 1}/{len(data_loader)}], Loss: [{avg_loss:.4f}]')
            print(f'Assignments: \t\t{assignments}')
            print(f'Target indices: \t{target_indices}')
            print(f'Similarity matrix: \n{F.softmax(similarity_matrix, dim=-1)}')
            # Log gradient norms
            # for name, param in pipeline.named_parameters():
            #     if param.grad is not None:
            #         print(f'Gradient norm for {name}: {param.grad.norm()}')
            total_loss = 0
            steps = 0
            if avg_loss < best_loss:
                best_loss = avg_loss
                print('Loss went down; Saving model...')
                torch.save(pipeline.state_dict(), WEIGHTS_PATH + 'MaskAssign_weights.pth')
                # Save loss value to file
                with open(WEIGHTS_PATH + 'loss.txt', 'w') as f:
                    f.write(str(best_loss))
            with open(WEIGHTS_PATH + 'log.txt', 'a+') as f:
                f.write(f'Epoch [{epoch + 1}/{num_epochs}], Step [{batch_idx + 1}/{len(data_loader)}], Loss: [{avg_loss:.4f}]')
                f.write(' - new best loss\n' if avg_loss < best_loss else '\n')
