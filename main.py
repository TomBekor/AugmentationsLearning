import os
import glob

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parameter import Parameter

import torchvision
from torchvision import transforms

import kornia as K

import matplotlib.pyplot as plt
from PIL import Image

from AugmentationsDataset import *

from utils import plot_tensor, show_sample, learning_grid



### --- Config --- ###
# dataset
target_rotation_deg = 90.

# training
init_rotation_deg = 50.
batch_size = 64
epochs = 1
lr=0.1
# momentum=0.9

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# paths:
run_name = f'init-{init_rotation_deg}_target-{target_rotation_deg}'

figures_dir = f'figures'
run_figures_dir = f'{figures_dir}/{run_name}'

gifs_dir = f'gifs'
run_gif_name = f'{gifs_dir}/{run_name}.gif'

os.makedirs(run_figures_dir, exist_ok=True)
os.makedirs(gifs_dir, exist_ok=True)
### -------------- ###


source_transform = transforms.Compose([
    transforms.ToTensor()
])

target_augmentations = transforms.Compose([
    K.augmentation.RandomRotation(degrees=[target_rotation_deg,target_rotation_deg], p=1), # nn.Module
])

target_transform = transforms.Compose([
    transforms.ToTensor(),
    target_augmentations
])

glob_path = '../data/tiny-imagenet-200/train/*/images/*'
dataset = AugmentationsDataset(glob_path,
                               source_transform, target_transform,)

# show_sample(dataset[0])

dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=True)

# Model Class
class AugmentationNetwork(nn.Module):
    def __init__(self, init_rot_degree):
        super().__init__()
        self.trans_param = Parameter(torch.Tensor([init_rot_degree]),
                                     requires_grad=True)

    def forward(self, x):
        rand_rotation = K.augmentation.RandomRotation(
                        torch.cat([self.trans_param, self.trans_param]), p=1) # torch cat prevernts parameter from casting to float.
        out = rand_rotation(x)
        return out

# Model initialization
model = AugmentationNetwork(init_rot_degree=init_rotation_deg)
model.to(device)

for name, param in model.named_parameters():
    if param.requires_grad:
        print(name, param.data)

criterion = nn.MSELoss()
# optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
optimizer = optim.Adam(model.parameters(), lr=lr)

losses = []
for epoch in range(epochs):

    running_loss = 0.0
    for i, data in enumerate(dataloader):
        source_images, target_images = data
        source_images, target_images = source_images.to(device), target_images.to(device)
        
        optimizer.zero_grad()

        output_images = model(source_images)
        loss = criterion(output_images, target_images)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        running_loss += loss.item()
        print_every = 10
        if i % print_every == print_every - 1:
            print(f'[Epoch: {epoch+1} | Batch: {i+1} | MSE Loss: {running_loss/print_every:.3f} | {model.trans_param.item()}]')
            running_loss = 0.0

            with torch.no_grad():
                img_dict = {
                    "source": dataset[0][0],
                    "model's output": model(dataset[0][0]).squeeze(),
                    "target": dataset[0][1]
                }
                learning_grid(img_dict, save=f'{run_figures_dir}/epoch-{epoch+1:02}_batch-{i+1:05}.png')
                
        if i == 1000:
            break

print('Finished Training.')

fp_in = f"{run_figures_dir}/epoch-*"
fp_out = run_gif_name

imgs = (Image.open(f) for f in sorted(glob.glob(fp_in)))
img = next(imgs)  # extract first image from iterator
img.save(fp=fp_out, format='GIF', append_images=imgs,
         save_all=True, duration=75, loop=0)