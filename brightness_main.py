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

from utils import plot_tensor, show_sample, learning_grid, create_loss_map

### --- Config --- ###
augmentation = 'brightness'
# dataset
target_param_val = 10.

init_param_val = 0.
# training
batch_size = 64
epochs = 1
lr=50
momentum=0.99
scheduler_iters = 10
use_scheduler=False

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# paths:
run_name = f'{augmentation}-init-{init_param_val}_target-{target_param_val}'

figures_dir = f'figures'
run_figures_dir = f'{figures_dir}/{run_name}'

gifs_dir = f'gifs'
run_gif_name = f'{gifs_dir}/{run_name}.gif'

os.makedirs(run_figures_dir, exist_ok=True)
os.makedirs(gifs_dir, exist_ok=True)

source_transform = transforms.Compose([
    transforms.ToTensor()
])

target_augmentations = transforms.Compose([
    K.augmentation.container.ImageSequential(
        K.augmentation.ColorJiggle(brightness=target_param_val)
    )
])
target_transform = transforms.Compose([
    transforms.ToTensor(),
    target_augmentations
])

glob_path = '../data/tiny-imagenet-200/train/*/images/*'
dataset = AugmentationsDataset(glob_path,
                               source_transform, target_transform,)

dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=True)

class AugmentationNetwork(nn.Module):
    def __init__(self, init_param_val):
        super().__init__()
        self.trans_param = Parameter(torch.Tensor([init_param_val]))

    def forward(self, x):
        clamped_param = torch.clamp(self.trans_param, min=0.0)
        # clamped_param = torch.clamp(self.trans_param, min=-360.0, max=360.0)
        # clamped_param = torch.remainder(self.trans_param, 360.0)
        augmenter = K.augmentation.container.ImageSequential(
            K.augmentation.ColorJiggle(brightness=target_param_val)
        )
        out = augmenter(x)
        return out


model = AugmentationNetwork(init_param_val=init_param_val)
model.to(device)

criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
# optimizer = optim.Adam(model.parameters(), lr=lr)
# scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.93, last_epoch=-1, verbose=False)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=0.1, verbose=True)

losses = []
lrs = []
p_progress = []
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
        
        lrs.append(scheduler.get_last_lr())
        p_progress.append(model.trans_param.item())
        if use_scheduler:
            if i % scheduler_iters == scheduler_iters - 1 and i > 100:
                scheduler.step()


        if i == 1000:
            break

print('Finished Training.')