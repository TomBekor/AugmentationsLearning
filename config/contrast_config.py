import os
import numpy as np
import torch
from torch import nn, optim
from torch.nn.parameter import Parameter
import kornia as K
from kAugmentations.kContrast import *


### --- Config --- ###
augmentation_name = 'contrast'
main_parameter_name = 'contrast_factor'
aug_bounds = (0.0, None)
param_linspace = np.linspace(0.0,20.,400)

# dataset
target_param_val = 3.
target_aug_constructor = K.enhance.AdjustContrast
target_aug_constructor_args = {
    'contrast_factor': target_param_val,
}

kAugmentation = kContrast
init_param_val = 0.

# training
criterion_constructor = nn.MSELoss
criterion_constructor_args = {}

batch_size = 64
epochs = 1

optimizer_constructor = optim.SGD
optimizer_constructor_args = {
    'lr': 0.01,
    'momentum': 0.9,
}

scheduler_constructor = optim.lr_scheduler.CosineAnnealingLR
scheduler_constructor_args = {
    'T_max':100,
    'eta_min':0.1,
    'verbose':True,
}

scheduler_warmup = 100
scheduler_freq = 10
use_scheduler=False

early_stopping = 500

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# paths:
run_name = f'{augmentation_name}-init-{init_param_val}_target-{target_param_val}'

figures_dir = f'figures'
run_figures_dir = f'{figures_dir}/{run_name}'

gifs_dir = f'gifs'
run_gif_name = f'{gifs_dir}/{run_name}.gif'

os.makedirs(run_figures_dir, exist_ok=True)
os.makedirs(gifs_dir, exist_ok=True)
os.makedirs(f'{run_figures_dir}/results', exist_ok=True)
os.makedirs(f'{run_figures_dir}/learning_progress', exist_ok=True)
