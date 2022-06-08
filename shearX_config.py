import os
import numpy as np
import torch
from torch import nn, optim
from torch.nn.parameter import Parameter
import kornia as K


### --- Config --- ###
augmentation_name = 'shearX'
main_parameter_name = 'shear'
aug_bounds = (None, None)
param_linspace = np.linspace(-2,2,400)

# dataset
target_param_val = 0.4
target_aug_constructor = K.geometry.transform.Shear
target_aug_constructor_args = {
    'shear': torch.Tensor([[target_param_val, 0]]),
}

init_param_val = 0.0
training_aug_constructor = K.geometry.transform.Shear
aug_learnable_params = {
    'shear': Parameter(torch.Tensor([[init_param_val,0]]))
}
training_aug_constructor_args = {}
aug_learnable_params_names = list(aug_learnable_params.keys())

# training
criterion_constructor = nn.MSELoss
criterion_constructor_args = {}

batch_size = 64
epochs = 1

optimizer_constructor = optim.SGD
optimizer_constructor_args = {
    'lr': 0.1,
    'momentum': 0.9,
}

scheduler_constructor = optim.lr_scheduler.CosineAnnealingLR
scheduler_constructor_args = {
    'T_max':200,
    'eta_min':0.1,
    'verbose':True,
}

use_scheduler=True
scheduler_warmup = 100
scheduler_freq = 10


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