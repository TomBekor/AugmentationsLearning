import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

import torch
from torch.nn.parameter import Parameter
import torchvision
import kornia as K


def plot_tensor(t):
    plt.imshow(np.array(t.permute(1,2,0)))

def kornia_rotation(img, degrees):
    # unsqueeze img
    img = img.unsqueeze(0)

    # create transformation (rotation)
    alpha: float = 45.0  # in degrees
    angle: torch.tensor = torch.ones(1) * alpha

    # define the rotation center
    center: torch.tensor = torch.ones(1, 2)
    center[..., 0] = img.shape[3] / 2  # x
    center[..., 1] = img.shape[2] / 2  # y

    # define the scale factor
    scale: torch.tensor = torch.ones(1, 2)

    # compute the transformation matrix
    M: torch.tensor = K.geometry.get_rotation_matrix2d(center, angle, scale)

    _, _, h, w = img.shape
    img_warped: torch.tensor = K.geometry.warp_affine(img, M, dsize=(h, w))

    return img_warped.squeeze(0)

def show_sample(input, size: tuple = None):
    images = torch.stack(input, dim=0)
    out = torchvision.utils.make_grid(images, nrow=4, padding=5, pad_value=1)
    out_np: np.ndarray = K.utils.tensor_to_image(out)
    plt.figure(figsize=size)
    plt.imshow(out_np)
    plt.axis('off')
    plt.show()


def learning_grid(img_dict: dict, title, save=None):
    fig = plt.figure(figsize=(15,10))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                    nrows_ncols=(1, 3),  # creates 2x2 grid of axes
                    axes_pad=(2,10),  # pad between axes in inch.
                    )

    for ax, im_name in zip(grid, img_dict.keys()):
        # Iterating over the grid returns the Axes.
        ax.set_title(im_name, fontsize=20)
        im = np.array(img_dict[im_name].permute(1,2,0).detach().cpu())
        ax.imshow(im)

        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    fig.suptitle(title, fontsize=25)
    
    if save:
        plt.savefig(save)
        plt.close()
    else:
        plt.show()


def old_create_loss_map(model_constructor, linspace, dataloader, loss_function, save):
    loss_maps_path = 'loss_maps'
    os.makedirs(loss_maps_path, exist_ok=True)
    map_path = f'{loss_maps_path}/{save}.npy'
    if not os.path.exists(map_path):
        with torch.no_grad():
            p_losses = []
            for p in linspace:
                p_model = model_constructor(init_param_val=p)
                for i, data in enumerate(dataloader): # calc for one batch
                    source_images, target_images = data
                    source_images, target_images = source_images, target_images
                    output_images = p_model(source_images)
                    loss = loss_function(output_images, target_images)
                    p_losses.append(loss.item())
                    break
        if save:
            np.save(map_path, p_losses)
    else:
        p_losses = np.load(map_path)
    return p_losses


def create_loss_map(model_constructor, training_aug_constructor, training_aug_constructor_args,
                    aug_learnable_params, main_parameter_name, aug_bounds,
                    linspace, dataloader, loss_function, save):
    loss_maps_path = 'loss_maps'
    os.makedirs(loss_maps_path, exist_ok=True)
    map_path = f'{loss_maps_path}/{save}.npy'
    if not os.path.exists(map_path):
        with torch.no_grad():
            p_losses = []
            for p in linspace:
                p_aug_learnable_params = aug_learnable_params.copy()
                p_aug_learnable_params[main_parameter_name] = Parameter(torch.Tensor([p]))
                p_model = model = model_constructor(
                                    aug_constructor=training_aug_constructor,
                                    learnable_params=p_aug_learnable_params,
                                    aug_constructor_args=training_aug_constructor_args,
                                    aug_bounds=aug_bounds
                                )
                for i, data in enumerate(dataloader): # calc for one batch
                    source_images, target_images = data
                    source_images, target_images = source_images, target_images
                    output_images = p_model(source_images)
                    loss = loss_function(output_images, target_images)
                    p_losses.append(loss.item())
                    break
        if save:
            np.save(map_path, p_losses)
    else:
        p_losses = np.load(map_path)
    return p_losses
