import imp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

import torch
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


def learning_grid(img_dict: dict, save=None):
    fig = plt.figure(figsize=(15,20))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                    nrows_ncols=(1, 3),  # creates 2x2 grid of axes
                    axes_pad=(2,10),  # pad between axes in inch.
                    )

    for ax, im_name in zip(grid, img_dict.keys()):
        # Iterating over the grid returns the Axes.
        ax.set_title(im_name, fontsize=20)
        im = np.array(img_dict[im_name].permute(1,2,0).detach())
        ax.imshow(im)

        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    
    if save:
        plt.savefig(save)
        plt.close()
    else:
        plt.show()
