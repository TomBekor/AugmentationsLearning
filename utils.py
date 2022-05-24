import imp
import numpy as np
import matplotlib.pyplot as plt

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
    out = torchvision.utils.make_grid(images, nrow=4, padding=2)
    out_np: np.ndarray = K.utils.tensor_to_image(out)
    plt.figure(figsize=size)
    plt.imshow(out_np)
    plt.axis('off')
    plt.show()