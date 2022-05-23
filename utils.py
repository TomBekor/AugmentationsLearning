import numpy as np
import torch
import matplotlib.pyplot as plt


def plot_tensor(t):
    plt.imshow(np.array(t.permute(1,2,0)))