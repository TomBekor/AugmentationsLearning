import glob
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset



class AugmentationsDataset(Dataset):
    def __init__(self, dataset_glob_path, source_transform, target_transform):
        super().__init__()
        self.dataset_image_paths = np.array(glob.glob(dataset_glob_path))
        self.source_transform = source_transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.dataset_image_paths)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image_path = self.dataset_image_paths[idx]
        image = Image.open(image_path)
        source_image = self.source_transform(image) # input image
        target_image = self.target_transform(image) # augmented image

        return source_image, target_image