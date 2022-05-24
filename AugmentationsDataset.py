import glob
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset

class AugmentationsDataset(Dataset):
    def __init__(self, dataset_glob_path,
                 source_transform, target_transform,
                 augmentations):
        super().__init__()
        self.dataset_image_paths = np.array(glob.glob(dataset_glob_path))
        self.source_transform = source_transform
        self.target_transform = target_transform
        self.augmentations = augmentations

    def __len__(self):
        return len(self.dataset_image_paths)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image_path = self.dataset_image_paths[idx]
        image = Image.open(image_path)
        source_image = self.source_transform(image) # tensored image
        target_image = self.target_transform(image) # tensored image

        for augment in self.augmentations:
            target_image = augment(target_image)        

        return source_image, target_image