from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os

import numpy as np


class ImageData(Dataset):
    def __init__(self, image_paths_a, image_paths_b, transforms_a, transforms_b):
        self.image_paths = list(zip(image_paths_a, image_paths_b))
        self.transforms_a = transforms_a
        self.transforms_b = transforms_b

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        A, B = self.image_paths[index]
        im_a = Image.open(A).convert("RGB")
        im_b = Image.open(B).convert("RGB")
        return self.transforms_a(im_a), self.transforms_b(im_b)


def get_image_data_loader(data_path_a, data_path_b, transforms_a, transforms_b, batch=4, shuffle=True):
    image_paths_a = [os.path.join(data_path_a, img) for img in os.listdir(data_path_a)]
    image_paths_b = [os.path.join(data_path_b, img) for img in os.listdir(data_path_b)]
    return DataLoader(ImageData(image_paths_a, image_paths_b, transforms_a, transforms_b), batch_size=batch, shuffle=shuffle)


def calculate_mean_and_std(data_loader):
    mean = 0.
    std = 0.
    for images in data_loader:
        batch_samples = images.size(0)  # batch size (the last batch can have smaller size!)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)

    mean /= len(data_loader.dataset)
    std /= len(data_loader.dataset)

    return mean, std