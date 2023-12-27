import os
import pandas as pd
import torch
from torchvision.io import read_image
import numpy as np
from torchvision import transforms

class CustomImageDataset(torch.utils.data.Dataset):
    def __init__(self, annotations_file, img_dir, transform=True, target_transform=None):
        super(CustomImageDataset, self).__init__()

        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):


        self.transform = transforms.Compose(
            [
                transforms.Resize((512, 512),antialias=True),
                transforms.RandomHorizontalFlip(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image.float())
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
    
