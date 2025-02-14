import pandas as pd
import torch
from torch.utils.data import Dataset
import torchvision.transforms.v2 as T
from PIL import Image

class CompetitionDataset(Dataset):
    def __init__(self, dataset, split='train', resolution=224):
        self.dataset = dataset
        self.split = split
        self.resolution = resolution
        self.mode = 'single'
        
        images_paths = dataset['img_path'].tolist()
        labels = dataset['label'].tolist()
        self.items = [{"image_path": image_path, "is_real": label} for image_path, label in zip(images_paths, labels)]

    def __len__(self):
        return len(self.dataset)

    def read_image(self, path):
        image = Image.open(path).convert('RGB')
        if self.split == "train":
            transforms = T.Compose([
                T.Resize(self.resolution + self.resolution // 8, interpolation=T.InterpolationMode.BILINEAR),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomVerticalFlip(p=0.5),
                T.RandomChoice([
                    T.RandomRotation(degrees=(-90, -90)),
                    T.RandomRotation(degrees=(90, 90)),
                    ], p=[0.5, 0.5]),
                T.RandomCrop(self.resolution),
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
                T.ToImage(),
                T.ToDtype(torch.float32, scale=True),
                ])
        else:
            transforms = T.Compose([
                T.Resize(self.resolution + self.resolution // 8, interpolation=T.InterpolationMode.BILINEAR),
                T.CenterCrop(self.resolution),
                T.ToImage(),
                T.ToDtype(torch.float32, scale=True),
            ])
        image = transforms(image)
        return image
    
    def __getitem__(self, i):
        sample = {
            "image_path": self.items[i]["image_path"],
            "image": self.read_image(self.items[i]["image_path"]),
            "is_real": torch.as_tensor([0 if self.items[i]["is_real"] is True else 1]),
        }
        return sample