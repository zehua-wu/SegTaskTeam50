# dataset.py
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from PIL import Image
import os
import torch
import numpy as np

# Transform
class SegmentationTransform:
    def __init__(self, size):
        self.image_transform = transforms.Compose([
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.label_transform = transforms.Compose([
            transforms.Resize(size, interpolation=transforms.InterpolationMode.NEAREST),
        ])

    def __call__(self, image, label):
        image = self.image_transform(image)
        label = self.label_transform(label)

        label = np.array(label)

        if len(label.shape) == 3:
            label = label[:, :, 0]
        
        return image, label




# Customed dataset class

class SegmentationDataset(Dataset):
    def __init__(self, image_dir, label_dir, transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.image_files = sorted(os.listdir(image_dir))
        self.label_files = sorted(os.listdir(label_dir))
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_files[idx])
        label_path = os.path.join(self.label_dir, self.label_files[idx])

        # open image and annotation
        image = Image.open(image_path).convert("RGB")
        label = Image.open(label_path)

        # apply transform
        if self.transform:
            image, label = self.transform(image, label)  

        # squeeze dimension
        label = torch.tensor(label, dtype=torch.long)

        return image, label





# def get_dataloader(img_path, label_path, batch_size, transform=None):
#     dataset = SegmentationDataset(img_path, label_path, img_transform, seg_transform)
#     # val_dataset = SegmentationDataset(CONFIG["val_images_dir"], CONFIG["val_annotations_dir"], img_transform, seg_transform)

#     return DataLoader(dataset, batch_size, shuffle=True)



# Get dataloader
def get_dataloader(img_path, label_path, batch_size):
    transform = SegmentationTransform(size=(224, 224))
    dataset = SegmentationDataset(image_dir=img_path, label_dir=label_path, transform=transform)
    
    return DataLoader(dataset, batch_size, shuffle=True)