# dataset.py
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from PIL import Image
import os


# Preprocessing
img_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

seg_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.squeeze(0).long())
])

class SegmentationDataset(Dataset):
    def __init__(self, image_dir, annotation_dir, img_transform=None, seg_transform=None):
        self.image_dir = image_dir
        self.annotation_dir = annotation_dir
        self.img_transform = img_transform
        self.seg_transform = seg_transform
        self.images = sorted(os.listdir(image_dir))
        self.segmentations = sorted(os.listdir(annotation_dir))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        seg_path = os.path.join(self.annotation_dir, self.segmentations[idx])

        img = Image.open(img_path).convert("RGB")
        seg = Image.open(seg_path).convert("L")

        if self.img_transform:
            img = self.img_transform(img)
        if self.seg_transform:
            seg = self.seg_transform(seg)

        return img, seg


def get_dataloader(img_path, label_path, batch_size, transform=None):
    dataset = SegmentationDataset(img_path, label_path, img_transform, seg_transform)
    # val_dataset = SegmentationDataset(CONFIG["val_images_dir"], CONFIG["val_annotations_dir"], img_transform, seg_transform)

    return DataLoader(dataset, batch_size, shuffle=True)

