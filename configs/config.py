# config.py
import os

BASE_DIR = './data/Zdata'  # Path to your dataset directory

CONFIG = {
    "train_images_dir": os.path.join(BASE_DIR, 'img_dir/train'),
    "train_annotations_dir": os.path.join(BASE_DIR, 'ann_dir/train'),
    "val_images_dir": os.path.join(BASE_DIR, 'img_dir/val'),
    "val_annotations_dir": os.path.join(BASE_DIR, 'ann_dir/val'),
    "batch_size": 16,
    "num_classes": 12,
    "learning_rate": 1e-3,
    "momentum": 0.9,
    "weight_decay": 5e-4,
    "nesterov": True,
    "num_epochs": 2
}


CLASS_NAMES = [
    "class0", "class1", "class2", "class3", "class4", "class5",
    "class6", "class7", "class8", "class9", "class10", "class11",
    
]