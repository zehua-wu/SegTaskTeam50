# config.py
import os

BASE_DIR = './Zdata'  # Path to your dataset directory

CONFIG = {
    "train_images_dir": os.path.join(BASE_DIR, 'img_dir/train'),
    "train_annotations_dir": os.path.join(BASE_DIR, 'ann_dir/train'),
    "val_images_dir": os.path.join(BASE_DIR, 'img_dir/val'),
    "val_annotations_dir": os.path.join(BASE_DIR, 'ann_dir/val'),
    "batch_size": 64,
    "num_classes": 12,
    "learning_rate": 1e-2,
    "num_epochs": 5
}
