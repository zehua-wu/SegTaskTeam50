# config.py
import os

# BASE_DIR = "./data/bdd100k"  # Path to your dataset directory
BASE_DIR = "/media/titan3/File_HuaYiT/VMamba/segmentation/data/bdd100k"  # Path to your dataset directory

CONFIG = {
    "train_images_dir": os.path.join(BASE_DIR, "images/10k/train"),
    "train_annotations_dir": os.path.join(BASE_DIR, "labels/sem_seg/masks/train"),
    "val_images_dir": os.path.join(BASE_DIR, "images/10k/val"),
    "val_annotations_dir": os.path.join(BASE_DIR, "labels/sem_seg/masks/val"),
    "batch_size": 12,
    "H": 512,
    "W": 512,
    "num_classes": 19,
    "learning_rate": 1e-5,
    "momentum": 0.9,
    "weight_decay": 5e-4,
    "nesterov": True,
    "num_epochs": 200,
    "resume_training": True,
    "visualize": True,
    "num_visualize_samples": 5,
    "visualize_output_dir": "./outputs/visualizations",
}


CLASS_NAMES = [
    "class0: road",
    "class1: sidewalk",
    "class2: building",
    "class3: wall",
    "class4: fence",
    "class5: pole",
    "class6: traffic light",
    "class7: traffic sign",
    "class8: vegetation",
    "class9: terrain",
    "class10: sky",
    "class11: person",
    "class12: rider",
    "class13: car",
    "class14: truck",
    "class15: bus",
    "class16: train",
    "class17: motocycle",
    "class18: bicycle",
]
