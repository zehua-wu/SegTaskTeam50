# config.py
import os

BASE_DIR = '/content/drive/MyDrive/BDD100k'  # Path to your dataset directory

CONFIG = {
    "train_images_dir": os.path.join(BASE_DIR, 'img_dir/train'),
    "train_annotations_dir": os.path.join(BASE_DIR, 'ann_dir/train'),
    "val_images_dir": os.path.join(BASE_DIR, 'img_dir/val'),
    "val_annotations_dir": os.path.join(BASE_DIR, 'ann_dir/val'),
    "batch_size": 16,
    "num_classes": 19,
    "learning_rate": 1e-3,
    "momentum": 0.9,
    "weight_decay": 5e-4,
    "nesterov": True,
    "num_epochs": 200,
    "resume_training": False,
    "visualize":True,
    "num_visualize_samples":5,
    "visualize_output_dir": "./outputs/visualizations"
}


CLASS_NAMES = [
    "class0: road", "class1: sidewalk", "class2: building", "class3: wall", "class4: fence", "class5: pole",
    "class6: traffic light", "class7: traffic sign", "class8: vegetation", "class9: terrain", "class10: sky", "class11: person",
    "class12: rider", "class13: car", "class14: truck", "class15: bus", "class16: train", "class17: motocycle", 
    "class18: bicycle"
]
