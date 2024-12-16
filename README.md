# ESE5460 2024fall Team50

## Project Overview

**SegTaskTeam50** is a semantic segmentation project based on **State Space Model, SegFormer and DeepLabV3+**. It provides a modular and extensible code structure, covering the complete workflow from data loading to model training. The project is designed for various semantic segmentation scenarios.

---

## Project Structure

```bash
project/
├── data/
│   └── dataset.py       # Data loading and dataset definition
├── models/
│   └── deeplabv3.py     # DeepLabV3 model definition
├── utils/
│   └── helpers.py       # Utility functions
├── configs/
│   └── config.py        # Configuration settings
├── train.py             # Training script
└── requirements.txt     # Project dependencies
```

To clone the repository and conduct experiments in Colab, here is the instruction:
Step 1: Clone the specific branch branch6, run the following command:
```bash
branch_name="branch6"
repo_url="https://github.com/zehua-wu/SegTaskTeam50.git"
git clone -b ${branch_name} ${repo_url}

```

Step 2: Mount the drive for dataset:
```bash
from google.colab import drive
drive.mount('/content/drive')

```
You are suggested to put BDD100k semantic segmentation dataset into your Google drive and organize in the way what we setup in config.py.



