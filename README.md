# ESE5460 2024fall Team50

## Project Overview

**SegTaskTeam50** is a semantic segmentation project based on **State Space Model, SegFormer and DeepLabV3+**. It provides a modular and extensible code structure, covering the complete workflow from data loading to model training. The project is designed for various semantic segmentation scenarios.

---

## Project Structure

```bash
SegTaskTeam50/
├── configs/                    
│   ├── config.py              # General configuration settings
│   └── config_mamba.py        # Mamba-specific configurations
│   
│
├── data/                      
│   └── dataset.py             # Dataset loading and preparation
│   
│
├── models/                    
│   ├── csm_triton.py          # Triton-based segmentation model
│   ├── csm65s.py              # Custom segmentation model version 65s
│   ├── deeplab.py             # DeepLab model implementation
│   ├── deeplabv3.py           # DeepLabV3 model implementation
│   ├── mamba2.py              # Mamba model version 2
│   ├── mamba2seg.py           # Mamba segmentation model
│   └── segformer.py           # SegFormer model definition
│   
│
├── outputs/
│   └── visualizations/        # Model predictions and visual outputs
│       
│
├── utils/                     
│   ├── checkpoint.py          # Checkpoint management for saving models
│   ├── metrics.py             # Evaluation metrics functions
│   ├── training.py            # Training utilities and scripts
│   ├── utils_logger.py        # Logging utilities
│   └── visualize.py           # Functions to visualize predictions/results
│   
│
├── train_deeplab.py           # Training script for DeepLab model
├── train_mamba2.py            # Training script for Mamba2 model
├── train_segformer.py         # Training script for SegFormer model
├── requirements.txt           # Dependencies and libraries
└── README.md                  # Project documentation


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

Step 3: Navigate to the cloned project directory and install the required Python dependencies:

```bash
pip install -r requirements.txt

```

Step 4: How to Train the Model
Ensure the dataset is prepared and stored in the specified directory.
Configure the configs/config.py file to adjust hyperparameters and paths.
Run the training script:
```bash
python train.py

```


Contribution Guidelines
Contributions are welcome! To contribute:

Fork the repository.
Create a new branch.
Submit a pull request with your changes.
Please ensure your code follows the project’s coding standards.

For any questions or feedback, feel free to contact the project author:
Huayi Tang [huayit@seas.upenn.edu],
Zehua Wu [zehuawu@seas.upenn.edu],
Haojia WU [wuhaojia@seas.upenn.edu]

