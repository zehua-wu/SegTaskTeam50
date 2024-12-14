# SegTaskTeam50

project/
├── data/
│ └── dataset.py # Dataset and data loaders
|
├── models/
│ └── deeplabv3.py # Model definitions
|
├── utils/
│ └── helpers.py # Helper functions
|
├── configs/
│ └── config.py # Configuration settings
|
├── train.py # Training script
└── requirements.txt # Dependencies



branch_name = "branch6"
repo_url = "https://github.com/zehua-wu/SegTaskTeam50.git"

!git clone -b {branch_name} {repo_url}


%cd SegTaskTeam50 


from google.colab import drive
drive.mount('/content/drive')


!pip install -r requirements.txt



pip install --upgrade torch torchvision


!python train.py





function ClickConnect(){
    console.log("Working");
    const button = document.querySelector("YOUR-SELECTOR-HERE");
    if (button) button.click();
}
setInterval(ClickConnect,60000);


