# train.py
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from data.dataset import get_dataloader
from configs.config import CONFIG, CLASS_NAMES
from utils.training import train_one_epoch, validate_one_epoch
from utils.utils_logger import setup_logger

from models.deeplabv3 import create_deeplabv3


# Get data from DataLoader
train_loader = get_dataloader(CONFIG["train_images_dir"], CONFIG["train_annotations_dir"], CONFIG["batch_size"])
#train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True)
test_loader = get_dataloader(CONFIG["val_images_dir"],CONFIG["val_annotations_dir"], batch_size=CONFIG["batch_size"])


# Model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"using device: {device}")



model = create_deeplabv3(num_classes=12, pretrained=False).to(device)
#model = MyDeepLab(CONFIG["num_classes"])


# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), 
                    lr=CONFIG["learning_rate"], 
                    momentum=CONFIG["momentum"], 
                    weight_decay=CONFIG["weight_decay"], 
                    nesterov=CONFIG["nesterov"])




# Training Loop

num_epochs = CONFIG["num_epochs"]


# Initialize logger
logger = setup_logger(log_file="training_validation.log")

for epoch in range(num_epochs):
    logger.info(f"Epoch {epoch + 1}/{num_epochs}")

    # Call train function
    train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
    logger.info(f"Train Loss: {train_loss:.4f}")

    # Call validation function
    val_loss, val_miou, val_accuracy = validate_one_epoch(model, 
                                                         test_loader, 
                                                         criterion, 
                                                         device, 
                                                         num_classes=12, 
                                                         logger=logger,
                                                         class_names=CLASS_NAMES)

   
