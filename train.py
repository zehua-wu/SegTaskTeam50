# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from models.deeplab import MyDeepLab
from torchvision import transforms
from data.dataset import SegmentationDataset
from data.dataset import get_dataloader
from configs.config import CONFIG
from utils.training import train_one_epoch, validate
from utils.metrics import compute_iou


# Get data from DataLoader
train_loader = get_dataloader(CONFIG["train_images_dir"], CONFIG["train_annotations_dir"], CONFIG["batch_size"])
#train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True)
val_loader = get_dataloader(CONFIG["val_images_dir"],CONFIG["val_annotations_dir"], batch_size=CONFIG["batch_size"])


# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"using device: {device}")

# Model
model = MyDeepLab(num_classes=CONFIG["num_classes"]).to(device)


# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), 
                    lr=CONFIG["learning_rate"], 
                    momentum=CONFIG["momentum"], 
                    weight_decay=CONFIG["weight_decay"], 
                    nesterov=CONFIG["nesterov"])


# Training Loop

num_epochs = CONFIG["num_epochs"]

for epoch in range(num_epochs):
    
    train_loss, train_accuracy = train_one_epoch(model, train_loader, criterion, optimizer, device)

    
    val_loss, val_accuracy, iou_score = validate(model, val_loader, criterion, device, CONFIG["num_classes"])
    

    
    print(f"Epoch {epoch + 1}/{num_epochs}")
    print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%")
    print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%, IoU: {iou_score:.4f}")
