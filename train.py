# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from models.deeplab import MyDeepLab
from torchvision import transforms
from data.dataset import SegmentationDataset
from configs.config import CONFIG



# Get data from DataLoader
train_loader = get_dataloader(CONFIG["train_images_dir"], CONFIG["train_annotations_dir"], CONFIG["batch_size"])
#train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True)
val_loader = get_dataloader(CONFIG["val_images_dir"],CONFIG["val_annotations_dir"], batch_size=CONFIG["batch_size"])


# Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    model.train()
    train_loss = 0
    correct = 0
    total = 0

    for images, masks in train_loader:
        images, masks = images.to(device), masks.to(device)

        outputs = model(images)
        loss = criterion(outputs, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += (predicted == masks).sum().item()
        total += masks.numel()

    train_accuracy = 100. * correct / total
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.2f}%")
