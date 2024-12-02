# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.models.segmentation import deeplabv3_resnet50
from dataset import SegmentationDataset
from config import CONFIG

# Preprocessing
img_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
seg_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.squeeze(0).long())
])

# Dataset and DataLoader
train_dataset = SegmentationDataset(CONFIG["train_images_dir"], CONFIG["train_annotations_dir"], img_transform, seg_transform)
val_dataset = SegmentationDataset(CONFIG["val_images_dir"], CONFIG["val_annotations_dir"], img_transform, seg_transform)

train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=CONFIG["batch_size"], shuffle=False)

# Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = deeplabv3_resnet50(pretrained=False, num_classes=CONFIG["num_classes"]).to(device)

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=CONFIG["learning_rate"], momentum=0.9, weight_decay=5e-4, nesterov=True)

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
        loss = criterion(outputs['out'], masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs['out'].max(1)
        correct += (predicted == masks).sum().item()
        total += masks.numel()

    train_accuracy = 100. * correct / total
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.2f}%")
