import torch
import torch.nn as nn
import torch.optim as optim
from utils.metrics import compute_iou

def train_one_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    train_loss = 0
    correct = 0
    total = 0

    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += (predicted == targets).sum().item()
        total += targets.numel()

    accuracy = 100. * correct / total
    return train_loss, accuracy



def validate(model, val_loader, criterion, device, num_classes):
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    iou_score = 0

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            val_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += (predicted == targets).sum().item()
            total += targets.numel()

            iou_score += compute_iou(predicted, targets, num_classes)

    accuracy = 100. * correct / total
    avg_loss = val_loss / len(val_loader)
    miou = iou_score / len(val_loader)
    return val_loss, accuracy, miou
