import torch
import torch.nn as nn
import torch.optim as optim

def train_one_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    train_loss = 0
    correct = 0
    total = 0

    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs['out'], targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs['out'].max(1)
        correct += (predicted == targets).sum().item()
        total += targets.numel()

    accuracy = 100. * correct / total
    return train_loss, accuracy

def validate(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs['out'], targets)

            val_loss += loss.item()
            _, predicted = outputs['out'].max(1)
            correct += (predicted == targets).sum().item()
            total += targets.numel()

    accuracy = 100. * correct / total
    return val_loss, accuracy
