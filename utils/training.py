import torch
import numpy as np
from utils.iou_computing import calculate_IoU


# def train_one_epoch(model, train_loader, criterion, optimizer, device):
#     model.train()
#     train_loss = 0
#     correct = 0
#     total = 0

#     for inputs, targets in train_loader:
#         inputs, targets = inputs.to(device), targets.to(device)
#         outputs = model(inputs)
#         loss = criterion(outputs['out'], targets)

#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         train_loss += loss.item()
#         _, predicted = outputs['out'].max(1)
#         correct += (predicted == targets).sum().item()
#         total += targets.numel()

#     accuracy = 100. * correct / total
#     return train_loss, accuracy

# def validate(model, val_loader, criterion, device):
#     model.eval()
#     val_loss = 0
#     correct = 0
#     total = 0

#     with torch.no_grad():
#         for inputs, targets in val_loader:
#             inputs, targets = inputs.to(device), targets.to(device)
#             outputs = model(inputs)
#             loss = criterion(outputs['out'], targets)

#             val_loss += loss.item()
#             _, predicted = outputs['out'].max(1)
#             correct += (predicted == targets).sum().item()
#             total += targets.numel()

#     accuracy = 100. * correct / total
#     return val_loss, accuracy



def train_one_epoch(model, dataloader, optimizer, criterion, device):
    """
    Train the model for one epoch.
    
    Args:
        model: The PyTorch model.
        dataloader: DataLoader for the training dataset.
        optimizer: Optimizer for the model.
        criterion: Loss function.
        device: Device to perform computation on ('cuda' or 'cpu').
    
    Returns:
        avg_loss: Average loss over the training dataset.
    """
    model.train()  # 设置为训练模式
    total_loss = 0  # 累加损失

    for batch, (images, labels) in enumerate(dataloader):
        # 将数据移动到设备
        images, labels = images.to(device), labels.to(device)

        # 清空梯度
        optimizer.zero_grad()

        # 前向传播
        outputs = model(images)
        loss = criterion(outputs["out"], labels)

        # 反向传播和优化
        loss.backward()
        optimizer.step()

        # 累加损失
        total_loss += loss.item()

        # 打印中间过程（每 100 个 batch 输出一次）
        if batch % 10 == 0:
            current = batch * len(images)  # 已处理的样本数
            print(f"Batch {batch}/{len(dataloader)} - Loss: {loss.item():.4f}")

    # 返回平均损失
    avg_loss = total_loss / len(dataloader)
    print(f"Average Training Loss: {avg_loss:.4f}")

    return avg_loss




# def validate_one_epoch(model, dataloader, criterion, device, num_classes):
#     """
#     Validate the model for one epoch.
    
#     Args:
#         model
#         dataloader: DataLoader for the validation dataset.
#         criterion: Loss function.
#         device: Device to perform computation on ('cuda' or 'cpu').
#         num_classes: Number of classes in the dataset.
    
#     Returns:
#         avg_loss: Average loss over the validation dataset.
#         mean_iou: Mean IoU score for all classes.
#         pixel_accuracy: Overall pixel accuracy.
#     """
#     model.eval()  # Set model to evaluation mode
#     total_loss = 0
#     total_pixels = 0
#     correct_pixels = 0

#     # Initialize intersection and union arrays
#     total_intersection = np.zeros(num_classes)
#     total_union = np.zeros(num_classes)

#     with torch.no_grad():
#         for images, labels in dataloader:
#             # Move data to device
#             images, labels = images.to(device), labels.to(device)

#             # Forward pass
#             outputs = model(images)['out']  # 'out' for torchvision segmentation models
#             loss = criterion(outputs, labels)

#             # Update total loss
#             total_loss += loss.item()

#             # Compute pixel-wise predictions
#             preds = torch.argmax(outputs, dim=1)  # Shape: [batch_size, height, width]

#             # Calculate pixel accuracy
#             correct_pixels += (preds == labels).sum().item()
#             total_pixels += labels.numel()

#             # Calculate intersection and union for each class
#             for cls in range(num_classes):
#                 intersection = ((preds == cls) & (labels == cls)).sum().item()
#                 union = ((preds == cls) | (labels == cls)).sum().item()
#                 total_intersection[cls] += intersection
#                 total_union[cls] += union

#     # Calculate average loss
#     avg_loss = total_loss / len(dataloader)

#     # Calculate mean IoU
#     iou_scores = [
#         total_intersection[cls] / total_union[cls] if total_union[cls] > 0 else 0
#         for cls in range(num_classes)
#     ]
#     mean_iou = np.mean(iou_scores)

#     # Calculate pixel accuracy
#     pixel_accuracy = correct_pixels / total_pixels

#     return avg_loss, mean_iou, pixel_accuracy


def validate_one_epoch(model, dataloader, criterion, device, num_classes, logger):
    model.eval()  # Set model to evaluation mode
    total_loss = 0
    total_pixels = 0
    correct_pixels = 0

    # Initialize arrays for IoU
    total_intersection = np.zeros(num_classes)
    total_union = np.zeros(num_classes)

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)['out']
            loss = criterion(outputs, labels)

            # Update total loss
            total_loss += loss.item()

            # Compute predictions
            preds = torch.argmax(outputs, dim=1)

            # Pixel accuracy
            correct_pixels += (preds == labels).sum().item()
            total_pixels += labels.numel()

            # Update IoU arrays
            iou_scores, mean_iou = calculate_IoU(preds.cpu().numpy(), labels.cpu().numpy(), num_classes)

    # Calculate overall metrics
    avg_loss = total_loss / len(dataloader)
    pixel_accuracy = correct_pixels / total_pixels

    # Log results
    logger.info(f"Validation Loss: {avg_loss:.4f}")
    logger.info(f"Pixel Accuracy: {pixel_accuracy:.4f}")
    logger.info(f"Mean IoU: {mean_iou:.4f}")
    logger.info("Per-Class IoU:")
    for cls_idx, iou in enumerate(iou_scores):
        logger.info(f"  Class {cls_idx}: IoU = {iou:.4f}")

    return avg_loss, mean_iou, pixel_accuracy