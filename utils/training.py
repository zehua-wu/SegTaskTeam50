import torch
import numpy as np
from utils.metrics import calculate_iou_for_class, calculate_accuracy_for_class



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
    # Set into train mode
    model.train()  

    size = len(dataloader)
    
    # Store total loss
    total_loss = 0  

    # Traverse batches in training dataset
    for batch, (images, labels) in enumerate(dataloader):
        
        # Attach to device
        images, labels = images.to(device), labels.to(device)

        # Zero-grading
        optimizer.zero_grad()

        ### Forward
        # Get predicts
        outputs = model(images)

        # Compute loss
        loss = criterion(outputs["out"], labels)

        ### Back
        loss.backward()
        optimizer.step()

        # Acumulate loss
        total_loss += loss.item()

        
        if batch % 100 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Batch {batch}/{len(dataloader)} - Loss: {loss.item():.4f} - LR: {current_lr:.6f}")


        
    avg_loss = total_loss / size 

    print(f"Average Training Loss: {avg_loss:.4f}")

    return avg_loss



def validate_one_epoch(model, dataloader, criterion, device, num_classes, logger, class_names):
    model.eval()  
    size = len(dataloader)

    total_loss = 0
    total_pixels = 0
    correct_pixels = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            assert images.device == labels.device == next(model.parameters()).device, "Device mismatch detected!"


            outputs = model(images)['out']
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)
            correct_pixels += (preds == labels).sum().item()
            total_pixels += labels.numel()

            # Collect predictions and labels for each batch in TestDataLoader
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    # Concatenate
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    # Compute IoU and Accuracy for test dataset
    iou_scores, mean_iou = calculate_iou_for_class(all_preds, all_labels, num_classes)
    acc_scores = calculate_accuracy_for_class(all_preds, all_labels, num_classes)

    
    avg_loss = total_loss / size
    pixel_accuracy = correct_pixels / total_pixels

    # Logging
    logger.info("+--------------------------+---------+--------+")
    logger.info("| Class                    |   IoU   |   Acc  |")
    logger.info("+--------------------------+---------+--------+")
    for cls_idx, class_name in enumerate(class_names[:len(iou_scores)]):
        logger.info(f"| {class_name:<24}| {iou_scores[cls_idx]:7.2f} | {acc_scores[cls_idx]:7.2f} |")
    logger.info("+--------------------------+---------+--------+")
    logger.info(f"Overall Pixel Accuracy: {pixel_accuracy:.4f}")
    logger.info(f"Mean IoU: {mean_iou:.4f}")

    print()
    print()


    return avg_loss, mean_iou, pixel_accuracy

    print()
    print()

    return avg_loss, mean_iou, pixel_accuracy
