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

        
        if batch % 10 == 0:
            current = batch * len(images)  
            print(f"Batch {batch}/{len(dataloader)} - Loss: {loss.item():.4f}")

    
    avg_loss = total_loss / size 

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


def validate_one_epoch(model, dataloader, criterion, device, num_classes, logger, class_names):
    
    # Set model to evaluation mode
    model.eval()  

    size = len(dataloader)

    total_loss = 0
    total_pixels = 0
    correct_pixels = 0

    # Initialize arrays for IoU
    total_intersection = np.zeros(num_classes)
    total_union = np.zeros(num_classes)


    with torch.no_grad():

        # Traverse each datum in test dataset
        for images, labels in dataloader:

            # Attach to device
            images, labels = images.to(device), labels.to(device)

            # Forward 
            outputs = model(images)['out']
            loss = criterion(outputs, labels)

            # Accumulate total loss
            total_loss += loss.item()


            ### Argmax logits(model predictions)

            # [batch, num_classes, height, width] -> [batch, height, width]
            preds = torch.argmax(outputs, dim=1)



            ### Pixel accuracy

            # Correct pixel number in test dataset
            correct_pixels += (preds == labels).sum().item()

            # Total pixel number in test dataset
            total_pixels += labels.numel()

           

        ### Compute IoU and Accuracy for each class

        predicts_feed = preds.cpu().numpy()
        labels_feed = labels.cpu().numpy()

        iou_scores, mean_iou = calculate_iou_for_class(predicts_feed, labels_feed, num_classes)

        acc_scores = calculate_accuracy_for_class(predicts_feed, labels_feed, num_classes)
        


    # Calculate overall metrics
    avg_loss = total_loss / size
    pixel_accuracy = correct_pixels / total_pixels


    # Logging per-class results in table format
    logger.info("+-------------------+--------+--------+")
    logger.info("| Class            | IoU    | Acc    |")
    logger.info("+-------------------+--------+--------+")

    for cls_idx, class_name in enumerate(class_names):
        logger.info(f"| {class_name:<16} | {iou_scores[cls_idx]:.2f} | {acc_scores[cls_idx]:.2f} |")
    logger.info("+-------------------+--------+--------+")
    logger.info(f"Overall Pixel Accuracy: {pixel_accuracy:.4f}")
    logger.info(f"Mean IoU: {mean_iou:.4f}")

    return avg_loss, mean_iou, pixel_accuracy