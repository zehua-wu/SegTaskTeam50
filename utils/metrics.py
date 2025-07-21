import numpy as np

# def calculate_IoU(preds, labels, num_classes):
#     """
#     Calculate Intersection over Union (IoU) for each class.

#     Args:
#         preds: Predicted segmentation map (tensor or array).
#         labels: Ground truth segmentation map (tensor or array).
#         num_classes: Number of classes in the dataset.

#     Returns:
#         iou_scores: List of IoU scores for each class.
#         mean_iou: Mean IoU score across all classes.
#     """
#     total_intersection = np.zeros(num_classes)
#     total_union = np.zeros(num_classes)

#     # Compute IoU for each class
#     for cls in range(num_classes):
#         intersection = ((preds == cls) & (labels == cls)).sum().item()
#         union = ((preds == cls) | (labels == cls)).sum().item()
#         total_intersection[cls] += intersection
#         total_union[cls] += union

#     iou_scores = [
#         total_intersection[cls] / total_union[cls] if total_union[cls] > 0 else 0
#         for cls in range(num_classes)
#     ]
#     mean_iou = np.mean(iou_scores)
#     return iou_scores, mean_iou


def calculate_iou_for_class(preds, labels, num_classes):
    
    """
    Calculate IoU for each class and the mean IoU for each epoch
    
    Args:
        pred: model predictions
        labels: ground truth labels
        num_classes: total labels in this semantic segmentation task
    
    Returns:
        iou_scores: IoU score for each class for each epoch
        mean_iou: mean IoU across all classes for each epoch
    """
    

    # Store intersection for each class
    total_intersection = np.zeros(num_classes)

    # Store union for each class
    total_union = np.zeros(num_classes)


    # Compute IoU for each class
    for cls in range(num_classes):

        mask = labels != 255
        
        intersection = ((preds == cls) & (labels == cls) & mask).sum().item()
        union = ((preds == cls) | (labels == cls) & mask).sum().item()
        total_intersection[cls] += intersection
        total_union[cls] += union


    iou_scores = [
        total_intersection[cls] / total_union[cls] if total_union[cls] > 0 else 0
        for cls in range(num_classes)
    ]
    mean_iou = np.mean(iou_scores)

    return iou_scores, mean_iou



def calculate_accuracy_for_class(preds, labels, num_classes):
    """
    Args:
        preds: moel predicts 
        labels: Ground truth labels
        num_classes: total labels in this semantic segmentation task

    Returns:
        acc_scores (list): Accuracy score for each class.
    """

    
    total_correct = np.zeros(num_classes)
    total_label = np.zeros(num_classes)

    # Compute IoU and Accuracy for each class
    for cls in range(num_classes):
        mask = labels != 255
        correct = ((preds[mask] == cls) & (labels[mask] == cls)).sum()
        label_count = (labels[mask] == cls).sum()
        total_correct[cls] += correct
        total_label[cls] += label_count


    # Accuracy scores
    acc_scores = [
        total_correct[cls] / total_label[cls] if total_label[cls] > 0 else 0
        for cls in range(num_classes)
    ]

    
    return acc_scores


