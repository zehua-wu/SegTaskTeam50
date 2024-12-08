import numpy as np

def calculate_IoU(preds, labels, num_classes):
    """
    Calculate Intersection over Union (IoU) for each class.

    Args:
        preds: Predicted segmentation map (tensor or array).
        labels: Ground truth segmentation map (tensor or array).
        num_classes: Number of classes in the dataset.

    Returns:
        iou_scores: List of IoU scores for each class.
        mean_iou: Mean IoU score across all classes.
    """
    total_intersection = np.zeros(num_classes)
    total_union = np.zeros(num_classes)

    # Compute IoU for each class
    for cls in range(num_classes):
        intersection = ((preds == cls) & (labels == cls)).sum().item()
        union = ((preds == cls) | (labels == cls)).sum().item()
        total_intersection[cls] += intersection
        total_union[cls] += union

    iou_scores = [
        total_intersection[cls] / total_union[cls] if total_union[cls] > 0 else 0
        for cls in range(num_classes)
    ]
    mean_iou = np.mean(iou_scores)
    return iou_scores, mean_iou
