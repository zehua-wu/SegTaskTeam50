import torch

def compute_iou(predictions, targets, num_classes):
    iou = []
    for cls in range(num_classes):
        intersection = torch.logical_and(predictions == cls, targets == cls).sum().item()
        union = torch.logical_or(predictions == cls, targets == cls).sum().item()
        if union > 0:
            iou.append(intersection / union)
    return sum(iou) / len(iou)