from transformers import SegformerForSemanticSegmentation
import torch
import torch.nn as nn
import torch.nn.functional as F


class SegformerWrapper(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = create_segformer(num_classes)

    def forward(self, x):
        input_height, input_width = x.shape[2], x.shape[3]

        outputs = self.model(pixel_values=x)
        logits = outputs.logits

        logits = F.interpolate(
            logits,
            size=(input_height, input_width),
            mode="bilinear",
            align_corners=False,
        )

        return {"out": logits}


def create_segformer(num_classes):
    """
    Create a SegFormer model for semantic segmentation.

    Args:
        num_classes (int): Number of output classes
        pretrained (bool): Whether to use pretrained weights
    """
    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/mit-b0",  # change to mit-b1, mit-b2, etc.
        num_labels=num_classes,
        ignore_mismatched_sizes=True,
        reshape_last_stage=True,
    )
    return model


def create_wrapped_segformer(num_classes):
    return SegformerWrapper(num_classes)
