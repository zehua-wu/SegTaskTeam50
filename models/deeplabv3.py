import torch.nn as nn
import torchvision.models.segmentation


def create_deeplabv3(num_classes, pretrained=False):
    """
    Create a DeepLabv3 model and modify its classifier for the specified number of classes.

    Args:
        num_classes (int): Number of output classes.
        pretrained (bool): Whether to load pretrained weights.

    Returns:
        model (torch.nn.Module): Modified DeepLabv3 model.
    """
    model = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=pretrained)

    # Modify the classifier for the specific number of classes
    model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=(1, 1))

    # Modify auxiliary classifier if it exists
    if hasattr(model, "aux_classifier") and model.aux_classifier is not None:
        model.aux_classifier[4] = nn.Conv2d(256, num_classes, kernel_size=(1, 1))

    return model
