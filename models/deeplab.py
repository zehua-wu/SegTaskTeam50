import torch
from torchvision.models.segmentation import deeplabv3_resnet50, deeplabv3_resnet101

def get_deeplab_model(num_classes=21, backbone='resnet50', pretrained=True):
    """
    Returns a DeepLabV3 model with a ResNet backbone.

    Args:
        num_classes (int): Number of output classes for the segmentation task.
        backbone (str): Backbone type, 'resnet50' or 'resnet101'.
        pretrained (bool): Whether to load pretrained weights.

    Returns:
        model (torch.nn.Module): The DeepLabV3 model.
    """
    if backbone == 'resnet50':
        model = deeplabv3_resnet50(pretrained=pretrained)
    elif backbone == 'resnet101':
        model = deeplabv3_resnet101(pretrained=pretrained)
    else:
        raise ValueError("Backbone must be 'resnet50' or 'resnet101'")

    # Modify the final classifier for the desired number of classes
    model.classifier[4] = torch.nn.Conv2d(256, num_classes, kernel_size=(1, 1))
    return model
