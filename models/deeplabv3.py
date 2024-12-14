import torch.nn as nn
import torchvision.models.segmentation


def create_deeplabv3_mobilenet(num_classes, pretrained=True):
    """
    Create a DeepLabv3 model with MobileNetV3-Large backbone and modify its classifier for the specified number of classes.

    Args:
        num_classes (int): Number of output classes.
        pretrained (bool): Whether to load pretrained weights.

    Returns:
        model (torch.nn.Module): Modified DeepLabv3 model.
    """
    # Use weights if pretrained=True, otherwise set to None
    weights = DeepLabV3_MobileNet_V3_Large_Weights.COCO_WITH_VOC_LABELS_V1 if pretrained else None

    # Load the DeepLabv3 model with MobileNetV3-Large backbone
    model = torchvision.models.segmentation.deeplabv3_mobilenet_v3_large(weights=weights)

    # Modify the main classifier for the specific number of classes
    model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=(1, 1))

    # Modify the auxiliary classifier for the specific number of classes
    if hasattr(model, "aux_classifier") and model.aux_classifier is not None:
        # The MobileNetV3 backbone outputs 40 channels for auxiliary features
        model.aux_classifier = nn.Sequential(
            nn.Conv2d(40, 256, kernel_size=(1, 1)),  # Adjust input from 40 to 256 channels
            nn.ReLU(),
            nn.Conv2d(256, num_classes, kernel_size=(1, 1))  # Output the required number of classes
        )

    return model
