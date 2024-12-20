import torch
import torch.nn as nn
from torchvision.models import resnet50

# Include your ResNetBackbone, ASPP, and MyDeepLab here
class ResNetBackbone(nn.Module):
    # Backbone implementation...
    def __init__(self):
        super(ResNetBackbone, self).__init__()
        resnet = resnet50(pretrained=False)  # Use pre-trained weights if needed
        # Remove the fully connected and avgpool layers
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])  # Outputs [Batch, 2048, 7, 7]

    def forward(self, x):
        return self.backbone(x)

class ASPP(nn.Module):
    # ASPP implementation...
    def __init__(self, in_channels, out_channels):
        super(ASPP, self).__init__()
        self.conv_aspp1 = nn.Conv2d(in_channels, 256, kernel_size=1, stride=1, padding=0)
        self.conv_aspp2 = nn.Conv2d(in_channels, 256, kernel_size=3, stride=1, padding=6, dilation=6)
        self.conv_aspp3 = nn.Conv2d(in_channels, 256, kernel_size=3, stride=1, padding=12, dilation=12)
        self.conv_aspp4 = nn.Conv2d(in_channels, 256, kernel_size=3, stride=1, padding=18, dilation=18)
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)

        # After global average pooling
        self.conv_aspp5 = nn.Conv2d(in_channels, 256, kernel_size=1, stride=1, padding=0)

        # Reduce channels to the desired output (e.g., 256 for standard DeepLabv3)
        self.out_conv = nn.Conv2d(1280, out_channels, kernel_size=1, stride=1, padding=0)

        self.bn = nn.BatchNorm2d(256)
        self.relu = nn.ReLU()

    def forward(self, x):
        # Apply ASPP branches
        x1 = self.relu(self.bn(self.conv_aspp1(x)))
        x2 = self.relu(self.bn(self.conv_aspp2(x)))
        x3 = self.relu(self.bn(self.conv_aspp3(x)))
        x4 = self.relu(self.bn(self.conv_aspp4(x)))

        # Global average pooling
        x5 = self.global_avg_pool(x)
        x5 = self.conv_aspp5(x5)
        x5 = nn.functional.interpolate(x5, size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=True)

        # Concatenate and reduce dimensions
        out = torch.cat([x1, x2, x3, x4, x5], dim=1)  # Shape: [Batch, 1280, H, W]
        out = self.out_conv(out)  # Shape: [Batch, 256, H, W]
        return out

class MyDeepLab(nn.Module):
    # MyDeepLab implementation...
    def __init__(self, num_classes):
        super(MyDeepLab, self).__init__()
        self.backbone = ResNetBackbone()
        self.aspp = ASPP(in_channels=2048, out_channels=256)

        # Decoder
        self.decoder_conv = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.decoder_bn = nn.BatchNorm2d(256)
        self.relu = nn.ReLU()

        # Classification layer
        self.classifier = nn.Conv2d(256, num_classes, kernel_size=1, stride=1)

        # Upsample to original image size
        self.upsample = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=64, stride=32, padding=16)

    def forward(self, x):
        # Backbone
        x = self.backbone(x)  # Shape: [Batch, 2048, 7, 7]

        # ASPP
        x = self.aspp(x)  # Shape: [Batch, 256, 7, 7]

        # Decoder
        x = self.decoder_conv(x)
        x = self.relu(self.decoder_bn(x))

        # Classification
        x = self.classifier(x)  # Shape: [Batch, num_classes, 7, 7]

        # Upsample to match input resolution
        x = self.upsample(x)  # Shape: [Batch, num_classes, H, W]
        return x
