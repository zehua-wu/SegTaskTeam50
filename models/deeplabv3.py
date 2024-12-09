import torch.nn as nn
import torchvision.models.segmentation


class MyDeepLab(nn.Module):
    def __init__(self, num_classes):
        super(MyDeepLab, self).__init__()
        resnet = torchvision.models.resnet50(pretrained=True)

        # Backbone with layers up to layer4
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])  # ResNet backbone
        self.layer3 = nn.Sequential(*list(resnet.children())[:-3])   # Intermediate layer3

        # ASPP Module
        self.aspp = ASPP(in_channels=2048, out_channels=256)

        # Decoder module (for main output)
        self.decoder = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        # Main classifier
        self.classifier = nn.Conv2d(256, num_classes, kernel_size=1)

        # Auxiliary classifier (on layer3 output)
        self.aux_classifier = nn.Sequential(
            nn.Conv2d(1024, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, num_classes, kernel_size=1)
        )

    def forward(self, x):
        input_height, input_width = x.shape[2], x.shape[3]

        # Backbone feature extractor
        features = self.backbone(x)

        # Auxiliary output from intermediate layer (layer3)
        aux_features = self.layer3(x)  # Use features from layer3
        aux_out = self.aux_classifier(aux_features)

        # ASPP and Decoder
        x = self.aspp(features)
        x = self.decoder(x)
        x = self.classifier(x)

        # Upsample to input size
        x = F.interpolate(x, size=(input_height, input_width), mode='bilinear', align_corners=True)
        aux_out = F.interpolate(aux_out, size=(input_height, input_width), mode='bilinear', align_corners=True)

        return OrderedDict({'out': x, 'aux': aux_out})
