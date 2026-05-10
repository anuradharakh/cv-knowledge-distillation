import torch
import torch.nn as nn
import torch.nn.functional as F


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class Preprocess(nn.Module):
    def __init__(self, net, size=96, mean=IMAGENET_MEAN, std=IMAGENET_STD):
        super().__init__()
        self.net = net
        self.size = size
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))

    def forward(self, x):
        x = F.interpolate(x, size=self.size, mode="bilinear", align_corners=False)
        x = (x - self.mean) / self.std
        return self.net(x)


class ConvBNAct(nn.Sequential):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )


class DWConvBNAct(nn.Sequential):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__(
            nn.Conv2d(in_ch, in_ch, 3, stride=stride, padding=1, groups=in_ch, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )


class SmallCNN(nn.Module):
    """
    Compact MobileNet-style CNN for deployment.

    Uses depthwise separable convolutions to increase representational capacity
    while staying safely below the 500K parameter limit.
    """

    def __init__(self, num_classes=7, dropout=0.3):
        super().__init__()

        self.features = nn.Sequential(
            ConvBNAct(3, 32, stride=2),       # 96 -> 48
            DWConvBNAct(32, 64, stride=1),
            DWConvBNAct(64, 96, stride=2),   # 48 -> 24
            DWConvBNAct(96, 128, stride=1),
            DWConvBNAct(128, 192, stride=2), # 24 -> 12
            DWConvBNAct(192, 256, stride=1),
            DWConvBNAct(256, 320, stride=2), # 12 -> 6
            DWConvBNAct(320, 384, stride=1),
            nn.AdaptiveAvgPool2d(1),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(384, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def build_student(num_classes=7, image_size=96, dropout=0.3):
    student = SmallCNN(num_classes=num_classes, dropout=dropout)
    return Preprocess(student, size=image_size)