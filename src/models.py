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
            nn.Conv2d(
                in_ch,
                in_ch,
                3,
                stride=stride,
                padding=1,
                groups=in_ch,
                bias=False,
            ),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

class SmallCNN(nn.Module):
    """
    Compact MobileNet-style student CNN.

    Designed for deployment under the 500,000 parameter limit.
    Uses depthwise separable convolutions, BatchNorm, ReLU,
    dropout, and global average pooling.
    """

    def __init__(self, num_classes=7, dropout=0.3, channels=None):
        super().__init__()

        if channels is None:
            channels = [32, 64, 96, 128, 192, 256, 320, 384]

        c1, c2, c3, c4, c5, c6, c7, c8 = channels

        self.features = nn.Sequential(
            ConvBNAct(3, c1, stride=2),       # 96 -> 48
            DWConvBNAct(c1, c2, stride=1),
            DWConvBNAct(c2, c3, stride=2),   # 48 -> 24
            DWConvBNAct(c3, c4, stride=1),
            DWConvBNAct(c4, c5, stride=2),   # 24 -> 12
            DWConvBNAct(c5, c6, stride=1),
            DWConvBNAct(c6, c7, stride=2),   # 12 -> 6
            DWConvBNAct(c7, c8, stride=1),
            nn.AdaptiveAvgPool2d(1),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(c8, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))

def build_student(num_classes=7, image_size=96, dropout=0.3, channels=None):
    student = SmallCNN(
        num_classes=num_classes,
        dropout=dropout,
        channels=channels,
    )
    return Preprocess(student, size=image_size)