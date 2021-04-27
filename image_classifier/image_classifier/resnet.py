# Copyright (c) 2021 Massachusetts Institute of Technology
import torch
from torchvision import models


class ResNet(models.resnet.ResNet):
    """TorchVision ResNet Adapted for CIFAR10 Image Sizes"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        del self.maxpool

        self.conv1 = torch.nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False
        )

        self.fc = torch.nn.Linear(self.inplanes, 10)

    def _forward_impl(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


def resnet18():
    return ResNet(models.resnet.BasicBlock, [2, 2, 2, 2])


def resnet50():
    return ResNet(models.resnet.Bottleneck, [3, 4, 6, 3])
