"""Native PyTorch ResNet-50 model used by DLIO compute workloads."""

import torch
from torch import nn


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(
            out_channels, out_channels * self.expansion, 1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, inputs):
        residual = inputs
        outputs = self.relu(self.bn1(self.conv1(inputs)))
        outputs = self.relu(self.bn2(self.conv2(outputs)))
        outputs = self.bn3(self.conv3(outputs))
        if self.downsample is not None:
            residual = self.downsample(inputs)
        return self.relu(outputs + residual)


class ResNet50(nn.Module):
    """ResNet-50 that preserves DLIO's existing model dimensions."""

    def __init__(self, num_classes=1000):
        super().__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, stride=2)
        self.layer1 = self._make_layer(64, 3)
        self.layer2 = self._make_layer(128, 4, stride=2)
        self.layer3 = self._make_layer(256, 6, stride=2)
        self.layer4 = self._make_layer(512, 3, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * Bottleneck.expansion, num_classes)

    def _make_layer(self, out_channels, blocks, stride=1):
        downsample = None
        expanded_channels = out_channels * Bottleneck.expansion
        if stride != 1 or self.in_channels != expanded_channels:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels,
                    expanded_channels,
                    1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(expanded_channels),
            )

        layers = [
            Bottleneck(self.in_channels, out_channels, stride, downsample)
        ]
        self.in_channels = expanded_channels
        layers.extend(
            Bottleneck(self.in_channels, out_channels) for _ in range(1, blocks)
        )
        return nn.Sequential(*layers)

    def forward(self, inputs):
        outputs = self.maxpool(self.relu(self.bn1(self.conv1(inputs))))
        outputs = self.layer1(outputs)
        outputs = self.layer2(outputs)
        outputs = self.layer3(outputs)
        outputs = self.layer4(outputs)
        outputs = self.avgpool(outputs)
        outputs = torch.flatten(outputs, 1)
        return self.fc(outputs)
