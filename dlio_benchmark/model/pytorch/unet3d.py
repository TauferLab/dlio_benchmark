"""Native PyTorch UNet3D model used by DLIO compute workloads."""

import torch
from torch import nn
from torch.nn import functional as functional


class ConvBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv = nn.Conv3d(
            in_channels,
            out_channels,
            3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.norm = nn.InstanceNorm3d(out_channels, affine=True)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, inputs):
        return self.activation(self.norm(self.conv(inputs)))


class InputBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = ConvBlock3D(in_channels, out_channels)
        self.conv2 = ConvBlock3D(out_channels, out_channels)

    def forward(self, inputs):
        return self.conv2(self.conv1(inputs))


class DownsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = ConvBlock3D(in_channels, out_channels, stride=2)
        self.conv2 = ConvBlock3D(out_channels, out_channels)

    def forward(self, inputs):
        return self.conv2(self.conv1(inputs))


class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upsample = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size=2, stride=2
        )
        self.conv1 = ConvBlock3D(2 * out_channels, out_channels)
        self.conv2 = ConvBlock3D(out_channels, out_channels)

    @staticmethod
    def _match_spatial_shape(tensor, reference):
        tensor_shape = tensor.shape[2:]
        reference_shape = reference.shape[2:]
        target = [max(left, right) for left, right in zip(tensor_shape, reference_shape)]
        if tuple(tensor_shape) != tuple(target):
            padding = []
            for dimension, target_dimension in reversed(list(zip(tensor_shape, target))):
                padding.extend([0, target_dimension - dimension])
            tensor = functional.pad(tensor, padding)
        return tensor[:, :, : target[0], : target[1], : target[2]]

    def forward(self, inputs, skip):
        outputs = self.upsample(inputs)
        if outputs.shape[2:] != skip.shape[2:]:
            outputs = self._match_spatial_shape(outputs, skip)
            skip = self._match_spatial_shape(skip, outputs)
        outputs = torch.cat((outputs, skip), dim=1)
        return self.conv2(self.conv1(outputs))


class UNet3D(nn.Module):
    def __init__(self, in_channels=1, num_classes=3):
        super().__init__()
        filters = [32, 64, 128, 256, 320]
        self.input_block = InputBlock(in_channels, filters[0])
        self.down_blocks = nn.ModuleList(
            DownsampleBlock(filters[index], filters[index + 1])
            for index in range(len(filters) - 1)
        )
        self.bottleneck = InputBlock(filters[-1], filters[-1])
        self.up_blocks = nn.ModuleList(
            UpsampleBlock(filters[index], filters[index - 1])
            for index in range(len(filters) - 1, 0, -1)
        )
        self.output_layer = nn.Conv3d(filters[0], num_classes, 1)

    def forward(self, inputs):
        outputs = self.input_block(inputs)
        skip_connections = []
        for downsample in self.down_blocks:
            skip_connections.append(outputs)
            outputs = downsample(outputs)
        outputs = self.bottleneck(outputs)
        for upsample, skip in zip(self.up_blocks, reversed(skip_connections)):
            outputs = upsample(outputs, skip)
        return self.output_layer(outputs)
