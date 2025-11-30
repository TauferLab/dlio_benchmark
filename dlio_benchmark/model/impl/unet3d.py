from dlio_benchmark.common.enumerations import FrameworkType, Loss
from dlio_benchmark.model.model import UnifiedModel
from typing import Any, Tuple


class ConvBlock3D:
    """3D Convolution block with normalization and activation"""

    def __init__(self, layer_factory, framework: FrameworkType, in_channels: int,
                 out_channels: int, kernel_size: int = 3, stride: int = 1,
                 padding: int = 1, normalization: str = "instancenorm",
                 activation: str = "relu"):
        self.framework = framework
        self.conv = layer_factory.conv3d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, bias=(normalization == "none")
        )

        # Normalization
        if normalization == "instancenorm":
            self.norm = layer_factory.instance_norm3d(out_channels)
        elif normalization == "batchnorm":
            self.norm = layer_factory.batch_norm3d(out_channels)
        elif normalization == "syncbatchnorm":
            self.norm = layer_factory.sync_batch_norm(out_channels)
        else:
            self.norm = layer_factory.identity()

        # Activation
        if activation == "relu":
            self.activation = layer_factory.relu()
        elif activation == "leaky_relu":
            self.activation = layer_factory.leaky_relu(0.01)
        elif activation == "sigmoid":
            self.activation = layer_factory.sigmoid()
        else:
            self.activation = layer_factory.identity()

    def __call__(self, x: Any) -> Any:
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        return x


class InputBlock(ConvBlock3D):
    """Input block with two 3D convolutions"""

    def __init__(self, layer_factory, framework: FrameworkType, in_channels: int,
                 out_channels: int, normalization: str = "instancenorm",
                 activation: str = "relu"):
        self.conv1 = ConvBlock3D(layer_factory, framework, in_channels, out_channels,
                                 normalization=normalization, activation=activation)
        self.conv2 = ConvBlock3D(layer_factory, framework, out_channels, out_channels,
                                 normalization=normalization, activation=activation)

    def __call__(self, x: Any) -> Any:
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class DownsampleBlock:
    """Downsample block with strided convolution"""

    def __init__(self, layer_factory, framework: FrameworkType, in_channels: int,
                 out_channels: int, normalization: str = "instancenorm",
                 activation: str = "relu"):
        self.conv1 = ConvBlock3D(layer_factory, framework, in_channels, out_channels,
                                 stride=2, normalization=normalization, activation=activation)
        self.conv2 = ConvBlock3D(layer_factory, framework, out_channels, out_channels,
                                 normalization=normalization, activation=activation)

    def __call__(self, x: Any) -> Any:
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class UpsampleBlock:
    """Upsample block with transpose convolution and skip connection"""

    def __init__(self, layer_factory, framework: FrameworkType, in_channels: int,
                 out_channels: int, normalization: str = "instancenorm",
                 activation: str = "relu"):
        self.framework = framework
        self.upsample = layer_factory.conv_transpose3d(
            in_channels, out_channels, kernel_size=2, stride=2, padding=0
        )
        self.conv1 = ConvBlock3D(layer_factory, framework, 2 * out_channels, out_channels,
                                 normalization=normalization, activation=activation)
        self.conv2 = ConvBlock3D(layer_factory, framework, out_channels, out_channels,
                                 normalization=normalization, activation=activation)

    def __call__(self, x: Any, skip: Any) -> Any:
        x = self.upsample(x)

        # Handle size mismatches between upsampled and skip connection
        if self.framework == FrameworkType.PYTORCH:
            import torch
            import torch.nn.functional as F

            # Get spatial dimensions (D, H, W)
            x_size = x.shape[2:]
            skip_size = skip.shape[2:]

            # If sizes don't match, pad or crop to match
            if x_size != skip_size:
                # Calculate padding needed
                pad_d = skip_size[0] - x_size[0]
                pad_h = skip_size[1] - x_size[1]
                pad_w = skip_size[2] - x_size[2]

                if pad_d >= 0 and pad_h >= 0 and pad_w >= 0:
                    # Pad x to match skip size
                    # F.pad expects (left, right, top, bottom, front, back)
                    x = F.pad(x, (0, pad_w, 0, pad_h, 0, pad_d))
                else:
                    # Crop skip to match x size (shouldn't happen in typical U-Net)
                    skip = skip[:, :, :x_size[0], :x_size[1], :x_size[2]]

            x = torch.cat((x, skip), dim=1)
        else:  # TensorFlow
            import tensorflow as tf

            # Get spatial dimensions (D, H, W)
            x_size = x.shape[1:4]
            skip_size = skip.shape[1:4]

            # If sizes don't match, pad or crop to match
            if x_size != skip_size:
                # Calculate padding needed
                pad_d = skip_size[0] - x_size[0]
                pad_h = skip_size[1] - x_size[1]
                pad_w = skip_size[2] - x_size[2]

                if pad_d >= 0 and pad_h >= 0 and pad_w >= 0:
                    # Pad x to match skip size
                    # TensorFlow padding: [[top, bottom], [left, right], ...]
                    paddings = [[0, 0], [0, pad_d], [0, pad_h], [0, pad_w], [0, 0]]
                    x = tf.pad(x, paddings)
                else:
                    # Crop skip to match x size
                    skip = skip[:, :x_size[0], :x_size[1], :x_size[2], :]

            x = tf.concat([x, skip], axis=-1)

        x = self.conv1(x)
        x = self.conv2(x)
        return x


class OutputLayer:
    """Output layer with 1x1x1 convolution"""

    def __init__(self, layer_factory, framework: FrameworkType, in_channels: int,
                 n_class: int):
        self.framework = framework
        self.conv = layer_factory.conv3d(in_channels, n_class, kernel_size=1,
                                         stride=1, padding=0, bias=True)

    def __call__(self, x: Any) -> Any:
        return self.conv(x)


class UNet3D(UnifiedModel):
    """UNet3D implementation for 3D medical imaging segmentation"""

    def __init__(self, framework: FrameworkType, communication, gpu_id,
                 in_channels: int = 1, n_class: int = 3,
                 normalization: str = "instancenorm", activation: str = "relu",
                 weights_init_scale: float = 1.0):
        super().__init__(framework, Loss.CE, communication, gpu_id)
        self.in_channels = in_channels
        self.n_class = n_class
        self.normalization = normalization
        self.activation = activation
        self.weights_init_scale = weights_init_scale

        self.build_model()
        self._model = self.layer_factory.get_model(self.forward)

        if framework == FrameworkType.PYTORCH:
            import torch
            self.layer_factory.set_optimizer(torch.optim.Adam, 1e-4)
        else:
            import tensorflow as tf
            self.layer_factory.set_optimizer(tf.optimizers.Adam, 1e-4)

    def build_model(self):
        """Build UNet3D architecture"""
        # Filters for each level
        filters = [32, 64, 128, 256, 320]

        # Input block
        self.input_block = InputBlock(
            self.layer_factory, self.framework,
            self.in_channels, filters[0],
            normalization=self.normalization,
            activation=self.activation
        )

        # Downsample path (encoder)
        self.downsample_blocks = []
        for i in range(len(filters) - 1):
            block = DownsampleBlock(
                self.layer_factory, self.framework,
                filters[i], filters[i + 1],
                normalization=self.normalization,
                activation=self.activation
            )
            self.downsample_blocks.append(block)

        # Bottleneck
        self.bottleneck = DownsampleBlock(
            self.layer_factory, self.framework,
            filters[-1], filters[-1],
            normalization=self.normalization,
            activation=self.activation
        )

        # Upsample path (decoder)
        self.upsample_blocks = []
        # First upsample from bottleneck
        self.upsample_blocks.append(
            UpsampleBlock(
                self.layer_factory, self.framework,
                filters[-1], filters[-1],
                normalization=self.normalization,
                activation=self.activation
            )
        )
        # Remaining upsamples
        for i in range(len(filters) - 1, 0, -1):
            block = UpsampleBlock(
                self.layer_factory, self.framework,
                filters[i], filters[i - 1],
                normalization=self.normalization,
                activation=self.activation
            )
            self.upsample_blocks.append(block)

        # Output layer
        self.output_layer = OutputLayer(
            self.layer_factory, self.framework,
            filters[0], self.n_class
        )

    def forward(self, x: Any) -> Any:
        """Forward pass through UNet3D"""
        # Input block
        x = self.input_block(x)

        # Encoder path - save outputs for skip connections
        outputs = [x]
        for downsample in self.downsample_blocks:
            x = downsample(x)
            outputs.append(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder path with skip connections
        for upsample, skip in zip(self.upsample_blocks, reversed(outputs)):
            x = upsample(x, skip)

        # Output
        x = self.output_layer(x)

        return x

    def validate_data(self, data: Any) -> Tuple[Any, Any]:
        """Validate and preprocess 3D medical imaging data"""
        try:
            if self.framework == FrameworkType.PYTORCH:
                import torch
                import numpy as np

                # Convert numpy array to torch tensor if needed
                if isinstance(data, np.ndarray):
                    data = torch.from_numpy(data)

                if isinstance(data, torch.Tensor):
                    # Handle different input shapes
                    if len(data.shape) == 3:
                        # Shape: (B, H, W) - 2D data, convert to 3D by adding depth and channel dims
                        # Convert to (B, C, D, H, W) with D=1 (single slice)
                        data = data.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, H, W)
                    elif len(data.shape) == 4:
                        # Could be (B, C, H, W) or (B, D, H, W)
                        # Assume it's (B, D, H, W) and add channel dimension
                        data = data.unsqueeze(1)  # (B, 1, D, H, W)
                    elif len(data.shape) == 5:
                        # Already in correct format (B, C, D, H, W)
                        pass
                    else:
                        raise ValueError(f"Expected 3D, 4D or 5D tensor, got shape {data.shape}")

                    input_data = data.float()
                    # Generate dummy target with same spatial dimensions
                    # Target shape: (B, D, H, W) for cross entropy
                    batch_size = input_data.shape[0]
                    spatial_shape = input_data.shape[2:]  # (D, H, W)
                    target = torch.zeros((batch_size, *spatial_shape), dtype=torch.long)
                else:
                    input_data, target = data
            else:  # TensorFlow
                import tensorflow as tf
                if isinstance(data, tf.Tensor):
                    # Handle different input shapes
                    if len(data.shape) == 3:
                        # Shape: (B, H, W) - 2D data, convert to 3D
                        # TensorFlow uses (B, D, H, W, C) format
                        data = tf.expand_dims(data, axis=1)  # (B, 1, H, W)
                        data = tf.expand_dims(data, axis=-1)  # (B, 1, H, W, 1)
                    elif len(data.shape) == 4:
                        # Add channel dimension
                        data = tf.expand_dims(data, axis=-1)  # (B, D, H, W, 1)
                    elif len(data.shape) == 5:
                        # Check if channels are in wrong position and transpose
                        if data.shape[1] <= 4:  # Likely (B, C, D, H, W)
                            data = tf.transpose(data, perm=[0, 2, 3, 4, 1])
                    else:
                        raise ValueError(f"Expected 3D, 4D or 5D tensor, got shape {data.shape}")

                    input_data = tf.cast(data, tf.float32)
                    # Generate dummy target
                    batch_size = input_data.shape[0]
                    spatial_shape = input_data.shape[1:4]  # (D, H, W)
                    target = tf.zeros((batch_size, *spatial_shape), dtype=tf.int32)
                else:
                    input_data, target = data
        except Exception as e:
            raise ValueError(f"Invalid data format for UNet3D: {e}")

        return input_data, target
