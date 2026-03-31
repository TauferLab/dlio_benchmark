from dlio_benchmark.common.enumerations import FrameworkType, Loss
from dlio_benchmark.model.model import UnifiedModel
from typing import Any, Tuple
from dlio_benchmark.utils.utility import Profile

dlp = Profile("UNET3d")


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

            # Size matching for skip connections

            # If sizes don't match, make them match
            if x_size != skip_size:
                # We need to make both tensors have the same spatial dimensions
                # Strategy: pad the smaller one or crop the larger one to match
                
                # For each dimension, determine target size (use the larger of the two)
                target_d = max(x_size[0], skip_size[0])
                target_h = max(x_size[1], skip_size[1])
                target_w = max(x_size[2], skip_size[2])
                
                # Pad x if needed
                if x_size[0] < target_d or x_size[1] < target_h or x_size[2] < target_w:
                    pad_d = target_d - x_size[0]
                    pad_h = target_h - x_size[1]
                    pad_w = target_w - x_size[2]
                    x = F.pad(x, (0, pad_w, 0, pad_h, 0, pad_d))
                
                # Crop x if needed
                if x.shape[2] > target_d or x.shape[3] > target_h or x.shape[4] > target_w:
                    x = x[:, :, :target_d, :target_h, :target_w]
                
                # Pad skip if needed
                if skip_size[0] < target_d or skip_size[1] < target_h or skip_size[2] < target_w:
                    pad_d = target_d - skip_size[0]
                    pad_h = target_h - skip_size[1]
                    pad_w = target_w - skip_size[2]
                    skip = F.pad(skip, (0, pad_w, 0, pad_h, 0, pad_d))
                
                # Crop skip if needed
                if skip.shape[2] > target_d or skip.shape[3] > target_h or skip.shape[4] > target_w:
                    skip = skip[:, :, :target_d, :target_h, :target_w]

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
        self._target_spatial_shape = None

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


        # Bottleneck - use InputBlock instead of DownsampleBlock to avoid extra downsampling
        self.bottleneck = InputBlock(
            self.layer_factory, self.framework,
            filters[-1], filters[-1],
            normalization=self.normalization,
            activation=self.activation
        )

        # Upsample path (decoder)
        self.upsample_blocks = []
        # Upsamples should match the number of downsamples
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

        # Encoder path - save outputs BEFORE downsampling for skip connections
        outputs = []
        for downsample in self.downsample_blocks:
            outputs.append(x)  # Save BEFORE downsampling
            x = downsample(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder path with skip connections
        # Skip connections come from outputs in reverse order
        for upsample, skip in zip(self.upsample_blocks, reversed(outputs)):
            x = upsample(x, skip)

        # Output
        x = self.output_layer(x)

        return x

    @dlp.log
    def compute(self, batch):
        """Override compute to handle target resizing to match prediction dimensions"""
        import torch
        import torch.nn.functional as F
        
        input_data, target_data = self.validate_data(batch)
        
        # Do a forward pass to get the prediction shape if not already cached
        if self._target_spatial_shape is None:
            # Move input to GPU for shape inference
            if self.layer_factory.gpu_id >= 0 and torch.cuda.is_available():
                input_data = input_data.cuda("cuda:{}".format(self.layer_factory.gpu_id))
            
            with torch.no_grad():
                pred_shape = self._model(input_data).shape
            self._target_spatial_shape = pred_shape[2:]

        # Resize target to match prediction spatial dimensions
        if target_data.shape[1:] != self._target_spatial_shape:
            # Target should be (B, D, H, W), pred is (B, C, D, H, W)
            target_size = self._target_spatial_shape  # (D, H, W)
            # Interpolate target to match prediction size
            # Add channel dim for interpolation, then remove it
            target_data = target_data.unsqueeze(1).float()  # (B, 1, D, H, W)
            target_data = F.interpolate(target_data, size=target_size, mode='nearest')
            target_data = target_data.squeeze(1).long()  # (B, D, H, W)
        
        # Now do the actual compute
        self.layer_factory.compute(input_data, target_data)

    @dlp.log
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
                    # Generate dummy target with spatial dimensions matching model output
                    # UNet3D has 4 downsample blocks (each with stride=2) and 4 upsample blocks
                    # Net effect: spatial dimensions stay roughly the same but may vary slightly
                    # due to padding/cropping in skip connections
                    # Target shape: (B, D, H, W) for cross entropy (no channel dim)
                    batch_size = input_data.shape[0]
                    input_spatial = input_data.shape[2:]  # (D, H, W)
                    
                    # The output spatial dims are affected by downsampling and upsampling
                    # With 4 downsamples (stride 2) and 4 upsamples (stride 2), 
                    # the net effect depends on padding/cropping
                    # For simplicity, estimate output size based on the architecture:
                    # After 4 downsamples: D/16, H/16, W/16
                    # After 4 upsamples: back to ~D, ~H, ~W (with some padding)
                    # The actual output will be slightly larger due to padding in upsamples
                    # Use a heuristic: output is roughly input size * 16 / 16 with some padding
                    output_d = input_spatial[0] * 16  # Rough estimate
                    output_h = input_spatial[1]  # Stays roughly the same
                    output_w = input_spatial[2]  # Stays roughly the same

                    #TODO: can we do this
                    target = torch.zeros((batch_size, output_d, output_h, output_w), dtype=torch.long)

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
