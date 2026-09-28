"""Native TensorFlow UNet3D model used by DLIO compute workloads."""

import tensorflow as tf


def _normalization_layer(normalization, channels):
    if normalization == "instancenorm":
        return tf.keras.layers.GroupNormalization(groups=channels, axis=-1)
    if normalization == "batchnorm":
        return tf.keras.layers.BatchNormalization()
    if normalization == "syncbatchnorm":
        return tf.keras.layers.BatchNormalization(synchronized=True)
    if normalization == "none":
        return None
    raise ValueError("Unsupported normalization: {}".format(normalization))


def _activation_layer(activation):
    if activation == "relu":
        return tf.keras.layers.ReLU()
    if activation == "leaky_relu":
        return tf.keras.layers.LeakyReLU(alpha=0.01)
    if activation == "sigmoid":
        return tf.keras.layers.Activation("sigmoid")
    if activation == "none":
        return None
    raise ValueError("Unsupported activation: {}".format(activation))


class ConvBlock3D(tf.keras.layers.Layer):
    """3D convolution, normalization, and activation block."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        normalization="instancenorm",
        activation="relu",
    ):
        super().__init__()
        del in_channels
        self.conv = tf.keras.layers.Conv3D(
            out_channels,
            kernel_size,
            strides=stride,
            padding="same" if padding > 0 else "valid",
            use_bias=normalization == "none",
        )
        self.norm = _normalization_layer(normalization, out_channels)
        self.activation = _activation_layer(activation)

    def call(self, inputs, training=None):
        outputs = self.conv(inputs)
        if isinstance(self.norm, tf.keras.layers.BatchNormalization):
            outputs = self.norm(outputs, training=training)
        elif self.norm is not None:
            outputs = self.norm(outputs)
        if self.activation is not None:
            outputs = self.activation(outputs)
        return outputs


class InputBlock(tf.keras.layers.Layer):
    """Two convolutions at a single UNet resolution."""

    def __init__(
        self,
        in_channels,
        out_channels,
        normalization="instancenorm",
        activation="relu",
    ):
        super().__init__()
        self.conv1 = ConvBlock3D(
            in_channels,
            out_channels,
            normalization=normalization,
            activation=activation,
        )
        self.conv2 = ConvBlock3D(
            out_channels,
            out_channels,
            normalization=normalization,
            activation=activation,
        )

    def call(self, inputs, training=None):
        outputs = self.conv1(inputs, training=training)
        return self.conv2(outputs, training=training)


class DownsampleBlock(tf.keras.layers.Layer):
    """UNet encoder block with a stride-two convolution."""

    def __init__(
        self,
        in_channels,
        out_channels,
        normalization="instancenorm",
        activation="relu",
    ):
        super().__init__()
        self.conv1 = ConvBlock3D(
            in_channels,
            out_channels,
            stride=2,
            normalization=normalization,
            activation=activation,
        )
        self.conv2 = ConvBlock3D(
            out_channels,
            out_channels,
            normalization=normalization,
            activation=activation,
        )

    def call(self, inputs, training=None):
        outputs = self.conv1(inputs, training=training)
        return self.conv2(outputs, training=training)


class UpsampleBlock(tf.keras.layers.Layer):
    """UNet decoder block with a transpose convolution and skip input."""

    def __init__(
        self,
        in_channels,
        out_channels,
        normalization="instancenorm",
        activation="relu",
    ):
        super().__init__()
        self.upsample = tf.keras.layers.Conv3DTranspose(
            out_channels, 2, strides=2, padding="valid"
        )
        self.conv1 = ConvBlock3D(
            2 * out_channels,
            out_channels,
            normalization=normalization,
            activation=activation,
        )
        self.conv2 = ConvBlock3D(
            out_channels,
            out_channels,
            normalization=normalization,
            activation=activation,
        )

    @staticmethod
    def _match_spatial_shape(tensor, reference):
        """Crop or pad an NDHWC tensor to the reference spatial shape."""
        target_shape = tf.shape(reference)[1:4]
        tensor_shape = tf.shape(tensor)
        cropped_shape = tf.minimum(tensor_shape[1:4], target_shape)
        slice_size = tf.concat(
            ([tensor_shape[0]], cropped_shape, [tensor_shape[4]]), axis=0
        )
        tensor = tf.slice(tensor, tf.zeros(5, dtype=tf.int32), slice_size)

        padding = target_shape - cropped_shape
        paddings = tf.stack(
            (
                [0, 0],
                [0, padding[0]],
                [0, padding[1]],
                [0, padding[2]],
                [0, 0],
            )
        )
        return tf.pad(tensor, paddings)

    def call(self, inputs, skip, training=None):
        outputs = self.upsample(inputs)
        outputs = self._match_spatial_shape(outputs, skip)
        outputs = tf.concat((outputs, skip), axis=-1)
        outputs = self.conv1(outputs, training=training)
        return self.conv2(outputs, training=training)


class UNet3D(tf.keras.Model):
    """DLIO's five-level UNet3D architecture for NDHWC inputs."""

    def __init__(
        self,
        in_channels=1,
        num_classes=3,
        normalization="instancenorm",
        activation="relu",
        weights_init_scale=1.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.normalization = normalization
        self.activation_name = activation
        self.weights_init_scale = weights_init_scale

        filters = [32, 64, 128, 256, 320]
        self.input_block = InputBlock(
            in_channels,
            filters[0],
            normalization=normalization,
            activation=activation,
        )
        self.down_blocks = [
            DownsampleBlock(
                filters[index],
                filters[index + 1],
                normalization=normalization,
                activation=activation,
            )
            for index in range(len(filters) - 1)
        ]
        self.bottleneck = InputBlock(
            filters[-1],
            filters[-1],
            normalization=normalization,
            activation=activation,
        )
        self.up_blocks = [
            UpsampleBlock(
                filters[index],
                filters[index - 1],
                normalization=normalization,
                activation=activation,
            )
            for index in range(len(filters) - 1, 0, -1)
        ]
        self.output_layer = tf.keras.layers.Conv3D(num_classes, 1)

    def call(self, inputs, training=None):
        outputs = self.input_block(inputs, training=training)
        skip_connections = []
        for downsample in self.down_blocks:
            skip_connections.append(outputs)
            outputs = downsample(outputs, training=training)

        outputs = self.bottleneck(outputs, training=training)
        for upsample, skip in zip(self.up_blocks, reversed(skip_connections)):
            outputs = upsample(outputs, skip, training=training)

        return self.output_layer(outputs)
