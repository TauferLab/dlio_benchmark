"""Native TensorFlow ResNet-50 model used by DLIO compute workloads."""

import tensorflow as tf


class Projection(tf.keras.layers.Layer):
    """Project a residual tensor to the bottleneck output shape."""

    def __init__(self, out_channels, stride=1):
        super().__init__()
        self.conv = tf.keras.layers.Conv2D(
            out_channels, 1, strides=stride, use_bias=False
        )
        self.norm = tf.keras.layers.BatchNormalization()

    def call(self, inputs, training=None):
        outputs = self.conv(inputs)
        return self.norm(outputs, training=training)


class Bottleneck(tf.keras.layers.Layer):
    """ResNet-50 bottleneck block for channels-last tensors."""

    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = tf.keras.layers.Conv2D(
            out_channels, 1, use_bias=False
        )
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.conv2 = tf.keras.layers.Conv2D(
            out_channels,
            3,
            strides=stride,
            padding="same",
            use_bias=False,
        )
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.conv3 = tf.keras.layers.Conv2D(
            out_channels * self.expansion, 1, use_bias=False
        )
        self.bn3 = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.ReLU()

        expanded_channels = out_channels * self.expansion
        if stride != 1 or in_channels != expanded_channels:
            self.downsample = Projection(expanded_channels, stride)
        else:
            self.downsample = None

    def call(self, inputs, training=None):
        residual = inputs

        outputs = self.conv1(inputs)
        outputs = self.bn1(outputs, training=training)
        outputs = self.relu(outputs)

        outputs = self.conv2(outputs)
        outputs = self.bn2(outputs, training=training)
        outputs = self.relu(outputs)

        outputs = self.conv3(outputs)
        outputs = self.bn3(outputs, training=training)

        if self.downsample is not None:
            residual = self.downsample(inputs, training=training)

        return self.relu(outputs + residual)


class ResNet50(tf.keras.Model):
    """DLIO's ResNet-50 architecture for NHWC inputs."""

    def __init__(self, num_classes=1000):
        super().__init__()
        self.conv1 = tf.keras.layers.Conv2D(
            64,
            7,
            strides=2,
            padding="same",
            use_bias=False,
        )
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.ReLU()
        self.maxpool = tf.keras.layers.MaxPooling2D(pool_size=3, strides=2)

        self.layer1 = self._make_layer(64, 64, 3)
        self.layer2 = self._make_layer(256, 128, 4, stride=2)
        self.layer3 = self._make_layer(512, 256, 6, stride=2)
        self.layer4 = self._make_layer(1024, 512, 3, stride=2)

        self.avgpool = tf.keras.layers.GlobalAveragePooling2D()
        self.flatten = tf.keras.layers.Flatten()
        self.fc = tf.keras.layers.Dense(num_classes)

    @staticmethod
    def _make_layer(in_channels, out_channels, blocks, stride=1):
        layers = [Bottleneck(in_channels, out_channels, stride)]
        layers.extend(
            Bottleneck(out_channels * Bottleneck.expansion, out_channels)
            for _ in range(1, blocks)
        )
        return layers

    @staticmethod
    def _apply_layer(blocks, inputs, training):
        outputs = inputs
        for block in blocks:
            outputs = block(outputs, training=training)
        return outputs

    def call(self, inputs, training=None):
        outputs = self.conv1(inputs)
        outputs = self.bn1(outputs, training=training)
        outputs = self.relu(outputs)
        outputs = self.maxpool(outputs)

        outputs = self._apply_layer(self.layer1, outputs, training)
        outputs = self._apply_layer(self.layer2, outputs, training)
        outputs = self._apply_layer(self.layer3, outputs, training)
        outputs = self._apply_layer(self.layer4, outputs, training)

        outputs = self.avgpool(outputs)
        outputs = self.flatten(outputs)
        return self.fc(outputs)
