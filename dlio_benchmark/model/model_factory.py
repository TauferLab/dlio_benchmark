"""Construct native framework models without importing unused backends."""

from dlio_benchmark.common.enumerations import FrameworkType, Model


class ModelFactory:
    @staticmethod
    def create_model(framework: FrameworkType, model_type: Model):
        """Return a raw ``torch.nn.Module`` or ``tf.keras.Model``.

        Device placement, distributed wrapping, optimization, and training are
        framework concerns and intentionally do not belong to this factory.
        """
        if model_type in (Model.SLEEP, Model.DEFAULT):
            return None

        if framework == FrameworkType.PYTORCH:
            return ModelFactory.create_pytorch_model(model_type)
        if framework == FrameworkType.TENSORFLOW:
            return ModelFactory.create_tensorflow_model(model_type)
        raise ValueError(f"Unsupported framework: {framework}")

    @staticmethod
    def create_pytorch_model(model_type: Model):
        """Return a native PyTorch model, importing only the PyTorch backend."""
        if model_type == Model.RESNET:
            from dlio_benchmark.model.pytorch.resnet import ResNet50

            return ResNet50()
        if model_type == Model.UNET:
            from dlio_benchmark.model.pytorch.unet3d import UNet3D

            return UNet3D()
        if model_type in (Model.SLEEP, Model.DEFAULT):
            return None
        raise ValueError(f"Unsupported model type: {model_type}")

    @staticmethod
    def create_tensorflow_model(model_type: Model):
        """Return a native Keras model, importing only the TensorFlow backend."""
        if model_type == Model.RESNET:
            from dlio_benchmark.model.tensorflow.resnet import ResNet50

            return ResNet50()
        if model_type == Model.UNET:
            from dlio_benchmark.model.tensorflow.unet3d import UNet3D

            return UNet3D()
        if model_type in (Model.SLEEP, Model.DEFAULT):
            return None
        raise ValueError(f"Unsupported model type: {model_type}")
