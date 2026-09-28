"""Focused tests for native TensorFlow models and TFRecord batch handling."""

import pytest

# DFTracer initializes PyTorch hooks; load it before TensorFlow.
from dlio_benchmark.utils.utility import DLIOMPI


def _initialize_mpi():
    mpi = DLIOMPI.get_instance()
    try:
        mpi.size()
    except Exception:
        mpi.initialize()


def test_resnet50_forward_shape_and_trainable_variables():
    tf = pytest.importorskip("tensorflow")
    from dlio_benchmark.model.tensorflow.resnet import ResNet50

    model = ResNet50()
    output = model(tf.zeros((1, 64, 64, 3)), training=False)

    assert output.shape == (1, 1000)
    assert model.trainable_variables


def test_unet3d_preserves_odd_spatial_shape():
    tf = pytest.importorskip("tensorflow")
    from dlio_benchmark.model.tensorflow.unet3d import UNet3D

    model = UNet3D()
    output = model(tf.zeros((1, 17, 17, 17, 1)), training=False)

    assert output.shape == (1, 17, 17, 17, 3)
    assert model.trainable_variables


def test_tfrecord_parser_preserves_record_batch():
    tf = pytest.importorskip("tensorflow")
    from dlio_benchmark.reader.tf_reader import TFReader

    example = tf.train.Example(
        features=tf.train.Features(
            feature={
                "image": tf.train.Feature(bytes_list=tf.train.BytesList(value=[b"x"])),
                "size": tf.train.Feature(int64_list=tf.train.Int64List(value=[8])),
            }
        )
    ).SerializeToString()
    reader = TFReader.__new__(TFReader)
    reader._resized_image = tf.zeros((8, 8), dtype=tf.uint8)

    output = reader._parse_image(tf.constant([example] * 4))

    assert output.shape == (4, 8, 8)


def test_factory_constructs_native_tensorflow_models():
    tf = pytest.importorskip("tensorflow")
    from dlio_benchmark.common.enumerations import FrameworkType, Model
    from dlio_benchmark.model.model_factory import ModelFactory

    assert isinstance(
        ModelFactory.create_model(FrameworkType.TENSORFLOW, Model.RESNET),
        tf.keras.Model,
    )
    assert isinstance(
        ModelFactory.create_model(FrameworkType.TENSORFLOW, Model.UNET),
        tf.keras.Model,
    )
    assert ModelFactory.create_model(FrameworkType.TENSORFLOW, Model.SLEEP) is None


def test_tfrecord_batch_prepares_for_resnet():
    tf = pytest.importorskip("tensorflow")
    _initialize_mpi()
    from dlio_benchmark.common.enumerations import Model
    from dlio_benchmark.framework.tf_framework import TFFramework

    framework = TFFramework.__new__(TFFramework)
    framework.model_type = Model.RESNET
    inputs, target = framework._prepare_resnet_batch(
        (tf.zeros((4, 8, 8), dtype=tf.uint8),)
    )

    assert inputs.shape == (4, 8, 8, 3)
    assert target.shape == (4,)
    assert inputs.dtype == tf.float32
    assert target.dtype == tf.int32


def test_unet_target_uses_nearest_resize_and_scalar_labels_become_masks():
    tf = pytest.importorskip("tensorflow")
    _initialize_mpi()
    from dlio_benchmark.common.enumerations import Model
    from dlio_benchmark.framework.tf_framework import TFFramework

    framework = TFFramework.__new__(TFFramework)
    framework.model_type = Model.UNET
    inputs, target = framework._prepare_unet_batch(
        (tf.zeros((4, 1, 5, 6, 1)), tf.zeros((4, 1)))
    )
    assert inputs.shape == (4, 1, 5, 6, 1)
    assert target.shape == (4, 1, 5, 6)

    source = tf.reshape(tf.range(8, dtype=tf.int32), (1, 2, 2, 2))
    prediction = tf.zeros((1, 4, 4, 4, 3))
    resized = TFFramework._align_unet_target(prediction, source)
    assert resized.shape == (1, 4, 4, 4)
    assert resized[0, 0, 0, 0] == source[0, 0, 0, 0]
    assert resized[0, -1, -1, -1] == source[0, -1, -1, -1]


def test_optimizer_is_selected_for_each_tensorflow_architecture():
    tf = pytest.importorskip("tensorflow")
    _initialize_mpi()
    from dlio_benchmark.common.enumerations import Model
    from dlio_benchmark.framework.tf_framework import TFFramework

    framework = TFFramework.__new__(TFFramework)
    framework._configure_training(Model.RESNET)
    assert isinstance(framework._optimizer, tf.keras.optimizers.SGD)
    framework._configure_training(Model.UNET)
    assert isinstance(framework._optimizer, tf.keras.optimizers.Adam)
    with pytest.raises(ValueError, match="Unsupported TensorFlow model"):
        framework._configure_training(Model.BERT)


@pytest.mark.parametrize("model_type", ["resnet50", "unet3d"])
def test_native_tensorflow_training_step_updates_weights(model_type):
    tf = pytest.importorskip("tensorflow")
    _initialize_mpi()
    from dlio_benchmark.common.enumerations import Model
    from dlio_benchmark.framework.tf_framework import TFFramework

    framework = TFFramework.__new__(TFFramework)
    framework.model_type = Model(model_type)
    if framework.model_type == Model.RESNET:
        framework._model = tf.keras.Sequential(
            [tf.keras.layers.Flatten(), tf.keras.layers.Dense(1000)]
        )
        batch = tf.ones((2, 8, 8), dtype=tf.uint8)
    else:
        framework._model = tf.keras.Sequential([tf.keras.layers.Conv3D(3, 1)])
        batch = tf.ones((2, 1, 8, 8, 1), dtype=tf.float32)
    framework._configure_training(framework.model_type)
    prepared, _ = framework._prepare_batch(batch)
    framework._model(prepared)
    before = [variable.numpy().copy() for variable in framework._model.trainable_variables]

    prediction, loss = framework._train_batch(batch)

    assert prediction.shape[0] == 2
    assert float(loss.numpy()) > 0
    assert any(
        (old != variable.numpy()).any()
        for old, variable in zip(before, framework._model.trainable_variables)
    )
