import numpy as np
import pytest

from dlio_benchmark.common.enumerations import Model
from dlio_benchmark.utils.utility import DLIOMPI


def _initialize_mpi():
    mpi = DLIOMPI.get_instance()
    try:
        mpi.size()
    except Exception:
        mpi.initialize()


class _FakeDaliTensorList:
    def __init__(self, array):
        self.array = array

    def as_array(self):
        return self.array


def test_torch_native_and_dali_batches_are_normalized():
    torch = pytest.importorskip("torch")
    _initialize_mpi()

    from dlio_benchmark.framework.torch_framework import TorchFramework

    dali_batch = _FakeDaliTensorList(
        np.zeros((8, 8, 8), dtype=np.uint8)
    )
    converted = TorchFramework._as_tensor(dali_batch)
    assert tuple(converted.shape) == (8, 8, 8)

    inputs, target = TorchFramework._split_batch(
        [{"data": torch.zeros((8, 8, 8), dtype=torch.uint8)}]
    )
    assert tuple(inputs.shape) == (8, 8, 8)
    assert target is None


def test_torch_unet_scalar_labels_become_segmentation_masks():
    torch = pytest.importorskip("torch")
    _initialize_mpi()

    from dlio_benchmark.framework.torch_framework import TorchFramework

    framework = TorchFramework.__new__(TorchFramework)
    framework.device = torch.device("cpu")
    framework.model_type = Model.UNET

    inputs, target = framework._prepare_unet_batch(
        (torch.zeros((8, 8, 8)), torch.zeros((8, 1)))
    )

    assert tuple(inputs.shape) == (8, 1, 1, 8, 8)
    assert tuple(target.shape) == (8, 1, 8, 8)


def test_tensorflow_tfrecord_batch_is_preserved():
    tf = pytest.importorskip("tensorflow")
    _initialize_mpi()

    from dlio_benchmark.framework.tf_framework import TFFramework
    from dlio_benchmark.reader.tf_reader import TFReader

    example = tf.train.Example(
        features=tf.train.Features(
            feature={
                "image": tf.train.Feature(
                    bytes_list=tf.train.BytesList(value=[b"x"])
                ),
                "size": tf.train.Feature(
                    int64_list=tf.train.Int64List(value=[8])
                ),
            }
        )
    ).SerializeToString()

    reader = TFReader.__new__(TFReader)
    reader._resized_image = tf.zeros((8, 8), dtype=tf.uint8)
    record_batch = reader._parse_image(tf.constant([example] * 8))
    assert tuple(record_batch.shape) == (8, 8, 8)

    framework = TFFramework.__new__(TFFramework)
    framework.model_type = Model.RESNET
    inputs, target = framework._prepare_resnet_batch((record_batch,))

    assert tuple(inputs.shape) == (8, 8, 8, 3)
    assert tuple(target.shape) == (8,)


def test_tensorflow_unet_target_uses_nearest_resize():
    tf = pytest.importorskip("tensorflow")
    _initialize_mpi()

    from dlio_benchmark.framework.tf_framework import TFFramework

    framework = TFFramework.__new__(TFFramework)
    framework.model_type = Model.UNET
    inputs, one_hot_target = framework._prepare_unet_batch(
        (tf.zeros((4, 1, 5, 6, 1)), tf.zeros((4, 1, 5, 6, 3)))
    )
    assert tuple(inputs.shape) == (4, 1, 5, 6, 1)
    assert tuple(one_hot_target.shape) == (4, 1, 5, 6)
    target = tf.reshape(tf.range(8, dtype=tf.int32), (1, 2, 2, 2))
    prediction = tf.zeros((1, 4, 4, 4, 3))
    resized = TFFramework._align_unet_target(prediction, target)

    assert tuple(resized.shape) == (1, 4, 4, 4)
    assert resized[0, 0, 0, 0] == target[0, 0, 0, 0]
    assert resized[0, -1, -1, -1] == target[0, -1, -1, -1]
