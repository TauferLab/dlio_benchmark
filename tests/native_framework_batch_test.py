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
