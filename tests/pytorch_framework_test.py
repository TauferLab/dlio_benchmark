"""PyTorch model-factory and batch-adapter contracts."""

import numpy as np
import pytest

from dlio_benchmark.common.enumerations import FrameworkType, Model
from dlio_benchmark.utils.utility import DLIOMPI


def _framework(model_type):
    torch = pytest.importorskip("torch")
    mpi = DLIOMPI.get_instance()
    try:
        mpi.size()
    except Exception:
        mpi.initialize()
    from dlio_benchmark.framework.torch_framework import TorchFramework

    framework = TorchFramework.__new__(TorchFramework)
    framework.device = torch.device("cpu")
    framework.model_type = model_type
    return framework


def test_factory_constructs_native_models_and_rejects_unsupported():
    torch = pytest.importorskip("torch")
    from dlio_benchmark.model import ModelFactory

    assert isinstance(ModelFactory.create_model(FrameworkType.PYTORCH, Model.RESNET), torch.nn.Module)
    assert isinstance(ModelFactory.create_model(FrameworkType.PYTORCH, Model.UNET), torch.nn.Module)
    assert ModelFactory.create_model(FrameworkType.PYTORCH, Model.SLEEP) is None
    with pytest.raises(ValueError, match="Unsupported model type"):
        ModelFactory.create_model(FrameworkType.PYTORCH, Model.BERT)


class _DaliTensorList:
    def __init__(self, array):
        self.array = array

    def as_array(self):
        return self.array


def test_resnet_batch_normalizes_dali_and_channel_last():
    torch = pytest.importorskip("torch")
    framework = _framework(Model.RESNET)

    dali_batch = [{"data": _DaliTensorList(np.zeros((2, 16, 12, 3), dtype=np.uint8)),
                   "label": _DaliTensorList(np.array([[4], [7]], dtype=np.int64))}]
    images, labels = framework._prepare_batch(dali_batch)
    assert images.shape == (2, 3, 16, 12)
    assert images.dtype == torch.float32
    assert labels.tolist() == [4, 7]

    grayscale, labels = framework._prepare_batch(torch.zeros((2, 16, 12), dtype=torch.uint8))
    assert grayscale.shape == (2, 3, 16, 12)
    assert labels.tolist() == [0, 0]
    with pytest.raises(ValueError, match="ResNet input"):
        framework._prepare_batch(torch.zeros((2, 4, 16, 12)))


def test_unet_batch_masks_and_channel_last():
    torch = pytest.importorskip("torch")
    framework = _framework(Model.UNET)

    volume = torch.zeros((2, 8, 16, 16, 1), dtype=torch.uint8)
    masks = torch.nn.functional.one_hot(torch.ones((2, 8, 16, 16), dtype=torch.long), 3)
    inputs, target = framework._prepare_batch((volume, masks))
    assert inputs.shape == (2, 1, 8, 16, 16)
    assert inputs.dtype == torch.float32
    assert target.shape == (2, 8, 16, 16)
    assert torch.all(target == 1)

    inputs, target = framework._prepare_batch((torch.zeros((2, 8, 16, 16)), torch.zeros((2, 1))))
    assert inputs.shape == (2, 1, 8, 16, 16)
    assert target.shape == (2, 8, 16, 16)
    assert torch.all(target == 0)
    with pytest.raises(ValueError, match="UNet3D input"):
        framework._prepare_batch(torch.zeros((2, 4, 8, 16, 16)))


def test_unet_target_alignment_uses_nearest_classes():
    torch = pytest.importorskip("torch")
    framework = _framework(Model.UNET)
    target = torch.arange(8, dtype=torch.long).reshape(1, 2, 2, 2)
    prediction = torch.zeros((1, 3, 4, 4, 4))
    aligned = framework._align_unet_target(prediction, target)
    assert aligned.shape == (1, 4, 4, 4)
    assert aligned[0, 0, 0, 0] == target[0, 0, 0, 0]
    assert aligned[0, -1, -1, -1] == target[0, -1, -1, -1]
    assert set(aligned.unique().tolist()) == set(target.flatten().tolist())


@pytest.mark.parametrize("model_type", [Model.RESNET, Model.UNET])
def test_framework_native_step_updates_parameters(monkeypatch, model_type):
    torch = pytest.importorskip("torch")
    mpi = DLIOMPI.get_instance()
    try:
        mpi.size()
    except Exception:
        mpi.initialize()
    from dlio_benchmark.framework import torch_framework as module
    torch.set_num_threads(1)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(module, "DFTRACER_ENABLE", False)

    framework = module.TorchFramework(False, model_type)
    if model_type is Model.RESNET:
        batch = (torch.randn(2, 3, 64, 64), torch.tensor([0, 1]))
        parameter = framework.native_model.fc.weight
        expected_shape = (2, 1000)
    else:
        batch = (torch.randn(1, 1, 16, 32, 32), torch.zeros((1, 16, 32, 32), dtype=torch.long))
        parameter = framework.native_model.output_layer.weight
        expected_shape = (1, 3, 16, 32, 32)
    before = parameter.detach().clone()
    prediction, loss = framework.compute(batch, 1, 1, 0)
    assert tuple(prediction.shape) == expected_shape
    assert torch.isfinite(loss)
    assert parameter.grad is not None and torch.count_nonzero(parameter.grad) > 0
    assert not torch.equal(before, parameter.detach())
    framework.finalize()


def test_batch_factor_default_keeps_the_whole_batch(monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.delenv("BATCH_FACTOR", raising=False)
    framework = _framework(Model.RESNET)
    images, labels = framework._prepare_batch((torch.zeros((4, 3, 8, 8)), torch.arange(4)))
    assert images.shape == (4, 3, 8, 8)
    assert labels.tolist() == [0, 1, 2, 3]
    inputs = torch.zeros((4, 3, 8, 8))
    same_inputs, same_target = framework._take_compute_share(inputs, None)
    assert same_inputs is inputs and same_target is None


@pytest.mark.parametrize("factor", [2, 4])
def test_batch_factor_resnet_trains_on_the_first_share(monkeypatch, factor):
    torch = pytest.importorskip("torch")
    monkeypatch.setenv("BATCH_FACTOR", str(factor))
    framework = _framework(Model.RESNET)
    loaded = 8
    inputs = torch.arange(loaded, dtype=torch.float32).reshape(loaded, 1, 1, 1).expand(loaded, 3, 4, 4).contiguous()
    images, labels = framework._prepare_batch((inputs, torch.arange(loaded)))
    share = loaded // factor
    assert images.shape == (share, 3, 4, 4)
    assert torch.equal(images, inputs[:share])
    assert labels.tolist() == list(range(share))


def test_batch_factor_cuts_before_any_copy(monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.setenv("BATCH_FACTOR", "4")
    framework = _framework(Model.RESNET)
    inputs = torch.zeros((8, 16, 16), dtype=torch.uint8)
    cut, target = framework._take_compute_share(inputs, None)
    assert cut.shape == (2, 16, 16) and target is None
    assert cut.data_ptr() == inputs.data_ptr()  # a view of the loaded batch, nothing copied
    images, _ = framework._prepare_batch(inputs)
    assert images.shape == (2, 3, 16, 16)  # the 2-D to 3-channel repeat ran on the share only


def test_batch_factor_unet_trains_on_the_first_share(monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.setenv("BATCH_FACTOR", "2")
    framework = _framework(Model.UNET)
    volume = torch.arange(4, dtype=torch.float32).reshape(4, 1, 1, 1, 1).expand(4, 1, 4, 4, 4).contiguous()
    masks = torch.arange(4).reshape(4, 1, 1, 1).expand(4, 4, 4, 4).contiguous()
    inputs, target = framework._prepare_batch((volume, masks))
    assert inputs.shape == (2, 1, 4, 4, 4)
    assert torch.equal(inputs, volume[:2])
    assert target[:, 0, 0, 0].tolist() == [0, 1]


def test_batch_factor_rejects_an_indivisible_batch(monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.setenv("BATCH_FACTOR", "4")
    framework = _framework(Model.RESNET)
    with pytest.raises(ValueError, match="not divisible by BATCH_FACTOR=4"):
        framework._prepare_batch((torch.zeros((6, 3, 8, 8)), torch.arange(6)))


@pytest.mark.parametrize("value", ["0", "-1", "two", "1.5", ""])
def test_batch_factor_rejects_bad_values(monkeypatch, value):
    monkeypatch.setenv("BATCH_FACTOR", value)
    framework = _framework(Model.RESNET)
    with pytest.raises(ValueError, match="BATCH_FACTOR must be a positive integer"):
        framework._batch_factor()


def test_framework_native_step_with_batch_factor(monkeypatch):
    torch = pytest.importorskip("torch")
    mpi = DLIOMPI.get_instance()
    try:
        mpi.size()
    except Exception:
        mpi.initialize()
    from dlio_benchmark.framework import torch_framework as module
    torch.set_num_threads(1)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(module, "DFTRACER_ENABLE", False)
    monkeypatch.setenv("BATCH_FACTOR", "2")

    framework = module.TorchFramework(False, Model.RESNET)
    prediction, loss = framework.compute((torch.randn(4, 3, 64, 64), torch.tensor([0, 1, 2, 3])), 1, 1, 0)
    assert tuple(prediction.shape) == (2, 1000)  # 4 loaded samples, 2 trained
    assert torch.isfinite(loss)
    framework.finalize()
