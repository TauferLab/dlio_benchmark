"""Focused coverage for the native PyTorch model architectures.

Framework batch adaptation and optimizer behavior are covered by the
integration tests added after the compute/communication base is available.
"""

import pytest


def test_resnet50_forward_and_parameter_update():
    torch = pytest.importorskip("torch")
    from dlio_benchmark.model.pytorch.resnet import ResNet50

    torch.manual_seed(7)
    torch.set_num_threads(1)
    model = ResNet50()
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    images = torch.randn(2, 3, 64, 64)
    labels = torch.tensor([0, 1])
    before = model.fc.weight.detach().clone()

    optimizer.zero_grad(set_to_none=True)
    logits = model(images)
    loss = torch.nn.functional.cross_entropy(logits, labels)
    loss.backward()
    assert logits.shape == (2, 1000)
    assert torch.isfinite(loss)
    assert model.fc.weight.grad is not None
    assert torch.count_nonzero(model.fc.weight.grad) > 0
    optimizer.step()
    assert not torch.equal(before, model.fc.weight.detach())


def test_unet3d_forward_and_parameter_update():
    torch = pytest.importorskip("torch")
    from dlio_benchmark.model.pytorch.unet3d import UNet3D

    torch.manual_seed(7)
    torch.set_num_threads(1)
    model = UNet3D()
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    volume = torch.randn(1, 1, 16, 32, 32)
    labels = torch.zeros((1, 16, 32, 32), dtype=torch.long)
    before = model.output_layer.weight.detach().clone()

    optimizer.zero_grad(set_to_none=True)
    logits = model(volume)
    loss = torch.nn.functional.cross_entropy(logits, labels)
    loss.backward()
    assert logits.shape == (1, 3, 16, 32, 32)
    assert torch.isfinite(loss)
    assert model.output_layer.weight.grad is not None
    assert torch.count_nonzero(model.output_layer.weight.grad) > 0
    optimizer.step()
    assert not torch.equal(before, model.output_layer.weight.detach())


def test_unet3d_odd_size_produces_finite_logits():
    torch = pytest.importorskip("torch")
    from dlio_benchmark.model.pytorch.unet3d import UNet3D

    torch.set_num_threads(1)
    model = UNet3D().eval()
    with torch.no_grad():
        logits = model(torch.randn(1, 1, 17, 33, 33))
    assert logits.shape[0:2] == (1, 3)
    assert all(actual >= requested for actual, requested in zip(logits.shape[2:], (17, 33, 33)))
    assert torch.isfinite(logits).all()
