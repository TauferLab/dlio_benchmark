import pytest

from dlio_benchmark.common.enumerations import FrameworkType, Model
from dlio_benchmark.model import ModelFactory


def test_pytorch_resnet_is_directly_callable():
    torch = pytest.importorskip("torch")

    model = ModelFactory.create_model(FrameworkType.PYTORCH, Model.RESNET)
    model.eval()

    assert isinstance(model, torch.nn.Module)
    with torch.no_grad():
        output = model(torch.randn(1, 3, 64, 64))
    assert output.shape == (1, 1000)


@pytest.mark.parametrize("framework", list(FrameworkType))
@pytest.mark.parametrize("model_type", [Model.DEFAULT, Model.SLEEP])
def test_non_model_compute_types_return_none(framework, model_type):
    assert ModelFactory.create_model(framework, model_type) is None


@pytest.mark.parametrize("framework", list(FrameworkType))
def test_unsupported_model_is_rejected(framework):
    with pytest.raises(ValueError, match="Unsupported model type"):
        ModelFactory.create_model(framework, Model.BERT)
