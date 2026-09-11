from dlio_benchmark.model import torch_layer


def test_dftracer_dynamo_backend_is_applied_when_enabled(monkeypatch):
    """DFTracer's public torch.compile backend replaces the removed legacy API."""
    backend = object()
    backend_calls = []
    compile_calls = []

    def create_test_backend(**kwargs):
        backend_calls.append(kwargs)
        return backend

    def compile_model(model, *, backend):
        compile_calls.append((model, backend))
        return model

    monkeypatch.setattr(torch_layer, "DFTRACER_ENABLE", True)
    monkeypatch.setattr(torch_layer, "create_backend", create_test_backend)
    monkeypatch.setattr(torch_layer.torch, "compile", compile_model)

    layers = torch_layer.PyTorchLayers(loss_function=lambda prediction, target: prediction)
    model = layers.get_model(lambda inputs: inputs)

    assert backend_calls == [
        {"name": "PyTorchLayers", "enable": True, "autograd": True}
    ]
    assert compile_calls == [(layers._raw_model, backend)]
    assert model is layers._raw_model


def test_dftracer_dynamo_backend_is_not_applied_when_disabled(monkeypatch):
    def fail_if_called(*args, **kwargs):
        raise AssertionError("DFTracer backend should not be created when disabled")

    monkeypatch.setattr(torch_layer, "DFTRACER_ENABLE", False)
    monkeypatch.setattr(torch_layer, "create_backend", fail_if_called)
    monkeypatch.setattr(torch_layer.torch, "compile", fail_if_called)

    layers = torch_layer.PyTorchLayers(loss_function=lambda prediction, target: prediction)
    model = layers.get_model(lambda inputs: inputs)

    assert model is layers._raw_model
