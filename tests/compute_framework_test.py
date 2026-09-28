"""Single-rank compute contract before architecture registrations land."""

import numpy as np
import pytest

from dlio_benchmark.common.enumerations import FrameworkType, Model
from dlio_benchmark.utils.config import ConfigArguments
from dlio_benchmark.utils.utility import DLIOMPI


def _args():
    mpi = DLIOMPI.get_instance()
    try:
        mpi.size()
    except Exception:
        mpi.initialize()
    return ConfigArguments.get_instance()


@pytest.mark.parametrize("framework", list(FrameworkType))
@pytest.mark.parametrize("model_type", [Model.DEFAULT, Model.SLEEP])
def test_factory_non_model_modes_return_none(framework, model_type):
    from dlio_benchmark.model import ModelFactory

    assert ModelFactory.create_model(framework, model_type) is None


@pytest.mark.parametrize("framework", list(FrameworkType))
def test_factory_rejects_unregistered_model(framework):
    from dlio_benchmark.model import ModelFactory

    with pytest.raises(ValueError, match="Unsupported model type"):
        ModelFactory.create_model(framework, Model.BERT)


def test_torch_forward_backward_and_optimizer(monkeypatch):
    torch = pytest.importorskip("torch")
    _args()
    from dlio_benchmark.framework.torch_framework import TorchFramework
    from dlio_benchmark.model.model_factory import ModelFactory

    monkeypatch.setattr(
        ModelFactory,
        "create_pytorch_model",
        lambda model_type: torch.nn.Linear(3, 2),
    )
    # Keep the engine test independent of later architecture-specific hooks.
    def configure_training(self, model_type):
        self._loss_function = torch.nn.CrossEntropyLoss()
        self._optimizer = torch.optim.SGD(self._training_model.parameters(), lr=0.1)

    def prepare_batch(self, batch):
        return (
            torch.as_tensor(batch[0], dtype=torch.float32, device=self.device),
            torch.as_tensor(batch[1], dtype=torch.long, device=self.device),
        )

    monkeypatch.setattr(TorchFramework, "_configure_training", configure_training)
    monkeypatch.setattr(TorchFramework, "_prepare_batch", prepare_batch)
    framework = TorchFramework(False, Model.BERT)
    before = framework.native_model.weight.detach().clone()
    prediction, loss = framework.compute(
        (np.ones((4, 3), dtype=np.float32), np.zeros(4, dtype=np.int64)),
        1, 1, 0,
    )
    assert tuple(prediction.shape) == (4, 2)
    assert loss.detach().item() > 0
    assert not torch.equal(framework.native_model.weight, before)
    framework.finalize()


def test_tensorflow_forward_backward_and_optimizer(monkeypatch):
    tf = pytest.importorskip("tensorflow")
    _args()
    from dlio_benchmark.framework.tf_framework import TFFramework
    from dlio_benchmark.model.model_factory import ModelFactory

    model = tf.keras.Sequential([
        tf.keras.Input(shape=(3,)),
        tf.keras.layers.Dense(2),
    ])
    monkeypatch.setattr(
        ModelFactory, "create_tensorflow_model", lambda model_type: model
    )
    def configure_training(self, model_type):
        self._loss_function = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True
        )
        self._optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)

    def prepare_batch(self, batch):
        return tf.convert_to_tensor(batch[0]), tf.convert_to_tensor(batch[1])

    monkeypatch.setattr(TFFramework, "_configure_training", configure_training)
    monkeypatch.setattr(TFFramework, "_prepare_batch", prepare_batch)
    framework = TFFramework(False, Model.BERT)
    before = model.trainable_variables[0].numpy().copy()
    prediction, loss = framework.compute(
        (np.ones((4, 3), dtype=np.float32), np.zeros(4, dtype=np.int32)),
        1, 1, 0,
    )
    assert tuple(prediction.shape) == (4, 2)
    assert float(loss) > 0
    assert not np.array_equal(model.trainable_variables[0].numpy(), before)
    framework.finalize()


@pytest.mark.parametrize("backend", ["torch", "tensorflow"])
def test_default_and_missing_batch_keep_sleep_contract(monkeypatch, backend):
    _args()
    if backend == "torch":
        from dlio_benchmark.framework import torch_framework as module
        framework = module.TorchFramework(False, Model.DEFAULT)
    else:
        from dlio_benchmark.framework import tf_framework as module
        framework = module.TFFramework(False, Model.DEFAULT)

    calls = []
    monkeypatch.setattr(module, "sleep", lambda duration: calls.append(duration))
    assert framework.compute(None, 1, 1, 0.25) is None
    assert calls == [0.25]
    framework.finalize()


def test_cpu_synthetic_loader_yields_local_batches(monkeypatch):
    torch = pytest.importorskip("torch")
    args = _args()
    args.total_samples_train = 8
    args.batch_size = 4
    args.resized_image = np.zeros((3, 5), dtype=np.uint8)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    from dlio_benchmark.common.enumerations import DatasetType, FormatType
    from dlio_benchmark.data_loader.synthetic_data_loader import SyntheticDataLoader

    loader = SyntheticDataLoader(FormatType.SYNTHETIC, DatasetType.TRAIN, 0)
    assert tuple(loader.getitem().shape) == (4, 3, 5)
    assert not loader.getitem().is_pinned()
    assert len(list(loader.next())) == 2


def test_indexed_binary_generator_lookup_does_not_reference_loader_enum():
    from dlio_benchmark.common.enumerations import FormatType
    from dlio_benchmark.data_generator.generator_factory import GeneratorFactory
    from dlio_benchmark.data_generator.indexed_binary_generator import IndexedBinaryGenerator

    assert isinstance(
        GeneratorFactory.get_generator(FormatType.INDEXED_BINARY),
        IndexedBinaryGenerator,
    )



def test_load_mem_matches_actual_torch_loader_on_cpu(monkeypatch):
    torch = pytest.importorskip("torch")
    args = _args()
    args.total_samples_train = 8
    args.batch_size = 4
    args.resized_image = np.arange(15, dtype=np.uint8).reshape(3, 5)
    args.file_list_train = [f"dummy_{i}" for i in range(8)]
    args.train_global_index_map = {}
    args.train_file_map = {}
    args.training_steps = 2
    args.epochs = 1
    args.read_threads = 0
    args.prefetch_size = 2
    args.pin_memory = False
    args.reader_class = None
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    from dlio_benchmark.common.enumerations import DatasetType, FormatType
    from dlio_benchmark.data_loader.torch_data_loader import TorchDataLoader
    from dlio_benchmark.data_loader import load_mem_data_loader as module

    monkeypatch.setattr(module.time, "sleep", lambda duration: None)
    source_loader = TorchDataLoader(FormatType.SYNTHETIC, DatasetType.TRAIN, 0)
    cached_loader = module.LoadMemDataLoader(
        FormatType.SYNTHETIC, DatasetType.TRAIN, 0
    )
    source_loader.read()
    source = list(source_loader.next())
    cached = list(cached_loader.next())
    assert len(source) == len(cached) == 2
    for source_batch, cached_batch in zip(source, cached):
        assert cached_batch.shape == source_batch.shape
        assert cached_batch.dtype == source_batch.dtype
        assert torch.equal(cached_batch, source_batch)
        assert not cached_batch.is_pinned()
    source_loader.finalize()
    cached_loader.finalize()


def test_torch_trace_backend_and_phase_events(monkeypatch):
    torch = pytest.importorskip("torch")
    args = _args()
    from dlio_benchmark.framework import torch_framework as module
    from dftracer.python.ai_common import dftracer as tracer
    from importlib import import_module
    dynamo = import_module("dftracer.python.dynamo")

    backend_calls = []
    compiled = []
    backend = object()
    monkeypatch.setattr(module, "DFTRACER_ENABLE", True)
    monkeypatch.setattr(dynamo, "create_backend", lambda **kw: backend_calls.append(kw) or backend)
    monkeypatch.setattr(torch, "compile", lambda model, backend: compiled.append(backend) or model)

    model = torch.nn.Linear(3, 2)
    assert module.TorchFramework._configure_tracing(model) is model
    assert compiled == [backend]
    assert backend_calls == [{"name": "TorchFramework", "enable": True, "autograd": True}]

    events = []
    class FakeTracer:
        tick = 0
        def get_time(self):
            self.tick += 100
            return self.tick
        def log_event(self, name, category, start, duration):
            events.append((name, category, start, duration))

    monkeypatch.setattr(tracer, "get_instance", lambda: FakeTracer())
    framework = module.TorchFramework.__new__(module.TorchFramework)
    framework.args = args
    framework.device = torch.device("cpu")
    framework._model = model
    framework._optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    framework._loss_function = torch.nn.CrossEntropyLoss()
    monkeypatch.setattr(
        framework, "_prepare_batch",
        lambda batch: (torch.ones((4, 3)), torch.zeros(4, dtype=torch.long)),
    )
    framework._train_batch(None)
    assert [event[0] for event in events] == ["forward_pass", "backward_pass"]
    assert all(event[1] == "TorchFramework" and event[3] > 0 for event in events)



def test_unknown_native_model_is_rejected_but_legacy_name_sleeps():
    from dlio_benchmark.utils.config import LoadConfig

    args = _args()
    with pytest.raises(ValueError, match="Unsupported workload.model.name"):
        LoadConfig(args, {"train": {"compute": True}, "model": {"name": "legacy"}})
    LoadConfig(args, {"train": {"compute": False}, "model": {"name": "legacy"}})
    assert args.model is Model.SLEEP



def test_compute_config_preserves_train_and_eval_batch_sizes():
    from dlio_benchmark.utils.config import LoadConfig

    args = _args()
    LoadConfig(args, {
        "train": {"compute": True},
        "reader": {"batch_size": 5, "batch_size_eval": 3},
    })
    assert args.compute is True
    assert args.batch_size == 5
    assert args.batch_size_eval == 3
