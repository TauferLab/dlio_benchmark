"""Communication behavior that does not depend on ResNet or UNet."""

import os
import shutil
import socket
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _free_port():
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return listener.getsockname()[1]


def _run_mpi_sync(backend, force_cpu):
    if shutil.which("mpirun") is None:
        pytest.skip("mpirun is unavailable")

    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    env["MASTER_ADDR"] = "127.0.0.1"
    env["MASTER_PORT"] = str(_free_port())
    env["TORCH_DISTRIBUTED_BACKEND"] = backend
    env["DLIO_COMM_TEST_BACKEND"] = backend
    if force_cpu:
        env["CUDA_VISIBLE_DEVICES"] = ""
    env["OMPI_ALLOW_RUN_AS_ROOT"] = "1"
    env["OMPI_ALLOW_RUN_AS_ROOT_CONFIRM"] = "1"
    command = [
        "mpirun", "-n", "2", sys.executable,
        "-m", "tests.communication_mpi_worker",
    ]
    result = subprocess.run(
        command,
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_mpi_gloo_sync():
    _run_mpi_sync("gloo", force_cpu=True)


def test_mpi_nccl_sync():
    import torch

    if torch.cuda.device_count() < 2:
        pytest.skip("two CUDA devices are required for the NCCL smoke test")
    _run_mpi_sync("nccl", force_cpu=False)


def _initialize_mpi():
    from dlio_benchmark.utils.utility import DLIOMPI

    mpi = DLIOMPI.get_instance()
    try:
        mpi.size()
    except Exception:
        mpi.initialize()
    return mpi


def _torch_stub():
    from types import SimpleNamespace
    from unittest.mock import Mock

    import torch

    _initialize_mpi()
    from dlio_benchmark.framework.torch_framework import TorchFramework

    framework = object.__new__(TorchFramework)
    framework.args = SimpleNamespace(logger=Mock())
    framework.device = torch.device("cpu")
    framework.gpu_id = 0
    framework.communication = True
    framework._distributed_initialized_here = False
    return framework


def test_disabled_communication_never_initializes_process_group(monkeypatch):
    import torch
    import torch.distributed as distributed

    framework = _torch_stub()
    monkeypatch.setattr(
        distributed, "init_process_group",
        lambda *args, **kwargs: pytest.fail("unexpected process group"),
    )
    module = torch.nn.Linear(1, 1)
    assert framework._configure_distributed(module, False) is module
    framework.finalize()


def test_one_rank_direct_construction_disables_communication(monkeypatch):
    import torch
    import torch.distributed as distributed
    from types import SimpleNamespace
    from unittest.mock import Mock

    from dlio_benchmark.utils.utility import DLIOMPI

    framework = _torch_stub()
    monkeypatch.setattr(
        DLIOMPI, "get_instance", staticmethod(lambda: SimpleNamespace(size=lambda: 1))
    )
    monkeypatch.setattr(
        distributed, "init_process_group",
        lambda *args, **kwargs: pytest.fail("unexpected process group"),
    )
    module = torch.nn.Linear(1, 1)
    assert framework._configure_distributed(module, True) is module
    assert framework.communication is False
    framework.args.logger.warning.assert_called_once()
    framework.finalize()


def test_compatible_external_group_is_not_destroyed(monkeypatch):
    import torch
    import torch.distributed as distributed
    from types import SimpleNamespace
    from unittest.mock import Mock

    from dlio_benchmark.utils.utility import DLIOMPI

    framework = _torch_stub()
    mpi = SimpleNamespace(size=lambda: 2, rank=lambda: 0)
    monkeypatch.setattr(DLIOMPI, "get_instance", staticmethod(lambda: mpi))
    monkeypatch.setattr(distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(distributed, "get_backend", lambda: "gloo")
    monkeypatch.setattr(distributed, "get_rank", lambda: 0)
    monkeypatch.setattr(distributed, "get_world_size", lambda: 2)
    destroy = Mock()
    monkeypatch.setattr(distributed, "destroy_process_group", destroy)
    monkeypatch.setattr(
        torch.nn.parallel, "DistributedDataParallel", lambda module: module
    )
    module = torch.nn.Linear(1, 1)
    assert framework._configure_distributed(module, True) is module
    framework.finalize()
    destroy.assert_not_called()


def test_incompatible_external_group_is_rejected(monkeypatch):
    import torch
    import torch.distributed as distributed
    from types import SimpleNamespace
    from unittest.mock import Mock

    from dlio_benchmark.utils.utility import DLIOMPI

    framework = _torch_stub()
    mpi = SimpleNamespace(size=lambda: 2, rank=lambda: 0)
    monkeypatch.setattr(DLIOMPI, "get_instance", staticmethod(lambda: mpi))
    monkeypatch.setattr(distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(distributed, "get_backend", lambda: "gloo")
    monkeypatch.setattr(distributed, "get_rank", lambda: 0)
    monkeypatch.setattr(distributed, "get_world_size", lambda: 3)
    destroy = Mock()
    monkeypatch.setattr(distributed, "destroy_process_group", destroy)
    with pytest.raises(ValueError, match="world size"):
        framework._configure_distributed(torch.nn.Linear(1, 1), True)
    framework.finalize()
    destroy.assert_not_called()


def test_owned_group_is_destroyed_if_ddp_wrap_fails(monkeypatch):
    import torch
    import torch.distributed as distributed
    from types import SimpleNamespace
    from unittest.mock import Mock

    from dlio_benchmark.utils.utility import DLIOMPI

    framework = _torch_stub()
    comm = Mock()
    comm.allgather.return_value = [(None, None, None), (None, None, None)]
    comm.bcast.return_value = "localhost"
    mpi = SimpleNamespace(size=lambda: 2, rank=lambda: 0, comm=lambda: comm)
    monkeypatch.setattr(DLIOMPI, "get_instance", staticmethod(lambda: mpi))
    monkeypatch.setenv("MASTER_ADDR", "localhost")
    monkeypatch.setenv("MASTER_PORT", "23455")
    initialized = False

    def init_group(**kwargs):
        nonlocal initialized
        initialized = True

    def destroy_group():
        nonlocal initialized
        initialized = False

    monkeypatch.setattr(distributed, "is_initialized", lambda: initialized)
    monkeypatch.setattr(distributed, "init_process_group", init_group)
    destroy = Mock(side_effect=destroy_group)
    monkeypatch.setattr(distributed, "destroy_process_group", destroy)

    def bad_wrapper(module):
        raise RuntimeError("DDP wrap failed")

    monkeypatch.setattr(torch.nn.parallel, "DistributedDataParallel", bad_wrapper)
    with pytest.raises(RuntimeError, match="DDP wrap failed"):
        framework._configure_distributed(torch.nn.Linear(1, 1), True)
    assert not initialized
    assert not framework._distributed_initialized_here
    destroy.assert_called_once()


@pytest.mark.parametrize("communication", [False, True])
def test_tensorflow_native_model_rejects_multiple_ranks(monkeypatch, communication):
    from types import SimpleNamespace

    _initialize_mpi()
    from dlio_benchmark.utils.config import ConfigArguments
    ConfigArguments.get_instance()
    from dlio_benchmark.framework.tf_framework import TFFramework
    from dlio_benchmark.common.enumerations import Model
    from dlio_benchmark.utils.utility import DLIOMPI

    monkeypatch.setattr(
        DLIOMPI, "get_instance", staticmethod(lambda: SimpleNamespace(size=lambda: 2, rank=lambda: 0))
    )
    with pytest.raises(NotImplementedError, match="one MPI rank"):
        TFFramework(False, Model.BERT, communication)


@pytest.mark.parametrize("model_type", ["sleep", "default"])
def test_non_native_compute_never_initializes_ddp(monkeypatch, model_type):
    import torch
    import torch.distributed as distributed
    from types import SimpleNamespace

    from dlio_benchmark.common.enumerations import Model
    from dlio_benchmark.utils.config import ConfigArguments
    from dlio_benchmark.utils.utility import DLIOMPI

    _initialize_mpi()
    ConfigArguments.get_instance()
    from dlio_benchmark.framework.torch_framework import TorchFramework

    mpi = SimpleNamespace(size=lambda: 2, rank=lambda: 0, local_rank=lambda: 0)
    monkeypatch.setattr(DLIOMPI, "get_instance", staticmethod(lambda: mpi))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        distributed, "init_process_group",
        lambda **kwargs: pytest.fail("unexpected process group"),
    )
    framework = TorchFramework(False, Model(model_type), communication=True)
    assert framework.native_model is None
    assert framework._training_model is None
    framework.finalize()


def test_owned_group_is_destroyed_if_training_setup_fails(monkeypatch):
    import torch
    import torch.distributed as distributed
    from types import SimpleNamespace
    from unittest.mock import Mock

    from dlio_benchmark.common.enumerations import Model
    from dlio_benchmark.utils.config import ConfigArguments
    from dlio_benchmark.utils.utility import DLIOMPI

    _initialize_mpi()
    ConfigArguments.get_instance()
    from dlio_benchmark.framework.torch_framework import TorchFramework
    from dlio_benchmark.model.model_factory import ModelFactory

    comm = Mock()
    comm.bcast.return_value = "localhost"
    mpi = SimpleNamespace(
        size=lambda: 2, rank=lambda: 0, local_rank=lambda: 0, comm=lambda: comm
    )
    monkeypatch.setattr(DLIOMPI, "get_instance", staticmethod(lambda: mpi))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setenv("MASTER_ADDR", "localhost")
    monkeypatch.setenv("MASTER_PORT", "23456")
    initialized = False

    def init_group(**kwargs):
        nonlocal initialized
        initialized = True

    def destroy_group():
        nonlocal initialized
        initialized = False

    monkeypatch.setattr(distributed, "is_initialized", lambda: initialized)
    monkeypatch.setattr(distributed, "init_process_group", init_group)
    destroy = Mock(side_effect=destroy_group)
    monkeypatch.setattr(distributed, "destroy_process_group", destroy)
    monkeypatch.setattr(
        torch.nn.parallel, "DistributedDataParallel", lambda module: module
    )
    monkeypatch.setattr(
        ModelFactory, "create_pytorch_model", lambda model_type: torch.nn.Linear(1, 1)
    )

    def fail_training(self, model_type):
        raise RuntimeError("optimizer setup failed")

    monkeypatch.setattr(TorchFramework, "_configure_training", fail_training)
    with pytest.raises(RuntimeError, match="optimizer setup failed"):
        TorchFramework(False, Model.BERT, communication=True)
    assert not initialized
    destroy.assert_called_once()


def test_tracing_wraps_ddp_training_model(monkeypatch):
    import torch
    import torch.distributed as distributed
    from types import SimpleNamespace
    from unittest.mock import Mock

    from dlio_benchmark.common.enumerations import Model
    from dlio_benchmark.utils.config import ConfigArguments
    from dlio_benchmark.utils.utility import DLIOMPI

    _initialize_mpi()
    ConfigArguments.get_instance()
    from dlio_benchmark.framework.torch_framework import TorchFramework
    from dlio_benchmark.model.model_factory import ModelFactory

    mpi = SimpleNamespace(size=lambda: 2, rank=lambda: 0, local_rank=lambda: 0)
    monkeypatch.setattr(DLIOMPI, "get_instance", staticmethod(lambda: mpi))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(distributed, "get_backend", lambda: "gloo")
    monkeypatch.setattr(distributed, "get_rank", lambda: 0)
    monkeypatch.setattr(distributed, "get_world_size", lambda: 2)
    destroy = Mock()
    monkeypatch.setattr(distributed, "destroy_process_group", destroy)
    monkeypatch.setattr(
        ModelFactory, "create_pytorch_model", lambda model_type: torch.nn.Linear(1, 2)
    )

    class FakeDDP(torch.nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, *args):
            return self.module(*args)

    monkeypatch.setattr(torch.nn.parallel, "DistributedDataParallel", FakeDDP)
    traced = []

    def trace_model(model):
        traced.append(model)
        return SimpleNamespace(module=model)

    monkeypatch.setattr(TorchFramework, "_configure_tracing", staticmethod(trace_model))
    framework = TorchFramework(False, Model.BERT, communication=True)
    assert isinstance(framework._training_model, FakeDDP)
    assert traced == [framework._training_model]
    assert framework._model.module is framework._training_model
    assert list(framework._optimizer.param_groups[0]["params"]) == list(
        framework._training_model.parameters()
    )
    framework.finalize()
    destroy.assert_not_called()


def test_benchmark_failure_releases_framework_group(monkeypatch):
    from unittest.mock import Mock

    _initialize_mpi()
    from dlio_benchmark import main as module

    benchmark = Mock()
    benchmark.run.side_effect = RuntimeError("training failed")
    monkeypatch.setattr(module, "DLIOBenchmark", lambda config: benchmark)
    with pytest.raises(RuntimeError, match="training failed"):
        module.run_benchmark.__wrapped__({"workload": {}})
    benchmark.framework.finalize.assert_called_once()
    benchmark.finalize.assert_not_called()
