"""End-to-end PyTorch benchmark cases from the preserved model test matrix.

Each case constructs a fresh benchmark and checks a trainable parameter after a
real epoch. The communication cases require a two-rank pytest invocation.
"""

from pathlib import Path
import socket
import shutil
import tempfile

from hydra import compose, initialize_config_dir
from mpi4py import MPI
import pytest


def _run_native_benchmark(model_name, communication, monkeypatch):
    torch = pytest.importorskip("torch")
    from dlio_benchmark.utils.utility import DLIOMPI
    import dlio_benchmark

    mpi = DLIOMPI.get_instance()
    try:
        mpi.size()
    except Exception:
        mpi.initialize()
    from dlio_benchmark.common.enumerations import Model
    from dlio_benchmark.framework import torch_framework
    from dlio_benchmark.main import DLIOBenchmark
    from dlio_benchmark.utils.config import ConfigArguments

    comm = MPI.COMM_WORLD
    if communication and comm.Get_size() < 2:
        pytest.skip("communication benchmark requires at least two MPI ranks")
    if communication:
        # A fresh rendezvous port keeps consecutive DDP model cases independent.
        if comm.Get_rank() == 0:
            with socket.socket() as listener:
                listener.bind(("127.0.0.1", 0))
                port = listener.getsockname()[1]
        else:
            port = None
        monkeypatch.setenv("MASTER_PORT", str(comm.bcast(port, root=0)))

    # Each test gets one shared filesystem root; both MPI ranks use the same
    # generated synthetic files, as in the actual benchmark.
    root = comm.bcast(tempfile.mkdtemp(prefix="dlio-pytorch-benchmark-") if comm.Get_rank() == 0 else None, root=0)
    monkeypatch.chdir(root)
    torch.set_num_threads(1)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch_framework, "DFTRACER_ENABLE", False)
    torch_framework.TorchFramework._TorchFramework__instance = None
    ConfigArguments.reset()

    overrides = [
        "++workload.workflow.generate_data=true",
        "++workload.workflow.train=true",
        "++workload.workflow.evaluation=false",
        f"++workload.model.name={model_name}",
        "++workload.framework=pytorch",
        "++workload.dataset.format=synthetic",
        "++workload.reader.data_loader=synthetic",
        "++workload.dataset.data_folder=data/model-smoke",
        f"++workload.dataset.num_files_train={comm.Get_size() * (2 if model_name == 'resnet50' else 1)}",
        "++workload.dataset.num_files_eval=0",
        "++workload.dataset.num_subfolders_train=0",
        "++workload.dataset.num_subfolders_eval=0",
        "++workload.dataset.record_length_bytes=16384",
        "++workload.reader.batch_size=" + ("2" if model_name == "resnet50" else "1"),
        "++workload.train.epochs=1",
        "++workload.train.compute=true",
        "++workload.train.communication=" + ("true" if communication else "false"),
        "++workload.train.computation_time=0",
    ]
    if model_name == "resnet50":
        overrides.append("++workload.dataset.record_length_bytes_resize=4096")
    else:
        overrides.append("++workload.reader.transformed_record_dims=[16,32,32]")

    config_dir = str(Path(dlio_benchmark.__file__).parent / "configs")
    try:
        with initialize_config_dir(version_base=None, config_dir=config_dir):
            cfg = compose(config_name="config", overrides=overrides)
        benchmark = DLIOBenchmark(cfg["workload"])
        assert benchmark.framework.model_type == (Model.RESNET if model_name == "resnet50" else Model.UNET)
        model = benchmark.framework.native_model
        assert model is not None
        parameter = model.fc.bias if model_name == "resnet50" else model.output_layer.bias
        before = parameter.detach().clone()
        benchmark.initialize()
        benchmark.run()
        benchmark.finalize()
        if communication:
            assert not torch.distributed.is_initialized()
        assert parameter.grad is not None
        assert torch.count_nonzero(parameter.grad) > 0
        assert not torch.equal(before, parameter.detach())
        if communication:
            values = comm.allgather(parameter.detach().cpu().numpy())
            for value in values[1:]:
                assert (values[0] == value).all()
    finally:
        torch_framework.TorchFramework._TorchFramework__instance = None
        comm.Barrier()
        if comm.Get_rank() == 0:
            shutil.rmtree(root)
        comm.Barrier()


def test_resnet_model_with_compute_enabled(monkeypatch):
    _run_native_benchmark("resnet50", False, monkeypatch)


def test_unet3d_model_with_compute_enabled(monkeypatch):
    _run_native_benchmark("unet3d", False, monkeypatch)


def test_resnet_model_with_comms_enabled(monkeypatch):
    _run_native_benchmark("resnet50", True, monkeypatch)


def test_unet3d_model_with_comms_enabled(monkeypatch):
    _run_native_benchmark("unet3d", True, monkeypatch)
