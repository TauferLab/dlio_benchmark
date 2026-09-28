"""Small end-to-end TensorFlow cases from the preserved model test matrix.

Each case uses a fresh benchmark and confirms native optimizer steps during a
real epoch. Native TensorFlow compute is single-rank; the multi-rank rejection
is covered by the communication branch's rank-guard test.
"""

from pathlib import Path
import shutil
import tempfile

from hydra import compose, initialize_config_dir
from mpi4py import MPI
import pytest

# DLIO utility initializes DFTracer's PyTorch hooks before TensorFlow loads.
from dlio_benchmark.utils.utility import DLIOMPI


def _run_native_benchmark(model_name, communication, monkeypatch):
    tf = pytest.importorskip("tensorflow")
    import dlio_benchmark
    comm = MPI.COMM_WORLD
    if comm.Get_size() > 1:
        pytest.skip("Native TensorFlow benchmark runs on one MPI rank")

    # Earlier loader tests may leave the process-global singleton with child
    # state after unpickling configuration. Start each benchmark with a fresh
    # main-process MPI wrapper, while keeping mpi4py's world communicator.
    DLIOMPI.reset()
    DLIOMPI.get_instance().initialize()
    from dlio_benchmark.common.enumerations import Model
    from dlio_benchmark.framework.tf_framework import TFFramework
    from dlio_benchmark.main import DLIOBenchmark
    from dlio_benchmark.utils.config import ConfigArguments

    root = tempfile.mkdtemp(prefix="dlio-tensorflow-benchmark-")
    monkeypatch.chdir(root)
    TFFramework._TFFramework__instance = None
    ConfigArguments.reset()

    overrides = [
        "++workload.workflow.generate_data=true",
        "++workload.workflow.train=true",
        "++workload.workflow.evaluation=false",
        f"++workload.model.name={model_name}",
        "++workload.framework=tensorflow",
        "++workload.dataset.format=synthetic",
        "++workload.reader.data_loader=synthetic",
        "++workload.dataset.data_folder=data/model-smoke",
        "++workload.dataset.num_files_train=" + ("2" if model_name == "resnet50" else "1"),
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
        framework = benchmark.framework
        assert framework.model_type == (Model.RESNET if model_name == "resnet50" else Model.UNET)
        assert isinstance(framework.native_model, tf.keras.Model)
        benchmark.initialize()
        benchmark.run()
        benchmark.finalize()
        assert int(framework._optimizer.iterations.numpy()) > 0
        assert framework.native_model.trainable_variables
    finally:
        TFFramework._TFFramework__instance = None
        shutil.rmtree(root)


def test_resnet_model_with_compute_enabled(monkeypatch):
    _run_native_benchmark("resnet50", False, monkeypatch)


def test_unet3d_model_with_compute_enabled(monkeypatch):
    _run_native_benchmark("unet3d", False, monkeypatch)


def test_resnet_model_with_comms_enabled(monkeypatch):
    _run_native_benchmark("resnet50", True, monkeypatch)


def test_unet3d_model_with_comms_enabled(monkeypatch):
    _run_native_benchmark("unet3d", True, monkeypatch)
