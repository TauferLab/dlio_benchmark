"""Two-rank CPU synchronization probe, launched by communication_test.py."""

import os

import torch
import torch.distributed as distributed
from torch.nn.parallel import DistributedDataParallel

from dlio_benchmark.utils.utility import DLIOMPI


def main():
    torch.set_num_threads(1)
    mpi = DLIOMPI.get_instance()
    mpi.initialize()
    from dlio_benchmark.common.enumerations import Model
    from dlio_benchmark.framework.torch_framework import TorchFramework
    from dlio_benchmark.model.model_factory import ModelFactory

    def tiny_model(model_type):
        model = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            model.weight.fill_(1.0)
        return model

    ModelFactory.create_pytorch_model = staticmethod(tiny_model)

    def configure_tiny_training(self, model_type):
        self._loss_function = torch.nn.MSELoss()
        self._optimizer = torch.optim.SGD(self._training_model.parameters(), lr=0.1)

    def prepare_tiny_batch(self, batch):
        sample = batch.to(self.device)
        return sample, torch.zeros_like(sample)

    TorchFramework._configure_training = configure_tiny_training
    TorchFramework._prepare_batch = prepare_tiny_batch
    framework = None
    completed = False
    try:
        framework = TorchFramework(False, Model.BERT, communication=True)
        assert isinstance(framework._training_model, DistributedDataParallel)
        backend = os.environ.get("DLIO_COMM_TEST_BACKEND", "gloo")
        assert distributed.get_backend() == backend
        if backend == "nccl":
            assert framework.device.index == mpi.local_rank()
            assert torch.cuda.current_device() == mpi.local_rank()
        assert distributed.get_rank() == mpi.rank()
        assert distributed.get_world_size() == 2

        # Different rank-local inputs produce different local gradients; DDP
        # must average them before the generic train kernel updates weights.
        sample = torch.tensor(
            [[float(mpi.rank() + 1)]], device=framework.device
        )
        framework.compute(sample, 1, 1, 0)
        parameter = float(framework.native_model.weight.item())
        gathered = mpi.comm().allgather(parameter)
        # Local gradients would be 2 and 8; DDP averages to 5, so both
        # weights must become 0.5. Merely checking DDP construction misses this.
        assert all(abs(weight - 0.5) < 1e-6 for weight in gathered), gathered
        completed = True
    finally:
        if framework is not None:
            framework.finalize()
        assert not distributed.is_initialized()
        if completed:
            mpi.finalize()


if __name__ == "__main__":
    main()
