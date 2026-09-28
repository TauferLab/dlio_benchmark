"""
   Copyright (c) 2025, UChicago Argonne, LLC
   All Rights Reserved
   
   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""

"""PyTorch framework integration for DLIO's native models."""

import os
import socket
from typing import Any, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as functional

from dlio_benchmark.common.constants import MODULE_AI_FRAMEWORK
from dlio_benchmark.common.enumerations import (
    DataLoaderType,
    DatasetType,
    FrameworkType,
    Model,
)
from dlio_benchmark.framework.framework import DummyTraceObject, Framework
from dlio_benchmark.model.model_factory import ModelFactory
from dlio_benchmark.utils.utility import (
    DFTRACER_ENABLE,
    DLIOMPI,
    Profile,
    dft_ai,
    sleep,
)

dlp = Profile(MODULE_AI_FRAMEWORK)

class TorchFramework(Framework):
    """Own device placement and native PyTorch training."""

    __instance = None

    @dlp.log_init
    def __init__(
        self,
        profiling,
        model: Model = Model.SLEEP,
        communication: bool = False,
    ):
        super().__init__()
        self.profiling = profiling
        self.reader_handler = None
        self.model_type = model
        self.communication = communication
        self._distributed_initialized_here = False
        self._optimizer = None
        self._loss_function = None
        self.native_model = None
        self._training_model = None
        self._model = None

        mpi = DLIOMPI.get_instance()
        self.gpu_id = mpi.local_rank()
        if torch.cuda.is_available():
            torch.cuda.set_device(self.gpu_id)
            self.device = torch.device("cuda", self.gpu_id)
        else:
            self.device = torch.device("cpu")

        # Retain the diagnostic from the source branch for rank/device setup.
        print(
            f"Creating model for framework {FrameworkType.PYTORCH}, "
            f"model_type {model}, communication {communication}, "
            f"gpu_id {self.gpu_id}"
        )
        if model not in (Model.SLEEP, Model.DEFAULT):
            try:
                self.native_model = ModelFactory.create_pytorch_model(model)
                self.native_model.to(self.device)
                self._training_model = self._configure_distributed(
                    self.native_model, communication
                )
                self._configure_training(model)
                self._model = self._configure_tracing(self._training_model)
            except BaseException:
                # DDP may have initialized a process group before a later
                # optimizer or tracing step fails.
                self.finalize()
                raise
        elif communication:
            self.args.logger.warning(
                "Communication requested without native compute; no gradients will be exchanged"
            )

    def _configure_distributed(self, model, communication):
        """Wrap a native module with MPI-sized DDP when requested."""
        if not communication:
            return model

        mpi = DLIOMPI.get_instance()
        world_size = mpi.size()
        if world_size <= 1:
            self.args.logger.warning(
                "Disabling PyTorch distributed communication for a one-rank run"
            )
            self.communication = False
            return model

        import torch.distributed as distributed
        from torch.nn.parallel import DistributedDataParallel

        backend = os.environ.get(
            "TORCH_DISTRIBUTED_BACKEND",
            "nccl" if self.device.type == "cuda" else "gloo",
        )
        if distributed.is_initialized():
            actual = (
                distributed.get_backend(),
                distributed.get_rank(),
                distributed.get_world_size(),
            )
            expected = (backend, mpi.rank(), world_size)
            if actual != expected:
                raise ValueError(
                    "Existing PyTorch process group backend, rank, or world size "
                    f"does not match MPI: expected {expected}, found {actual}"
                )
        else:
            master_addr = socket.gethostname() if mpi.rank() == 0 else None
            master_addr = mpi.comm().bcast(master_addr, root=0)
            os.environ.setdefault("MASTER_ADDR", master_addr)
            os.environ.setdefault("MASTER_PORT", "2345")
            try:
                distributed.init_process_group(
                    backend=backend,
                    rank=mpi.rank(),
                    world_size=world_size,
                )
            except BaseException:
                if distributed.is_initialized():
                    distributed.destroy_process_group()
                raise
            self._distributed_initialized_here = True

        try:
            if self.device.type == "cuda":
                return DistributedDataParallel(
                    model,
                    device_ids=[self.gpu_id],
                    output_device=self.gpu_id,
                )
            return DistributedDataParallel(model)
        except BaseException:
            self.finalize()
            raise

    def _configure_training(self, model_type):
        self._loss_function = torch.nn.CrossEntropyLoss()
        if model_type == Model.RESNET:
            self._optimizer = torch.optim.SGD(
                self._training_model.parameters(),
                lr=1,
                momentum=1,
                weight_decay=1,
            )
        elif model_type == Model.UNET:
            self._optimizer = torch.optim.Adam(
                self._training_model.parameters(), lr=1e-4
            )
        else:
            raise ValueError(f"Unsupported PyTorch model: {model_type}")

    @staticmethod
    def _configure_tracing(model):
        if not DFTRACER_ENABLE:
            return model

        from dftracer.python.dynamo import create_backend

        backend = create_backend(
            name="TorchFramework", enable=True, autograd=True
        )
        return torch.compile(model, backend=backend)

    @dlp.log
    def init_loader(self, format_type, epoch=0, data_loader=None):
        if data_loader is None:
            data_loader = DataLoaderType.PYTORCH
        super().init_loader(format_type, epoch, data_loader)

    @dlp.log
    def get_type(self):
        return FrameworkType.PYTORCH

    @staticmethod
    def get_instance(
        profiling,
        model: Model = Model.SLEEP,
        communication: bool = False,
    ):
        """Static access method."""
        if TorchFramework.__instance is None:
            TorchFramework.__instance = TorchFramework(
                profiling, model, communication
            )
        return TorchFramework.__instance

    @dlp.log
    def start_framework_profiler(self):
        pass

    @dlp.log
    def stop_framework_profiler(self):
        pass

    @dlp.log
    def trace_object(self, string, step, r):
        return DummyTraceObject(string, step, r)

    @dft_ai.compute
    def compute(self, batch, epoch_number, step, computation_time):
        self.args.logger.debug(
            f"Computing for epoch {epoch_number}, step {step}, "
            f"computation_time {computation_time}"
        )
        return self.model(epoch_number, batch, computation_time)

    def model(self, epoch, batch, computation_time):
        if self._model is None or batch is None:
            sleep(computation_time)
            return None
        return self._train_batch(batch)

    @staticmethod
    def _split_batch(batch) -> Tuple[Any, Optional[Any]]:
        while isinstance(batch, (tuple, list)) and len(batch) == 1:
            batch = batch[0]
        if isinstance(batch, dict):
            inputs = None
            target = None
            for key in ("data", "image", "input", "inputs"):
                if key in batch:
                    inputs = batch[key]
                    break
            for key in ("label", "labels", "target", "targets"):
                if key in batch:
                    target = batch[key]
                    break
            if inputs is None and len(batch) == 1:
                inputs = next(iter(batch.values()))
            if inputs is None:
                raise ValueError(
                    f"Could not find model input in batch keys {tuple(batch)}"
                )
            return inputs, target
        if isinstance(batch, (tuple, list)) and len(batch) == 2:
            return batch[0], batch[1]
        return batch, None

    @staticmethod
    def _as_tensor(value):
        if isinstance(value, torch.Tensor):
            return value
        if hasattr(value, "as_array"):
            value = value.as_array()
        elif hasattr(value, "as_tensor"):
            value = value.as_tensor()
            if hasattr(value, "as_array"):
                value = value.as_array()
        if isinstance(value, np.ndarray):
            return torch.from_numpy(value)
        if hasattr(value, "__dlpack__"):
            return torch.utils.dlpack.from_dlpack(value)
        return torch.as_tensor(value)

    def _prepare_resnet_batch(self, batch):
        inputs, target = self._split_batch(batch)
        inputs = self._as_tensor(inputs)

        if inputs.ndim == 2:
            inputs = inputs.unsqueeze(1).unsqueeze(2).repeat(1, 3, 1, 1)
        elif inputs.ndim == 3:
            inputs = inputs.unsqueeze(1).repeat(1, 3, 1, 1)
        elif inputs.ndim == 4:
            if inputs.shape[1] == 1:
                inputs = inputs.repeat(1, 3, 1, 1)
            elif inputs.shape[1] != 3 and inputs.shape[-1] in (1, 3):
                inputs = inputs.permute(0, 3, 1, 2)
                if inputs.shape[1] == 1:
                    inputs = inputs.repeat(1, 3, 1, 1)
            elif inputs.shape[1] != 3:
                raise ValueError(
                    "ResNet input must be BHW, BCHW, or BHWC with 1/3 channels; "
                    f"got {tuple(inputs.shape)}"
                )
        else:
            raise ValueError(
                f"ResNet input must have 2-4 dimensions; got {tuple(inputs.shape)}"
            )

        if target is not None:
            target = self._as_tensor(target)
        inputs = inputs.to(self.device, dtype=torch.float32, non_blocking=True)

        if target is None:
            target = torch.zeros(
                inputs.shape[0], dtype=torch.long, device=self.device
            )
        else:
            target = target.to(self.device, non_blocking=True)
            if target.ndim == 2 and target.shape[1] == 1:
                target = target[:, 0]
            if target.ndim == 1:
                target = target.long()
            else:
                target = target.float()
        return inputs, target

    def _prepare_unet_batch(self, batch):
        inputs, target = self._split_batch(batch)
        inputs = self._as_tensor(inputs)

        if inputs.ndim == 3:
            inputs = inputs.unsqueeze(1).unsqueeze(2)
        elif inputs.ndim == 4:
            inputs = inputs.unsqueeze(1)
        elif inputs.ndim == 5:
            if inputs.shape[1] != 1 and inputs.shape[-1] == 1:
                inputs = inputs.permute(0, 4, 1, 2, 3)
            elif inputs.shape[1] != 1:
                raise ValueError(
                    "UNet3D input must be BDHW, BCDHW, or BDHWC with one "
                    f"channel; got {tuple(inputs.shape)}"
                )
        else:
            raise ValueError(
                f"UNet3D input must have 3-5 dimensions; got {tuple(inputs.shape)}"
            )

        if target is not None:
            target = self._as_tensor(target)
        inputs = inputs.to(self.device, dtype=torch.float32, non_blocking=True)

        # DALI sample labels are classification IDs, not segmentation masks.
        # For those scalar labels, synthesize the same valid zero mask used
        # when a loader supplies input data only.
        if target is None or target.ndim <= 2:
            target = torch.zeros(
                (inputs.shape[0], *inputs.shape[2:]),
                dtype=torch.long,
                device=self.device,
            )
        else:
            target = target.to(self.device, non_blocking=True)
            if target.ndim == 3:
                target = target.unsqueeze(1)
            elif target.ndim == 5:
                if target.shape[1] == 1:
                    target = target[:, 0]
                elif target.shape[1] == 3:
                    target = target.argmax(dim=1)
                elif target.shape[-1] == 1:
                    target = target[..., 0]
                elif target.shape[-1] == 3:
                    target = target.argmax(dim=-1)
                else:
                    raise ValueError(
                        "UNet3D target must have one label per voxel or "
                        f"three class channels; got {tuple(target.shape)}"
                    )
            elif target.ndim != 4:
                raise ValueError(
                    "UNet3D target must be BHW, BDHW, BCDHW, or BDHWC; "
                    f"got {tuple(target.shape)}"
                )
            target = target.long()
        return inputs, target

    def _prepare_batch(self, batch):
        if self.model_type == Model.RESNET:
            return self._prepare_resnet_batch(batch)
        if self.model_type == Model.UNET:
            return self._prepare_unet_batch(batch)
        raise ValueError(f"Unsupported PyTorch model: {self.model_type}")

    @staticmethod
    def _align_unet_target(prediction, target):
        if tuple(target.shape[1:]) == tuple(prediction.shape[2:]):
            return target
        target = functional.interpolate(
            target.unsqueeze(1).float(),
            size=prediction.shape[2:],
            mode="nearest",
        )
        return target[:, 0].long()

    def _train_batch(self, batch):
        inputs, target = self._prepare_batch(batch)
        self._optimizer.zero_grad(set_to_none=True)

        if DFTRACER_ENABLE:
            from dftracer.python.ai_common import dftracer as tracer

            trace = tracer.get_instance()
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            started = trace.get_time()

        prediction = self._model(inputs)
        if getattr(self, "model_type", None) == Model.UNET:
            target = self._align_unet_target(prediction, target)
        loss = self._loss_function(prediction, target)

        if DFTRACER_ENABLE:
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            ended = trace.get_time()
            trace.log_event(
                "forward_pass",
                "TorchFramework",
                int(started),
                int(ended - started),
            )
            backward_started = trace.get_time()

        loss.backward()
        self._optimizer.step()

        if DFTRACER_ENABLE:
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            backward_ended = trace.get_time()
            trace.log_event(
                "backward_pass",
                "TorchFramework",
                int(backward_started),
                int(backward_ended - backward_started),
            )
        return prediction, loss

    def finalize(self):
        """Release only a PyTorch process group created by this framework."""
        if self._distributed_initialized_here:
            import torch.distributed as distributed

            try:
                if distributed.is_initialized():
                    distributed.destroy_process_group()
            finally:
                self._distributed_initialized_here = False

    @dlp.log
    def get_loader(self, dataset_type=DatasetType.TRAIN):
        if dataset_type == DatasetType.TRAIN:
            return self.reader_train
        else:
            return self.reader_valid

    @dlp.log
    def is_nativeio_available(self):
        return False
