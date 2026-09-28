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

from typing import Any, Optional, Tuple

import numpy as np
import torch

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
    ):
        super().__init__()
        self.profiling = profiling
        self.reader_handler = None
        self.model_type = model
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

        if model not in (Model.SLEEP, Model.DEFAULT):
            self.native_model = ModelFactory.create_pytorch_model(model)
            self.native_model.to(self.device)
            self._training_model = self.native_model
            self._configure_training(model)
            self._model = self._configure_tracing(self._training_model)

    def _configure_training(self, model_type):
        """Defaults for native classifiers; architecture PRs may specialize."""
        self._loss_function = torch.nn.CrossEntropyLoss()
        self._optimizer = torch.optim.SGD(self._training_model.parameters(), lr=0.1)

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
    ):
        """Static access method."""
        if TorchFramework.__instance is None:
            TorchFramework.__instance = TorchFramework(
                profiling, model
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
        self.args.logger.debug("is model None? %s", self._model is None)
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

    def _prepare_batch(self, batch):
        """Convert a labeled batch without assuming an architecture or layout."""
        inputs, target = self._split_batch(batch)
        if target is None:
            raise ValueError("Native model training requires a target")
        inputs = self._as_tensor(inputs).to(self.device, dtype=torch.float32)
        target = self._as_tensor(target).to(self.device)
        return inputs, target.long()

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
        """No framework resources are allocated by single-rank training."""
        return None

    @dlp.log
    def get_loader(self, dataset_type=DatasetType.TRAIN):
        if dataset_type == DatasetType.TRAIN:
            return self.reader_train
        else:
            return self.reader_valid

    @dlp.log
    def is_nativeio_available(self):
        return False
