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

"""TensorFlow framework integration for DLIO's native Keras models."""

from typing import Any, Optional, Tuple

import numpy as np

from dlio_benchmark.common.constants import MODULE_AI_FRAMEWORK
from dlio_benchmark.common.enumerations import (
    DataLoaderType,
    DatasetType,
    FrameworkType,
    MetadataType,
    Model,
    Profiler,
)
from dlio_benchmark.framework.framework import Framework
from dlio_benchmark.model.model_factory import ModelFactory
from dlio_benchmark.profiler.profiler_factory import ProfilerFactory
from dlio_benchmark.utils.utility import Profile, dft_ai, sleep

# DFTracer currently imports PyTorch/Triton while utility is initialized. Load
# it before TensorFlow to avoid a Triton/TensorFlow initialization crash in
# direct TFFramework imports.
import tensorflow as tf
from tensorflow.python.framework import errors

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

dlp = Profile(MODULE_AI_FRAMEWORK)


class TFFramework(Framework):
    """Own optimization and native GradientTape training for Keras models."""

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
        self.native_model = None
        self._model = None
        self._optimizer = None
        self._loss_function = None

        # Temporary behavior retained from the existing profiler integration.
        if profiling:
            if self.args.profiler != Profiler.IOSTAT:
                self.tensorboard = ProfilerFactory.get_profiler(Profiler.NONE)
            else:
                self.tensorboard = ProfilerFactory.get_profiler(
                    Profiler.TENSORBOARD
                )


        if model not in (Model.SLEEP, Model.DEFAULT):
            self.native_model = ModelFactory.create_tensorflow_model(model)
            self._model = self.native_model
            self._configure_training(model)

    def _configure_training(self, model_type):
        """Defaults for native classifiers; architecture PRs may specialize."""
        self._loss_function = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True
        )
        self._optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)

    @dlp.log
    def init_loader(self, format_type, epoch=0, data_loader=None):
        if data_loader is None:
            data_loader = DataLoaderType.TENSORFLOW
        super().init_loader(format_type, epoch, data_loader)

    @dlp.log
    def get_type(self):
        return FrameworkType.TENSORFLOW

    @staticmethod
    def get_instance(
        profiling,
        model: Model = Model.SLEEP,
    ):
        """Static access method."""
        if TFFramework.__instance is None:
            TFFramework.__instance = TFFramework(
                profiling, model
            )
        return TFFramework.__instance

    @dlp.log
    def start_framework_profiler(self):
        if self.profiling:
            self.tensorboard.start()

    @dlp.log
    def stop_framework_profiler(self):
        if self.profiling:
           self.tensorboard.stop()
        pass

    @dlp.log
    def trace_object(self, string, step, r):
        pass

    @dft_ai.compute
    def compute(self, batch, epoch_number, step, computation_time):
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
        if isinstance(value, tf.Tensor):
            return value
        if hasattr(value, "detach") and hasattr(value, "cpu"):
            value = value.detach().cpu().numpy()
        elif hasattr(value, "as_array"):
            value = value.as_array()
        elif hasattr(value, "as_tensor"):
            value = value.as_tensor()
            if hasattr(value, "as_array"):
                value = value.as_array()
        if isinstance(value, np.ndarray):
            return tf.convert_to_tensor(value)
        if hasattr(value, "__dlpack__"):
            return tf.experimental.dlpack.from_dlpack(value.__dlpack__())
        return tf.convert_to_tensor(value)

    def _prepare_batch(self, batch):
        """Convert a labeled batch without assuming an architecture or layout."""
        inputs, target = self._split_batch(batch)
        if target is None:
            raise ValueError("Native model training requires a target")
        return tf.cast(self._as_tensor(inputs), tf.float32), tf.cast(
            self._as_tensor(target), tf.int32
        )

    def _train_batch(self, batch):
        inputs, target = self._prepare_batch(batch)
        with tf.GradientTape() as tape:
            prediction = self._model(inputs, training=True)
            loss = self._loss_function(target, prediction)

        gradients = tape.gradient(loss, self._model.trainable_variables)
        gradients_and_variables = [
            (gradient, variable)
            for gradient, variable in zip(
                gradients, self._model.trainable_variables
            )
            if gradient is not None
        ]
        if gradients_and_variables:
            self._optimizer.apply_gradients(gradients_and_variables)
        else:
            self.args.logger.warning(
                "No TensorFlow gradients were produced; optimizer step skipped"
            )
        return prediction, loss

    def finalize(self):
        cleanup = getattr(self.native_model, "finalize", None)
        if callable(cleanup):
            cleanup()

    @dlp.log
    def get_loader(self, dataset_type=DatasetType.TRAIN):
        if dataset_type == DatasetType.TRAIN:
            return self.reader_train
        else:
            return self.reader_valid

    @dlp.log
    def is_nativeio_available(self):
        return True

    @dlp.log
    def create_node(self, id, exist_ok=False):
        tf.io.gfile.makedirs(id)
        return True

    @dlp.log
    def get_node(self, id):
        if tf.io.gfile.exists(id):
            if tf.io.gfile.isdir(id):
                return MetadataType.DIRECTORY
            else:
                return MetadataType.FILE
        else:
            return None

    @dlp.log
    def walk_node(self, id, use_pattern=False):
        try:
            if not use_pattern:
                return tf.io.gfile.listdir(id)
            else:
                return tf.io.gfile.glob(id)
        except errors.NotFoundError:
            return []

    @dlp.log
    def delete_node(self, id):
        tf.io.gfile.rmtree(id)
        return True

    @dlp.log
    def put_data(self, id, data, offset=None, length=None):
        with tf.io.gfile.GFile(id, "w") as fd:
            fd.write(data)

    @dlp.log
    def get_data(self, id, data, offset=None, length=None):
        with tf.io.gfile.GFile(id, "r") as fd:
            data = fd.read()
        return data

    @dlp.log
    def isfile(self, id):
        return tf.io.gfile.exists(id) and not tf.io.gfile.isdir(id)
