"""Optional Llama framework; heavyweight initialization is explicit and deferred."""

from dlio_benchmark.common.enumerations import DatasetType, FrameworkType
from dlio_benchmark.framework.framework import Framework, DummyTraceObject
from dlio_benchmark.utils.utility import dft_ai


class LlamaFramework(Framework):
    def __init__(self, profiling=False):
        super().__init__()
        self.profiling = profiling
        self.runtime = None
        self.context = None
        self.readers = {}

    def initialize(self, config, model_config):
        from dlio_benchmark.llama.runtime import DistributedContext, FSDPRuntime, DeepSpeedRuntime
        from dlio_benchmark.utils.utility import DLIOMPI
        self.context = DistributedContext(DLIOMPI.get_instance(), config)
        try:
            error = None
            try:
                self.context.validate()
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
            errors = self.context.mpi.comm().allgather(error)
            if any(errors):
                raise ValueError("Llama rank/device validation failed: " + "; ".join(x for x in errors if x))
            self.context.initialize()
            if config.backend == "fsdp":
                self.runtime = FSDPRuntime(config, self.context, model_config)
            else:
                self.runtime = DeepSpeedRuntime(config, self.context, self.args.batch_size)
        except BaseException:
            self.finalize()
            raise

    @property
    def native_model(self):
        return self.runtime.model if self.runtime is not None else None

    def init_loader(self, format_type=None, epoch=0, data_loader=None):
        for loader in self.readers.values():
            loader.set_epoch(epoch)

    def get_loader(self, dataset_type=DatasetType.TRAIN):
        return self.readers[dataset_type]

    def get_type(self):
        return FrameworkType.PYTORCH

    @dft_ai.compute
    def compute(self, batch, epoch_number=0, step=0, computation_time=0):
        return self.runtime.train_window(batch)

    def model(self, epoch, batch, computation_time=0):
        return self.compute(batch, epoch)

    def start_framework_profiler(self):
        pass

    def stop_framework_profiler(self):
        pass

    def trace_object(self, string, step, r):
        return DummyTraceObject(string, step, r)

    def is_nativeio_available(self):
        return False

    def finalize(self):
        if self.context is not None:
            self.context.close()
