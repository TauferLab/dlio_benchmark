"""Native sample/step metrics, without synthetic file-count assumptions."""

from pathlib import Path
from time import perf_counter

from dlio_benchmark.llama.data import atomic_json


class LlamaStats:
    def __init__(self, args, config):
        self.args = args
        self.rank = args.my_rank
        self.started = None
        self.windows = []
        self.checkpoints = []
        self.evaluations = []
        self.summary = {"native_llama": True, "backend": config.backend,
                        "zero_stage": config.zero_stage, "world_size": args.comm_size,
                        "configuration": config.to_dict(), "metric": {}}

    def start_run(self):
        self.started = perf_counter()

    def end_run(self, runtime, loader):
        from mpi4py import MPI
        duration = self.args.comm.allreduce(perf_counter() - self.started, op=MPI.MAX)
        self.summary["duration_seconds"] = duration
        if runtime is not None:
            self.summary["progress"] = runtime.progress()
            total_samples = sum(len(x["sample_ids"]) for x in self.windows) * self.args.comm_size
            tokens = sum(x["valid_tokens"] for x in self.windows)
            self.summary["metric"] = {"samples_per_second": total_samples / duration,
                                      "valid_tokens_per_second": tokens / duration,
                                      "samples_this_run": total_samples,
                                      "valid_tokens_this_run": tokens}
            self.summary["dataset"] = {
                "global_samples": len(loader.dataset), "local_microbatches_per_epoch": loader.num_batches,
                "dropped_samples_per_epoch": len(loader.dataset) - loader.num_batches * loader.batch_size * loader.world_size,
                "fingerprint": loader.dataset.manifest["fingerprint"],
            }

    def save_data(self):
        folder = Path(self.args.output_folder)
        atomic_json(folder / f"{self.rank}_output.json", {
            "native_llama": True, "windows": self.windows,
            "checkpoints": self.checkpoints, "evaluations": self.evaluations,
        })
        if self.rank == 0:
            self.summary["checkpoints"] = self.checkpoints
            self.summary["evaluations"] = self.evaluations
            atomic_json(folder / "summary.json", self.summary)
            atomic_json(folder / "per_epoch_stats.json", {"native_llama": True, "windows": self.windows})
