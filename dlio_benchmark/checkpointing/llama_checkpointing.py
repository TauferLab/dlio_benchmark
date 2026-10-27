"""Synchronous live-state checkpoints with collective completion and rank RNG."""

import json
import os
from pathlib import Path
import random
from time import perf_counter

import numpy as np
import torch
import torch.distributed as dist

from dlio_benchmark.llama.data import atomic_json
from dlio_benchmark.utils.utility import dft_ai

CHECKPOINT_VERSION = 1


def capture_rng():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state() if torch.cuda.is_available() else None}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None:
        torch.cuda.set_rng_state(state["cuda"])


def collective_local(function):
    """Propagate local filesystem/validation failures before the next collective."""
    result, error = None, None
    try:
        result = function()
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    errors = [error]
    if dist.is_initialized():
        errors = [None] * dist.get_world_size()
        dist.all_gather_object(errors, error)
    if any(errors):
        raise RuntimeError("Native checkpoint operation failed: " + "; ".join(x for x in errors if x))
    return result


def sync_tree(folder):
    for path in sorted(Path(folder).rglob("*")):
        if path.is_file():
            with path.open("rb") as stream:
                os.fsync(stream.fileno())
    for path in sorted([p for p in Path(folder).rglob("*") if p.is_dir()], reverse=True) + [Path(folder)]:
        descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


class LlamaCheckpointing:
    def __init__(self, runtime, config, folder, identity, fsync=False):
        self.runtime, self.config = runtime, config
        self.folder, self.identity, self.fsync = Path(folder), identity, fsync
        self.rank, self.world_size = runtime.rank, runtime.world_size
        self.last_saved_step = None

    def barrier(self):
        if dist.is_initialized():
            dist.barrier()

    def elapsed(self, started):
        duration = torch.tensor(perf_counter() - started, dtype=torch.float64, device=self.runtime.device)
        if dist.is_initialized():
            dist.all_reduce(duration, op=dist.ReduceOp.MAX)
        return duration.item()

    @dft_ai.checkpoint.capture
    def save_checkpoint(self, loader, label="periodic"):
        if loader.cursor % loader.accumulation:
            raise ValueError("Cannot checkpoint an incomplete gradient accumulation window")
        if label not in {"periodic", "epoch", "final"}:
            raise ValueError("Unknown native checkpoint label")
        step = self.runtime.optimizer_steps
        if dist.is_initialized():
            local = {"progress": self.runtime.progress(), "loader": loader.state_dict()}
            states = [None] * self.world_size
            dist.all_gather_object(states, local)
            if any(state != local for state in states):
                raise ValueError("Ranks disagree on checkpoint progress or consumed data")
        target = self.folder / f"step-{step:012d}-{label}"
        self.runtime.synchronize()
        self.barrier()
        started = perf_counter()
        collective_local(lambda: target.mkdir(parents=True, exist_ok=False) if self.rank == 0 else None)
        self.barrier()
        if self.config.backend == "deepspeed":
            result = self.runtime.model.save_checkpoint(str(target / "engine"), tag="state",
                                                       save_latest=False)
            collective_local(lambda: self._require_save(result))
        else:
            import torch.distributed.checkpoint as dcp
            dcp.save(self.runtime.checkpoint_state(), storage_writer=dcp.FileSystemWriter(
                target / "state", sync_files=self.fsync))
        state = {"rng": capture_rng(), "loader": loader.state_dict(),
                 "progress": self.runtime.progress()}
        if self.config.backend != "deepspeed" and self.runtime.scheduler is not None:
            state["scheduler"] = self.runtime.scheduler.state_dict()
        collective_local(lambda: torch.save(state, target / f"rank-{self.rank:06d}.pt"))
        self.barrier()

        def complete():
            if self.rank != 0:
                return
            if self.fsync:
                sync_tree(target)
            files = {str(path.relative_to(target)): path.stat().st_size
                     for path in target.rglob("*") if path.is_file()}
            manifest = {"version": CHECKPOINT_VERSION, "identity": self.identity,
                        "files": files, "payload_bytes": sum(files.values()),
                        "progress": self.runtime.progress(), "loader": loader.state_dict()}
            atomic_json(target / "complete.json", manifest, fsync=self.fsync)
            if self.fsync:
                descriptor = os.open(target.parent, os.O_RDONLY | os.O_DIRECTORY)
                try:
                    os.fsync(descriptor)
                finally:
                    os.close(descriptor)
        collective_local(complete)
        self.barrier()
        duration = self.elapsed(started)
        self.last_saved_step = step
        metrics = self._metrics(target, duration, "save")
        metrics["path"] = str(target.resolve())
        return metrics

    @staticmethod
    def _require_save(result):
        if result is False:
            raise RuntimeError("DeepSpeed checkpoint save returned failure")

    def _validate_manifest(self, target):
        manifest = json.loads((target / "complete.json").read_text())
        if manifest.get("version") != CHECKPOINT_VERSION:
            raise ValueError("Unsupported native checkpoint schema")
        if manifest.get("identity") != self.identity:
            raise ValueError("Checkpoint backend, world size, versions, model, data or training settings differ")
        for name, expected in manifest["files"].items():
            path = Path(name)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError("Invalid file path in checkpoint manifest")
            if (target / path).stat().st_size != expected:
                raise ValueError(f"Incomplete/truncated checkpoint file: {name}")
        if sum(manifest["files"].values()) != manifest["payload_bytes"]:
            raise ValueError("Checkpoint byte manifest is inconsistent")
        for rank in range(self.world_size):
            if f"rank-{rank:06d}.pt" not in manifest["files"]:
                raise ValueError("Checkpoint is missing a rank's resume state")
        return manifest

    @dft_ai.checkpoint.restart
    def load_checkpoint(self, path, loader):
        target = Path(path)
        self.runtime.synchronize()
        self.barrier()
        started = perf_counter()
        manifest = collective_local(lambda: self._validate_manifest(target))
        # Load and validate cursor metadata before mutating model/optimizer state.
        state = collective_local(lambda: torch.load(target / f"rank-{self.rank:06d}.pt",
                                                    map_location="cpu", weights_only=False))
        def validate_rank_state():
            if state["progress"] != manifest["progress"] or state["loader"] != manifest["loader"]:
                raise ValueError("Rank resume state disagrees with completion manifest")
        collective_local(validate_rank_state)
        collective_local(lambda: loader.load_state_dict(state["loader"]))
        if self.config.backend == "deepspeed":
            loaded, _ = self.runtime.model.load_checkpoint(str(target / "engine"), tag="state")
            collective_local(lambda: self._require_load(loaded))
        else:
            import torch.distributed.checkpoint as dcp
            tensors = self.runtime.checkpoint_state()
            dcp.load(tensors, checkpoint_id=target / "state")
            self.runtime.restore_checkpoint_state(tensors)
            if "scheduler" in state:
                if self.runtime.scheduler is None:
                    raise ValueError("Checkpoint requires a scheduler")
                self.runtime.scheduler.load_state_dict(state["scheduler"])
        self.runtime.restore_progress(state["progress"])
        # Restore after loader/model/optimizer setup so initialization RNG is discarded.
        restore_rng(state["rng"])
        self.barrier()
        duration = self.elapsed(started)
        self.last_saved_step = self.runtime.optimizer_steps
        metrics = self._metrics(target, duration, "load")
        metrics["path"] = str(target.resolve())
        return metrics

    @staticmethod
    def _require_load(loaded):
        if loaded is None:
            raise RuntimeError("DeepSpeed checkpoint load returned no checkpoint")

    def _metrics(self, target, duration, operation):
        # Manifest/files are shared; count unique paths once, rather than sum
        # one global size on every rank. Stored bytes are not physical I/O.
        manifest = json.loads((target / "complete.json").read_text())
        size = manifest["payload_bytes"] + (target / "complete.json").stat().st_size
        return {"operation": operation, "checkpoint_bytes": size,
                "duration_seconds": duration, "throughput_GiB_s": size / (1024 ** 3) / duration,
                "rank_metadata_bytes": (target / f"rank-{self.rank:06d}.pt").stat().st_size}

    def finalize(self):
        # Synchronous adapters have no pending background writes.
        pass
