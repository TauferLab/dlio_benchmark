"""Native Llama lifecycle selected by workload.model.name=llama in DLIO."""

import fcntl
import hashlib
from importlib.metadata import version
import os
from pathlib import Path
import random
import shutil
import tempfile
from time import perf_counter, sleep

import numpy as np
from omegaconf import OmegaConf

from dlio_benchmark.common.enumerations import (DataLoaderType, DatasetType, FrameworkType, Model,
                                               StorageType)
from dlio_benchmark.framework.framework_factory import FrameworkFactory
from dlio_benchmark.llama.config import LlamaConfig, check_dependencies
from dlio_benchmark.llama.data import (TokenDataset, fingerprint, generate_tokens,
                                      prepare_openhermes)
from dlio_benchmark.utils.llama_stats import LlamaStats
from dlio_benchmark.utils.utility import DFTRACER_ENABLE, DLIOMPI, dft_ai


def local_asset_identity(source, weights=True):
    """Local assets must be immutable for a run; hash outside measured training."""
    path = Path(source)
    if not path.is_dir():
        return None
    hashes = {}
    for file in sorted(path.iterdir()):
        if file.is_file() and (file.suffix in {".json", ".model"} or (weights and file.suffix == ".safetensors")):
            digest = hashlib.sha256()
            with file.open("rb") as stream:
                for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                    digest.update(chunk)
            hashes[file.name] = digest.hexdigest()
    return hashes


def local_dataset_identity(source):
    path = Path(source)
    if not path.is_dir():
        return None
    hashes = {}
    for file in sorted(path.rglob("*")):
        if file.is_file() and file.suffix in {".json", ".arrow", ".parquet"}:
            digest = hashlib.sha256()
            with file.open("rb") as stream:
                for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                    digest.update(chunk)
            hashes[str(file.relative_to(path))] = digest.hexdigest()
    return hashes


class LlamaBenchmark:
    def __init__(self, args, cfg, initialize_tracer=True, finalize_tracer=True):
        values = OmegaConf.to_container(cfg, resolve=True) if OmegaConf.is_config(cfg) else dict(cfg)
        self.config = LlamaConfig.from_mapping(values.get("llama", {}))
        self.args = args
        self.mpi = DLIOMPI.get_instance()
        self.comm, self.rank, self.world_size = self.mpi.comm(), self.mpi.rank(), self.mpi.size()
        args.comm = self.comm
        self._validate_workflow(values)
        # Only explicit workload.reader settings count; ConfigArguments' image defaults
        # (TensorFlow loader, one worker, a one-second delay, pinning) never reach Llama.
        from dlio_benchmark.data_loader.token_loader import resolve_delivery
        self.delivery = resolve_delivery(values.get("reader"), self.config.data_source,
                                         self.config.tokenization, self.config.packing,
                                         self.config.distributed_timeout_seconds)
        args.multiprocessing_context = self.delivery.multiprocessing_context  # worker tracing needs a non-fork start
        Path(args.output_folder).mkdir(parents=True, exist_ok=True)
        args.configure_dlio_logging()
        self.logger = args.logger
        self.tracer = args.configure_dftracer() if initialize_tracer else None
        self.finalize_tracer = finalize_tracer
        self.framework = FrameworkFactory.get_framework(FrameworkType.PYTORCH, args.do_profiling, Model.LLAMA)
        self.stats = LlamaStats(args, self.config)
        self.checkpointing = None
        self.profiler = None
        self.profiler_started = False
        self.train_loader = None
        self.eval_loader = None
        self.prepared_folders = []
        self.closed = False
        self.generation_only = not args.do_train

    def _validate_workflow(self, values):
        args = self.args
        if args.framework != FrameworkType.PYTORCH:
            raise ValueError("Native Llama requires framework=pytorch")
        if args.storage_type not in {StorageType.LOCAL_FS, StorageType.PARALLEL_FS}:
            raise ValueError("Native Llama currently requires shared POSIX storage")
        if not args.keep_files:
            raise ValueError("Native token caches are retained for resume; use keep_files=true and remove caches explicitly")
        if not args.do_train and not args.generate_data:
            raise ValueError("Native Llama requires training or token generation")
        if args.do_train and not args.compute:
            raise ValueError("Native Llama training requires train.compute=true")
        if not args.do_train and (args.do_checkpoint or args.do_eval):
            raise ValueError("Native checkpoint/evaluation requires a live training runtime")
        if args.tensor_parallelism != 1 or args.pipeline_parallelism != 1 or args.zero_stage:
            raise ValueError("Synthetic tensor/pipeline/ZeRO options cannot configure native Llama")
        if min(args.batch_size, args.batch_size_eval, args.epochs) < 1:
            raise ValueError("Batch sizes and epochs must be positive")
        if args.total_training_steps == 0 or args.total_training_steps < -1:
            raise ValueError("train.total_training_steps must be -1 or a positive optimizer-step limit")
        if args.do_checkpoint:
            if args.steps_between_checkpoints == 0 or args.epochs_between_checkpoints < 1:
                raise ValueError("Invalid native checkpoint interval")
        if args.do_eval and (args.eval_after_epoch < 1 or args.epochs_between_evals < 1):
            raise ValueError("Invalid evaluation interval")
        checkpoint = values.get("checkpoint", {})
        if checkpoint.get("recovery_rank_shift", False) or checkpoint.get("mode", "default") != "default":
            raise ValueError("Native checkpoints do not support synthetic rank-shift/subset modes")
        if args.do_eval and self.config.data_source == "openhermes" and not self.config.eval_split:
            raise ValueError("OpenHermes evaluation requires an explicit llama.eval_split")
        if args.do_eval and self.config.data_source == "dlio_tokens" and not self.config.generated_eval_samples:
            raise ValueError("Generated-token evaluation requires generated_eval_samples>0")

    def root_call(self, function):
        result, error = None, None
        if self.rank == 0:
            try:
                result = function()
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
        result, error = self.comm.bcast((result, error), root=0)
        if error:
            raise RuntimeError(f"Llama preparation failed: {error}")
        return result

    def local_call(self, function):
        result, error = None, None
        try:
            result = function()
        except Exception as exc:
            error = f"rank {self.rank}: {type(exc).__name__}: {exc}"
        errors = self.comm.allgather(error)
        if any(errors):
            raise RuntimeError("Llama initialization failed: " + "; ".join(x for x in errors if x))
        return result

    def initialize(self):
        started = perf_counter()
        self.local_call(lambda: check_dependencies(self.config, training=not self.generation_only))
        import torch
        from transformers import AutoConfig, AutoTokenizer

        if not self.generation_only:
            self.local_call(lambda: self._check_gpu(torch))
            if self.config.backend == "deepspeed":
                from dlio_benchmark.llama.runtime import deepspeed_settings
                self.local_call(lambda: deepspeed_settings(self.config, self.args.batch_size, self.world_size))
        self.model_config = self.local_call(lambda: AutoConfig.from_pretrained(
            self.config.model_name_or_path, revision=self.config.model_revision,
            local_files_only=self.config.local_files_only))
        if self.model_config.model_type != "llama":
            raise ValueError("Native Llama requires a Llama model configuration")
        if self.config.sequence_length > self.model_config.max_position_embeddings:
            raise ValueError("sequence_length exceeds model position limit")
        from dataclasses import replace
        commit = getattr(self.model_config, "_commit_hash", None)
        if commit:
            self.config = replace(self.config, model_revision=commit)
        tokenizer_source = self.config.tokenizer_name_or_path or self.config.model_name_or_path
        tokenizer_revision = self.config.tokenizer_revision if self.config.tokenizer_name_or_path else self.config.model_revision
        if not Path(tokenizer_source).is_dir():
            from transformers.utils.hub import cached_file, extract_commit_hash
            token_config_path = self.local_call(lambda: cached_file(
                tokenizer_source, "tokenizer_config.json", revision=tokenizer_revision,
                local_files_only=self.config.local_files_only))
            tokenizer_revision = extract_commit_hash(token_config_path, None) or tokenizer_revision
        self.tokenizer = self.local_call(lambda: AutoTokenizer.from_pretrained(
            tokenizer_source, revision=tokenizer_revision, local_files_only=self.config.local_files_only))
        if self.tokenizer.pad_token_id is None:
            if self.tokenizer.eos_token_id is None:
                raise ValueError("Tokenizer requires a PAD or EOS token")
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if len(self.tokenizer) > self.model_config.vocab_size or self.model_config.vocab_size >= 2 ** 31:
            raise ValueError("Tokenizer vocabulary must fit model embeddings and int32 records")
        source_identity = self.root_call(lambda: {
            "model": local_asset_identity(self.config.model_name_or_path) or
                     getattr(self.model_config, "_commit_hash", None) or self.config.model_revision,
            "tokenizer": local_asset_identity(tokenizer_source, weights=False) or
                         self.tokenizer.init_kwargs.get("_commit_hash") or tokenizer_revision,
        })
        dataset_revision = self.config.dataset_revision
        dataset_assets = None
        if self.config.data_source == "openhermes":
            dataset_assets = self.root_call(lambda: local_dataset_identity(self.config.dataset_name_or_path))
        if self.config.data_source == "openhermes" and not Path(self.config.dataset_name_or_path).is_dir():
            from huggingface_hub import HfApi
            dataset_revision = self.root_call(lambda: HfApi().dataset_info(
                self.config.dataset_name_or_path, revision=self.config.dataset_revision).sha)
        self.data_identity = {"source": self.config.data_source, "assets": source_identity,
                              "sequence_length": self.config.sequence_length,
                              "vocab_size": self.model_config.vocab_size,
                              "pad_token_id": self.tokenizer.pad_token_id, "seed": self.args.seed,
                              "dataset": self.config.dataset_name_or_path,
                              "dataset_revision": dataset_revision,
                              "dataset_assets": dataset_assets,
                              "formatter": self.config.conversation_format, "loss": self.config.loss,
                              "packing": self.config.packing, "remainder": self.config.packing_remainder,
                              "max_samples": self.config.max_samples}
        self.tokenizer_source, self.tokenizer_revision = tokenizer_source, tokenizer_revision
        sources = [self._prepare_source("train", self.config.dataset_split,
                                        self.config.generated_samples, self.args.seed)]
        if self.args.do_eval:
            sources.append(self._prepare_source("valid", self.config.eval_split,
                                                self.config.generated_eval_samples, self.args.seed + 1))
        if self.generation_only:
            self.stats.summary["preparation_seconds"] = perf_counter() - started
            return
        # Preload (RAM mode) happens here, before any GPU allocation and outside timed training.
        preload_started = perf_counter()
        providers = self._build_providers(sources)
        self.train_loader = self._build_loader(providers[0], self.args.batch_size,
                                               self.config.gradient_accumulation_steps, self.args.seed)
        self.train_loader.set_epoch(1)
        self.framework.readers[DatasetType.TRAIN] = self.train_loader
        if self.args.do_eval:
            self.eval_loader = self._build_loader(providers[1], self.args.batch_size_eval, 1, self.args.seed + 1)
            self.framework.readers[DatasetType.VALID] = self.eval_loader
        self.stats.summary["delivery"] = {
            "loader": self.delivery.loader.value, "tokenization": self.config.tokenization,
            "read_threads": self.delivery.read_threads, "prefetch_size": self.delivery.prefetch_size,
            "pin_memory": self.delivery.pin_memory, "iter_time": self.delivery.iter_time,
            "multiprocessing_context": self.delivery.multiprocessing_context,
            "host_sync": self.config.host_sync, "preload_seconds": perf_counter() - preload_started,
            "cache_resident_bytes": sum(getattr(p, "nbytes", 0) for p in providers),
            "train_records": len(self.train_loader.dataset),
            "train_dropped_records_per_epoch": self.train_loader.sampler.dropped_records,
        }
        random.seed(self.args.seed + self.rank)
        np.random.seed(self.args.seed + self.rank)
        torch.manual_seed(self.args.seed + self.rank)
        self.framework.initialize(self.config, self.model_config)
        self.stats.summary["preparation_seconds"] = perf_counter() - started
        packages = ["torch", "transformers", "datasets", "pydftracer"]
        if self.config.backend == "deepspeed":
            packages += ["deepspeed", "accelerate"]
        identity = {
            "backend": self.config.backend, "zero_stage": self.config.zero_stage,
            "world_size": self.world_size, "versions": {p: version(p) for p in packages},
            "assets": source_identity, "train_data": self.train_loader.dataset.manifest["fingerprint"],
            "eval_data": self.eval_loader.dataset.manifest["fingerprint"] if self.eval_loader else None,
            "precision": self.config.precision, "attention": self.config.attention_implementation,
            "activation_checkpointing": self.config.activation_checkpointing,
            "microbatch_size": self.args.batch_size, "accumulation": self.config.gradient_accumulation_steps,
            "learning_rate": self.config.learning_rate, "weight_decay": self.config.weight_decay,
            "max_grad_norm": self.config.max_grad_norm,
            "deepspeed": self.config.deepspeed_config,
        }
        identities = self.comm.allgather(identity)
        if any(value != identity for value in identities):
            raise ValueError("Llama model/data/package identity differs across ranks")
        self.stats.summary["identity"] = identity
        from dlio_benchmark.checkpointing.checkpointing_factory import CheckpointingFactory
        folder = self._storage_path(self.args.checkpoint_folder)
        # Load is permitted even when periodic saving is disabled.
        self.checkpointing = CheckpointingFactory.get_native_llama_mechanism(
            self.framework.runtime, self.config, folder, identity,
            fsync=self.args.checkpoint_fsync,
        )
        if self.config.resume_from:
            self.stats.checkpoints.append(self.checkpointing.load_checkpoint(
                self._storage_path(self.config.resume_from), self.train_loader))
        if self.args.do_profiling:
            from dlio_benchmark.profiler.profiler_factory import ProfilerFactory
            self.profiler = ProfilerFactory().get_profiler(self.args.profiler)

    def _check_gpu(self, torch):
        if not torch.cuda.is_available():
            raise RuntimeError("Native Llama training requires CUDA/ROCm GPUs; generation-only can run on CPU")

    def _storage_path(self, path):
        path = Path(path)
        return path if path.is_absolute() else Path(self.args.storage_root) / path

    def _split_identity(self, name, split, count):
        identity = dict(self.data_identity, split=split, generated_samples=count, logical_split=name)
        if self.config.tokenization == "online":
            identity["tokenization"] = "online"  # offline identities stay byte-identical
        return identity

    def _prepare_source(self, name, split, count, seed):
        """Prepare what the delivery mode needs; synthetic delivery needs no files."""
        needs_cache = (self.generation_only or self.args.generate_data or
                       self.delivery.loader in (DataLoaderType.PYTORCH, DataLoaderType.LOAD_MEM))
        manifest = self._prepare_split(name, split, count) if needs_cache else None
        return {"name": name, "split": split, "count": count, "seed": seed, "manifest": manifest,
                "identity": self._split_identity(name, split, count)}

    def _build_providers(self, sources):
        """Record providers for every split, after one collective memory check in RAM mode."""
        kind, length = self.delivery.loader, self.config.sequence_length
        if kind is DataLoaderType.SYNTHETIC:
            from dlio_benchmark.data_loader.token_loader import GeneratedTokenProvider
            return [GeneratedTokenProvider(s["seed"], s["count"], self.model_config.vocab_size, length,
                                           self.tokenizer.pad_token_id, s["identity"]) for s in sources]
        if self.config.tokenization == "online":
            from functools import partial
            from dlio_benchmark.llama.online import (OnlineTokenProvider, load_hf_tokenizer,
                                                     load_online_manifest)
            loader = partial(load_hf_tokenizer, self.tokenizer_source, self.tokenizer_revision,
                             self.config.local_files_only)
            providers = []
            for source in sources:
                manifest, examples = self.local_call(lambda s=source: load_online_manifest(s["manifest"]))
                providers.append(OnlineTokenProvider(self.config.dataset_name_or_path, source["split"],
                                                     manifest, examples, loader))
            return providers
        datasets = [self.local_call(lambda s=s: TokenDataset(s["manifest"])) for s in sources]
        if kind is not DataLoaderType.LOAD_MEM:
            return datasets
        from mpi4py import MPI
        from dlio_benchmark.data_loader.token_loader import (check_ram_budget, load_token_arrays,
                                                             ram_payload_bytes)
        need = sum(ram_payload_bytes(len(d), length) + 8 * len(d) for d in datasets)  # + sampler permutation
        node = self.comm.Split_type(MPI.COMM_TYPE_SHARED)
        try:
            check_ram_budget(self.comm, need, self.delivery.memory_budget_bytes, node_comm=node)
        finally:
            node.Free()
        return [self.local_call(lambda d=d: load_token_arrays(d)) for d in datasets]

    def _build_loader(self, provider, batch_size, accumulation, seed):
        from dlio_benchmark.data_loader.token_loader import WorkerTracing, build_token_loader
        tracing = WorkerTracing.from_current_config() if self.delivery.read_threads and DFTRACER_ENABLE else None
        return self.local_call(lambda: build_token_loader(
            self.delivery.loader, provider, self.delivery, batch_size=batch_size, rank=self.rank,
            world_size=self.world_size, accumulation=accumulation, seed=seed, tracing=tracing))

    def _close_loaders(self):
        for loader in (self.train_loader, self.eval_loader):
            if loader is not None:
                loader.close()

    def _prepare_split(self, name, split, count):
        identity = self._split_identity(name, split, count)
        root = self._storage_path(self.args.data_folder) / "llama_tokens"
        target = root / fingerprint(identity)
        self.prepared_folders.append(target)

        def prepare():
            root.mkdir(parents=True, exist_ok=True)
            lock_path = root / (target.name + ".lock")
            with lock_path.open("a") as lock:
                deadline = perf_counter() + self.config.distributed_timeout_seconds
                while True:
                    try:
                        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                        break
                    except BlockingIOError:
                        if perf_counter() >= deadline:
                            raise TimeoutError("Timed out waiting for token-cache preparation")
                        sleep(0.1)
                manifest = target / "manifest.json"
                online = self.config.tokenization == "online"
                if manifest.is_file():
                    if online:
                        from dlio_benchmark.llama.online import load_online_manifest
                        cached = load_online_manifest(manifest)[0]
                    else:
                        cached = TokenDataset(manifest).manifest
                    if cached["identity"] != identity:
                        raise ValueError("Cached token identity differs")
                    return str(manifest)
                if self.config.data_source == "dlio_tokens" and not self.args.generate_data:
                    raise FileNotFoundError("No token cache; set workload.workflow.generate_data=true")
                temporary = Path(tempfile.mkdtemp(prefix=".prepare-", dir=root))
                try:
                    if self.config.data_source == "dlio_tokens":
                        generate_tokens(temporary, self.config, self.model_config.vocab_size,
                                        self.tokenizer.pad_token_id, count,
                                        self.args.seed + (1 if name == "valid" else 0), identity)
                    else:
                        # Resolve main before cache generation to make preparation immutable.
                        from dataclasses import replace
                        resolved = replace(self.config, dataset_revision=self.data_identity["dataset_revision"])
                        if online:
                            from dlio_benchmark.llama.online import prepare_online_index
                            prepare_online_index(temporary, resolved, self.tokenizer,
                                                 self.model_config.vocab_size, split, identity)
                        else:
                            prepare_openhermes(temporary, resolved, self.tokenizer,
                                              self.model_config.vocab_size, split, identity)
                    os.rename(temporary, target)
                except BaseException:
                    shutil.rmtree(temporary, ignore_errors=True)
                    raise
                return str(manifest)
        return self.root_call(prepare)

    @dft_ai.dataloader.fetch
    def fetch_window(self, loader):
        if self.config.host_sync == "step":
            # No cross-rank error exchange: a failure here surfaces on the other ranks as a
            # collective timeout (llama.distributed_timeout_seconds). See LLAMA_TODO L09.4.
            return loader.window()
        return self.local_call(loader.window)

    @dft_ai.pipeline.train
    def run(self):
        self.stats.start_run()
        if self.generation_only:
            self.stats.end_run(None, None)
            return
        if self.profiler:
            self.profiler.start()
            self.profiler_started = True
        runtime = self.framework.runtime
        loader = self.train_loader
        first_epoch = loader.epoch + (1 if loader.cursor == loader.num_batches else 0)
        limit = self.args.total_training_steps
        for epoch in dft_ai.pipeline.epoch.iter(range(first_epoch, self.args.epochs + 1), include_iter=False):
            if loader.epoch != epoch:
                loader.set_epoch(epoch)
            dft_ai.update(epoch=epoch)
            # Never dispatch (or prefetch) more windows than a global step limit will consume.
            loader.limit_windows(max(0, limit - runtime.optimizer_steps) if limit > 0 else None)
            while loader.cursor < loader.num_batches:
                if limit > 0 and runtime.optimizer_steps >= limit:
                    break
                started = perf_counter()
                before_bytes = loader.dataset.bytes_read
                batches = self.fetch_window(loader)
                fetch_time = perf_counter() - started
                if self.config.host_sync != "step":
                    runtime.synchronize()
                compute_started = perf_counter()
                result = self.framework.compute(batches, epoch, runtime.optimizer_steps + 1)
                compute_time = perf_counter() - compute_started
                loader.commit_window()
                dft_ai.update(step=runtime.optimizer_steps)
                self.stats.windows.append(dict(result, epoch=epoch,
                    cursor=loader.cursor, fetch_seconds=fetch_time, compute_seconds=compute_time,
                    application_read_bytes=loader.dataset.bytes_read - before_bytes,
                    sample_ids=[sample for batch in batches for sample in batch["sample_ids"]]))
                interval = self.args.steps_between_checkpoints
                if self.args.do_checkpoint and interval > 0 and runtime.optimizer_steps % interval == 0:
                    self.stats.checkpoints.append(self.checkpointing.save_checkpoint(loader))
            loader.close()  # idle prefetch workers between epochs only hold memory
            completed_epoch = loader.cursor == loader.num_batches
            if completed_epoch:
                if self.args.do_eval and epoch >= self.args.eval_after_epoch and (
                        epoch - self.args.eval_after_epoch) % self.args.epochs_between_evals == 0:
                    self._evaluate(epoch)
                if self.args.do_checkpoint and self.args.steps_between_checkpoints < 0 and (
                        epoch >= self.args.checkpoint_after_epoch and
                        (epoch - self.args.checkpoint_after_epoch) % self.args.epochs_between_checkpoints == 0):
                    self.stats.checkpoints.append(self.checkpointing.save_checkpoint(loader, "epoch"))
            if limit > 0 and runtime.optimizer_steps >= limit:
                break
        if self.args.do_checkpoint and self.config.save_final:
            if self.checkpointing.last_saved_step != runtime.optimizer_steps:
                self.stats.checkpoints.append(self.checkpointing.save_checkpoint(loader, "final"))
        self.stats.summary["delivery"]["intentional_delay_seconds"] = loader.delay_seconds
        self.stats.end_run(runtime, loader)

    def _evaluate(self, epoch):
        from dlio_benchmark.checkpointing.llama_checkpointing import capture_rng, restore_rng
        rng = capture_rng()
        loader = self.eval_loader
        loader.set_epoch(epoch)
        def batches():
            while loader.cursor < loader.num_batches:
                yield self.fetch_window(loader)[0]
                loader.commit_window()
        try:
            loss = self.framework.runtime.evaluate(batches())
            self.stats.evaluations.append({"epoch": epoch, "loss": loss,
                                           "samples": loader.num_batches * loader.batch_size * loader.world_size})
        finally:
            loader.close()
            restore_rng(rng)

    def finalize(self):
        if self.closed:
            return
        try:
            self._close_loaders()  # workers stop before the caches, groups and tracer go away
            if self.checkpointing:
                self.checkpointing.finalize()
            if self.profiler_started:
                self.profiler.stop()
                self.profiler_started = False
            self.stats.save_data()
        finally:
            self.framework.finalize()
            if self.finalize_tracer:
                self.args.finalize_dftracer(self.tracer)
            self.closed = True

    def abort(self):
        # Failure cleanup must not introduce more MPI collectives.
        try:
            try:
                self._close_loaders()
            except Exception:  # cleanup keeps the original failure visible
                pass
            if self.profiler_started:
                self.profiler.stop()
            self.stats.summary["failed"] = True
            self.stats.save_data()
        finally:
            self.framework.finalize()
            if self.finalize_tracer:
                self.args.finalize_dftracer(self.tracer)
            self.closed = True
