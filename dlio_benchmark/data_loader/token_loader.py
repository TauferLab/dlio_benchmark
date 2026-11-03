"""Token delivery for native Llama through DLIO's loader modes.

Disk, RAM, generated and online-tokenized records share one sampler and one collator,
so a seed gives the same microbatches in every mode, for any worker count or prefetch
depth. Only where the records come from differs:

* ``pytorch``   : records read from the prepared token files (optionally by workers).
* ``load_mem``  : the whole split read once into int32 arrays, then served from memory.
* ``synthetic`` : records generated from ``SeedSequence([seed, sample_id])``, no files.
* online       : raw text tokenized inside the workers (see ``llama/online.py``).

The loader keeps the interface the native runner and the v1 checkpoint code already
use (``epoch``, ``cursor``, ``set_epoch``, ``window``, ``commit_window``, ``state_dict``,
``load_state_dict``, ``dataset``). The cursor counts committed microbatches only;
prefetching never advances it, and an iterator is rebuilt from the committed cursor.

DALI and FFCV are tracked in LLAMA_TODO Q03. They plug in through
``register_token_loader`` and must take their sample order from ``TokenSampler``
rather than shuffling themselves.
"""

from dataclasses import dataclass
import math
import os
import pickle
import time

import numpy as np
from torch.utils.data import Dataset

from dlio_benchmark.common.constants import MODULE_DATA_LOADER
from dlio_benchmark.common.enumerations import DataLoaderType
from dlio_benchmark.llama.data import fingerprint
from dlio_benchmark.utils.utility import Profile

dlp = Profile(MODULE_DATA_LOADER)

SAMPLER_VERSION = 1
SUPPORTED_LOADERS = (DataLoaderType.PYTORCH, DataLoaderType.LOAD_MEM, DataLoaderType.SYNTHETIC)


# ---------------------------------------------------------------- sampling and collation

class TokenSampler:
    """Epoch permutation, rank/microbatch reshape and tail drop, shared by every mode.

    Reproduces ``TokenLoader``: ``SeedSequence([seed, epoch])`` permutes all records;
    ``floor(N / (B * W * A)) * B * W * A`` of them are used so every rank runs the same
    number of complete accumulation windows. This is the only owner of sharding.
    """

    def __init__(self, num_records, batch_size, rank, world_size, accumulation=1, seed=0):
        self.num_records, self.batch_size = num_records, batch_size
        self.rank, self.world_size = rank, world_size
        self.accumulation, self.seed = accumulation, seed
        self.num_batches = num_records // (batch_size * world_size * accumulation) * accumulation
        if not self.num_batches:
            raise ValueError("Dataset is too small for one global accumulation window")

    @property
    def dropped_records(self):
        return self.num_records - self.num_batches * self.batch_size * self.world_size

    def order(self, epoch):
        """Record indices of this rank, shape ``[num_batches, batch_size]``."""
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, epoch]))
        order = rng.permutation(self.num_records)
        usable = self.num_batches * self.batch_size * self.world_size
        return order[:usable].reshape(self.num_batches, self.world_size, self.batch_size)[:, self.rank]


def collate_tokens(samples):
    """Stack records into int64 model inputs; sample IDs stay metadata, not model inputs."""
    import torch
    batch = {key: torch.as_tensor(np.stack([x[key] for x in samples]), dtype=torch.long)
             for key in ("input_ids", "attention_mask", "labels")}
    batch["sample_ids"] = [x["sample_id"] for x in samples]
    return batch


# ---------------------------------------------------------------- record providers

class RamTokenProvider:
    """Whole-split cache: int32 IDs and labels plus the original lengths.

    Masks come from the stored length, never from ``id != pad``: EOS may equal PAD.
    """

    bytes_read = 0  # steady-state reads from storage are zero by construction

    def __init__(self, ids, labels, lengths, manifest):
        self.ids, self.labels, self.lengths, self.manifest = ids, labels, lengths, manifest
        self.length = ids.shape[1]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        mask = np.arange(self.length) < int(self.lengths[index])
        return {"input_ids": self.ids[index].astype(np.int64), "attention_mask": mask.astype(np.int64),
                "labels": self.labels[index].astype(np.int64), "sample_id": int(index)}

    @property
    def nbytes(self):
        return self.ids.nbytes + self.labels.nbytes + self.lengths.nbytes


def ram_payload_bytes(num_records, sequence_length):
    """Lower bound for one cached split: ``N * (2L + 1) * 4``, before sampler and batches."""
    return num_records * (2 * sequence_length + 1) * 4


def load_token_arrays(dataset):
    """Read a verified ``TokenDataset`` into contiguous arrays with the reader's checks."""
    length, vocab = dataset.length, dataset.manifest["vocab_size"]
    total = len(dataset)
    ids = np.empty((total, length), dtype=np.int32)
    labels = np.empty((total, length), dtype=np.int32)
    lengths = np.empty(total, dtype=np.int32)
    start = 0
    for path in dataset.paths:
        records = np.fromfile(path, dtype="<i4").reshape(-1, 2 * length + 1)
        stop = start + len(records)
        sizes, file_ids, file_labels = records[:, 0], records[:, 1:1 + length], records[:, 1 + length:]
        if np.any((sizes < 2) | (sizes > length)):
            raise ValueError("Invalid token record length")
        if np.any((file_ids < 0) | (file_ids >= vocab)):
            raise ValueError("Token ID outside model vocabulary")
        mask = np.arange(length)[None, :] < sizes[:, None]
        if np.any(file_labels[~mask] != -100) or np.any((file_labels != -100) & (file_labels != file_ids)):
            raise ValueError("Invalid causal labels in token record")
        if not np.all((file_labels[:, 1:] != -100).any(axis=1)):
            raise ValueError("Token record has no valid prediction target")
        ids[start:stop], labels[start:stop], lengths[start:stop] = file_ids, file_labels, sizes
        start = stop
    if start != total:
        raise ValueError("Token files hold a different record count than the manifest")
    return RamTokenProvider(ids, labels, lengths, dataset.manifest)


def available_memory_bytes():
    """Usable bytes on this node: MemAvailable, capped by the cgroup limit if one is set."""
    limits = []
    try:
        with open("/proc/meminfo") as stream:
            for line in stream:
                if line.startswith("MemAvailable:"):
                    limits.append(int(line.split()[1]) * 1024)
    except OSError:
        pass
    for limit_file, usage_file in (("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory.current"),
                                   ("/sys/fs/cgroup/memory/memory.limit_in_bytes",
                                    "/sys/fs/cgroup/memory/memory.usage_in_bytes")):
        try:
            limit = open(limit_file).read().strip()
            if limit.isdigit() and int(limit) < 2 ** 60:
                limits.append(int(limit) - int(open(usage_file).read().strip()))
        except (OSError, ValueError):
            pass
    return min(limits) if limits else None


def check_ram_budget(comm, need_bytes, budget_bytes, node_comm=None, available_bytes=None, headroom=0.9):
    """Fail on every rank, before any allocation, if the caches cannot fit.

    Each rank holds a full replica, so the node-wide demand is the sum over the ranks
    that share the node. ``node_comm`` groups those ranks; ``available_bytes`` can be
    supplied for tests.
    """
    problems = []
    if budget_bytes is None or budget_bytes <= 0:
        problems.append("reader.memory_budget_bytes must be a positive per-rank limit for load_mem")
    elif need_bytes > budget_bytes:
        problems.append(f"needs {need_bytes} B but reader.memory_budget_bytes is {budget_bytes} B")
    node_need = node_comm.allreduce(need_bytes) if node_comm is not None else need_bytes
    available = available_memory_bytes() if available_bytes is None else available_bytes
    if available is not None and node_need > headroom * available:
        problems.append(f"node demand {node_need} B exceeds {headroom:.0%} of the {available} B available")
    reported = comm.allgather(problems)
    if any(reported):
        detail = "; ".join(f"rank {rank}: {', '.join(p)}" for rank, p in enumerate(reported) if p)
        raise ValueError("load_mem cannot hold the token cache: " + detail)


class GeneratedTokenProvider:
    """Fileless records equal to ``generate_tokens`` output for the same seed and ID."""

    bytes_read = 0
    GENERATOR = "numpy_default_rng_seedsequence_v1"

    def __init__(self, seed, count, vocab_size, sequence_length, pad_token_id, identity=None):
        if sequence_length < 2:
            raise ValueError("At least two tokens are needed for causal prediction")
        self.seed, self.count, self.vocab = seed, count, vocab_size
        self.length, self.pad = sequence_length, pad_token_id
        logical = {"generator": self.GENERATOR, "numpy": np.__version__, "seed": seed,
                   "num_records": count, "vocab_size": vocab_size, "sequence_length": sequence_length,
                   "pad_token_id": pad_token_id, "loss": "full", "identity": identity}
        self.manifest = {"fingerprint": fingerprint(logical), "num_records": count,
                         "sequence_length": sequence_length, "vocab_size": vocab_size,
                         "pad_token_id": pad_token_id, "logical_identity": logical}

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        if not 0 <= index < self.count:
            raise IndexError(index)
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, int(index)]))
        ids = rng.integers(0, self.vocab, self.length)
        return {"input_ids": ids.astype(np.int64), "attention_mask": np.ones(self.length, dtype=np.int64),
                "labels": ids.astype(np.int64), "sample_id": int(index)}


# ---------------------------------------------------------------- torch datasets

class _TokenSource(Dataset):
    """Map-style wrapper: traces ``__getitem__`` and counts completed reads across workers.

    Reads made inside DataLoader workers are invisible to the parent's Python objects, so
    each process adds to its own slot of a shared int64 tensor. ``bytes_read`` is the sum
    at the time of the call: completed reads including read-ahead that was never consumed.
    """

    kind = "token"

    def __init__(self, provider, workers=0):
        import torch
        self.provider = provider
        self._reads = torch.zeros(max(workers, 0) + 1, dtype=torch.int64).share_memory_()

    def __len__(self):
        return len(self.provider)

    @property
    def manifest(self):
        return self.provider.manifest

    @property
    def length(self):
        return self.provider.length

    @property
    def bytes_read(self):
        return int(self._reads.sum())

    def _fetch(self, index):
        import torch
        before = self.provider.bytes_read
        record = self.provider[index]
        delta = self.provider.bytes_read - before
        if delta:
            info = torch.utils.data.get_worker_info()
            self._reads[(info.id + 1) if info is not None else 0] += delta
        return record


class DiskTokenDataset(_TokenSource):
    kind = "pytorch"

    @dlp.log
    def __getitem__(self, index):
        return self._fetch(index)


class RamTokenDataset(_TokenSource):
    kind = "load_mem"

    @dlp.log
    def __getitem__(self, index):
        return self._fetch(index)


class GeneratedTokenDataset(_TokenSource):
    kind = "synthetic"

    @dlp.log
    def __getitem__(self, index):
        return self._fetch(index)


class OnlineTokenDataset(_TokenSource):
    kind = "online"

    @dlp.log
    def __getitem__(self, index):
        return self._fetch(index)


class WorkerTracing:
    """Picklable DataLoader ``worker_init_fn``: one DFTracer log per worker, flushed on exit.

    Runs only inside a spawned worker, where restoring the pickled ``ConfigArguments``
    cannot disturb the training process. Spawn workers exit through multiprocessing,
    which skips atexit callbacks, so the flush is registered with ``Finalize``.
    """

    _logger = None  # keeps the worker's logger alive for the life of the process

    def __init__(self, serial_args):
        self.serial_args = serial_args

    @classmethod
    def from_current_config(cls):
        """Snapshot of the shared config for workers, without any MPI object.

        ``LlamaBenchmark`` keeps its communicator on the shared config as ``comm``. Unpickling it
        in a spawned worker imports mpi4py and calls MPI_Init inside the parent's Slurm step; the
        PMI socket is not the worker's to use, so MVAPICH fails and the worker segfaults
        (smoke job 472287). Workers only need the plain settings, so ``comm`` is left out while
        pickling and put back for the parent.
        """
        from dlio_benchmark.utils.config import ConfigArguments
        args = ConfigArguments.get_instance()
        absent = object()
        communicator = args.__dict__.pop("comm", absent)
        try:
            blob = pickle.dumps(args)
        finally:
            if communicator is not absent:
                args.comm = communicator
        if b"mpi4py" in blob:  # fail here, in the parent, not as a worker crash under Slurm
            raise RuntimeError("The worker tracing spec embeds an MPI object; workers must not unpickle MPI state")
        return cls(blob)

    def __call__(self, worker_id):
        if self.serial_args is None:
            return
        from multiprocessing.util import Finalize
        from dlio_benchmark.utils.config import ConfigArguments
        pickle.loads(self.serial_args)
        args = ConfigArguments.get_instance()
        args.configure_dlio_logging(is_child=True)
        logger = args.configure_dftracer(is_child=True, use_pid=True)
        if logger:
            WorkerTracing._logger = logger
            Finalize(None, logger.finalize, exitpriority=10)


# ---------------------------------------------------------------- loader

class TokenDataLoader:
    """Windows of ``accumulation`` microbatches with a committed cursor."""

    def __init__(self, dataset, batch_size, rank, world_size, accumulation=1, seed=0, *,
                 num_workers=0, prefetch_factor=None, pin_memory=False, multiprocessing_context="spawn",
                 iter_time=0.0, timeout=0, worker_init=None):
        if num_workers and multiprocessing_context != "spawn":
            raise ValueError("Token workers require multiprocessing_context=spawn: the iterator "
                             "can be created after CUDA/NCCL initialization, where fork is unsafe")
        self.dataset = dataset
        self.sampler = TokenSampler(len(dataset), batch_size, rank, world_size, accumulation, seed)
        self.batch_size, self.rank, self.world_size = batch_size, rank, world_size
        self.accumulation, self.seed = accumulation, seed
        self.num_batches = self.sampler.num_batches
        self.num_workers, self.prefetch_factor = num_workers, prefetch_factor
        self.pin_memory, self.multiprocessing_context = pin_memory, multiprocessing_context
        self.iter_time, self.timeout, self.worker_init = iter_time, timeout, worker_init
        self.epoch = 0
        self.cursor = 0
        self.order = None
        self.delay_seconds = 0.0
        self._iterator = None
        self._max_windows = None

    # -- epoch and cursor
    def set_epoch(self, epoch, cursor=0):
        if cursor < 0 or cursor > self.num_batches or cursor % self.accumulation:
            raise ValueError("Resume cursor must be at a complete accumulation boundary")
        self._release_iterator()
        self.epoch, self.cursor = epoch, cursor
        self.order = self.sampler.order(epoch)

    def limit_windows(self, windows):
        """Dispatch at most ``windows`` more windows (a step limit), so a short run does
        not prefetch an epoch it will not consume. ``None`` removes the bound."""
        self._release_iterator()
        self._max_windows = windows

    def commit_window(self):
        self.cursor += self.accumulation

    # -- reading
    def _make_iterator(self):
        import torch
        from torch.utils.data import DataLoader
        end = self.num_batches
        if self._max_windows is not None:
            end = min(end, self.cursor + self._max_windows * self.accumulation)
        batches = [[int(i) for i in row] for row in self.order[self.cursor:end]]
        # A private generator keeps iterator creation (the DataLoader base seed) from
        # advancing the global RNG that model dropout and checkpoints depend on.
        generator = torch.Generator().manual_seed((self.seed * 1_000_003 + self.epoch) % (2 ** 63))
        options = dict(batch_sampler=batches, collate_fn=collate_tokens, num_workers=self.num_workers,
                       pin_memory=self.pin_memory and torch.cuda.is_available(), generator=generator)
        if self.num_workers:
            options.update(multiprocessing_context=self.multiprocessing_context, persistent_workers=False,
                           prefetch_factor=self.prefetch_factor or 2, timeout=self.timeout,
                           worker_init_fn=self.worker_init)
        self._iterator = iter(DataLoader(self.dataset, **options))

    def window(self):
        if self.order is None:
            raise RuntimeError("Call set_epoch before reading tokens")
        if self.cursor >= self.num_batches:
            raise RuntimeError("The epoch has no windows left")
        if self._iterator is None:
            self._make_iterator()
        batches = []
        for _ in range(self.accumulation):
            if self.iter_time:
                time.sleep(self.iter_time)
                self.delay_seconds += self.iter_time
            try:
                batches.append(next(self._iterator))
            except StopIteration:
                raise RuntimeError("Scheduled windows are exhausted; call limit_windows(None) "
                                   "or set_epoch before reading more") from None
        return batches

    # -- lifecycle
    def _release_iterator(self):
        iterator, self._iterator = self._iterator, None
        if iterator is not None and getattr(iterator, "_workers", None):
            # DataLoader has no public close(); this private call is the one place that
            # depends on the pinned torch version and is covered by a test.
            iterator._shutdown_workers()

    def close(self):
        self._release_iterator()

    @property
    def bytes_read(self):
        return self.dataset.bytes_read

    # -- checkpoint state (same keys as the v1 TokenLoader)
    def state_dict(self):
        return {"epoch": self.epoch, "cursor": self.cursor, "seed": self.seed,
                "dataset_fingerprint": self.dataset.manifest["fingerprint"],
                "batch_size": self.batch_size, "accumulation": self.accumulation,
                "world_size": self.world_size}

    def load_state_dict(self, state):
        expected = self.state_dict()
        for key in ("seed", "dataset_fingerprint", "batch_size", "accumulation", "world_size"):
            if state[key] != expected[key]:
                raise ValueError(f"Checkpoint loader setting mismatch: {key}")
        self.set_epoch(state["epoch"], state["cursor"])


# ---------------------------------------------------------------- reader settings

@dataclass(frozen=True)
class DeliverySettings:
    loader: DataLoaderType = DataLoaderType.PYTORCH
    read_threads: int = 0
    prefetch_size: int = 2
    pin_memory: bool = False
    iter_time: float = 0.0
    multiprocessing_context: str = "spawn"
    memory_budget_bytes: int = 0
    timeout_seconds: int = 300

    @property
    def prefetch_factor(self):
        if not self.read_threads:
            return None
        return max(1, math.ceil(self.prefetch_size / self.read_threads)) if self.prefetch_size else 2


def resolve_delivery(reader, data_source, tokenization="offline", packing=False, timeout_seconds=300):
    """Explicit ``workload.reader`` settings first; DLIO's image defaults must not leak in.

    Omitted native settings mean: pytorch loader, no workers, no delay, no pinning.
    ConfigArguments' own defaults (TensorFlow loader, one worker, one second per batch,
    pinned memory) would change today's native behavior, so they are never consulted.
    """
    reader = dict(reader or {})
    try:
        loader = DataLoaderType(reader.get("data_loader", "pytorch"))
    except ValueError:
        raise ValueError(f"Unknown reader.data_loader {reader.get('data_loader')!r}") from None
    if loader not in SUPPORTED_LOADERS:
        raise ValueError(f"reader.data_loader={loader.value} is not supported for Llama tokens; use one of "
                         f"{', '.join(x.value for x in SUPPORTED_LOADERS)} (DALI/FFCV: LLAMA_TODO Q03)")
    threads = reader.get("read_threads", 0)
    if type(threads) is not int or threads < 0:
        raise ValueError("reader.read_threads must be a nonnegative integer")
    prefetch = reader.get("prefetch_size", 2)
    if type(prefetch) is not int or prefetch < 0:
        raise ValueError("reader.prefetch_size must be a nonnegative integer")
    iter_time = float(reader.get("iter_time", 0.0))
    if not math.isfinite(iter_time) or iter_time < 0:
        raise ValueError("reader.iter_time must be a finite non-negative number of seconds")
    pin = reader.get("pin_memory", False)
    if type(pin) is not bool:
        raise ValueError("reader.pin_memory must be true or false")
    context = reader.get("multiprocessing_context", "spawn")
    if threads and context != "spawn":
        raise ValueError("reader.multiprocessing_context must be spawn for token workers; fork is unsafe "
                         "after CUDA/NCCL initialization")
    budget = reader.get("memory_budget_bytes", 0)
    if type(budget) is not int or budget < 0:
        raise ValueError("reader.memory_budget_bytes must be a nonnegative integer")
    if loader is DataLoaderType.LOAD_MEM:
        if threads:
            raise ValueError("load_mem keeps one cache per rank; read_threads must be 0 "
                             "(workers would duplicate it)")
        if budget <= 0:
            raise ValueError("load_mem needs a positive per-rank reader.memory_budget_bytes")
    if loader is DataLoaderType.SYNTHETIC and data_source == "openhermes":
        raise ValueError("synthetic delivery generates random token IDs and cannot represent OpenHermes text")
    if tokenization == "online":
        if data_source != "openhermes":
            raise ValueError("llama.tokenization=online needs llama.data_source=openhermes")
        if loader is not DataLoaderType.PYTORCH:
            raise ValueError("llama.tokenization=online runs in the pytorch loader (tokenization is the workers' job)")
        if packing:
            raise ValueError("online tokenization needs llama.packing=false: a packed record depends on "
                             "every earlier example, so it cannot be built from its index alone")
    return DeliverySettings(loader, threads, prefetch, pin, iter_time, context, budget, timeout_seconds)


# ---------------------------------------------------------------- factory hooks

_TOKEN_LOADERS = {}


def register_token_loader(loader_type, builder):
    """Add a delivery mode. ``builder(provider, settings, **sampling)`` returns a loader
    exposing the TokenDataLoader interface and taking its order from ``TokenSampler``."""
    _TOKEN_LOADERS[loader_type] = builder


def _builder(source_class):
    def build(provider, settings, *, batch_size, rank, world_size, accumulation, seed, tracing=None):
        dataset = source_class(provider, settings.read_threads)
        if tracing is None and settings.read_threads:
            tracing = WorkerTracing(None)
        return TokenDataLoader(dataset, batch_size, rank, world_size, accumulation, seed,
                               num_workers=settings.read_threads, prefetch_factor=settings.prefetch_factor,
                               pin_memory=settings.pin_memory, iter_time=settings.iter_time,
                               multiprocessing_context=settings.multiprocessing_context,
                               timeout=settings.timeout_seconds if settings.read_threads else 0,
                               worker_init=tracing)
    return build


def _pytorch_builder(provider, settings, **sampling):
    # Online tokenization runs in the pytorch loader; it gets its own event name for the trace.
    source = OnlineTokenDataset if getattr(provider, "online", False) else DiskTokenDataset
    return _builder(source)(provider, settings, **sampling)


register_token_loader(DataLoaderType.PYTORCH, _pytorch_builder)
register_token_loader(DataLoaderType.LOAD_MEM, _builder(RamTokenDataset))
register_token_loader(DataLoaderType.SYNTHETIC, _builder(GeneratedTokenDataset))


def build_token_loader(loader_type, provider, settings, **sampling):
    if loader_type not in _TOKEN_LOADERS:
        raise ValueError(f"No token loader is registered for {loader_type}")
    if loader_type is DataLoaderType.LOAD_MEM and settings.read_threads:
        raise ValueError("load_mem requires read_threads=0")
    return _TOKEN_LOADERS[loader_type](provider, settings, **sampling)
