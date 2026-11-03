"""Real, versioned token records shared by generated and OpenHermes inputs."""

import bisect
import hashlib
import json
import os
from pathlib import Path

import numpy as np

FORMAT_VERSION = 1
FORMATTER_VERSION = "role_lines_v1"


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def atomic_json(path, value, fsync=False):
    path = Path(path)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, indent=2)
        stream.write("\n")
        stream.flush()
        if fsync:
            os.fsync(stream.fileno())
    os.replace(temporary, path)
    if fsync:
        descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


class TokenWriter:
    """One record: length, L int32 tokens, L int32 labels; little endian."""

    def __init__(self, folder, sequence_length, vocab_size, pad_token_id,
                 samples_per_file, identity):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.length = sequence_length
        self.vocab = vocab_size
        self.pad = pad_token_id
        self.samples_per_file = samples_per_file
        self.identity = identity
        self.files = []
        self.stream = None
        self.count = 0
        self.digest = None

    def append(self, input_ids, labels=None):
        ids = np.asarray(input_ids, dtype=np.int64)
        targets = ids if labels is None else np.asarray(labels, dtype=np.int64)
        if ids.ndim != 1 or len(ids) < 2 or len(ids) > self.length or ids.shape != targets.shape:
            raise ValueError("Token record must have 2..sequence_length aligned tokens and labels")
        if np.any((ids < 0) | (ids >= self.vocab)):
            raise ValueError("Token ID outside model vocabulary")
        if np.any((targets != -100) & (targets != ids)) or not np.any(targets[1:] != -100):
            raise ValueError("Token record needs valid, unshifted causal targets")
        if self.stream is None or self.files[-1]["num_records"] == self.samples_per_file:
            self._close_file()
            name = f"tokens-{len(self.files):06d}.bin"
            self.stream = (self.folder / name).open("xb")
            self.digest = hashlib.sha256()
            self.files.append({"path": name, "num_records": 0, "size_bytes": 0})
        record = np.empty(2 * self.length + 1, dtype="<i4")
        record[0] = len(ids)
        record[1:1 + self.length] = self.pad
        record[1:1 + len(ids)] = ids
        record[1 + self.length:] = -100
        record[1 + self.length:1 + self.length + len(ids)] = targets
        raw = record.tobytes()
        self.stream.write(raw)
        self.digest.update(raw)
        self.files[-1]["num_records"] += 1
        self.files[-1]["size_bytes"] += len(raw)
        self.count += 1

    def _close_file(self):
        if self.stream is not None:
            self.stream.close()
            self.files[-1]["sha256"] = self.digest.hexdigest()
            self.stream = None

    def finish(self):
        self._close_file()
        if not self.count:
            raise ValueError("No usable token records were produced")
        manifest = {
            "version": FORMAT_VERSION, "dtype": "<i4", "sequence_length": self.length,
            "vocab_size": self.vocab, "pad_token_id": self.pad,
            "num_records": self.count, "files": self.files, "identity": self.identity,
        }
        manifest["fingerprint"] = fingerprint(manifest)
        atomic_json(self.folder / "manifest.json", manifest)
        return self.folder / "manifest.json"


class TokenDataset:
    def __init__(self, manifest_path):
        self.manifest_path = Path(manifest_path)
        self.manifest = json.loads(self.manifest_path.read_text())
        manifest = self.manifest
        signature = {k: v for k, v in manifest.items() if k != "fingerprint"}
        if manifest.get("fingerprint") != fingerprint(signature):
            raise ValueError("Token manifest fingerprint mismatch")
        if manifest["version"] != FORMAT_VERSION or manifest["dtype"] != "<i4":
            raise ValueError("Unsupported token record format")
        self.length = manifest["sequence_length"]
        self.record_bytes = (2 * self.length + 1) * 4
        self.ends = []
        self.paths = []
        total = 0
        for info in manifest["files"]:
            name = Path(info["path"])
            if name.is_absolute() or ".." in name.parts:
                raise ValueError("Token file path must be relative to its manifest")
            path = self.manifest_path.parent / name
            expected = info["num_records"] * self.record_bytes
            if expected != info["size_bytes"] or path.stat().st_size != expected:
                raise ValueError(f"Truncated or inconsistent token file: {path}")
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            if digest.hexdigest() != info["sha256"]:
                raise ValueError(f"Token file checksum mismatch: {path}")
            total += info["num_records"]
            self.paths.append(path)
            self.ends.append(total)
        if not total or total != manifest["num_records"]:
            raise ValueError("Token manifest sample count is inconsistent")
        self.bytes_read = 0

    def __len__(self):
        return self.manifest["num_records"]

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        file_index = bisect.bisect_right(self.ends, index)
        start = self.ends[file_index - 1] if file_index else 0
        with self.paths[file_index].open("rb") as stream:
            stream.seek((index - start) * self.record_bytes)
            raw = stream.read(self.record_bytes)
        if len(raw) != self.record_bytes:
            raise ValueError("Truncated token record")
        self.bytes_read += len(raw)
        record = np.frombuffer(raw, dtype="<i4")
        size = int(record[0])
        ids = record[1:1 + self.length].astype(np.int64)
        labels = record[1 + self.length:].astype(np.int64)
        if not 2 <= size <= self.length:
            raise ValueError("Invalid token record length")
        if np.any((ids < 0) | (ids >= self.manifest["vocab_size"])):
            raise ValueError("Token ID outside model vocabulary")
        mask = np.arange(self.length) < size
        if np.any(labels[~mask] != -100) or np.any((labels != -100) & (labels != ids)):
            raise ValueError("Invalid causal labels in token record")
        if not np.any(labels[1:] != -100):
            raise ValueError("Token record has no valid prediction target")
        return {"input_ids": ids, "attention_mask": mask.astype(np.int64),
                "labels": labels, "sample_id": index}


def generate_tokens(folder, config, vocab_size, pad_token_id, count, seed, identity):
    writer = TokenWriter(folder, config.sequence_length, vocab_size, pad_token_id,
                         config.samples_per_file, identity)
    for index in range(count):
        # Sample identity, rather than writer rank/chunk size, determines bytes.
        rng = np.random.default_rng(np.random.SeedSequence([seed, index]))
        writer.append(rng.integers(0, vocab_size, config.sequence_length))
    return writer.finish()


def conversation_tokens(example, tokenizer, loss):
    """A versioned text format for the base model, without a chat template."""
    roles = {"human": "User", "user": "User", "gpt": "Assistant",
             "assistant": "Assistant", "system": "System"}
    turns = example.get("conversations")
    if not isinstance(turns, list) or not turns:
        raise ValueError("OpenHermes record requires nonempty conversations")
    ids, labels = [], []
    if tokenizer.bos_token_id is not None:
        ids.append(tokenizer.bos_token_id)
        labels.append(tokenizer.bos_token_id if loss == "full" else -100)
    system = example.get("system_prompt")
    if isinstance(system, str) and system.strip():
        segment = tokenizer.encode(f"System:\n{system}\n", add_special_tokens=False)
        ids.extend(segment)
        labels.extend(segment if loss == "full" else [-100] * len(segment))
    for turn in turns:
        role = roles.get(turn.get("from"))
        value = turn.get("value")
        if role is None or not isinstance(value, str):
            raise ValueError("Unknown conversation role or non-string value")
        prefix = tokenizer.encode(f"{role}:\n", add_special_tokens=False)
        content = tokenizer.encode(value + "\n", add_special_tokens=False)
        ids.extend(prefix)
        labels.extend(prefix if loss == "full" else [-100] * len(prefix))
        ids.extend(content)
        labels.extend(content if loss == "full" or role == "Assistant" else [-100] * len(content))
    if tokenizer.eos_token_id is None:
        raise ValueError("A causal Llama tokenizer must provide EOS")
    ids.append(tokenizer.eos_token_id)
    last_assistant = roles.get(turns[-1].get("from")) == "Assistant"
    labels.append(tokenizer.eos_token_id if loss == "full" or last_assistant else -100)
    return ids, labels


def prepare_openhermes(folder, config, tokenizer, vocab_size, split, identity):
    from datasets import load_dataset, load_from_disk

    source = Path(config.dataset_name_or_path)
    if source.is_dir():
        loaded = load_from_disk(str(source))
        if hasattr(loaded, "keys"):
            dataset = loaded[split]
        elif split == "train":
            dataset = loaded
        else:
            raise ValueError("A local Dataset has only train; use a DatasetDict for evaluation splits")
    else:
        if config.local_files_only:
            raise ValueError("local_files_only OpenHermes input must be a save_to_disk directory")
        dataset = load_dataset(config.dataset_name_or_path, revision=config.dataset_revision,
                               split=split, streaming=True)
    writer = TokenWriter(folder, config.sequence_length, vocab_size, tokenizer.pad_token_id,
                         config.samples_per_file, identity)
    pending_ids, pending_labels = [], []
    for index, example in enumerate(dataset):
        if config.max_samples and index >= config.max_samples:
            break
        ids, labels = conversation_tokens(example, tokenizer, config.loss)
        if config.packing:
            pending_ids.extend(ids)
            pending_labels.extend(labels)
            while len(pending_ids) >= config.sequence_length:
                chunk = pending_ids[:config.sequence_length]
                targets = pending_labels[:config.sequence_length]
                if any(x != -100 for x in targets[1:]):
                    writer.append(chunk, targets)
                del pending_ids[:config.sequence_length]
                del pending_labels[:config.sequence_length]
        else:
            ids, labels = ids[:config.sequence_length], labels[:config.sequence_length]
            if len(ids) >= 2 and any(x != -100 for x in labels[1:]):
                writer.append(ids, labels)
    if config.packing and config.packing_remainder == "pad" and len(pending_ids) >= 2:
        if any(x != -100 for x in pending_labels[1:]):
            writer.append(pending_ids, pending_labels)
    return writer.finish()


class TokenLoader:
    """Single-owner sampling with a cursor advanced only by the runner.

    Uses synchronous file reads, so no worker prefetch/RNG has to be replayed.
    Remainders are dropped to complete global accumulation windows.

    This is the v1 reference implementation. The native runner now builds its
    loaders from ``dlio_benchmark.data_loader.token_loader``; this class stays as
    the independent oracle that the equivalence tests compare those loaders with.
    """

    def __init__(self, dataset, batch_size, rank, world_size, accumulation=1, seed=0):
        self.dataset = dataset
        self.batch_size = batch_size
        self.rank = rank
        self.world_size = world_size
        self.accumulation = accumulation
        self.seed = seed
        self.epoch = 0
        self.cursor = 0
        self.order = None
        self.num_batches = len(dataset) // (batch_size * world_size * accumulation) * accumulation
        if not self.num_batches:
            raise ValueError("Dataset is too small for one global accumulation window")

    def set_epoch(self, epoch, cursor=0):
        if cursor < 0 or cursor > self.num_batches or cursor % self.accumulation:
            raise ValueError("Resume cursor must be at a complete accumulation boundary")
        self.epoch, self.cursor = epoch, cursor
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, epoch]))
        order = rng.permutation(len(self.dataset))
        usable = self.num_batches * self.batch_size * self.world_size
        self.order = order[:usable].reshape(self.num_batches, self.world_size, self.batch_size)[:, self.rank]

    def window(self):
        import torch

        if self.order is None:
            raise RuntimeError("Call set_epoch before reading tokens")
        batches = []
        for position in range(self.cursor, min(self.cursor + self.accumulation, self.num_batches)):
            samples = [self.dataset[int(i)] for i in self.order[position]]
            batch = {key: torch.as_tensor(np.stack([x[key] for x in samples]), dtype=torch.long)
                     for key in ("input_ids", "attention_mask", "labels")}
            batch["sample_ids"] = [x["sample_id"] for x in samples]
            batches.append(batch)
        return batches

    def commit_window(self):
        self.cursor += self.accumulation

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
