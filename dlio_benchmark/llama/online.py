"""Online tokenization: raw OpenHermes text tokenized inside the loader workers.

Offline preparation tokenizes once, before timing, and training only reads fixed-size
records. Online delivery moves that work into the loader workers, so tokenizer CPU time,
its thread pool and the raw-text reads compete with the training process. Both modes
yield identical records, which makes the comparison a controlled one.

Equivalence needs record *i* to depend on its index alone. Packed records depend on every
earlier example, so online delivery requires ``llama.packing=false``: one example is one
record, truncated to ``sequence_length``. Examples that give no usable record (fewer than
two tokens, or no prediction target) are dropped exactly as offline preparation drops them;
``prepare_online_index`` finds them once, outside timed training, and stores the surviving
example indices so that sampling sees a fixed record count.
"""

import hashlib
import json
import os
from pathlib import Path

import numpy as np

from dlio_benchmark.llama.data import atomic_json, conversation_tokens, fingerprint

ONLINE_VERSION = 1


def open_raw_split(path, split):
    """Memory-mapped Arrow rows with random access (a save_to_disk directory)."""
    from datasets import load_from_disk
    source = Path(path)
    if not source.is_dir():
        raise ValueError("Online tokenization needs a local datasets.save_to_disk() directory: it reads "
                         "examples by index, which streaming cannot do")
    loaded = load_from_disk(str(source))
    if hasattr(loaded, "keys"):
        return loaded[split]
    if split == "train":
        return loaded
    raise ValueError("A local Dataset has only train; use a DatasetDict for evaluation splits")


def make_record(ids, labels, sequence_length, vocab_size, pad_token_id, sample_id):
    """The record ``TokenWriter`` + ``TokenDataset`` produce, validated the same way."""
    ids = np.asarray(ids, dtype=np.int64)
    targets = np.asarray(labels, dtype=np.int64)
    if ids.ndim != 1 or len(ids) < 2 or len(ids) > sequence_length or ids.shape != targets.shape:
        raise ValueError("Token record must have 2..sequence_length aligned tokens and labels")
    if np.any((ids < 0) | (ids >= vocab_size)):
        raise ValueError("Token ID outside model vocabulary")
    if np.any((targets != -100) & (targets != ids)) or not np.any(targets[1:] != -100):
        raise ValueError("Token record needs valid, unshifted causal targets")
    size = len(ids)
    padded = np.full(sequence_length, pad_token_id, dtype=np.int64)
    padded[:size] = ids
    padded_labels = np.full(sequence_length, -100, dtype=np.int64)
    padded_labels[:size] = targets
    mask = (np.arange(sequence_length) < size).astype(np.int64)
    return {"input_ids": padded, "attention_mask": mask, "labels": padded_labels, "sample_id": int(sample_id)}


def _example_ids(example, tokenizer, loss, sequence_length):
    ids, labels = conversation_tokens(example, tokenizer, loss)
    return ids[:sequence_length], labels[:sequence_length]


def prepare_online_index(folder, config, tokenizer, vocab_size, split, identity):
    """One pass over the raw split (outside timed training) listing the usable examples."""
    dataset = open_raw_split(config.dataset_name_or_path, split)
    usable = []
    for index, example in enumerate(dataset):
        if config.max_samples and index >= config.max_samples:
            break
        ids, labels = _example_ids(example, tokenizer, config.loss, config.sequence_length)
        if len(ids) >= 2 and any(x != -100 for x in labels[1:]):
            if max(ids) >= vocab_size:
                raise ValueError(f"Token ID outside model vocabulary in example {index}")
            usable.append(index)
    if not usable:
        raise ValueError("No usable examples were produced")
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    examples = folder / "examples.npy"
    np.save(examples, np.asarray(usable, dtype=np.int64))
    manifest = {
        "version": ONLINE_VERSION, "kind": "online_openhermes", "num_records": len(usable),
        "sequence_length": config.sequence_length, "vocab_size": vocab_size,
        "pad_token_id": tokenizer.pad_token_id, "loss": config.loss, "split": split,
        "examples_sha256": hashlib.sha256(examples.read_bytes()).hexdigest(), "identity": identity,
    }
    manifest["fingerprint"] = fingerprint(manifest)
    atomic_json(folder / "manifest.json", manifest)
    return folder / "manifest.json"


def load_online_manifest(path):
    path = Path(path)
    manifest = json.loads(path.read_text())
    if manifest.get("fingerprint") != fingerprint({k: v for k, v in manifest.items() if k != "fingerprint"}):
        raise ValueError("Online manifest fingerprint mismatch")
    if manifest["version"] != ONLINE_VERSION or manifest["kind"] != "online_openhermes":
        raise ValueError("Unsupported online manifest")
    examples = path.parent / "examples.npy"
    if hashlib.sha256(examples.read_bytes()).hexdigest() != manifest["examples_sha256"]:
        raise ValueError("Online example index checksum mismatch")
    array = np.load(examples)
    if len(array) != manifest["num_records"]:
        raise ValueError("Online example index count is inconsistent")
    return manifest, array


def load_hf_tokenizer(source, revision, local_files_only):
    """Default tokenizer loader; module-level so spawned workers can import it."""
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(source, revision=revision, local_files_only=local_files_only)


class OnlineTokenProvider:
    """Record ``i`` = tokenization of raw example ``examples[i]``, done on every read."""

    online = True

    def __init__(self, raw_path, split, manifest, examples, tokenizer_loader):
        self.raw_path, self.split = str(raw_path), split
        self.manifest, self.examples = manifest, examples
        self.length = manifest["sequence_length"]
        self.tokenizer_loader = tokenizer_loader  # picklable callable returning a tokenizer
        self.bytes_read = 0  # raw example text consumed, UTF-8 bytes
        self._dataset = None
        self._tokenizer = None

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_dataset"] = state["_tokenizer"] = None  # handles are reopened inside each worker
        return state

    def __len__(self):
        return len(self.examples)

    def _open(self):
        if self._dataset is None:
            self._dataset = open_raw_split(self.raw_path, self.split)
        if self._tokenizer is None:
            self._tokenizer = self.tokenizer_loader()

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        self._open()
        example = self._dataset[int(self.examples[index])]
        self.bytes_read += sum(len(turn["value"].encode("utf-8")) for turn in example["conversations"])
        ids, labels = _example_ids(example, self._tokenizer, self.manifest["loss"], self.length)
        return make_record(ids, labels, self.length, self.manifest["vocab_size"],
                           self.manifest["pad_token_id"], index)
