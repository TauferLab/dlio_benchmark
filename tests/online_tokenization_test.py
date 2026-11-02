"""Online tokenization must yield exactly the records offline preparation writes."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
datasets = pytest.importorskip("datasets")

from dlio_benchmark.common.enumerations import DataLoaderType
from dlio_benchmark.data_loader.token_loader import DeliverySettings, build_token_loader
from dlio_benchmark.llama.config import LlamaConfig
from dlio_benchmark.llama.data import TokenDataset, prepare_openhermes
from dlio_benchmark.llama.online import (OnlineTokenProvider, load_online_manifest, open_raw_split,
                                         prepare_online_index)

LENGTH, VOCAB = 64, 32


class SimpleTokenizer:
    bos_token_id = 1
    eos_token_id = 2
    pad_token_id = 2

    def encode(self, value, add_special_tokens=False):
        return [3 + ord(char) % 20 for char in value]


def load_simple_tokenizer():  # module-level so spawned workers can unpickle it
    return SimpleTokenizer()


def turns(*pairs):
    return {"conversations": [{"from": role, "value": value} for role, value in pairs]}


EXAMPLES = [
    turns(("human", "Hello"), ("gpt", "Hi")),
    turns(("human", "What is 2+2?"), ("gpt", "4"), ("human", "Thanks"), ("gpt", "Welcome")),
    turns(("human", "Tell me everything"), ("gpt", "a" * 100)),            # truncated to LENGTH
    turns(("human", "Only a question")),                                    # no target under response_only
    dict(turns(("human", "Hi"), ("gpt", "Hello there")), system_prompt="Be brief"),
    turns(("human", "Go"), ("gpt", "ok")),
] * 7


def options(source, **changes):
    values = dict(backend="fsdp", data_source="openhermes", sequence_length=LENGTH, precision="fp32",
                  attention_implementation="eager", conversation_format="role_lines_v1", loss="full",
                  dataset_name_or_path=str(source), packing=False, samples_per_file=5)
    values.update(changes)
    return LlamaConfig.from_mapping(values)


@pytest.fixture(scope="module")
def raw(tmp_path_factory):
    path = tmp_path_factory.mktemp("raw") / "arrow"
    datasets.Dataset.from_list(EXAMPLES).save_to_disk(str(path))
    return path


def online(tmp_path, raw, **changes):
    config = options(raw, **changes)
    tokenizer = SimpleTokenizer()
    manifest_path = prepare_online_index(tmp_path / "online", config, tokenizer, VOCAB, "train", {"t": 1})
    manifest, examples = load_online_manifest(manifest_path)
    return config, OnlineTokenProvider(raw, "train", manifest, examples, load_simple_tokenizer)


def offline(tmp_path, raw, **changes):
    config = options(raw, **changes)
    return TokenDataset(prepare_openhermes(tmp_path / "offline", config, SimpleTokenizer(), VOCAB, "train", {"t": 1}))


@pytest.mark.parametrize("loss", ["full", "response_only"])
@pytest.mark.parametrize("max_samples", [0, 9])
def test_online_records_equal_offline_records(tmp_path, raw, loss, max_samples):
    _, provider = online(tmp_path, raw, loss=loss, max_samples=max_samples)
    reference = offline(tmp_path, raw, loss=loss, max_samples=max_samples)
    assert len(provider) == len(reference) > 0
    for index in range(len(reference)):
        want, got = reference[index], provider[index]
        for key in ("input_ids", "attention_mask", "labels"):
            assert np.array_equal(want[key], got[key]) and want[key].dtype == got[key].dtype, (index, key)
        assert want["sample_id"] == got["sample_id"]


def test_unusable_examples_are_dropped_like_offline_preparation(tmp_path, raw):
    _, response_only = online(tmp_path, raw, loss="response_only")
    _, full = online(tmp_path / "full", raw, loss="full")
    # the human-only example has no assistant target under response_only, but is fine under full
    assert len(full) == len(EXAMPLES) and len(response_only) == len(EXAMPLES) - 7
    assert 3 not in response_only.examples.tolist()


def test_long_conversations_are_truncated_to_the_sequence_length(tmp_path, raw):
    _, provider = online(tmp_path, raw)
    longest = provider[provider.examples.tolist().index(2)]
    assert longest["attention_mask"].sum() == LENGTH and longest["input_ids"].shape == (LENGTH,)


def test_online_delivery_through_workers_matches_offline_disk_delivery(tmp_path, raw):
    _, provider = online(tmp_path, raw)
    reference = offline(tmp_path, raw)
    settings = DeliverySettings(DataLoaderType.PYTORCH, read_threads=2, prefetch_size=4)
    online_loader = build_token_loader(DataLoaderType.PYTORCH, provider, settings, batch_size=2, rank=0,
                                       world_size=1, accumulation=2, seed=5)
    disk_loader = build_token_loader(DataLoaderType.PYTORCH, reference, DeliverySettings(), batch_size=2,
                                     rank=0, world_size=1, accumulation=2, seed=5)
    try:
        for epoch in (1, 2):
            online_loader.set_epoch(epoch); disk_loader.set_epoch(epoch)
            while online_loader.cursor < online_loader.num_batches:
                for a, b in zip(online_loader.window(), disk_loader.window()):
                    assert a["sample_ids"] == b["sample_ids"]
                    for key in ("input_ids", "attention_mask", "labels"):
                        assert torch.equal(a[key], b[key])
                online_loader.commit_window(); disk_loader.commit_window()
        assert online_loader.bytes_read > 0, "raw text read inside workers must be counted"
    finally:
        online_loader.close()


def test_a_tampered_example_index_is_rejected(tmp_path, raw):
    config = options(raw)
    path = prepare_online_index(tmp_path / "o", config, SimpleTokenizer(), VOCAB, "train", {})
    np.save(path.parent / "examples.npy", np.arange(3))
    with pytest.raises(ValueError, match="checksum"):
        load_online_manifest(path)


def test_online_needs_a_local_random_access_dataset(tmp_path):
    with pytest.raises(ValueError, match="save_to_disk"):
        open_raw_split(tmp_path / "does-not-exist", "train")


def test_config_rejects_unsupported_online_combinations(raw):
    options(raw, tokenization="online")
    for changes in ({"packing": True}, {"data_source": "dlio_tokens", "loss": ""}):
        with pytest.raises(ValueError, match="online"):
            options(raw, tokenization="online", **changes)
    with pytest.raises(ValueError, match="offline or online"):
        options(raw, tokenization="lazy")
