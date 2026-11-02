"""Token delivery modes: one sampler and collator, whatever the source, worker count or prefetch."""

import random

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from dlio_benchmark.common.enumerations import DataLoaderType
from dlio_benchmark.data_loader.token_loader import (
    DeliverySettings, GeneratedTokenProvider, TokenSampler, build_token_loader, check_ram_budget,
    load_token_arrays, ram_payload_bytes, register_token_loader, resolve_delivery)
from dlio_benchmark.llama.config import LlamaConfig
from dlio_benchmark.llama.data import TokenDataset, TokenLoader, TokenWriter, generate_tokens

VOCAB, PAD, LENGTH, RECORDS, SEED = 32, 2, 8, 35, 91


def options(**changes):
    values = dict(backend="fsdp", data_source="dlio_tokens", sequence_length=LENGTH, precision="fp32",
                  attention_implementation="eager", generated_samples=RECORDS, samples_per_file=4)
    values.update(changes)
    return LlamaConfig.from_mapping(values)


@pytest.fixture(scope="module")
def manifest(tmp_path_factory):
    return generate_tokens(tmp_path_factory.mktemp("tokens"), options(), VOCAB, PAD, RECORDS, SEED, {"t": 1})


def settings(**changes):
    values = dict(loader=DataLoaderType.PYTORCH, read_threads=0)
    values.update(changes)
    return DeliverySettings(**values)


def make(loader_type, provider, rank=0, world=1, batch=2, accumulation=2, **changes):
    return build_token_loader(loader_type, provider, settings(loader=loader_type, **changes),
                              batch_size=batch, rank=rank, world_size=world, accumulation=accumulation, seed=SEED)


def windows(loader, epochs=(1, 2)):
    out = []
    for epoch in epochs:
        loader.set_epoch(epoch)
        while loader.cursor < loader.num_batches:
            out.append(loader.window())
            loader.commit_window()
    return out


def same(a, b):
    assert len(a) == len(b)
    for left, right in zip(a, b):
        for x, y in zip(left, right):
            assert x["sample_ids"] == y["sample_ids"]
            for key in ("input_ids", "attention_mask", "labels"):
                assert x[key].dtype == torch.long and torch.equal(x[key], y[key]), key


def legacy(manifest, rank, world):
    return TokenLoader(TokenDataset(manifest), 2, rank, world, accumulation=2, seed=SEED)


def legacy_windows(loader, epochs=(1, 2)):
    out = []
    for epoch in epochs:
        loader.set_epoch(epoch)
        while loader.cursor < loader.num_batches:
            out.append(loader.window())
            loader.commit_window()
    return out


@pytest.mark.parametrize("world", [1, 2])
def test_sampler_reproduces_the_v1_order_and_tail(manifest, world):
    for rank in range(world):
        old = legacy(manifest, rank, world)
        sampler = TokenSampler(RECORDS, 2, rank, world, 2, SEED)
        assert sampler.num_batches == old.num_batches
        for epoch in (1, 2, 3):
            old.set_epoch(epoch)
            assert np.array_equal(sampler.order(epoch), old.order)
    # 35 records, B*W*A = 4*world: the declared tail is dropped, never partially used.
    assert TokenSampler(RECORDS, 2, 0, world, 2, SEED).dropped_records == RECORDS % (4 * world)


def test_sampler_rejects_a_dataset_too_small_for_one_window():
    with pytest.raises(ValueError, match="too small"):
        TokenSampler(7, 2, 0, 2, 2, SEED)


@pytest.mark.parametrize("workers", [0, 2])
def test_disk_delivery_matches_the_v1_loader_on_two_ranks(manifest, workers):
    for rank in range(2):
        expected = legacy_windows(legacy(manifest, rank, 2))
        loader = make(DataLoaderType.PYTORCH, TokenDataset(manifest), rank, 2, read_threads=workers)
        try:
            same(windows(loader), expected)
        finally:
            loader.close()


def test_ram_delivery_matches_disk_and_makes_no_steady_state_reads(manifest):
    dataset = TokenDataset(manifest)
    provider = load_token_arrays(dataset)
    ram = make(DataLoaderType.LOAD_MEM, provider)
    disk = make(DataLoaderType.PYTORCH, TokenDataset(manifest))
    same(windows(ram), windows(disk))
    assert ram.bytes_read == 0 and disk.bytes_read > 0
    assert provider.nbytes == ram_payload_bytes(RECORDS, LENGTH)
    assert provider.ids.dtype == provider.labels.dtype == provider.lengths.dtype == np.int32


def test_ram_masks_come_from_lengths_not_from_the_pad_id(tmp_path):
    # EOS doubles as PAD here: id 2 is a real token inside the record and also the padding.
    writer = TokenWriter(tmp_path, LENGTH, VOCAB, PAD, 4, {"t": 2})
    writer.append([5, 2, 7, 2, 9], [-100, -100, 7, 2, 9])           # response-only labels, short
    writer.append([3, 4, 5, 6, 7, 2, 9, 2])
    for _ in range(6):
        writer.append([1, 2, 3, 4])
    dataset = TokenDataset(writer.finish())
    provider = load_token_arrays(dataset)
    for index in range(len(dataset)):
        want, got = dataset[index], provider[index]
        for key in ("input_ids", "attention_mask", "labels"):
            assert np.array_equal(want[key], got[key]) and got[key].dtype == np.int64, (index, key)
    assert provider[0]["attention_mask"].tolist() == [1, 1, 1, 1, 1, 0, 0, 0]
    # record 1 holds the PAD id at real positions 5 and 7; an id-based mask would drop them
    assert provider[1]["attention_mask"].sum() == LENGTH and provider[1]["input_ids"][[5, 7]].tolist() == [PAD, PAD]


def test_generated_delivery_matches_file_generation_without_touching_files(manifest):
    provider = GeneratedTokenProvider(SEED, RECORDS, VOCAB, LENGTH, PAD, {"t": 1})
    dataset = TokenDataset(manifest)
    for index in range(RECORDS):
        want, got = dataset[index], provider[index]
        for key in ("input_ids", "attention_mask", "labels"):
            assert np.array_equal(want[key], got[key]), (index, key)
    generated = make(DataLoaderType.SYNTHETIC, provider)
    same(windows(generated), windows(make(DataLoaderType.PYTORCH, TokenDataset(manifest))))
    assert generated.bytes_read == 0
    assert provider.manifest["fingerprint"] != dataset.manifest["fingerprint"]  # a different mode never resumes as disk


def test_generated_identity_records_generator_and_seed():
    a = GeneratedTokenProvider(SEED, RECORDS, VOCAB, LENGTH, PAD)
    assert a.manifest["fingerprint"] == GeneratedTokenProvider(SEED, RECORDS, VOCAB, LENGTH, PAD).manifest["fingerprint"]
    for other in (GeneratedTokenProvider(SEED + 1, RECORDS, VOCAB, LENGTH, PAD),
                  GeneratedTokenProvider(SEED, RECORDS + 1, VOCAB, LENGTH, PAD),
                  GeneratedTokenProvider(SEED, RECORDS, VOCAB + 1, LENGTH, PAD)):
        assert other.manifest["fingerprint"] != a.manifest["fingerprint"]
    logical = a.manifest["logical_identity"]
    assert logical["generator"] and logical["numpy"] == np.__version__ and logical["loss"] == "full"


@pytest.mark.parametrize("workers", [0, 2])
def test_creating_and_reading_an_iterator_does_not_advance_global_rng(manifest, workers):
    loader = make(DataLoaderType.PYTORCH, TokenDataset(manifest), read_threads=workers)
    loader.set_epoch(1)
    before = (torch.get_rng_state(), random.getstate(), np.random.get_state()[1].copy())
    try:
        loader.window()
    finally:
        loader.close()
    assert torch.equal(torch.get_rng_state(), before[0])
    assert random.getstate() == before[1]
    assert np.array_equal(np.random.get_state()[1], before[2])


@pytest.mark.parametrize("workers", [0, 2])
def test_resume_uses_the_committed_cursor_not_the_prefetched_one(manifest, workers):
    loader = make(DataLoaderType.PYTORCH, TokenDataset(manifest), read_threads=workers, prefetch_size=6)
    loader.set_epoch(1)
    first = loader.window()
    assert loader.cursor == 0, "dispatch and prefetch must not move the committed cursor"
    loader.commit_window()
    saved = loader.state_dict()
    expected = loader.window()
    loader.close()
    fresh = make(DataLoaderType.PYTORCH, TokenDataset(manifest), read_threads=workers)
    fresh.load_state_dict(saved)
    try:
        same([fresh.window()], [expected])
    finally:
        fresh.close()
    assert [b["sample_ids"] for b in first] != [b["sample_ids"] for b in expected]


def test_state_dict_keeps_the_v1_keys_and_rejects_a_different_dataset(manifest):
    loader = make(DataLoaderType.PYTORCH, TokenDataset(manifest))
    loader.set_epoch(1)
    state = loader.state_dict()
    assert set(state) == {"epoch", "cursor", "seed", "dataset_fingerprint", "batch_size", "accumulation", "world_size"}
    assert state == legacy(manifest, 0, 1).state_dict() | {"epoch": 1, "cursor": 0}
    other = make(DataLoaderType.SYNTHETIC, GeneratedTokenProvider(SEED, RECORDS, VOCAB, LENGTH, PAD))
    with pytest.raises(ValueError, match="dataset_fingerprint"):
        other.load_state_dict(state)
    with pytest.raises(ValueError, match="accumulation boundary"):
        loader.set_epoch(1, cursor=1)


@pytest.mark.parametrize("workers", [0, 2])
def test_a_step_limit_bounds_dispatch_and_worker_reads_are_counted(manifest, workers):
    dataset = TokenDataset(manifest)
    loader = make(DataLoaderType.PYTORCH, dataset, read_threads=workers, prefetch_size=8)
    loader.set_epoch(1)
    loader.limit_windows(1)
    try:
        loader.window()
        loader.commit_window()
        with pytest.raises(RuntimeError, match="exhausted"):
            loader.window()
    finally:
        loader.close()
    # one window = 2 microbatches x 2 records: nothing beyond the limit was dispatched
    assert loader.bytes_read == 4 * dataset.record_bytes


def test_close_stops_workers_and_is_idempotent(manifest):
    loader = make(DataLoaderType.PYTORCH, TokenDataset(manifest), read_threads=2)
    loader.set_epoch(1)
    loader.window()
    workers = list(loader._iterator._workers)
    assert workers and all(w.is_alive() for w in workers)
    loader.close()
    loader.close()
    for worker in workers:
        worker.join(timeout=30)
    assert not any(w.is_alive() for w in workers)


def test_worker_failure_surfaces_instead_of_hanging(tmp_path):
    private = generate_tokens(tmp_path / "private", options(), VOCAB, PAD, RECORDS, SEED, {"t": 3})
    dataset = TokenDataset(private)
    loader = make(DataLoaderType.PYTORCH, dataset, read_threads=1, timeout_seconds=60)
    dataset.paths[0].write_bytes(b"x")  # truncate a token file after validation
    loader.set_epoch(1)
    try:
        with pytest.raises(Exception):
            for _ in range(loader.num_batches // loader.accumulation):
                loader.window()
                loader.commit_window()
    finally:
        loader.close()


# --------------------------------------------------------------------- reader settings

def test_omitted_native_settings_do_not_inherit_the_image_defaults():
    result = resolve_delivery({}, "dlio_tokens")
    assert result == DeliverySettings(DataLoaderType.PYTORCH, 0, 2, False, 0.0, "spawn", 0, 300)
    assert result.prefetch_factor is None


def test_explicit_settings_are_resolved_and_validated():
    result = resolve_delivery({"data_loader": "pytorch", "read_threads": 4, "prefetch_size": 8,
                               "pin_memory": True, "iter_time": 0.25}, "openhermes")
    assert (result.read_threads, result.prefetch_factor, result.pin_memory, result.iter_time) == (4, 2, True, 0.25)
    bad = [({"data_loader": "dali"}, "Q03"), ({"data_loader": "tensorflow"}, "not supported"),
           ({"read_threads": 2, "multiprocessing_context": "fork"}, "spawn"),
           ({"read_threads": -1}, "nonnegative"), ({"iter_time": float("nan")}, "finite"),
           ({"data_loader": "load_mem"}, "memory_budget_bytes"),
           ({"data_loader": "load_mem", "memory_budget_bytes": 10, "read_threads": 1}, "read_threads must be 0"),
           ({"pin_memory": "yes"}, "true or false")]
    for reader, message in bad:
        with pytest.raises(ValueError, match=message):
            resolve_delivery(reader, "dlio_tokens")
    with pytest.raises(ValueError, match="OpenHermes"):
        resolve_delivery({"data_loader": "synthetic"}, "openhermes")
    assert resolve_delivery({"data_loader": "synthetic"}, "dlio_tokens").loader is DataLoaderType.SYNTHETIC


def test_online_tokenization_rules():
    assert resolve_delivery({}, "openhermes", "online").loader is DataLoaderType.PYTORCH
    for reader, source, packing, message in [
            ({}, "dlio_tokens", False, "openhermes"), ({"data_loader": "synthetic"}, "openhermes", False, "OpenHermes"),
            ({"data_loader": "load_mem", "memory_budget_bytes": 1}, "openhermes", False, "pytorch loader"),
            ({}, "openhermes", True, "packing=false")]:
        with pytest.raises(ValueError, match=message):
            resolve_delivery(reader, source, "online", packing)


def test_unknown_loader_types_fail_with_the_registration_hint():
    with pytest.raises(ValueError, match="registered"):
        build_token_loader(DataLoaderType.DALI, None, settings(), batch_size=2, rank=0, world_size=1,
                           accumulation=2, seed=SEED)
    calls = []
    register_token_loader(DataLoaderType.DALI, lambda provider, s, **kw: calls.append(kw) or "built")
    try:
        assert build_token_loader(DataLoaderType.DALI, None, settings(), batch_size=2, rank=0,
                                  world_size=1, accumulation=2, seed=SEED) == "built"
        assert calls[0]["seed"] == SEED
    finally:
        from dlio_benchmark.data_loader import token_loader
        token_loader._TOKEN_LOADERS.pop(DataLoaderType.DALI)


# --------------------------------------------------------------------- memory budget

class FakeComm:
    """allgather/allreduce over a fixed set of ranks, as one process sees them."""

    def __init__(self, all_problems_from_others=(), node_total=None):
        self.others, self.node_total = list(all_problems_from_others), node_total

    def allgather(self, mine):
        return [mine] + self.others

    def allreduce(self, value):
        return value if self.node_total is None else self.node_total


def test_ram_budget_accepts_a_fitting_cache():
    check_ram_budget(FakeComm(), 1_000, 10_000, available_bytes=1_000_000)


def test_ram_budget_rejects_a_missing_or_exceeded_budget_on_every_rank():
    with pytest.raises(ValueError, match="positive per-rank"):
        check_ram_budget(FakeComm(), 1_000, 0, available_bytes=1_000_000)
    with pytest.raises(ValueError, match="memory_budget_bytes is 500"):
        check_ram_budget(FakeComm(), 1_000, 500, available_bytes=1_000_000)
    # this rank fits, a peer does not: the check must still fail here, before any allocation
    with pytest.raises(ValueError, match="rank 1"):
        check_ram_budget(FakeComm(all_problems_from_others=[["needs 9 B"]]), 1_000, 10_000, available_bytes=1_000_000)


def test_ram_budget_counts_every_rank_sharing_the_node():
    with pytest.raises(ValueError, match="node demand 4000"):
        check_ram_budget(FakeComm(node_total=4_000), 1_000, 10_000, node_comm=FakeComm(node_total=4_000),
                         available_bytes=4_000)
    check_ram_budget(FakeComm(node_total=4_000), 1_000, 10_000, node_comm=FakeComm(node_total=4_000),
                     available_bytes=40_000)
