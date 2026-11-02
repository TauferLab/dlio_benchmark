"""End to end: every delivery mode trains, checkpoints mid-epoch and resumes exactly."""

from copy import deepcopy
import json
import pickle

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
datasets = pytest.importorskip("datasets")

from llama_test import config, initialize_mpi, make_runtime

@pytest.fixture
def tiny():
    from transformers import LlamaConfig as HFConfig, LlamaForCausalLM
    torch.set_num_threads(1)
    model_config = HFConfig(vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
                            num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=64,
                            attention_dropout=0.0, bos_token_id=1, eos_token_id=2)
    model_config._attn_implementation = "eager"
    return LlamaForCausalLM(model_config)


STEPS, SAMPLES = 4, 16  # 16 records / (batch 2 x accumulation 2) = 4 windows in the single epoch


def raw_conversations(path, count=20):
    rows = [{"conversations": [{"from": "human", "value": f"question {i}"},
                               {"from": "gpt", "value": f"answer {i}"}]} for i in range(count)]
    datasets.Dataset.from_list(rows).save_to_disk(str(path))
    return path


MODES = {
    "disk": dict(reader={}, llama={}, generate=True),
    "disk-workers-step-sync": dict(reader={"read_threads": 2, "prefetch_size": 4},
                                   llama={"host_sync": "step"}, generate=True),
    "ram": dict(reader={"data_loader": "load_mem", "memory_budget_bytes": 50_000_000}, llama={}, generate=True),
    "synthetic": dict(reader={"data_loader": "synthetic"}, llama={}, generate=False),
    "online": dict(reader={"read_threads": 2, "prefetch_size": 4}, generate=True,
                   llama={"data_source": "openhermes", "conversation_format": "role_lines_v1", "loss": "full",
                          "tokenization": "online", "generated_samples": 0}),
}


@pytest.fixture
def harness(tmp_path, monkeypatch, tiny):
    from transformers import PreTrainedTokenizerFast
    from tokenizers import Tokenizer, models
    from omegaconf import OmegaConf
    initialize_mpi()
    from dlio_benchmark.framework.llama_framework import LlamaFramework
    from dlio_benchmark.llama.benchmark import LlamaBenchmark
    from dlio_benchmark.main import DLIOBenchmark, set_dftracer_initialize, set_dftracer_finalize
    from dlio_benchmark.utils.config import ConfigArguments

    source = tmp_path / "model"
    tiny.save_pretrained(source)
    vocabulary = {f"t{index}": index for index in range(32)}
    PreTrainedTokenizerFast(tokenizer_object=Tokenizer(models.WordLevel(vocabulary, unk_token="t0")),
                            unk_token="t0", bos_token="t1", eos_token="t2", pad_token="t2").save_pretrained(source)
    raw = raw_conversations(tmp_path / "raw")
    monkeypatch.setattr("dlio_benchmark.llama.benchmark.check_dependencies", lambda *a, **kw: None)
    monkeypatch.setattr(LlamaBenchmark, "_check_gpu", lambda *a: None)

    def local_initialize(self, options, model_config):
        from dlio_benchmark.model.pytorch.llama import load_pretrained_llama
        self.runtime = make_runtime(torch, load_pretrained_llama(options), options)

    monkeypatch.setattr(LlamaFramework, "initialize", local_initialize)
    set_dftracer_initialize(False); set_dftracer_finalize(False)

    def run(name, mode, steps, resume_from=None):
        ConfigArguments.reset()
        llama = config(model_name_or_path=str(source), local_files_only=True, generated_samples=SAMPLES,
                       sequence_length=32, **{k: v for k, v in mode["llama"].items() if k != "generated_samples"})
        values = llama.to_dict()
        values.update({k: v for k, v in mode["llama"].items() if k == "generated_samples"})
        if mode["llama"].get("data_source") == "openhermes":
            values["dataset_name_or_path"] = str(raw)
        values["resume_from"] = str(resume_from) if resume_from else ""
        cfg = OmegaConf.create({
            "model": {"name": "llama"}, "framework": "pytorch",
            "workflow": {"train": True, "generate_data": mode["generate"], "checkpoint": True},
            "train": {"compute": True, "epochs": 1, "total_training_steps": steps},
            "reader": dict({"batch_size": 2}, **mode["reader"]),
            "dataset": {"data_folder": str(tmp_path / "data")}, "output": {"folder": str(tmp_path / name)},
            "checkpoint": {"checkpoint_folder": str(tmp_path / "ckpt" / name), "steps_between_checkpoints": 2},
            "llama": values})
        benchmark = DLIOBenchmark(cfg)
        benchmark.initialize(); benchmark.run(); benchmark.finalize()
        return (json.loads((tmp_path / name / "summary.json").read_text()),
                deepcopy(benchmark._native_llama.stats.windows))

    yield run, tmp_path
    ConfigArguments.reset()
    set_dftracer_initialize(True); set_dftracer_finalize(True)


@pytest.mark.parametrize("name", list(MODES))
def test_mode_trains_and_resumes_mid_epoch_like_an_uninterrupted_run(harness, name):
    run, tmp_path = harness
    mode = MODES[name]
    reference, reference_windows = run("reference", mode, STEPS)
    first, first_windows = run("first", mode, 2)
    resumed, resumed_windows = run("resumed", mode, STEPS, tmp_path / "ckpt" / "first" / "step-000000000002-periodic")

    assert reference["progress"]["optimizer_steps"] == STEPS
    assert [w["sample_ids"] for w in first_windows + resumed_windows] == [w["sample_ids"] for w in reference_windows]
    assert [w["loss"] for w in first_windows + resumed_windows] == pytest.approx(
        [w["loss"] for w in reference_windows], rel=1e-5)
    assert resumed["progress"] == reference["progress"]

    delivery = reference["delivery"]
    assert delivery["loader"] == mode["reader"].get("data_loader", "pytorch")
    assert delivery["read_threads"] == mode["reader"].get("read_threads", 0)
    reads = [w["application_read_bytes"] for w in reference_windows]
    if name in ("ram", "synthetic"):
        assert reads == [0] * STEPS, "no storage reads after preload / without files"
    elif name == "online":
        assert sum(reads) > 0, "raw text read inside workers must be visible"
    else:
        record_bytes = (2 * 32 + 1) * 4
        consumed = STEPS * 2 * 2 * record_bytes  # windows x accumulation x batch x record
        assert sum(reads) >= consumed, "worker reads must be visible and cover everything consumed"
        if mode["reader"].get("read_threads", 0) == 0:
            assert all(r > 0 for r in reads)
        # with workers a window can read zero bytes: its records were fetched ahead during an earlier one
    if name == "ram":
        assert delivery["cache_resident_bytes"] > 0 and delivery["preload_seconds"] >= 0
    if name == "synthetic":
        assert not (tmp_path / "data").exists(), "synthetic delivery must not need or create token files"


def test_every_mode_sees_the_same_microbatches_for_generated_data(harness):
    run, _ = harness
    ids = {name: [w["sample_ids"] for w in run(name, MODES[name], STEPS)[1]]
           for name in ("disk", "disk-workers-step-sync", "ram", "synthetic")}
    assert ids["disk"] == ids["disk-workers-step-sync"] == ids["ram"] == ids["synthetic"]


def test_a_resume_checkpoint_from_one_delivery_identity_is_rejected_by_another(harness):
    run, tmp_path = harness
    run("disk-first", MODES["disk"], 2)
    with pytest.raises(Exception, match="data or training settings differ"):
        run("synthetic-resumed", MODES["synthetic"], STEPS,
            tmp_path / "ckpt" / "disk-first" / "step-000000000002-periodic")


def test_worker_tracing_spec_is_picklable_for_spawn(harness):
    from dlio_benchmark.data_loader.token_loader import WorkerTracing
    initialize_mpi()
    spec = WorkerTracing.from_current_config()
    assert pickle.loads(pickle.dumps(spec)).serial_args == spec.serial_args
    WorkerTracing(None)(0)  # no tracing spec: a worker does nothing


def test_each_fetch_is_marked_with_the_optimizer_step_it_feeds(harness, monkeypatch):
    """The analysis numbers steps from fetch.iter's step; it must be set before the fetch, 1-based."""
    run, _ = harness
    from dlio_benchmark.llama import benchmark as bm
    events = []
    real_update, real_fetch = bm.dft_ai.update, bm.LlamaBenchmark._fetch

    def update(**kwargs):
        if "step" in kwargs:
            events.append(("step", kwargs["step"]))
        return real_update(**kwargs)

    def fetch(self, loader):
        events.append(("fetch",))
        return real_fetch(self, loader)

    monkeypatch.setattr(bm.dft_ai, "update", update)
    monkeypatch.setattr(bm.LlamaBenchmark, "_fetch", fetch)
    run("marked", MODES["disk"], STEPS)
    assert events == [item for k in range(1, STEPS + 1) for item in (("step", k), ("fetch",))]
