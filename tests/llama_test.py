"""Native Llama contracts; tiny local weights, no network or 8B download."""

from copy import deepcopy
import json
import os
from pathlib import Path
import random
import runpy
import subprocess
import sys
from unittest.mock import patch

import numpy as np
import pytest

from dlio_benchmark.llama.config import LlamaConfig
from dlio_benchmark.llama.data import (TokenDataset, TokenLoader, TokenWriter,
                                      conversation_tokens, generate_tokens)


def config(**changes):
    values = dict(backend="fsdp", data_source="dlio_tokens", sequence_length=8,
                  precision="fp32", attention_implementation="eager", generated_samples=24,
                  samples_per_file=3, gradient_accumulation_steps=2,
                  max_grad_norm=0, learning_rate=0.01)
    values.update(changes)
    return LlamaConfig.from_mapping(values)


@pytest.fixture(scope="module")
def torch():
    module = pytest.importorskip("torch")
    module.set_num_threads(1)
    return module


@pytest.fixture
def tiny(torch):
    from transformers import LlamaConfig as HFConfig, LlamaForCausalLM
    model_config = HFConfig(vocab_size=32, hidden_size=16, intermediate_size=32,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
        max_position_embeddings=32, attention_dropout=0.0, bos_token_id=1, eos_token_id=2)
    model_config._attn_implementation = "eager"
    return LlamaForCausalLM(model_config)


@pytest.fixture
def loader(tmp_path):
    manifest = generate_tokens(tmp_path / "tokens", config(), 32, 2, 24, 91, {"test": 1})
    result = TokenLoader(TokenDataset(manifest), 2, 0, 1, accumulation=2, seed=91)
    result.set_epoch(1)
    return result


def make_runtime(torch, model, options=None):
    from dlio_benchmark.llama.runtime import TorchRuntime
    options = options or config()
    optimizer = torch.optim.AdamW(model.parameters(), lr=options.learning_rate,
                                 weight_decay=options.weight_decay, betas=(0.9, 0.95), foreach=False)
    return TorchRuntime(model, optimizer, options, torch.device("cpu"))


def initialize_mpi():
    from dlio_benchmark.utils.utility import DLIOMPI
    mpi = DLIOMPI.get_instance()
    mpi.initialize()
    return mpi


def test_configuration_rejects_implicit_or_conflicting_choices():
    with pytest.raises(ValueError, match="explicitly"):
        config(backend="deepspeed")
    for stage in (2, 3):
        assert config(backend="deepspeed", zero_stage=stage).zero_stage == stage
    with pytest.raises(ValueError, match="conversation_format"):
        config(data_source="openhermes")
    for loss in ("full", "response_only"):
        assert config(data_source="openhermes", conversation_format="role_lines_v1", loss=loss).loss == loss
    with pytest.raises(ValueError, match="Unknown"):
        LlamaConfig.from_mapping({"backend": "fsdp", "data_source": "dlio_tokens", "typo": 1})
    with pytest.raises(ValueError, match="DeepSpeed settings"):
        config(zero_stage=3)


def test_packaging_extras_do_not_add_base_requirements():
    with patch("setuptools.setup") as setup:
        runpy.run_path(str(Path(__file__).parents[1] / "setup.py"))
    metadata = setup.call_args.kwargs
    base = metadata["install_requires"]
    optional = ("transformers", "datasets", "deepspeed", "accelerate", "flash-attn")
    assert not any(any(item.startswith(name) for name in optional) for item in base)
    extras = metadata["extras_require"]
    assert set(extras["llama"]) < set(extras["llama-deepspeed"])
    assert not any("flash" in item for item in extras["llama-deepspeed"])


def test_base_cli_import_does_not_require_llama_packages():
    script = """
import importlib.abc
import sys
class BlockOptional(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'transformers', 'datasets', 'deepspeed', 'accelerate'}:
            raise AssertionError('Base import requested ' + fullname)
sys.meta_path.insert(0, BlockOptional())
import dlio_benchmark.main
"""
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr


def test_checkpoint_configuration_reaches_native_flags():
    from unittest.mock import Mock
    from dlio_benchmark.utils.config import ConfigArguments, LoadConfig
    args = object.__new__(ConfigArguments)
    args.communication = False
    args.comm_size = 1
    args.logger = Mock()
    LoadConfig(args, {"checkpoint": {"fsync": True, "recovery_rank_shift": True}})
    assert args.checkpoint_fsync is True
    assert args.checkpoint_recovery_rank_shift is True


def test_token_bytes_and_labels_reach_the_loader(tmp_path, torch):
    writer = TokenWriter(tmp_path, 8, 32, 2, 2, {"fixture": True})
    writer.append([1, 17, 2, 5], [1, -100, 2, 5])
    writer.append([1, 2])
    dataset = TokenDataset(writer.finish())
    first = dataset[0]
    assert first["input_ids"].tolist() == [1, 17, 2, 5, 2, 2, 2, 2]
    assert first["labels"].tolist() == [1, -100, 2, 5, -100, -100, -100, -100]
    assert first["attention_mask"].tolist() == [1, 1, 1, 1, 0, 0, 0, 0]
    loader = TokenLoader(dataset, 1, 0, 1, 1)
    loader.set_epoch(1)
    batches = loader.window()
    assert batches[0]["input_ids"].dtype == torch.long
    assert dataset.bytes_read == 2 * (2 * 8 + 1) * 4


def test_generated_tokens_use_full_vocabulary_and_stable_sample_seed(tmp_path):
    options = config(sequence_length=128)
    first = TokenDataset(generate_tokens(tmp_path / "a", options, 128256, 2, 4, 7, {}))
    second = TokenDataset(generate_tokens(tmp_path / "b", options, 128256, 2, 4, 7, {}))
    assert first[0]["input_ids"].max() > 255
    for index in range(4):
        np.testing.assert_array_equal(first[index]["input_ids"], second[index]["input_ids"])
    assert first.manifest["fingerprint"] == second.manifest["fingerprint"]


def test_reader_rejects_truncated_and_invalid_records(tmp_path):
    with pytest.raises(ValueError, match="vocabulary"):
        TokenWriter(tmp_path / "bad", 8, 32, 2, 3, {}).append([1, 32])
    manifest = generate_tokens(tmp_path / "good", config(), 32, 2, 4, 7, {})
    dataset = TokenDataset(manifest)
    path = dataset.paths[0]
    with path.open("r+b") as stream:
        stream.truncate(path.stat().st_size - 1)
    with pytest.raises(ValueError, match="Truncated"):
        TokenDataset(manifest)


class SimpleTokenizer:
    bos_token_id = 1
    eos_token_id = 2
    pad_token_id = 2

    def encode(self, value, add_special_tokens=False):
        return [3 + ord(char) % 20 for char in value]


def test_openhermes_roles_and_response_loss_exclude_metadata():
    example = {"conversations": [{"from": "human", "value": "Hello"},
                                {"from": "gpt", "value": "Hi"}],
               "source": "must not be trained"}
    tokenizer = SimpleTokenizer()
    ids, full = conversation_tokens(example, tokenizer, "full")
    assert full == ids
    response_ids, response = conversation_tokens(example, tokenizer, "response_only")
    assert response_ids == ids
    assert response[-1] == tokenizer.eos_token_id
    assert response.count(-100) > 0
    assert sum(x != -100 for x in response) == len(tokenizer.encode("Hi\n")) + 1
    changed = dict(example, source="different metadata")
    assert conversation_tokens(changed, tokenizer, "full") == (ids, full)


def test_openhermes_preparation_packing_and_both_losses(tmp_path):
    from datasets import Dataset
    from dlio_benchmark.llama.data import prepare_openhermes
    source = tmp_path / "raw"
    Dataset.from_list([{"conversations": [{"from": "human", "value": "Hello"},
                                         {"from": "gpt", "value": "Hi"}]}] * 3).save_to_disk(str(source))
    for loss in ("full", "response_only"):
        for packing in (False, True):
            options = config(data_source="openhermes", conversation_format="role_lines_v1",
                             loss=loss, dataset_name_or_path=str(source),
                             sequence_length=64, packing=packing)
            manifest = prepare_openhermes(tmp_path / f"{loss}-{packing}", options,
                                         SimpleTokenizer(), 32, "train", {})
            data = TokenDataset(manifest)
            assert len(data) > 0
            assert all(data[index]["labels"][1:].max() >= 0 for index in range(len(data)))


def test_sampler_equal_counts_drop_partial_windows_and_resume(loader, torch):
    others = [TokenLoader(loader.dataset, 2, rank, 2, 2, 12) for rank in range(2)]
    for item in others:
        item.set_epoch(3)
    assert others[0].num_batches == others[1].num_batches == 6
    left, right = [set(item.order.flatten()) for item in others]
    assert left.isdisjoint(right)
    before = loader.window()
    assert loader.cursor == 0
    loader.commit_window()
    saved = loader.state_dict()
    expected = loader.window()
    fresh = TokenLoader(loader.dataset, 2, 0, 1, 2, 91)
    fresh.load_state_dict(saved)
    assert fresh.window()[0]["sample_ids"] == expected[0]["sample_ids"]
    assert before[0]["sample_ids"] != expected[0]["sample_ids"]
    with pytest.raises(ValueError, match="boundary"):
        fresh.set_epoch(3, 1)


@pytest.mark.parametrize("rope_type", [None, "llama3"])
def test_safetensors_loading_restores_parameters_and_rotary_buffers(tmp_path, tiny, torch, rope_type):
    from dlio_benchmark.model.pytorch.llama import load_pretrained_llama
    from dlio_benchmark.llama.runtime import initialize_rotary_buffers, load_parameter_slices, safetensor_sources
    if rope_type:
        model_config = deepcopy(tiny.config)
        model_config.rope_scaling = {"rope_type": "llama3", "factor": 8.0,
            "low_freq_factor": 1.0, "high_freq_factor": 4.0, "original_max_position_embeddings": 8}
        tiny = type(tiny)(model_config)
    tiny.save_pretrained(tmp_path, max_shard_size="3KB")
    options = config(model_name_or_path=str(tmp_path), local_files_only=True)
    restored = load_pretrained_llama(options, meta=True, model_config=tiny.config)
    restored.to_empty(device="cpu")
    initialize_rotary_buffers(restored, torch.device("cpu"))
    load_parameter_slices(restored, safetensor_sources(options))
    tiny.eval(); restored.eval()
    ids = torch.tensor([[1, 6, 2, 3]])
    with torch.no_grad():
        torch.testing.assert_close(restored(ids).logits, tiny(ids).logits, rtol=0, atol=0)


def test_accumulation_matches_combined_token_weighted_update(tiny, loader, torch):
    runtime = make_runtime(torch, deepcopy(tiny))
    reference = make_runtime(torch, deepcopy(tiny))
    # Compare accumulation algebra without Adam amplifying tiny near-zero
    # gradient roundoff into a larger first-step parameter difference.
    runtime.optimizer = torch.optim.SGD(runtime.model.parameters(), lr=0.01)
    reference.optimizer = torch.optim.SGD(reference.model.parameters(), lr=0.01)
    batches = loader.window()
    batches[0]["labels"][0, 4:] = -100
    result = runtime.train_window(batches)
    combined = {key: torch.cat([batch[key] for batch in batches])
                for key in ("input_ids", "attention_mask", "labels")}
    reference.optimizer.zero_grad()
    expected_loss = reference.model(**combined).loss
    expected_loss.backward()
    reference.optimizer.step()
    assert result["loss"] == pytest.approx(expected_loss.item(), rel=1e-6)
    for actual, expected in zip(runtime.model.parameters(), reference.model.parameters()):
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-7)
    assert runtime.progress() == {"microbatches": 2, "optimizer_steps": 1, "applied_updates": 1}


def test_evaluation_does_not_change_model_optimizer_or_progress(tiny, loader, torch):
    runtime = make_runtime(torch, tiny)
    runtime.train_window(loader.window())
    params = [p.detach().clone() for p in tiny.parameters()]
    optimizer = deepcopy(runtime.optimizer.state_dict())
    progress = runtime.progress()
    loss = runtime.evaluate(loader.window())
    assert np.isfinite(loss)
    for actual, expected in zip(tiny.parameters(), params):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for key, state in optimizer["state"].items():
        for name, tensor in state.items():
            torch.testing.assert_close(tensor, runtime.optimizer.state_dict()["state"][key][name], rtol=0, atol=0)
    assert runtime.progress() == progress
    assert tiny.training


def test_live_checkpoint_resume_matches_uninterrupted_training(tmp_path, tiny, loader, torch):
    from dlio_benchmark.checkpointing.llama_checkpointing import LlamaCheckpointing
    options = config()
    runtime = make_runtime(torch, tiny, options)
    # Include dropout so the test verifies RNG restoration as well as weights.
    runtime.model.config.attention_dropout = 0.2
    for layer in runtime.model.model.layers:
        layer.self_attn.attention_dropout = 0.2
    runtime.train_window(loader.window()); loader.commit_window()
    mechanism = LlamaCheckpointing(runtime, options, tmp_path / "checkpoints", {"backend": "fsdp"}, True)
    saved = mechanism.save_checkpoint(loader, "final")
    expected_ids = [batch["sample_ids"] for batch in loader.window()]
    expected = runtime.train_window(loader.window()); loader.commit_window()
    expected_random = (random.random(), np.random.random(), torch.rand(4))
    final_parameters = [p.detach().clone() for p in tiny.parameters()]
    fresh = make_runtime(torch, deepcopy(tiny), options)
    fresh_loader = TokenLoader(loader.dataset, 2, 0, 1, 2, 91)
    new_mechanism = LlamaCheckpointing(fresh, options, tmp_path / "other", {"backend": "fsdp"})
    loaded = new_mechanism.load_checkpoint(saved["path"], fresh_loader)
    assert [batch["sample_ids"] for batch in fresh_loader.window()] == expected_ids
    actual = fresh.train_window(fresh_loader.window()); fresh_loader.commit_window()
    actual_random = (random.random(), np.random.random(), torch.rand(4))
    assert actual["loss"] == expected["loss"]
    for parameter, expected_parameter in zip(fresh.model.parameters(), final_parameters):
        torch.testing.assert_close(parameter, expected_parameter, rtol=0, atol=0)
    assert actual_random[:2] == expected_random[:2]
    torch.testing.assert_close(actual_random[2], expected_random[2], rtol=0, atol=0)
    assert fresh.progress() == runtime.progress()
    assert loaded["checkpoint_bytes"] == saved["checkpoint_bytes"]
    assert loaded["throughput_GiB_s"] > 0


def test_incomplete_or_incompatible_checkpoint_rejected_before_mutation(tmp_path, tiny, loader, torch):
    from dlio_benchmark.checkpointing.llama_checkpointing import LlamaCheckpointing
    runtime = make_runtime(torch, tiny)
    runtime.train_window(loader.window()); loader.commit_window()
    mechanism = LlamaCheckpointing(runtime, config(), tmp_path / "ckpt", {"backend": "fsdp"})
    path = Path(mechanism.save_checkpoint(loader)["path"])
    before = next(tiny.parameters()).detach().clone()
    incompatible = LlamaCheckpointing(runtime, config(), tmp_path, {"backend": "deepspeed"})
    with pytest.raises(RuntimeError, match="differ"):
        incompatible.load_checkpoint(path, loader)
    state_file = next((path / "state").glob("*.distcp"))
    state_file.unlink()
    with pytest.raises(RuntimeError, match="FileNotFound"):
        mechanism.load_checkpoint(path, loader)
    torch.testing.assert_close(next(tiny.parameters()), before, rtol=0, atol=0)


def test_deepspeed_configuration_resolves_batch_size_and_rejects_auto():
    from dlio_benchmark.llama.runtime import deepspeed_settings
    for stage in (2, 3):
        settings = deepspeed_settings(config(backend="deepspeed", zero_stage=stage), 3, 4)
        assert settings["train_batch_size"] == 24
        assert settings["zero_optimization"]["stage"] == stage
    with pytest.raises(ValueError, match="auto"):
        deepspeed_settings(config(backend="deepspeed", zero_stage=3,
                                 deepspeed_config={"zero_optimization": {"reduce_bucket_size": "auto"}}), 2, 2)


@pytest.mark.parametrize("backend,precision", [("torch", "fp32"), ("fsdp", "fp32"),
    ("deepspeed2", "fp32"), ("deepspeed3", "fp32"), ("fsdp", "bf16"),
    ("deepspeed2", "bf16"), ("deepspeed3", "bf16")])
def test_fresh_process_backend_resume(tmp_path, tiny, backend, precision, torch):
    if backend.startswith("deepspeed"):
        pytest.importorskip("deepspeed")
    tiny.save_pretrained(tmp_path / "model", max_shard_size="3KB")
    generate_tokens(tmp_path / "tokens", config(), 32, 2, 24, 19, {})
    worker = Path(__file__).with_name("llama_restart_worker.py")
    device = os.environ.get("DLIO_LLAMA_TEST_DEVICE", "cpu") if backend != "torch" else "cpu"
    if device == "cuda" and torch.cuda.device_count() < 2:
        pytest.skip("Two GPUs are required for native backend qualification")
    command = [sys.executable]
    if backend != "torch":
        command += ["-m", "torch.distributed.run", "--standalone", "--nnodes=1", "--nproc_per_node=2"]
    command += [str(worker), str(tmp_path), backend]
    environment = dict(os.environ, DFTRACER_ENABLE="0", GLOO_SOCKET_IFNAME="lo",
                       DS_ACCELERATOR=device, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                       DLIO_LLAMA_TEST_PRECISION=precision)
    environment["PYTHONPATH"] = str(Path(__file__).parents[1]) + os.pathsep + environment.get("PYTHONPATH", "")
    environment["PATH"] = str(Path(sys.executable).parent) + os.pathsep + environment.get("PATH", "")
    for phase in ("save", "load"):
        result = subprocess.run(command + [phase, "--device", device], env=environment,
                                capture_output=True, text=True, timeout=180)
        assert result.returncode == 0, result.stdout + result.stderr
        if phase == "load":
            assert "fresh-process checkpoint continuation passed" in result.stdout


def test_dlio_entry_lifecycle_with_local_tiny_runtime(tmp_path, monkeypatch, tiny, torch):
    from transformers import PreTrainedTokenizerFast
    from tokenizers import Tokenizer, models
    from omegaconf import OmegaConf
    initialize_mpi()
    from dlio_benchmark.utils.config import ConfigArguments
    from dlio_benchmark.framework.llama_framework import LlamaFramework
    from dlio_benchmark.llama.benchmark import LlamaBenchmark
    from dlio_benchmark.main import DLIOBenchmark, set_dftracer_initialize, set_dftracer_finalize
    source = tmp_path / "model"
    tiny.save_pretrained(source)
    vocabulary = {f"t{index}": index for index in range(32)}
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=Tokenizer(models.WordLevel(vocabulary, unk_token="t0")),
                                       unk_token="t0", bos_token="t1", eos_token="t2", pad_token="t2")
    tokenizer.save_pretrained(source)
    monkeypatch.setattr("dlio_benchmark.llama.benchmark.check_dependencies", lambda *a, **kw: None)
    monkeypatch.setattr(LlamaBenchmark, "_check_gpu", lambda *a: None)
    def local_initialize(self, options, model_config):
        from dlio_benchmark.model.pytorch.llama import load_pretrained_llama
        self.runtime = make_runtime(torch, load_pretrained_llama(options), options)
    monkeypatch.setattr(LlamaFramework, "initialize", local_initialize)
    set_dftracer_initialize(False); set_dftracer_finalize(False)
    ConfigArguments.reset()
    cfg = OmegaConf.create({"model": {"name": "llama"}, "framework": "pytorch",
        "workflow": {"train": True, "generate_data": True, "checkpoint": True, "evaluation": True},
        "train": {"compute": True, "epochs": 1, "total_training_steps": 2},
        "reader": {"batch_size": 2, "batch_size_eval": 2},
        "dataset": {"data_folder": str(tmp_path / "data")},
        "output": {"folder": str(tmp_path / "output")},
        "checkpoint": {"checkpoint_folder": str(tmp_path / "ckpt"), "steps_between_checkpoints": 1},
        "llama": dict(config(model_name_or_path=str(source), local_files_only=True,
                              generated_samples=8, generated_eval_samples=8).to_dict())})
    benchmark = DLIOBenchmark(cfg)
    try:
        benchmark.initialize(); benchmark.run(); benchmark.finalize()
        summary = json.loads((tmp_path / "output" / "summary.json").read_text())
        assert summary["progress"]["optimizer_steps"] == 2
        assert len(summary["checkpoints"]) == 2
        assert len(summary["evaluations"]) == 1
        assert summary["metric"]["samples_this_run"] == 8
        assert all(item["application_read_bytes"] > 0 for item in benchmark._native_llama.stats.windows)
        # An epoch-boundary checkpoint resumes in the next epoch, without
        # repeating evaluation or trying to overwrite the previous save.
        ConfigArguments.reset()
        cfg.llama.resume_from = str(tmp_path / "ckpt" / "step-000000000002-periodic")
        cfg.train.epochs = 2
        cfg.train.total_training_steps = 4
        cfg.output.folder = str(tmp_path / "resumed-output")
        resumed = DLIOBenchmark(cfg)
        resumed.initialize(); resumed.run(); resumed.finalize()
        output = json.loads((tmp_path / "resumed-output" / "summary.json").read_text())
        assert output["progress"]["optimizer_steps"] == 4
        assert output["metric"]["samples_this_run"] == 8
        assert [item["epoch"] for item in output["evaluations"]] == [2]
        assert output["checkpoints"][0]["operation"] == "load"
    finally:
        ConfigArguments.reset()
        set_dftracer_initialize(True); set_dftracer_finalize(True)
