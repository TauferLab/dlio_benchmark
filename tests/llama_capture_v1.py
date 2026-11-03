"""Capture independent v1 fixtures using the frozen writer, then verify fresh-process continuation.

Run from an isolated baseline worktree. Artifacts deliberately live outside Git.
"""
import argparse
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import subprocess
import sys

CASES = (("torch", "fp32"), ("fsdp", "fp32"), ("deepspeed2", "fp32"),
         ("deepspeed3", "fp32"), ("fsdp", "bf16"),
         ("deepspeed2", "bf16"), ("deepspeed3", "bf16"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--case", choices=[f"{b}-{p}" for b, p in CASES], action="append")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    if output.exists():
        parser.error("Output must not exist: every fixture run needs a fresh directory")
    if args.device == "cuda" and not (os.environ.get("SLURM_JOB_ID") or os.environ.get("FLUX_JOB_ID")):
        parser.error("GPU qualification must run inside a user allocation")
    import torch
    import dlio_benchmark
    from transformers import LlamaConfig as HFConfig, LlamaForCausalLM
    from dlio_benchmark.checkpointing.llama_checkpointing import CHECKPOINT_VERSION
    from dlio_benchmark.llama.config import LlamaConfig, check_dependencies
    from dlio_benchmark.llama.data import generate_tokens
    assert Path(dlio_benchmark.__file__).resolve().is_relative_to(root)
    assert CHECKPOINT_VERSION == 1, "Capture must use the independent original v1 writer"
    if args.device == "cuda":
        assert torch.cuda.device_count() >= 2, "Two visible GPUs required"
    cases = [(b, p) for b, p in CASES if not args.case or f"{b}-{p}" in args.case]
    for backend in {b for b, _ in cases}:
        options = LlamaConfig.from_mapping(dict(
            backend="deepspeed" if backend.startswith("deepspeed") else "fsdp",
            zero_stage=int(backend[-1]) if backend.startswith("deepspeed") else 0,
            data_source="dlio_tokens", sequence_length=8))
        check_dependencies(options)
    output.mkdir(parents=True)
    packages = {name: version(name) for name in
                ("torch", "transformers", "datasets", "deepspeed", "accelerate", "numpy", "safetensors")}
    report = dict(source_root=str(root), source_sha=subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip(),
        source_status=subprocess.check_output(
        ["git", "-C", str(root), "status", "--porcelain"], text=True),
        python=sys.executable, packages=packages, device=args.device,
        writer_sha256=digest(root / "dlio_benchmark/checkpointing/llama_checkpointing.py"),
        worker_sha256=digest(root / "tests/llama_restart_worker.py"), cases=[])
    (output / "capture.json").write_text(json.dumps(report, indent=2) + "\n")
    torch.set_num_threads(1)
    for backend, precision in cases:
        folder = output / f"{backend}-{precision}"
        folder.mkdir()
        torch.manual_seed(20261027)
        model = LlamaForCausalLM(HFConfig(vocab_size=32, hidden_size=16, intermediate_size=32,
            num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
            max_position_embeddings=32, attention_dropout=0.1, bos_token_id=1, eos_token_id=2,
            pad_token_id=2, attn_implementation="eager"))
        model.save_pretrained(folder / "model", max_shard_size="3KB")
        del model
        options = LlamaConfig.from_mapping(dict(backend="fsdp", data_source="dlio_tokens",
            sequence_length=8, generated_samples=24, samples_per_file=3,
            gradient_accumulation_steps=2))
        generate_tokens(folder / "tokens", options, 32, 2, 24, 19, {})
        device = "cpu" if backend == "torch" else args.device
        command = [sys.executable]
        if backend != "torch":
            command += ["-m", "torch.distributed.run", "--standalone", "--nnodes=1", "--nproc_per_node=2"]
        command += [str(root / "tests/llama_restart_worker.py"), str(folder), backend]
        env = dict(os.environ, DFTRACER_ENABLE="0", GLOO_SOCKET_IFNAME="lo",
                   DS_ACCELERATOR=device, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                   DLIO_LLAMA_TEST_PRECISION=precision,
                   PYTHONPATH=str(root) + os.pathsep + os.environ.get("PYTHONPATH", ""))
        item = dict(backend=backend, precision=precision, device=device, phases=[])
        report["cases"].append(item)
        for phase in ("save", "load"):
            argv = command + [phase, "--device", device]
            with (folder / f"{phase}.log").open("w") as log:
                result = subprocess.run(argv, env=env, cwd=root, stdout=log,
                                        stderr=subprocess.STDOUT, timeout=240)
            item["phases"].append(dict(phase=phase, argv=argv, exit_code=result.returncode))
            (output / "capture.json").write_text(json.dumps(report, indent=2) + "\n")
            if result.returncode:
                raise RuntimeError(f"{backend}/{precision}/{phase} failed; see {folder}/{phase}.log")
        checkpoint = folder / "checkpoints/step-000000000001-periodic"
        assert json.loads((checkpoint / "complete.json").read_text())["version"] == 1
        for rank in range(1 if backend == "torch" else 2):
            expected = torch.load(folder / f"expected-{rank}.pt", map_location="cpu", weights_only=False)
            assert {"parameters", "progress", "result", "sample_ids", "optimizer", "rng", "next_rng"} <= expected.keys()
            assert expected["optimizer"], "Fixture must contain optimizer evidence"
        item["files"] = {str(p.relative_to(folder)): digest(p)
                         for p in sorted(folder.rglob("*")) if p.is_file()}
        item["status"] = "PASS"
        (output / "capture.json").write_text(json.dumps(report, indent=2) + "\n")
        print(f"{backend}/{precision}: independent v1 capture and continuation PASS", flush=True)
    assert len(report["cases"]) == len(cases)
    print(output / "capture.json", flush=True)


if __name__ == "__main__":
    main()
