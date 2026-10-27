"""Separate-process tiny-model save/restart qualification for native backends."""

import argparse
from contextlib import nullcontext
import os
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist

from dlio_benchmark.checkpointing.llama_checkpointing import LlamaCheckpointing
from dlio_benchmark.llama.config import LlamaConfig
from dlio_benchmark.llama.data import TokenDataset, TokenLoader
from dlio_benchmark.llama.runtime import FSDPRuntime, DeepSpeedRuntime, TorchRuntime


def parameter_snapshot(runtime, backend):
    from torch.distributed.tensor import DTensor
    if backend.startswith("deepspeed"):
        import deepspeed
        model = runtime.model.module
        manager = deepspeed.zero.GatheredParameters(list(model.parameters()))
    else:
        model = runtime.model
        manager = nullcontext()
    with manager:
        return {name: (value.to_local() if isinstance(value, DTensor) else value).detach().cpu().clone()
                for name, value in model.named_parameters()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("folder")
    parser.add_argument("backend", choices=["torch", "fsdp", "deepspeed2", "deepspeed3"])
    parser.add_argument("phase", choices=["save", "load"])
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    args = parser.parse_args()
    torch.set_num_threads(1)
    rank = int(os.environ.get("RANK", "0"))
    size = int(os.environ.get("WORLD_SIZE", "1"))
    device = torch.device(args.device, rank) if args.device == "cuda" else torch.device("cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    if args.backend != "torch":
        from datetime import timedelta
        dist.init_process_group("nccl" if device.type == "cuda" else "gloo",
                                timeout=timedelta(seconds=90))
    folder = Path(args.folder)
    try:
        options = LlamaConfig.from_mapping(dict(
            backend="deepspeed" if args.backend.startswith("deepspeed") else "fsdp",
            zero_stage=int(args.backend[-1]) if args.backend.startswith("deepspeed") else 0,
            data_source="dlio_tokens", model_name_or_path=str(folder / "model"),
            local_files_only=True, sequence_length=8,
            precision=os.environ.get("DLIO_LLAMA_TEST_PRECISION", "fp32"),
            attention_implementation="eager", gradient_accumulation_steps=2,
            learning_rate=0.001, max_grad_norm=1.0, activation_checkpointing=True,
        ))
        torch.manual_seed(123 + rank)
        from transformers import AutoConfig, AutoModelForCausalLM
        context = SimpleNamespace(device=device, rank=rank, world_size=size)
        if args.backend == "fsdp":
            runtime = FSDPRuntime(options, context, AutoConfig.from_pretrained(folder / "model"))
            from torch.distributed.checkpoint.state_dict import get_model_state_dict, StateDictOptions
            full = get_model_state_dict(runtime.model, options=StateDictOptions(full_state_dict=True))
            reference = AutoModelForCausalLM.from_pretrained(folder / "model", attn_implementation="eager")
            for name, value in reference.state_dict().items():
                torch.testing.assert_close(full[name].cpu(), value, rtol=0, atol=0)
            del full, reference
        elif args.backend.startswith("deepspeed"):
            if device.type == "cpu":
                import deepspeed
                # CPU qualification uses standard Gloo collectives rather than
                # an optional host-specific shared-memory C++ optimization.
                deepspeed.ops.__compatible_ops__["deepspeed_shm_comm"] = False
            runtime = DeepSpeedRuntime(options, context, 1)
        else:
            model = AutoModelForCausalLM.from_pretrained(folder / "model", attn_implementation="eager")
            optimizer = torch.optim.AdamW(model.parameters(), lr=options.learning_rate, betas=(0.9, 0.95))
            runtime = TorchRuntime(model, optimizer, options, device)
        loader = TokenLoader(TokenDataset(folder / "tokens" / "manifest.json"),
                             1, rank, size, accumulation=2, seed=19)
        loader.set_epoch(1)
        identity = {"backend": args.backend, "world_size": size, "torch": torch.__version__}
        checkpoint = LlamaCheckpointing(runtime, options, folder / "checkpoints", identity)
        if args.phase == "save":
            runtime.train_window(loader.window()); loader.commit_window()
            checkpoint.save_checkpoint(loader)
            sample_ids = [batch["sample_ids"] for batch in loader.window()]
            result = runtime.train_window(loader.window()); loader.commit_window()
            expected = {"parameters": parameter_snapshot(runtime, args.backend),
                        "progress": runtime.progress(), "result": result,
                        "sample_ids": sample_ids, "next_rng": torch.rand(4)}
            torch.save(expected, folder / f"expected-{rank}.pt")
        else:
            checkpoint.load_checkpoint(folder / "checkpoints" / "step-000000000001-periodic", loader)
            expected = torch.load(folder / f"expected-{rank}.pt", weights_only=False)
            assert [batch["sample_ids"] for batch in loader.window()] == expected["sample_ids"]
            result = runtime.train_window(loader.window()); loader.commit_window()
            actual = parameter_snapshot(runtime, args.backend)
            assert actual.keys() == expected["parameters"].keys()
            for name in actual:
                torch.testing.assert_close(actual[name], expected["parameters"][name], rtol=1e-6, atol=1e-7)
            assert abs(result["loss"] - expected["result"]["loss"]) < 1e-6
            assert runtime.progress() == expected["progress"]
            torch.testing.assert_close(torch.rand(4), expected["next_rng"], rtol=0, atol=0)
            if rank == 0:
                print(f"{args.backend}: fresh-process checkpoint continuation passed", flush=True)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
