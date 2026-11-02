"""Backend ownership of distributed state, pretrained loading and optimization."""

from copy import deepcopy
from datetime import timedelta
import json
import math
import os
from pathlib import Path
import socket

import torch
import torch.distributed as dist

from dlio_benchmark.common.enumerations import Model
from dlio_benchmark.model.model_factory import ModelFactory
from dlio_benchmark.utils.utility import dft_ai


class DistributedContext:
    def __init__(self, mpi, config):
        self.mpi = mpi
        self.config = config
        self.rank, self.world_size = mpi.rank(), mpi.size()
        self.owned = False
        self.original_environment = {}
        self.device = None

    def initialize(self):
        self.validate()
        hostname = socket.gethostname() if self.rank == 0 else None
        hostname = self.mpi.comm().bcast(hostname, root=0)
        for name, value in {"MASTER_ADDR": os.environ.get("MASTER_ADDR", hostname),
                            "MASTER_PORT": os.environ.get("MASTER_PORT", "2345"),
                            "RANK": str(self.rank), "WORLD_SIZE": str(self.world_size),
                            "LOCAL_RANK": str(self.device.index)}.items():
            self.original_environment[name] = os.environ.get(name)
            os.environ[name] = value
        if dist.is_initialized():
            if (dist.get_rank(), dist.get_world_size(), dist.get_backend()) != (self.rank, self.world_size, "nccl"):
                raise ValueError("Existing PyTorch process group disagrees with MPI or GPU backend")
        else:
            try:
                dist.init_process_group("nccl", rank=self.rank, world_size=self.world_size,
                                        timeout=timedelta(seconds=self.config.distributed_timeout_seconds))
            finally:
                self.owned = dist.is_initialized()

    def validate(self):
        for name, expected in (("RANK", self.rank), ("WORLD_SIZE", self.world_size)):
            if name in os.environ and int(os.environ[name]) != expected:
                raise ValueError(f"{name} disagrees with DLIO's MPI world; use one MPI launcher")
        if not torch.cuda.is_available():
            raise RuntimeError("Native FSDP/DeepSpeed Llama training requires a CUDA or ROCm GPU")
        count = torch.cuda.device_count()
        local_rank = self.mpi.local_rank()
        device_index = 0 if count == 1 else local_rank
        if device_index >= count:
            raise ValueError("MPI local rank exceeds visible GPUs; launch one rank per GPU")
        if "LOCAL_RANK" in os.environ and int(os.environ["LOCAL_RANK"]) not in {local_rank, device_index}:
            raise ValueError("LOCAL_RANK disagrees with MPI/device visibility")
        torch.cuda.set_device(device_index)
        self.device = torch.device("cuda", device_index)
        if self.config.precision == "bf16" and not torch.cuda.is_bf16_supported():
            raise ValueError("Selected GPU does not support BF16; select precision=fp32")

    def close(self):
        try:
            if self.owned and dist.is_initialized():
                dist.destroy_process_group()
        finally:
            self.owned = False
            for name, value in self.original_environment.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value
            self.original_environment.clear()


def safetensor_sources(config):
    """Resolve indexed or single-file HF weights, without loading a full model."""
    from transformers.utils.hub import cached_file
    options = {"revision": config.model_revision, "local_files_only": config.local_files_only}
    index = cached_file(config.model_name_or_path, "model.safetensors.index.json",
                        _raise_exceptions_for_missing_entries=False, **options)
    if index:
        weight_map = json.loads(Path(index).read_text())["weight_map"]
        files = {name: cached_file(config.model_name_or_path, name, **options)
                 for name in sorted(set(weight_map.values()))}
        return {key: files[name] for key, name in weight_map.items()}
    single = cached_file(config.model_name_or_path, "model.safetensors",
                         _raise_exceptions_for_missing_entries=False, **options)
    if not single:
        raise ValueError("FSDP initialization requires HF safetensors weights (single or indexed)")
    from safetensors import safe_open
    with safe_open(single, framework="pt", device="cpu") as stream:
        return {key: single for key in stream.keys()}


def load_parameter_slices(model, sources, rank=0, world_size=1):
    """Read only each DTensor's dim-0 shard from safetensors into its storage.

    FSDP2's 1-D mesh uses torch.chunk dim-0 placement. No full GPU or CPU model
    copy is created; transient memory is one local parameter slice.
    """
    from safetensors import safe_open
    from torch.distributed.tensor import DTensor, Shard

    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name not in sources:
                raise ValueError(f"Pretrained weights are missing parameter {name}")
            with safe_open(sources[name], framework="pt", device="cpu") as stream:
                view = stream.get_slice(name)
                if tuple(view.get_shape()) != tuple(parameter.shape):
                    raise ValueError(f"Pretrained shape mismatch: {name}")
                if isinstance(parameter, DTensor):
                    if len(parameter.placements) != 1 or parameter.placements[0] != Shard(0):
                        raise ValueError("Pretrained loader supports a 1-D dim-0 FSDP mesh")
                    local = parameter.to_local()
                    chunk = math.ceil(parameter.shape[0] / world_size)
                    start = min(rank * chunk, parameter.shape[0])
                    value = view[start:start + local.shape[0]]
                else:
                    local = parameter
                    value = view[:]
                local.copy_(value.to(device=local.device, dtype=local.dtype))


def initialize_rotary_buffers(model, device):
    # HF rotary inv_freq is nonpersistent and therefore absent from safetensors.
    from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding
    for module in model.modules():
        if isinstance(module, LlamaRotaryEmbedding):
            fresh = LlamaRotaryEmbedding(config=module.config, device=device)
            module.inv_freq = fresh.inv_freq
            module.original_inv_freq = fresh.original_inv_freq
            module.attention_scaling = fresh.attention_scaling


def deepspeed_settings(config, microbatch_size, world_size):
    settings = deepcopy(config.deepspeed_config)
    expected = {
        "train_micro_batch_size_per_gpu": microbatch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "train_batch_size": microbatch_size * world_size * config.gradient_accumulation_steps,
        "gradient_clipping": config.max_grad_norm,
    }
    for name, value in expected.items():
        if name in settings and settings[name] != value:
            raise ValueError(f"DeepSpeed {name} disagrees with DLIO Llama configuration")
        settings[name] = value
    stage = settings.setdefault("zero_optimization", {}).setdefault("stage", config.zero_stage)
    if stage != config.zero_stage:
        raise ValueError("DeepSpeed ZeRO stage disagrees with llama.zero_stage")
    bf16 = settings.setdefault("bf16", {}).setdefault("enabled", config.precision == "bf16")
    if bf16 != (config.precision == "bf16") or settings.get("fp16", {}).get("enabled", False):
        raise ValueError("DeepSpeed precision disagrees with llama.precision")
    if "optimizer" in settings:
        raise ValueError("Set optimizer hyperparameters under llama, not deepspeed_config.optimizer")
    settings["optimizer"] = {"type": "AdamW", "params": {
        "lr": config.learning_rate, "weight_decay": config.weight_decay,
        "betas": [0.9, 0.95], "eps": 1e-8, "torch_adam": True}}
    def reject_auto(value):
        if isinstance(value, dict):
            for item in value.values():
                reject_auto(item)
        elif isinstance(value, list):
            for item in value:
                reject_auto(item)
        elif value == "auto":
            raise ValueError("DeepSpeed 'auto' settings require Trainer; provide concrete values")
    reject_auto(settings)
    return settings


class TorchRuntime:
    """Common window accounting; also usable with a tiny CPU model in tests."""

    def __init__(self, model, optimizer, config, device, rank=0, world_size=1):
        self.model, self.optimizer, self.config = model, optimizer, config
        self.device, self.rank, self.world_size = device, rank, world_size
        self.microbatches = 0
        self.optimizer_steps = 0
        self.applied_updates = 0
        self.scheduler = None

    def synchronize(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def reduce_sum(self, value):
        tensor = torch.tensor(value, dtype=torch.float64, device=self.device)
        if dist.is_initialized():
            dist.all_reduce(tensor)
        return tensor.item()

    @property
    def step_sync(self):
        return self.config.host_sync == "step"

    def device_value(self, value):
        # A fill kernel, unlike torch.tensor(value, device=...), does not copy from the host.
        return torch.full((), float(value), dtype=torch.float64, device=self.device)

    def all_reduce_async(self, tensor, op=None):
        """Reduce on the device without reading the result on the host."""
        if dist.is_initialized():
            dist.all_reduce(tensor, op=op or dist.ReduceOp.SUM)
        return tensor

    @dft_ai.device.transfer
    def transfer(self, batch):
        # Blocking copies wait for the stream to drain; step mode lets the host run ahead.
        return {key: batch[key].to(self.device, non_blocking=self.step_sync)
                for key in ("input_ids", "attention_mask", "labels")}

    @dft_ai.compute.forward
    def forward_loss(self, batch):
        return self.model(**batch, use_cache=False).loss

    @dft_ai.compute.backward
    def backward(self, loss, boundary):
        loss.backward()

    def train_window(self, batches):
        if len(batches) != self.config.gradient_accumulation_steps:
            raise ValueError("Incomplete accumulation windows must be dropped before training")
        valid = [int((batch["labels"][:, 1:] != -100).sum()) for batch in batches]
        if not all(valid):
            raise ValueError("Every microbatch must contain a causal prediction target")
        if self.step_sync:
            return self._train_window_step(batches, valid)
        total = self.reduce_sum(sum(valid))
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        loss_sum = 0.0
        for index, (batch, count) in enumerate(zip(batches, valid)):
            boundary = index == len(batches) - 1
            if hasattr(self.model, "set_requires_gradient_sync"):
                self.model.set_requires_gradient_sync(boundary)
            loss = self.forward_loss(self.transfer(batch))
            finite = self.reduce_sum(int(torch.isfinite(loss.detach()).item()))
            if finite != self.world_size:
                raise FloatingPointError("Nonfinite causal loss on at least one rank")
            loss_sum += loss.detach().float().item() * count
            self.backward(loss * (count * self.world_size / total), boundary)
            self.microbatches += 1
        if self.config.max_grad_norm:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm,
                                           foreach=False, error_if_nonfinite=True)
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
        self.optimizer_steps += 1
        self.applied_updates += 1
        self.optimizer.zero_grad(set_to_none=True)
        self.synchronize()
        return {"loss": self.reduce_sum(loss_sum) / total, "valid_tokens": int(total),
                "optimizer_steps": self.optimizer_steps, "applied_updates": self.applied_updates}

    def _train_window_step(self, batches, valid):
        """Same update as train_window, with one host read per optimizer step.

        Losses, finiteness and target counts stay on the device. The single read
        happens after backward and before optimizer.step(), so a nonfinite loss or
        gradient on any rank still stops training before the weights change.
        """
        total = self.all_reduce_async(self.device_value(sum(valid)))
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        loss_sum = self.device_value(0)
        finite = self.device_value(1)
        for index, (batch, count) in enumerate(zip(batches, valid)):
            boundary = index == len(batches) - 1
            if hasattr(self.model, "set_requires_gradient_sync"):
                self.model.set_requires_gradient_sync(boundary)
            loss = self.forward_loss(self.transfer(batch))
            finite = torch.minimum(finite, torch.isfinite(loss.detach()).double())
            loss_sum = loss_sum + loss.detach().double() * count
            self.backward(loss * (count * self.world_size / total).to(loss.dtype), boundary)
            self.microbatches += 1
        # As in the microbatch mode, gradients are inspected only when clipping is on.
        # The norm stays on the device; a nonfinite norm is caught by the read below.
        norm = self.device_value(0)
        if self.config.max_grad_norm:
            norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm,
                                                  foreach=False)
            if hasattr(norm, "full_tensor"):
                norm = norm.full_tensor()
        self.all_reduce_async(finite, dist.ReduceOp.MIN if dist.is_initialized() else None)
        self.all_reduce_async(loss_sum)
        grad_norm, all_finite, loss_total, targets = torch.stack(
            [norm.double(), finite, loss_sum, total]).tolist()
        if all_finite < 1 or not math.isfinite(grad_norm):
            raise FloatingPointError("Nonfinite causal loss or gradient norm on at least one rank")
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
        self.optimizer_steps += 1
        self.applied_updates += 1
        self.optimizer.zero_grad(set_to_none=True)
        return {"loss": loss_total / targets, "valid_tokens": int(targets), "grad_norm": grad_norm,
                "optimizer_steps": self.optimizer_steps, "applied_updates": self.applied_updates}

    @torch.no_grad()
    @dft_ai.pipeline.evaluate
    def evaluate(self, batches):
        previous = self.model.training
        self.model.eval()
        loss_sum, count = 0.0, 0
        try:
            for batch in batches:
                valid = int((batch["labels"][:, 1:] != -100).sum())
                loss_sum += self.forward_loss(self.transfer(batch)).float().item() * valid
                count += valid
            return self.reduce_sum(loss_sum) / self.reduce_sum(count)
        finally:
            self.model.train(previous)

    def progress(self):
        return {"microbatches": self.microbatches, "optimizer_steps": self.optimizer_steps,
                "applied_updates": self.applied_updates}

    def restore_progress(self, state):
        for key in self.progress():
            setattr(self, key, state[key])

    def checkpoint_state(self):
        from torch.distributed.checkpoint.state_dict import get_state_dict
        model, optimizer = get_state_dict(self.model, self.optimizer)
        return {"model": model, "optimizer": optimizer}

    def restore_checkpoint_state(self, state):
        from torch.distributed.checkpoint.state_dict import set_state_dict
        set_state_dict(self.model, self.optimizer, model_state_dict=state["model"],
                       optim_state_dict=state["optimizer"])


class FSDPRuntime(TorchRuntime):
    def __init__(self, config, context, model_config):
        from torch.distributed.device_mesh import init_device_mesh
        from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy

        sources = safetensor_sources(config)
        model = ModelFactory.create_pytorch_model(Model.LLAMA, config=config, meta=True,
                                                  model_config=model_config)
        mesh = init_device_mesh(context.device.type, (context.world_size,))
        policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16 if config.precision == "bf16" else torch.float32,
                                      reduce_dtype=torch.float32)
        for layer in model.model.layers:
            fully_shard(layer, mesh=mesh, mp_policy=policy)
        fully_shard(model, mesh=mesh, mp_policy=policy)
        model.to_empty(device=context.device)
        initialize_rotary_buffers(model, context.device)
        load_parameter_slices(model, sources, context.rank, context.world_size)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate,
                                      weight_decay=config.weight_decay, betas=(0.9, 0.95), foreach=False)
        super().__init__(model, optimizer, config, context.device, context.rank, context.world_size)


class DeepSpeedRuntime(TorchRuntime):
    def __init__(self, config, context, microbatch_size):
        import deepspeed
        from transformers.integrations import HfDeepSpeedConfig

        settings = deepspeed_settings(config, microbatch_size, context.world_size)
        # Register the already-created Torch group with DeepSpeed before HF's
        # ZeRO-3 loader parses a world-size-dependent DeepSpeed configuration.
        deepspeed.init_distributed(dist_backend=dist.get_backend(), dist_init_required=False)
        # HF stores a weak reference; keep this alive through model construction.
        self.hf_deepspeed_config = HfDeepSpeedConfig(settings)
        model = ModelFactory.create_pytorch_model(Model.LLAMA, config=config)
        engine, optimizer, _, scheduler = deepspeed.initialize(
            model=model, model_parameters=model.parameters(), config=settings,
            dist_init_required=False,
        )
        super().__init__(engine, optimizer, config, context.device, context.rank, context.world_size)
        self.scheduler = scheduler
        self.effective_settings = settings

    def train_window(self, batches):
        if len(batches) != self.config.gradient_accumulation_steps:
            raise ValueError("Incomplete DeepSpeed accumulation window")
        valid = [int((batch["labels"][:, 1:] != -100).sum()) for batch in batches]
        if not all(valid):
            raise ValueError("Every microbatch must contain a prediction target")
        if self.step_sync:
            return self._train_window_step(batches, valid)
        total = self.reduce_sum(sum(valid))
        self.model.train()
        loss_sum = 0.0
        skipped_before = self.model.skipped_steps
        steps_before = self.model.global_steps
        for index, (batch, count) in enumerate(zip(batches, valid)):
            boundary = self.model.is_gradient_accumulation_boundary()
            if boundary != (index == len(batches) - 1):
                raise RuntimeError("DLIO and DeepSpeed accumulation counters disagree")
            loss = self.forward_loss(self.transfer(batch))
            if self.reduce_sum(int(torch.isfinite(loss.detach()).item())) != self.world_size:
                raise FloatingPointError("Nonfinite causal loss on at least one rank")
            loss_sum += loss.detach().float().item() * count
            # DeepSpeed backward divides by accumulation once internally.
            scale = count * self.world_size * len(batches) / total
            self.backward(loss * scale, boundary)
            self.microbatches += 1
        boundaries = self.model.global_steps - steps_before
        if boundaries != 1:
            raise RuntimeError("Expected one DeepSpeed optimizer boundary per window")
        self.optimizer_steps += boundaries
        self.applied_updates += boundaries - (self.model.skipped_steps - skipped_before)
        self.synchronize()
        return {"loss": self.reduce_sum(loss_sum) / total, "valid_tokens": int(total),
                "optimizer_steps": self.optimizer_steps, "applied_updates": self.applied_updates}

    def _train_window_step(self, batches, valid):
        """One host read per optimizer step, taken before the engine applies the update.

        DeepSpeed clips inside engine.step(), so this checks the losses of every
        microbatch on every rank, as the microbatch mode does, just later.
        """
        total = self.all_reduce_async(self.device_value(sum(valid)))
        self.model.train()
        state = {"loss_sum": self.device_value(0), "finite": self.device_value(1)}
        skipped_before = self.model.skipped_steps
        steps_before = self.model.global_steps

        def check_before_update():
            self.all_reduce_async(state["finite"], dist.ReduceOp.MIN if dist.is_initialized() else None)
            self.all_reduce_async(state["loss_sum"])
            all_finite, loss_total, targets = torch.stack(
                [state["finite"], state["loss_sum"], total]).tolist()
            if all_finite < 1:
                raise FloatingPointError("Nonfinite causal loss on at least one rank")
            state.update(loss=loss_total / targets, valid_tokens=int(targets))

        for index, (batch, count) in enumerate(zip(batches, valid)):
            boundary = self.model.is_gradient_accumulation_boundary()
            if boundary != (index == len(batches) - 1):
                raise RuntimeError("DLIO and DeepSpeed accumulation counters disagree")
            loss = self.forward_loss(self.transfer(batch))
            state["finite"] = torch.minimum(state["finite"], torch.isfinite(loss.detach()).double())
            state["loss_sum"] = state["loss_sum"] + loss.detach().double() * count
            # DeepSpeed backward divides by accumulation once internally.
            scale = (count * self.world_size * len(batches) / total).to(loss.dtype)
            self.backward(loss * scale, boundary, check_before_update if boundary else None)
            self.microbatches += 1
        boundaries = self.model.global_steps - steps_before
        if boundaries != 1:
            raise RuntimeError("Expected one DeepSpeed optimizer boundary per window")
        self.optimizer_steps += boundaries
        self.applied_updates += boundaries - (self.model.skipped_steps - skipped_before)
        return {"loss": state["loss"], "valid_tokens": state["valid_tokens"],
                "optimizer_steps": self.optimizer_steps, "applied_updates": self.applied_updates}

    @dft_ai.compute.backward
    def backward(self, loss, boundary, before_update=None):
        self.model.backward(loss)
        if before_update is not None:
            before_update()
        self.model.step()
