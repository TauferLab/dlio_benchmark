"""Validated native configuration, independent of optional ML packages."""

from dataclasses import asdict, dataclass, field, fields
import importlib.util
from importlib.metadata import version
import re
import sys


@dataclass(frozen=True)
class LlamaConfig:
    backend: str
    data_source: str
    model_name_or_path: str = "meta-llama/Llama-3.1-8B"
    model_revision: str = "main"
    tokenizer_name_or_path: str = ""
    tokenizer_revision: str = "main"
    local_files_only: bool = False
    sequence_length: int = 128
    precision: str = "bf16"
    attention_implementation: str = "sdpa"
    gradient_accumulation_steps: int = 1
    learning_rate: float = 2e-5
    weight_decay: float = 0.1
    max_grad_norm: float = 1.0
    activation_checkpointing: bool = False
    zero_stage: int = 0
    deepspeed_config: dict = field(default_factory=dict)
    generated_samples: int = 128
    generated_eval_samples: int = 0
    samples_per_file: int = 64
    dataset_name_or_path: str = "teknium/OpenHermes-2.5"
    dataset_revision: str = "main"
    dataset_split: str = "train"
    eval_split: str = ""
    max_samples: int = 0
    conversation_format: str = ""
    loss: str = ""
    packing: bool = False
    packing_remainder: str = "pad"
    resume_from: str = ""
    save_final: bool = True
    distributed_timeout_seconds: int = 300
    # microbatch: check every microbatch loss across ranks, blocking copies and
    # window-level synchronization. step: one host read per optimizer step,
    # before the update, so the host can queue work ahead of the GPU.
    host_sync: str = "microbatch"

    @classmethod
    def from_mapping(cls, mapping):
        values = dict(mapping)
        unknown = set(values) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown workload.llama settings: {sorted(unknown)}")
        config = cls(**values)
        config.validate()
        return config

    def validate(self):
        if self.backend not in {"fsdp", "deepspeed"}:
            raise ValueError("llama.backend must be fsdp or deepspeed")
        if self.data_source not in {"dlio_tokens", "openhermes"}:
            raise ValueError("llama.data_source must be dlio_tokens or openhermes")
        if self.precision not in {"fp32", "bf16"}:
            raise ValueError("llama.precision must be fp32 or bf16")
        if self.attention_implementation not in {"eager", "sdpa", "flash_attention_2"}:
            raise ValueError("Unsupported Llama attention implementation")
        for name in ("sequence_length", "gradient_accumulation_steps", "samples_per_file",
                     "distributed_timeout_seconds"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"llama.{name} must be a positive integer")
        if self.sequence_length < 2:
            raise ValueError("At least two tokens are needed for causal prediction")
        for name in ("generated_samples", "generated_eval_samples", "max_samples"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError(f"llama.{name} must be a nonnegative integer")
        if self.learning_rate <= 0 or self.weight_decay < 0 or self.max_grad_norm < 0:
            raise ValueError("Invalid Llama optimizer settings")
        if self.backend == "deepspeed" and self.zero_stage not in {2, 3}:
            raise ValueError("Select llama.zero_stage explicitly: 2 or 3")
        if self.backend == "fsdp" and (self.zero_stage or self.deepspeed_config):
            raise ValueError("DeepSpeed settings cannot be applied to FSDP")
        if self.data_source == "openhermes":
            if self.conversation_format != "role_lines_v1":
                raise ValueError("Select conversation_format=role_lines_v1 explicitly")
            if self.loss not in {"full", "response_only"}:
                raise ValueError("Select OpenHermes loss=full or response_only explicitly")
        elif self.loss not in {"", "full"}:
            raise ValueError("Generated token data supports full causal loss")
        if self.packing_remainder not in {"pad", "drop"}:
            raise ValueError("packing_remainder must be pad or drop")
        if self.host_sync not in {"microbatch", "step"}:
            raise ValueError("llama.host_sync must be microbatch or step")

    def to_dict(self):
        return asdict(self)


def check_dependencies(config, training=True):
    """Check availability before any model/network work; imports remain lazy."""
    extra = "llama-deepspeed" if config.backend == "deepspeed" else "llama"
    packages = ["torch", "transformers", "datasets"]
    if training and config.backend == "deepspeed":
        packages += ["deepspeed", "accelerate"]
    if config.attention_implementation == "flash_attention_2" and training:
        packages += ["flash_attn"]
    missing = [p for p in packages if importlib.util.find_spec(p) is None]
    if missing:
        raise ImportError(
            f"Missing Llama packages: {', '.join(missing)}. Install separately with "
            f"pip install 'dlio_benchmark[{extra}]'. FlashAttention, if selected, "
            "requires its own platform-specific installation."
        )
    if sys.version_info < (3, 10):
        raise RuntimeError("The native Llama stack requires Python >=3.10")
    if training:
        minimum = {"torch": (2, 9, 1), "transformers": (4, 57, 3), "datasets": (4, 4, 2)}
        if config.backend == "deepspeed":
            minimum.update({"deepspeed": (0, 18, 4), "accelerate": (1, 12, 0)})
        for package, required in minimum.items():
            found = tuple(int(x) for x in re.match(r"(\d+)\.(\d+)\.(\d+)", version(package)).groups())
            if found < required or found[:2] != required[:2]:
                upper = f"{required[0]}.{required[1] + 1}"
                raise RuntimeError(f"Native Llama requires {package}>={'.'.join(map(str, required))},<{upper}; "
                                   f"install 'dlio_benchmark[{extra}]' separately")
