"""Load a pretrained causal Llama only after its backend context is ready."""


def load_pretrained_llama(config, *, device=None, meta=False, model_config=None):
    import torch
    from transformers import AutoModelForCausalLM

    # FSDP retains FP32 optimizer/master parameters and casts computation via
    # its mixed-precision policy. DeepSpeed owns its own master-weight policy.
    dtype = torch.float32
    if config.backend == "deepspeed" and config.precision == "bf16":
        dtype = torch.bfloat16
    if meta:
        with torch.device("meta"):
            model = AutoModelForCausalLM.from_config(
                model_config, torch_dtype=dtype,
                attn_implementation=config.attention_implementation,
            )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            config.model_name_or_path,
            revision=config.model_revision,
            local_files_only=config.local_files_only,
            torch_dtype=dtype,
            attn_implementation=config.attention_implementation,
        )
    if model.config.model_type != "llama":
        raise ValueError("Native llama workloads require a pretrained Llama architecture")
    model.config.use_cache = False
    if config.activation_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if device is not None:
        model.to(device)
    return model
