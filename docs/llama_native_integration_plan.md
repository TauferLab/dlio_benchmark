# Native Llama training in DLIO: audited implementation plan

This design has been implemented in the working tree. See the [usage guide](source/llama.rst) and [implementation validation](llama_implementation_validation.md) for the resulting interfaces, test evidence, and remaining GPU qualification. The [original audit](llama_integration_audit.md) records the baseline evidence and corrections. DLIO was inspected at commit `1035482c4ba390e0b485008e706863a79cbbd0a7`; the reference branch was verified at commit `6ed121ef9749e84f32e3462f74bab04ca288b13c` (a commit ID, not a tree ID).

## Scope and decisions

The user confirmed real training with pretrained `meta-llama/Llama-3.1-8B`, multiple GPUs, selectable FSDP and DeepSpeed backends, both DLIO-generated tokens and OpenHermes data, and save/resume within each backend's own checkpoint format. The integration should participate in DLIO's native-model workflow like ResNet/UNet. New dependencies must require an explicit extra; ordinary installation must not add them.

The final choices and implementation boundaries are:

| Item | Treatment in this plan |
| --- | --- |
| DeepSpeed stage | User selected both stages 2 and 3; the example explicitly uses stage 3. |
| FSDP API | FSDP2 with `fully_shard`, meta construction and direct safetensors shard loading. Tiny-model loading and restart are tested; full 8B GPU qualification remains. |
| Checkpoint support boundary | Synchronous saves on shared POSIX storage, with same-world-size and exact recorded stack resume initially. Other storage, changed rank counts, and asynchronous saves need separate acceptance tests. |
| OpenHermes training objective | User selected both full-sequence and response-only loss, explicitly configured. `role_lines_v1` formats conversations without assuming an Instruct chat template. |
| Tuning | Precision, optimizer/state dtypes, sequence length, microbatch size, accumulation, activation checkpointing, and offload remain explicit workload choices. Multiple GPUs alone do not establish memory fit. |
| Branch tracing modes | User selected ordinary DLIO tracing and documentation of FULL/SELECTIVE aggregation for possible future implementation. |

ZeRO stage 2 partitions optimizer state and gradients; stage 3 additionally partitions parameters. For BF16 weights alone, this model has about 16.06 GB of parameters before other memory. Stage 2 keeps a full parameter replica per data-parallel GPU, while stage 3 gathers parameters as needed. Neither describes total training memory or implies an equivalent reduction in aggregate checkpoint bytes. Offload is an independent setting. [DeepSpeed configuration](https://www.deepspeed.ai/docs/config-json/), [model metadata](https://huggingface.co/api/models/meta-llama/Llama-3.1-8B).

## 1. Define configuration and initialization boundaries

Change `common/enumerations.py`, `utils/config.py`, `model/model_factory.py`, `framework/torch_framework.py`, and `main.py` together (paths under `dlio_benchmark/`). Register a native Llama model distinct from existing synthetic `llama_*` workloads. Model construction must receive validated configuration; merely adding an enum and a factory case is insufficient.

Define explicit fields for backend, model/tokenizer source and revision, data source, formatting, precision, optimizer, accumulation, checkpoint/restart path, and optional tracing. Resolve local weights and tokenizer assets or check access to the gated model before distributed construction. Record resolved revisions and effective settings. Preparation/download time must be distinguishable from timed training I/O.

Split cheap configuration validation from dataset-dependent validation. Initialize logging/tracing and validate optional packages, ranks, visible devices, backend settings, and storage compatibility before allocating weights. Preserve lightweight generation-only and synthetic runs. Do not let the existing constructor's full-model `.to(device)` or DDP wrapping run before native sharding.

Use one distributed context with consistent MPI and PyTorch global rank/world size. Validate local device selection under both all-visible and launcher-masked GPUs. Prevent separate launchers from creating mismatched MPI and Torch worlds. Refactor process-group initialization out of the DDP wrapper; preserve ownership so cleanup destroys only groups DLIO creates. Establish finite startup failure handling and perform checkpoint cleanup before distributed teardown.

## 2. Keep installation optional and platform-aware

The distribution is named `dlio_benchmark`. Installation commands are:

```sh
pip install dlio_benchmark
pip install 'dlio_benchmark[llama]'
pip install 'dlio_benchmark[llama-deepspeed]'
```

These extras are implemented in `setup.py`. Shared Llama requirements are the selected PyTorch constraint, Transformers, and Datasets. `llama-deepspeed` includes those requirements plus DeepSpeed and Accelerate, without also selecting `llama`. Accelerate is used by the Transformers/ZeRO-3 integration. FlashAttention remains a separate explicit installation and configuration choice. Unused TRL/PEFT dependencies from the reference requirements are excluded.

Do not add these packages, stricter PyTorch constraints, or optional requirement-file includes to base requirements. Imports in model, loader, checkpoint, and config registration must stay lazy, with actionable missing-extra errors. Keep model/dataset download out of installation and import paths. Embed DeepSpeed configuration in packaged YAML, or explicitly include JSON files in both wheel and sdist.

DLIO already requires `pydftracer[dynamo]>=2.0.3`; that extra requires PyTorch. Thus PyTorch is already a transitive base dependency, despite also appearing under `cuda`. This plan isolates new Llama requirements; making the existing base installation PyTorch-free is a separate change. [Local packaging](../setup.py), [DFTracer 2.0.3 metadata](https://pypi.org/pypi/pydftracer/2.0.3/json).

PyTorch 2.9.1, Transformers 4.57.3, Datasets 4.4.2, DeepSpeed 0.18.4, and Accelerate 1.12.0 form the CPU test baseline. This does not establish GPU kernel or 8B compatibility. PyTorch/Accelerate metadata requires Python 3.10 or newer; actual supported Python/platform combinations need installation tests. Select the platform's PyTorch wheel before extras installation, and verify the resolver preserves it. The reference's `2.9.1+rocm6.3` command is not the official 2.9.1 recipe; official instructions list ROCm 6.4. [PyTorch wheel instructions](https://pytorch.org/get-started/previous-versions/).

## 3. Implement real token data and sample accounting

Add a token-specific generator/reader mode with a versioned record manifest. A proposed fixed-length representation is little-endian int32 token IDs, with documented byte offsets and lengths, converted to `torch.long` batches. Derive vocabulary limits from the selected model/tokenizer and validate every record. Keep generated bytes deterministic from a recorded seed and sample identity. Existing indexed-binary generators write arbitrary uint8 bytes, and their readers return placeholder image arrays; changing the model's input cast cannot fix this.

For OpenHermes, explicitly map its `conversations` records (`from`/`value`) through the chosen formatter. Define special tokens, EOS, truncation, packing, remainder handling, and loss masking. Use bounded-memory preprocessing and a cache key covering dataset/tokenizer revisions, formatter version, and preprocessing options. Publish completed caches safely and propagate preparation failures. The audited dataset exposes only a training split: evaluation needs an explicit supplied split or reproducible held-out partition. [Dataset schema](https://datasets-server.huggingface.co/info?dataset=teknium%2FOpenHermes-2.5).

Both paths must yield actual `input_ids`, `attention_mask`, and `labels`. Preserve integer types, mask padding positions using the attention mask, and preserve genuine EOS targets even when PAD equals EOS. Let the causal-LM implementation shift targets exactly once. Require valid prediction targets; reject all-masked batches. If response-only loss is selected, combine its mask with padding explicitly.

Introduce a loader description containing global/local sample counts, sequence lengths, and resumable ordering. Adapt `main.py`, configuration derivation, and `StatsCounter` to use it. A Hugging Face loader cannot bypass current file discovery merely by being registered as a custom loader. Correct native sample accounting instead of using the existing file-count alignment formula.

DLIO owns sampling for both backends; do not pass a second training dataset to DeepSpeed to create another sampler. Implement epoch reseeding, equal collective step counts, explicit tail handling, and a consumed-sample cursor independent of worker prefetch. Read cached OpenHermes data through the loader's instrumented fetch path; report actual file reads separately from samples served from memory/mmap.

## 4. Introduce a backend-owned training interface

Add a runtime interface for initialization, one training microbatch, evaluation, checkpoint state, restoration, and teardown. Return loss and explicit progress information to DLIO. Keep synthetic sleep compute and existing image-model batch handling separate. Native evaluation must enter evaluation mode and disable gradients without changing parameters or optimizer state; the current PyTorch evaluation call goes through the training step.

For FSDP2, shard Llama decoder layers and then the root, and construct the optimizer afterward. Use the model call so sharding hooks execute. Prototype meta initialization, parameter loading, and materialization of nonpersistent buffers against a tiny pretrained Llama before scaling up. Compare weights and outputs with an unsharded reference and measure peak memory. Declare any full CPU-copy requirement; do not first construct a complete GPU model on every rank. A pretrained Hugging Face safetensors load is a separate operation from DCP restart. [FSDP2 contract](https://docs.pytorch.org/docs/2.9/distributed.fsdp.fully_shard.html).

For DeepSpeed ZeRO-3, create and retain `HfDeepSpeedConfig` before `from_pretrained`. Initialize one engine and let it own optimizer/backward/step behavior. Resolve batch settings explicitly from microbatch size, accumulation, and data-parallel world size; native integration cannot rely on Trainer resolving `"auto"` values. Stage 2 needs its own validated initialization path. Neither backend receives an additional DDP wrapper. [Transformers native DeepSpeed integration](https://huggingface.co/docs/transformers/v4.57.3/en/deepspeed).

Define distinct counters for consumed microbatches, optimizer boundaries, and applied updates, including overflow-skipped updates. Use globally monotonic checkpoint IDs and store all counters. Query DeepSpeed's accumulation boundary before `step()` advances it; do not assume `global_steps` counts only successful updates. FSDP accumulation must preserve gradients across microbatches, normalize loss consistently, and control synchronization intentionally. Define token weighting across unequal-length batches and ranks. Checkpoint only after a completed accumulation window; define how epoch/step limits handle incomplete windows.

Remove the synthetic per-batch allreduce-simulation barrier from the native path, or expose and record deliberate benchmark synchronization. Use synchronized device timing where needed. Make activation checkpointing, attention implementation, and precision policies explicit and tested; distinguish activation recomputation from persistent checkpoints.

## 5. Connect live checkpoint state to DLIO

Refactor checkpoint dispatch into a lightweight interface accepting the live runtime and run state. Do not instantiate the current `BaseCheckpointing` synthetic tensor allocator for native training. Adapt `checkpointing_factory.py`, custom-mechanism validation, and `main.py`; retain synthetic checkpoint behavior behind its existing implementation.

Use FSDP2-compatible distributed model/optimizer state APIs with synchronous DCP save/load. Use DeepSpeed engine save/load for its backend, with all required ranks participating and the returned load path checked. A ZeRO-3 checkpoint saved by an engine cannot simply be loaded straight back into that partitioned engine; test restart through a newly initialized engine. [DCP](https://docs.pytorch.org/docs/2.9/distributed.checkpoint.html), [DeepSpeed load semantics](https://github.com/deepspeedai/DeepSpeed/blob/v0.18.4/deepspeed/runtime/engine.py#L3287).

Save model, optimizer, scheduler/scaler if present, epoch, progress counters, consumed-data cursor, sampler state, and RNG state. Preserve rank-local RNG with distinct rank keys or sidecars, not one replicated key that checkpoint deduplication may discard. Record backend/stage, world size, package versions, model/tokenizer/data identity, configuration, and schema version. Final checkpoints need the same resume metadata as periodic ones. Restore data ordering and RNG before the next training operation, accounting for iterator initialization and worker RNG.

Publish a completion manifest only after required writes and participants succeed. Resume must reject incomplete, incompatible, synthetic, or other-backend checkpoints. Specify manifest contents, durability/fsync policy, and partial-checkpoint cleanup for the supported filesystem; a marker alone is not proof of crash durability. Do not inherit simulated rank-shift or synthetic tensor options as native features. Version compatibility is an explicit limit: DCP does not guarantee checkpoint compatibility across PyTorch versions.

## 6. Measure I/O and validate tracing separately

Replace fixed synthetic checkpoint-size estimates for native saves with per-save manifests of unique files/bytes. Report global stored bytes, per-rank contributions where attributable, save/load duration, and global throughput based on collective completion time. Use explicit bytes/GiB units. Distinguish logical state size, stored file size, observed read/write traffic, and durable persistence; mmap and caching make those different quantities.

Reuse DLIO events for fetch, preprocessing, transfer, forward, backward, save, and restore. Separate coarse instrumentation from the existing automatic Dynamo `torch.compile` path; do not compile FSDP/DeepSpeed merely because tracing is enabled. Test compilation separately before advertising it. Validate branch aggregation rules against actual DLIO event names and the installed DFTracer build. Attribute any configured offload I/O separately.

## 7. Acceptance gates before claiming support

| Gate | Required evidence |
| --- | --- |
| Packaging | Inspect built wheel/sdist metadata; clean resolver/install checks for base and both extras; base import/CLI succeeds without new optional packages; platform wheel remains selected. |
| Data | Known token fixture survives generated file -> reader -> model unchanged; vocabulary/boundary checks; OpenHermes schema/formatter fixture; real EOS preserved; padding masked; predictable tail and packing behavior. |
| Training | Tiny pretrained model matches the selected loading recipe; finite loss and parameter updates; accumulation greater than one; overflow accounting where applicable; evaluation leaves all training state unchanged. |
| Distributed | At least two ranks for FSDP2 and each advertised ZeRO stage; test rank/device mapping, uneven data, failure paths, and teardown. No double sharding or duplicated optimizer step. |
| Restart | In a fresh process, compare uninterrupted versus save/restart runs using the same data and stack. Check subsequent sample IDs, counters, optimizer/scheduler/RNG state, loss and parameters within justified numerical tolerances. Include mid-epoch and final saves. |
| Checkpoint failures | Incomplete/missing shards and incompatible metadata are rejected; all-rank participation succeeds; completion publication and actual byte accounting agree. |
| Regression | Run native ResNet/UNet, framework, communication, loader, and synthetic Llama checkpoint tests relevant to changed interfaces. |
| 8B qualification | Actual pretrained 8B, multiple GPUs, both input paths and every advertised backend/stage: train, save, stop, restart, continue. Record hardware/software, memory peaks and I/O. Tiny-model success does not establish 8B capacity or throughput. |

The sections above retain the design and acceptance gates for review. The implementation validation report distinguishes completed local checks from GPU, full-size model, and platform qualification that remains open. FULL/SELECTIVE aggregation is explicitly outside this implementation.
