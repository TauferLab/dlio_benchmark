# Verification of the Llama integration plan

The previous plan identified the broad goal correctly but was not implementation-ready. Static integration gaps, reference-script bugs, and unresolved scope were being treated too loosely. The [revised plan](llama_native_integration_plan.md) separates confirmed requirements from proposed choices and adds the missing implementation and acceptance work.

## Evidence and limits

- Local DLIO commit: `1035482c4ba390e0b485008e706863a79cbbd0a7`.
- Reference commit: [`6ed121ef9749e84f32e3462f74bab04ca288b13c`](https://github.com/OlgaKogiou/LLM-Checkpoints/tree/6ed121ef9749e84f32e3462f74bab04ca288b13c). Its tree is `360cc594c24411fe1e26dcb1fc33d0ec1f25771e`; the previous plan mislabeled the commit as a tree.
- Verified the cached bytes of all 15 inspected reference files against the GitHub tree's git-blob SHA1 values: training/data scripts, README, requirements, three DeepSpeed configs, aggregation rules, and launch scripts.
- Inspected versioned PyTorch/Transformers documentation, DeepSpeed 0.18.4 engine source, Transformers 4.57.3 source, six package metadata records, and public model/dataset metadata.
- Executed isolated source-function probes for the OpenHermes formatter and DeepSpeed boundary arithmetic. These reproduce specific logic errors; they are not distributed runtime tests.
- Installed packages here differ from the candidate stack: Python 3.13.2, Torch 2.7.0+rocm6.3, Transformers 4.49.0, Datasets 3.3.2. Accelerate, DeepSpeed, pydftracer, and TensorFlow are absent from this interpreter. No packages were installed and no multi-GPU training/checkpoint tests were run for this audit.

## DLIO findings

Paths and line numbers below refer to the inspected local commit and are under `dlio_benchmark/` unless stated otherwise.

| Previous assumption or omission | Verified evidence | Required correction |
| --- | --- | --- |
| Llama I/O implies native Llama training already exists | `model/model_factory.py` implements ResNet/UNet; `llama_8b_zero3.yaml` describes synthetic workload state. | Reuse lifecycle and instrumentation, but add the actual Llama compute/runtime path. |
| Existing checkpoint adapter can save/restore the live model | `checkpointing/base_checkpointing.py:47` allocates synthetic state. `checkpointing/pytorch_checkpointing.py:135` assigns loaded data to a local variable, not the live model or optimizer. | A lightweight native checkpoint interface must receive actual runtime state. Avoid allocating a second synthetic model-sized payload. |
| Validation happens before model creation | `main.py:80` constructs the framework before tracer setup around line 97 and full validation around line 254. `framework/torch_framework.py:86` immediately constructs and moves a model. | Defer heavyweight initialization; make rank, package, and config failures occur first. |
| FSDP/DeepSpeed can use the existing DDP setup unchanged | `framework/torch_framework.py:86` orders full-device placement, DDP, optimizer, then tracing. | Backends own placement, sharding and optimizer initialization; process-group setup needs separation from DDP. |
| Indexed data already contains usable Llama inputs | `data_generator/indexed_binary_generator.py:99` writes random uint8 bytes. `reader/reader_handler.py:120` and `reader/indexed_binary_mmap_reader.py:102` return `resized_image`. | Add real token records and returned token tensors; test disk-to-model identity. |
| A custom HF loader is sufficient | `main.py:180` still discovers DLIO-format files; `main.py:235` aligns a file count using a samples-per-step quantity. | Add dataset-specific sample accounting and bypass incompatible file assumptions for native data. |
| Existing sampler length and epoch loop provide resume | `data_loader/torch_data_loader.py:98` reports a global length despite local indices. `main.py:344` resets counters per epoch and schedules checkpoints from batches. | Define local lengths, epoch reseeding, consumed cursor, global counters, and accumulation-aware scheduling. |
| Native evaluation already supplies a correct template | `main.py:280` calls `framework.compute`; `framework/torch_framework.py:237` routes a real batch to `_train_batch`. | Add explicit inference-only evaluation and parameter/optimizer invariance tests. |
| Tracing and distributed compilation are interchangeable | `framework/torch_framework.py:183` enables Dynamo compilation when DFTracer is enabled. | Separate event tracing from backend compilation support. |
| Existing checkpoint metrics and flags transfer directly | `main.py:259` caches synthetic size; `utils/statscounter.py:355,372` divides it by local elapsed time. `utils/config.py:1117,1125` writes flag names different from the fields read by checkpointing. | Native adapters need explicit metrics and supported durability semantics; do not silently inherit ineffective flags. |
| PyTorch is optional only under `cuda` | Repository `setup.py:22` requires `pydftracer[dynamo]`, whose 2.0.3 metadata requires `torch>=2.5.1`. | Keep all new Llama dependencies optional and describe existing packaging accurately. A PyTorch-free base is separate scope. |

## Reference branch and library findings

The following source links are pinned so branch movement does not invalidate the audit.

| Finding | Consequence for integration |
| --- | --- |
| The script defaults to Llama-3.2-1B; `ds_config_zero3.json` actually selects stage 2, and `ds_config_zero3_ray_1node.json` stage 0. | Neither branch nor file names specify the requested 8B configuration. Create explicitly validated settings. [Reference tree](https://github.com/OlgaKogiou/LLM-Checkpoints/tree/6ed121ef9749e84f32e3462f74bab04ca288b13c). |
| OpenHermes uses `conversations`; `normalize_example_text` handles other fields and otherwise calls `str(example)`. | Implement intentional conversation formatting. An isolated probe with a schema-shaped record included its metadata in the training string. [Script formatter](https://github.com/OlgaKogiou/LLM-Checkpoints/blob/6ed121ef9749e84f32e3462f74bab04ca288b13c/train_llama3.py#L57), [dataset schema](https://datasets-server.huggingface.co/info?dataset=teknium%2FOpenHermes-2.5). |
| PAD is set equal to EOS; the pinned HF causal collator masks every token equal to PAD. | Real EOS targets are masked too. Mask padding positions rather than all equal token values. [Collator implementation](https://github.com/huggingface/transformers/blob/v4.57.3/src/transformers/data/data_collator.py#L1049). |
| Packing collects the entire token stream in one Python list; cache identity omits revisions and formatting version. | Use bounded preprocessing, complete cache identity, and explicit truncation/remainder policy. [Packing/cache code](https://github.com/OlgaKogiou/LLM-Checkpoints/blob/6ed121ef9749e84f32e3462f74bab04ca288b13c/train_llama3.py#L79). |
| The script checks accumulation boundaries after `engine.step()`. DeepSpeed increments `micro_steps` inside that step. | With accumulation 2, the branch observes boundaries after microbatches 1 and 3, while actual update boundaries are 2 and 4. Check before stepping or use a validated engine-result contract. [Training loop](https://github.com/OlgaKogiou/LLM-Checkpoints/blob/6ed121ef9749e84f32e3462f74bab04ca288b13c/train_llama3.py#L376), [engine boundary](https://github.com/deepspeedai/DeepSpeed/blob/v0.18.4/deepspeed/runtime/engine.py#L2517). |
| DeepSpeed increments `global_steps` even when an overflow skips an update. | Keep attempted boundaries and applied updates distinct. [Engine step accounting](https://github.com/deepspeedai/DeepSpeed/blob/v0.18.4/deepspeed/runtime/engine.py#L2614). |
| Resume ignores the returned load path, lacks complete RNG/sampler state, and final save omits client progress metadata. | Loading weights alone is not verified training continuation. Add fresh-process continuity tests, including final saves. [Script](https://github.com/OlgaKogiou/LLM-Checkpoints/blob/6ed121ef9749e84f32e3462f74bab04ca288b13c/train_llama3.py#L265). |
| The reference loads weights before DeepSpeed setup and has no `HfDeepSpeedConfig`. | Stage 3 needs a different initialization recipe. That HF helper also checks Accelerate, which the previous plan omitted from the proposed extra. [HF helper](https://github.com/huggingface/transformers/blob/v4.57.3/src/transformers/integrations/deepspeed.py#L75). |
| FSDP was an unspecified wrapper/API in the draft. | FSDP1 and FSDP2 state/loading contracts differ. Choose FSDP2 explicitly, then prove pretrained load, buffers, optimizer initialization, and DCP restart together. [FSDP2 documentation](https://docs.pytorch.org/docs/2.9/distributed.fsdp.fully_shard.html). |
| FlashAttention is hardcoded; README asks for a special DFTracer branch and a Torch/ROCm combination absent from the official 2.9.1 wheel recipe. | These are separate build/platform qualifications, not portable installation defaults. [Reference README](https://github.com/OlgaKogiou/LLM-Checkpoints/blob/6ed121ef9749e84f32e3462f74bab04ca288b13c/README.md), [official wheel matrix](https://pytorch.org/get-started/previous-versions/). |

## Reproduced accumulation result

This probe executed the actual DeepSpeed 0.18.4 boundary method with accumulation 2 and a stub counter, advancing it as `step()` does. It did not instantiate the engine.

| Microbatch | Boundary before step | Reference check after step |
| --- | --- | --- |
| 1 | False | True |
| 2 | True | False |
| 3 | False | True |
| 4 | True | False |

The reference configurations currently use accumulation 1, which conceals this error. A configurable DLIO integration needs a test greater than 1.

## What remains unverified

The revised design is grounded in source, but joint package compatibility, native pretrained sharding, resume equivalence, multi-node launch behavior, memory capacity, filesystem durability, and tracing aggregation require implementation and runtime qualification. User choices still outstanding are listed in the revised plan; they must not be presented as settled requirements.
