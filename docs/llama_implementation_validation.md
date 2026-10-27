# Native Llama implementation validation

The working tree implements real pretrained Llama training through DLIO's normal entry point. It includes FSDP2, DeepSpeed ZeRO 2/3, generated token records, OpenHermes full/response-only loss, evaluation, synchronous shared-POSIX save/resume, native token/I/O metrics, and existing DLIO tracing. The [usage guide](source/llama.rst) documents configuration and limitations. FULL/SELECTIVE DFTracer aggregation is documented as future work.

## Local evidence

Validation used an isolated `/tmp/dlio-llama-test-env` environment, leaving the shared Python installation unchanged: Python 3.13.2, Torch 2.9.1+cpu, Transformers 4.57.3, Datasets 4.4.2, DeepSpeed 0.18.4, Accelerate 1.12.0, and pydftracer 2.0.3. `pip check` reported no broken requirements. No GPUs were visible.

| Check | Result |
| --- | --- |
| Local native tests (`pytest tests/llama_test.py -k 'not fresh_process'`) | 18 passed. Includes real token bytes/labels, deterministic generated vocabulary IDs, OpenHermes packing and both losses, accumulation normalization, evaluation state preservation, pretrained safetensors/rotary restoration, checkpoint rejection and deterministic continuation, base optional-import isolation, and DLIO lifecycle/epoch-boundary restart. |
| Fresh-process backend tests (`pytest tests/llama_test.py -k fresh_process`) | 7 passed: shared Torch/DCP FP32 plus two-rank FSDP2 and DeepSpeed ZeRO 2/3, each in FP32 and BF16, with activation checkpointing. Subsequent sample IDs, parameters, loss, progress and RNG match uninterrupted continuation. FSDP initial full state also matches the HF snapshot exactly. |
| Existing factory/framework/communication unit tests | 36 passed, 16 deselected. PyTorch ResNet/UNet factory, input, update and communication ownership tests passed. |
| Existing PyTorch benchmark workflows and synthetic loader | 3 passed, 3 skipped: real ResNet/UNet single-process benchmark updates and synthetic loader passed; two MPI benchmark cases require a multi-rank invocation and MPI/Gloo smoke skipped because `mpirun` is unavailable. |
| Packaging | Wheel and sdist build successfully. Both contain the new code and all three workload configurations. Wheel metadata puts Transformers, Datasets, DeepSpeed and Accelerate only behind explicit extras; base requirements are unchanged. |
| Configuration and static checks | All three Hydra examples compose; Python compilation and `git diff --check` pass. |

The 25 distinct native tests were checked as the local subset and seven backend cases. Tests use local tiny snapshots and fixtures, without downloading gated 8B weights or the full OpenHermes corpus. DeepSpeed CPU qualification disables only its optional shared-memory C++ accelerator, retaining real engines, ZeRO optimizers and Torch/Gloo collectives. The host compiler is too old for that optional extension. A host C++ library-path adjustment was needed for the inherited SciPy/sklearn installation; this is test-environment setup, not a package requirement.

The regression command selected these files:

```sh
pytest tests/unit_model_factory.py tests/pytorch_framework_test.py \
  tests/native_framework_batch_test.py tests/config_communication_test.py \
  tests/compute_framework_test.py tests/communication_test.py -q \
  -k 'not mpi_gloo_sync and not mpi_nccl_sync and not tensorflow and not indexed_binary_generator_lookup'
pytest tests/pytorch_benchmark_test.py tests/synthetic_data_loader_test.py \
  tests/communication_test.py::test_mpi_gloo_sync -q
```

TensorFlow is absent from this test environment. An initial broader regression selection had one failure in an existing indexed-binary generator lookup test because its default framework imports TensorFlow; that environment-dependent test and TensorFlow cases were excluded from the final selection. NCCL tests and MPI-launched image benchmarks remain unqualified here. Base optional-import isolation blocks all new Llama packages during a fresh Python import; a pristine full base dependency installation was not performed.

## Remaining qualification

Run the fresh-process cases on the intended CUDA/ROCm stack with at least two GPUs:

```sh
DLIO_LLAMA_TEST_DEVICE=cuda DFTRACER_ENABLE=0 OMP_NUM_THREADS=1 \
  pytest tests/llama_test.py -q -k fresh_process
```

Then run pretrained `meta-llama/Llama-3.1-8B` train/save/stop/restart/continue through an MPI launcher for FSDP, ZeRO 2 and ZeRO 3, with both input paths. Record peak memory and actual storage behavior. CPU tiny-model checks do not prove GPU kernel compatibility, full-model memory fit, multi-node behavior, filesystem crash durability, or reference aggregation compatibility. Same-backend/stage, same-world-size and exact recorded stack resume are the current compatibility boundary.
