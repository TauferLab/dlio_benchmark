Native Llama training
====================

``workload.model.name=llama`` selects real pretrained causal-LM training through
DLIO's normal command and lifecycle. The existing ``llama_8b_zero3`` and other
synthetic checkpoint workloads remain separate. FSDP2 and DeepSpeed ZeRO stages
2 and 3 are selectable per run. Each backend restores its own checkpoints.

Installation
------------

Install the platform's PyTorch 2.9.1 wheel separately, following the official
`PyTorch installation recipes <https://pytorch.org/get-started/previous-versions/>`_.
If torchvision or torchaudio is already installed, align its version and wheel
build with the selected PyTorch stack; the Llama extras do not add these packages.
Then explicitly install the optional packages:

.. code-block:: sh

   pip install 'dlio_benchmark[llama]'
   # Includes all llama requirements plus DeepSpeed and Accelerate:
   pip install 'dlio_benchmark[llama-deepspeed]'

The extras constrain Torch 2.9.x, Transformers 4.57.x, Datasets 4.4.x,
DeepSpeed 0.18.x and Accelerate 1.12.x. The implementation baseline is
Torch 2.9.1 / Transformers 4.57.3 / Datasets 4.4.2 / DeepSpeed 0.18.4 /
Accelerate 1.12.0, with Python 3.10 or newer. GPU builds and kernels still need
qualification on the intended platform. FlashAttention is a separate optional
build; select it only after installing it for the chosen PyTorch/GPU stack.
The examples use PyTorch SDPA instead.

Plain ``pip install dlio_benchmark`` does not add Transformers, Datasets,
DeepSpeed or Accelerate. Existing base ``pydftracer[dynamo]`` already depends on
PyTorch; these changes do not make the base installation PyTorch-free.
Model weights and datasets are downloaded only during explicitly selected
preparation/training, never during package installation or import.

Launching and configuration
---------------------------

Use one MPI-aware launcher with one process per GPU. All ranks must see the same
POSIX token-cache and checkpoint paths. DLIO checks that any existing Torch rank
environment agrees with MPI; nesting an independent DeepSpeed/torchrun launcher
inside MPI is unnecessary. The Torch process group uses NCCL, including on ROCm.
Set ``MASTER_ADDR`` and ``MASTER_PORT`` when the environment needs explicit
rendezvous addressing.

.. code-block:: sh

   mpirun -np 4 dlio_benchmark workload=llama_native_fsdp
   mpirun -np 4 dlio_benchmark workload=llama_native_deepspeed
   mpirun -np 4 dlio_benchmark workload=llama_native_deepspeed workload.llama.zero_stage=2
   mpirun -np 4 dlio_benchmark workload=llama_native_openhermes
   mpirun -np 4 dlio_benchmark workload=llama_native_openhermes workload.llama.loss=response_only

The examples target pretrained ``meta-llama/Llama-3.1-8B``, at a pinned model
revision. Obtain access to this gated model through Hugging Face beforehand,
or provide a local snapshot with ``workload.llama.model_name_or_path=/shared/model``
and ``workload.llama.local_files_only=true``. A separate tokenizer directory can
be selected with ``tokenizer_name_or_path`` and ``tokenizer_revision``. Local
OpenHermes inputs must be saved with Hugging Face ``save_to_disk``; select their
directory with ``dataset_name_or_path``. A DatasetDict is needed for distinct
training/evaluation splits.

Microbatch size is ``workload.reader.batch_size``. Accumulation is
``workload.llama.gradient_accumulation_steps``. Global effective batch size is
their product with the data-parallel world size. Sequence length, precision,
attention implementation, activation checkpointing and optimizer settings are
explicit Llama options. These examples do not guarantee 8B memory fit on an
arbitrary collection of GPUs.

FSDP2 creates a meta model, shards decoder layers and the root, materializes
rotary buffers, then reads each rank's parameter slices from HF safetensors.
It keeps FP32 optimizer/master parameters and uses BF16 compute when selected.
It does not construct a complete GPU model before sharding. Single-file and
indexed safetensors snapshots are supported; PyTorch pickle weight files are
not an FSDP initialization format. DeepSpeed stage 3 establishes
``HfDeepSpeedConfig`` before pretrained loading; stage 2 retains parameter
replicas and generally needs more memory per GPU. DeepSpeed owns its optimizer
and checkpoint format. ``deepspeed_config`` can supply concrete engine settings,
but batch, stage, precision and optimizer options must agree with DLIO. Trainer
``auto`` values are rejected. Offload or custom kernel settings require their
own platform validation and can generate additional I/O.

Token data and loss
-------------------

Both inputs are prepared into versioned token records under
``dataset.data_folder/llama_tokens/<identity>``. Preparation occurs outside timed
training. A manifest records revisions, format settings, counts and checksums;
cache publication is locked and atomic. Local assets are fingerprinted and token
files are checked during initialization. Generation-only runs do not allocate
the 8B model and can run on CPU:

.. code-block:: sh

   dlio_benchmark workload=llama_native_fsdp workload.workflow.train=false workload.workflow.checkpoint=false

Token caches are retained for later reuse and resume. ``keep_files=false`` is
rejected for native workloads; remove an unused cache explicitly after runs that
share it have finished.

Generated records contain vocabulary-valid int32 IDs, deterministic from seed
and sample identity. They exercise real causal loss but are synthetic content.
OpenHermes reads ``conversations`` with ``from`` and ``value``. The explicit
``role_lines_v1`` formatter emits ``User:\n``, ``Assistant:\n`` and ``System:\n``
sections, with BOS/EOS where appropriate; it does not assume an Instruct chat
template. Metadata and per-turn ``weight`` fields do not affect this formatter.
``loss=full`` predicts every unpadded token; ``loss=response_only`` predicts
assistant content and a final assistant EOS, masking prompts and role headers.
PAD positions are masked by position, preserving real EOS even if PAD equals EOS.
HF performs the causal label shift once.

Without packing, each conversation is truncated to ``sequence_length`` and
padded. With packing, tokenized conversations are concatenated with EOS and
emitted incrementally; cross-conversation attention is allowed. The remainder
policy is explicitly ``pad`` or ``drop``. Records with no valid prediction targets
are omitted. ``max_samples`` limits source conversations before tokenization.

One sampler owns all sharding and ordering for every delivery mode (see
`Delivery modes`_). It drops the tail needed to make complete global
accumulation windows and reports how many samples were dropped. Its cursor
advances only after successful compute. Gradient normalization uses
the global number of valid target tokens across the entire accumulation window.
Evaluation uses no gradients and restores training RNG state. OpenHermes has no
implicit evaluation split: supply ``eval_split`` from a DatasetDict or source
that provides it, and enable ``workflow.evaluation`` explicitly.

Delivery modes
--------------
``workload.reader`` selects how records reach the training process, and only
explicitly set values count: omitted native settings mean the ``pytorch`` loader,
no workers, no per-batch delay and no pinning (not the image-workload defaults).
All modes use the same sampler and collator, so a seed gives identical
microbatches in every mode, for any worker count.

.. list-table::
   :header-rows: 1

   * - Setting
     - Behavior
   * - ``data_loader: pytorch``
     - Records read from the prepared token files. ``read_threads`` spawn
       workers (``fork`` is rejected), ``prefetch_size`` batches ahead.
       Workers write their own DFTracer logs.
   * - ``data_loader: load_mem``
     - The whole split is read once into int32 arrays on every rank before
       training, then served from memory. Needs ``read_threads: 0`` and a positive
       per-rank ``memory_budget_bytes``; the budget and the node's available
       memory are checked on every rank before anything is allocated.
   * - ``data_loader: synthetic``
     - Records are generated from ``SeedSequence([seed, sample_id])`` and equal the
       generated token files. No token files are needed (``generate_data: false``).
       Not available with OpenHermes.
   * - ``llama.tokenization: online``
     - OpenHermes raw text is tokenized inside the workers on every read, so the
       tokenizer's CPU time competes with training. Needs a local
       ``datasets.save_to_disk()`` directory and ``packing: false``; records equal
       offline preparation with ``packing: false``. Preparation lists the usable
       examples once, outside timed training.
   * - ``llama.host_sync: step``
     - Loss, finiteness and target counts stay on the device; one host read per
       optimizer step, before the update, so a nonfinite step is still rejected before
       any weight changes. Without per-window MPI error exchange, a loader failure on
       one rank reaches the others as a collective timeout
       (``distributed_timeout_seconds``). The default, ``microbatch``, checks every
       microbatch across ranks and synchronizes around each window.
   * - ``reader.iter_time``
     - An explicit delay per microbatch (default 0); reported separately.

Memory-cache payload per split is ``records * (2 * sequence_length + 1) * 4``
bytes, before the sampler's permutation and the batches. Reads inside workers are
counted in a shared counter: ``application_read_bytes`` is completed reads
including read-ahead that was not consumed yet, so one window can show zero.
Other loaders (DALI, FFCV) are not supported for Llama; they must take their
order from ``TokenSampler`` and register through ``register_token_loader``.

Checkpointing and statistics
----------------------------

``checkpoint.steps_between_checkpoints`` counts optimizer boundaries, not
microbatches. ``train.total_training_steps`` is a global optimizer-boundary
limit across epochs and restarts. Attempted optimizer boundaries and applied
updates are reported separately, accounting for DeepSpeed overflow skips.
Negative checkpoint step intervals use the normal epoch interval settings.

All required ranks participate in synchronous saves. A completion manifest is
published only after model/optimizer shards and rank-specific RNG, sampler and
progress state are present. ``checkpoint.fsync=true`` additionally syncs files
and directories. Incomplete saves are rejected and retained for inspection;
they are not automatically deleted. A save directory is never overwritten.
Resume accepts an explicit completed checkpoint directory:

.. code-block:: sh

   mpirun -np 4 dlio_benchmark workload=llama_native_fsdp \
     workload.llama.resume_from=/shared/checkpoints/step-000000000002-periodic

FSDP uses PyTorch Distributed Checkpoint; DeepSpeed uses its engine format and
resumes in a newly constructed engine. Resume requires the same backend/stage,
world size, exact recorded package versions, model/data identities and training
settings. Changed world size, cross-backend conversion, asynchronous saves,
synthetic rank-shift/subset checkpointing and object-storage checkpoints are not
supported by this implementation. Final saves carry the same resume metadata;
an already-saved final step is not duplicated.

Native JSON outputs include per-rank window/sample records, loss, valid-token
throughput, dropped samples, application bytes read, and checkpoint stored bytes
and throughput in GiB/s. Global checkpoint throughput uses the slowest rank's
collective completion time. The native output schema is marked ``native_llama``;
image-workload utilization metrics and the legacy postprocessor do not describe
these token metrics. Stored file bytes and synchronous reader bytes are not
claims about physical storage traffic or cache misses.

Tracing and later aggregation work
----------------------------------

The integration uses existing DLIO DFTracer events for fetch, transfer, forward,
backward, checkpoint capture and restart. It does not automatically enable
``torch.compile`` when tracing is on. GPU synchronization boundaries make window
timings include completed device work. No synthetic per-batch allreduce barrier
is added to native training.

The `reference branch <https://github.com/OlgaKogiou/LLM-Checkpoints/tree/6ed121ef9749e84f32e3462f74bab04ca288b13c>`_
also has FULL and SELECTIVE aggregation launch modes. Aggregation combines
repeated events into statistics, reducing trace size at the expense of individual
event timing. SELECTIVE applies rules to chosen event categories; FULL applies
aggregation broadly. Those modes depend on its special DFTracer build and are
future work here. Implementing them later requires pinning that build, mapping
rules to actual DLIO event names, testing event counts/durations and thread/rank
behavior, and comparing trace overhead with normal tracing. Their existence
does not imply current DLIO aggregation compatibility.

Validation
----------

``tests/llama_test.py`` exercises local tiny weights, actual token files,
OpenHermes preparation, both losses, accumulation, inference-only evaluation,
checkpoint rejection and continuity, and the DLIO entry lifecycle, including
epoch-boundary resume. Separate fresh processes test actual FSDP2 and DeepSpeed
ZeRO 2/3 engines with two ranks, FP32/BF16 and activation checkpointing. FSDP's
loaded weights are also compared with the original pretrained snapshot.

These checks passed on CPU with the installation baseline above. DeepSpeed CPU
tests disable its optional shared-memory C++ accelerator and use Gloo; they do
not substitute fake engines. The production training path requires GPUs.
To repeat the tests with the installed extras:

.. code-block:: sh

   DFTRACER_ENABLE=0 OMP_NUM_THREADS=1 pytest tests/llama_test.py -q
   # Qualify fresh-process native backend tests on at least two GPUs:
   DLIO_LLAMA_TEST_DEVICE=cuda DFTRACER_ENABLE=0 OMP_NUM_THREADS=1 \
     pytest tests/llama_test.py -q -k fresh_process

CPU results do not establish GPU correctness or 8B capacity. Qualify each
backend/stage on multiple GPUs with the target stack, then perform an actual
pretrained 8B train/save/stop/restart run for both data paths. The implementation
validation report in ``docs/llama_implementation_validation.md`` records local
results and environmental limitations.
