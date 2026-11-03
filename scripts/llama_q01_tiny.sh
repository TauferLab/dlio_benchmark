#!/usr/bin/env bash
# Tiny native-backend GPU qualification only. Never submits a scheduler job.
set -euo pipefail
if (( $# != 4 )); then
  echo "Usage: $0 matrix|tuolumne EXPECTED_COMMIT ENVIRONMENT_DIR FRESH_OUTPUT_DIR" >&2
  exit 2
fi
platform=$1
expected_commit=$2
environment_dir=$3
output_dir=$4
case "$platform" in
  matrix) : "${SLURM_JOB_ID:?Run inside a user-created Slurm allocation}" ;;
  tuolumne) : "${FLUX_JOB_ID:?Run inside a user-created Flux allocation}" ;;
  *) echo "Unknown platform" >&2; exit 2 ;;
esac
# Explicit environment argument permits a separately approved GPU-capable stack.
# The CPU-only llama-matrix-venv used for L00 must fail the GPU preflight below.
source "$environment_dir/bin/activate"
worktree=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
cd "$worktree"
test "$(git rev-parse HEAD)" = "$expected_commit"
test -z "$(git status --porcelain)"
test ! -e "$output_dir"
export PYTHONPATH="$worktree${PYTHONPATH:+:$PYTHONPATH}"
export DFTRACER_ENABLE=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
# One parent process only: the capture harness owns the sole two-rank torchrun.
python - "$worktree" "$platform" <<'PY'
from pathlib import Path
import json, os, socket, sys
import torch, dlio_benchmark
root = Path(sys.argv[1]).resolve()
assert Path(dlio_benchmark.__file__).resolve().is_relative_to(root)
assert int(os.environ.get("SLURM_PROCID", os.environ.get("FLUX_TASK_RANK", "0"))) == 0
assert int(os.environ.get("WORLD_SIZE", "1")) == 1, "Do not launch harness as a distributed rank"
assert torch.cuda.is_available() and torch.cuda.device_count() >= 2, "Two visible GPUs and a compatible GPU Torch build are required"
if sys.argv[2] == "matrix":
    assert torch.version.cuda is not None and torch.version.hip is None
else:
    assert torch.version.hip is not None
print(json.dumps({"hostname": socket.gethostname(), "python": sys.executable,
 "torch": torch.__version__, "cuda": torch.version.cuda, "hip": torch.version.hip,
 "visible_devices": [torch.cuda.get_device_properties(i).name for i in range(torch.cuda.device_count())],
 "allocation": os.environ.get("SLURM_JOB_ID", os.environ.get("FLUX_JOB_ID"))}, indent=2))
PY
python tests/llama_capture_v1.py "$output_dir" --device cuda
python - "$output_dir/capture.json" <<'PY'
import json, sys
r = json.load(open(sys.argv[1]))
assert len(r["cases"]) == 7
assert all(c.get("status") == "PASS" for c in r["cases"])
assert all(c["device"] == "cuda" for c in r["cases"] if c["backend"] != "torch")
print("Seven tiny-model cases passed; full-model/platform Q01 acceptance remains separate.")
PY
