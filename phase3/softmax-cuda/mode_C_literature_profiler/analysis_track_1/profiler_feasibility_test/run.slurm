#!/bin/bash
#SBATCH -A ACD115083
#SBATCH --job-name=softmax-prof-test
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:20:00
#SBATCH --output=/home/r14525078/HeCBench/phase3/softmax-cuda/mode_C_literature_profiler/analysis_track_1/profiler_feasibility_test/result.out
#SBATCH --error=/home/r14525078/HeCBench/phase3/softmax-cuda/mode_C_literature_profiler/analysis_track_1/profiler_feasibility_test/result.err

set -u
set -o pipefail

SRC=/home/r14525078/HeCBench/src/softmax-cuda
OUT=/home/r14525078/HeCBench/phase3/softmax-cuda/mode_C_literature_profiler/analysis_track_1/profiler_feasibility_test
RAW="$OUT/raw"
mkdir -p "$RAW"

module purge
module load cuda/12.8
cd "$SRC"

HOSTNAME_VALUE=$(hostname)
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1 | tr ',' ';')
CUDA_VERSION=$(nvcc --version | awk '/release/ {print $6}' | tr -d ',')
NCU_PATH=$(command -v ncu || true)
NCU_VERSION=$(ncu --version 2>&1 | tr '\n' ' ' | sed 's/,/;/g')
METRICS="launch__registers_per_thread,launch__shared_mem_per_block_static,launch__shared_mem_per_block_dynamic,sm__cycles_elapsed.avg"

{
  echo "slurm_job_id=${SLURM_JOB_ID:-unknown}"
  echo "hostname=${HOSTNAME_VALUE}"
  echo "gpu_name=${GPU_NAME}"
  echo "cuda_version=${CUDA_VERSION}"
  echo "ncu_path=${NCU_PATH:-not_found}"
  echo "ncu_version=${NCU_VERSION}"
  echo "metrics=${METRICS}"
  echo "diagnostic_case=numSlice=1000,sliceSize=784,impl=4,repeat=1"
  echo "launch_skip=0"
  echo "launch_count=1"
  echo "official_timing_used=false"
  echo "date=$(date -Is)"
} | tee "$OUT/environment.txt"

if [ ! -x "$SRC/main" ]; then
  echo "FAILED: existing binary missing; source not modified and build not attempted" | tee "$OUT/status.txt"
  exit 0
fi

ncu --target-processes all \
    --kernel-name-base function \
    --kernel-name softMax4 \
    --launch-skip 0 \
    --launch-count 1 \
    --metrics "$METRICS" \
    --force-overwrite \
    --export "$RAW/softmax4_metric_test.ncu-rep" \
    ./main 1000 784 4 1 \
    > "$RAW/softmax4_metric_test.stdout" \
    2> "$RAW/softmax4_metric_test.stderr"
rc=$?

if [ "$rc" -eq 0 ] && [ -s "$RAW/softmax4_metric_test.ncu-rep" ]; then
  echo "AVAILABLE" > "$OUT/status.txt"
else
  if grep -qi "ERR_NVGPUCTRPERM\\|permission\\|not supported\\|No metrics to collect\\|Unknown metric\\|Failed to prepare kernel" "$RAW/softmax4_metric_test.stdout" "$RAW/softmax4_metric_test.stderr"; then
    echo "UNAVAILABLE_OR_BAD_METRIC" > "$OUT/status.txt"
  else
    echo "FAILED" > "$OUT/status.txt"
  fi
fi

python3 - "$OUT" "$RAW" "${SLURM_JOB_ID:-unknown}" "$HOSTNAME_VALUE" "$GPU_NAME" "$CUDA_VERSION" "$NCU_VERSION" "$rc" <<'PY'
import re
import sys
from pathlib import Path

out = Path(sys.argv[1])
raw = Path(sys.argv[2])
job_id, host, gpu, cuda, ncu_version, rc = sys.argv[3:9]
stdout = raw / "softmax4_metric_test.stdout"
stderr = raw / "softmax4_metric_test.stderr"
text = ""
if stdout.exists():
    text += stdout.read_text(errors="replace")
if stderr.exists():
    text += "\n" + stderr.read_text(errors="replace")
status = (out / "status.txt").read_text(errors="replace").strip() if (out / "status.txt").exists() else "FAILED"

metrics_found = []
for name in [
    "launch__registers_per_thread",
    "launch__shared_mem_per_block_static",
    "launch__shared_mem_per_block_dynamic",
    "sm__cycles_elapsed.avg",
]:
    if name in text:
        metrics_found.append(name)

kernel = "n/a"
m = re.search(r'Profiling "([^"]+)"', text)
if m:
    kernel = m.group(1)

report = out / "profiler_feasibility_test.md"
report.write_text(f"""# Profiler Feasibility Test

Slurm job: `{job_id}` on `{host}`.

Status: `{status}`

This was a profiler usability test only. It is not official timing and does not
count as Submission 2.

## Diagnostic Case

- numSlice=1000
- sliceSize=784
- impl=4
- repeat=1
- kernel filter: `softMax4`
- launch_skip=0
- launch_count=1
- official_timing_used=false

## Environment

- GPU: `{gpu}`
- CUDA: `{cuda}`
- ncu: `{ncu_version}`
- ncu return code: `{rc}`
- actual kernel observed: `{kernel}`

## Metrics Requested

- `launch__registers_per_thread`
- `launch__shared_mem_per_block_static`
- `launch__shared_mem_per_block_dynamic`
- `sm__cycles_elapsed.avg`

Metrics seen in text output:

{chr(10).join('- `' + m + '`' for m in metrics_found) if metrics_found else '- none'}

## Artifacts

- stdout: `{stdout}`
- stderr: `{stderr}`
- report: `{raw / 'softmax4_metric_test.ncu-rep'}`

## Interpretation

If status is `AVAILABLE` and requested metrics appear in stdout, profiler metric
collection is usable. If status is unavailable or metrics are absent, do not use
profiler for causal claims without another approved diagnostic adjustment.
""")
PY

echo "Profiler feasibility test complete."
