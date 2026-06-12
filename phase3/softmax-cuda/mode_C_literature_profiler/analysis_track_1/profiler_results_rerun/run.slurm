#!/bin/bash
#SBATCH -A ACD115083
#SBATCH --job-name=softmax-c-prof2
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=/home/r14525078/HeCBench/phase3/softmax-cuda/mode_C_literature_profiler/analysis_track_1/profiler_results_rerun/result.out
#SBATCH --error=/home/r14525078/HeCBench/phase3/softmax-cuda/mode_C_literature_profiler/analysis_track_1/profiler_results_rerun/result.err

set -u
set -o pipefail

SRC=/home/r14525078/HeCBench/src/softmax-cuda
TRACK=/home/r14525078/HeCBench/phase3/softmax-cuda/mode_C_literature_profiler/analysis_track_1
RESULT="$TRACK/profiler_results_rerun"
RAW="$RESULT/profiler_raw"
CSV="$TRACK/profiler_summary.csv"
BUILD_LOG="$RESULT/build.log"
REPEAT_FOR_PROFILER=10
LAUNCH_SKIP=2
LAUNCH_COUNT=1
METRICS="launch__registers_per_thread,launch__shared_mem_per_block_static,launch__shared_mem_per_block_dynamic,sm__cycles_elapsed.avg,gpu__time_duration.sum,launch__waves_per_multiprocessor,launch__occupancy_limit_registers,launch__occupancy_limit_shared_mem,launch__occupancy_limit_warps"

mkdir -p "$RAW"

module purge
module load cuda/12.8
cd "$SRC"

HOSTNAME_VALUE=$(hostname)
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1 | tr ',' ';')
CUDA_VERSION=$(nvcc --version | awk '/release/ {print $6}' | tr -d ',')
NCU_PATH=$(command -v ncu || true)
NCU_VERSION=$(ncu --version 2>&1 | tr '\n' ' ' | sed 's/,/;/g')

{
  echo "slurm_job_id=${SLURM_JOB_ID:-unknown}"
  echo "hostname=${HOSTNAME_VALUE}"
  echo "gpu_name=${GPU_NAME}"
  echo "cuda_version=${CUDA_VERSION}"
  echo "ncu_path=${NCU_PATH:-not_found}"
  echo "ncu_version=${NCU_VERSION}"
  echo "repeat_for_profiler=${REPEAT_FOR_PROFILER}"
  echo "launch_skip=${LAUNCH_SKIP}"
  echo "launch_count=${LAUNCH_COUNT}"
  echo "metrics=${METRICS}"
  echo "official_timing_used=false"
  echo "source_modified=false"
  echo "date=$(date -Is)"
  nvidia-smi || true
  nvcc --version || true
} | tee "$RESULT/environment.txt"

make ARCH=sm_70 > "$BUILD_LOG" 2>&1
build_rc=$?
if [ "$build_rc" -ne 0 ]; then
  echo "Build failed; source was not modified. See $BUILD_LOG"
fi

run_profile() {
  local num_slice="$1"
  local slice="$2"
  local impl="$3"
  local kernel="$4"
  local base="slice_${slice}_impl_${impl}"
  local stdout_path="$RAW/${base}.stdout"
  local stderr_path="$RAW/${base}.stderr"
  local report_path="$RAW/${base}.ncu-rep"
  local details_path="$RAW/${base}_import_details.txt"
  local status_path="$RAW/${base}.status"

  if [ "$build_rc" -ne 0 ]; then
    echo "FAILED,build_failed,$report_path,$stdout_path,$stderr_path,$details_path" > "$status_path"
    return 0
  fi

  ncu --target-processes all \
      --kernel-name-base function \
      --kernel-name "$kernel" \
      --launch-skip "$LAUNCH_SKIP" \
      --launch-count "$LAUNCH_COUNT" \
      --metrics "$METRICS" \
      --force-overwrite \
      --export "$report_path" \
      ./main "$num_slice" "$slice" "$impl" "$REPEAT_FOR_PROFILER" \
      > "$stdout_path" 2> "$stderr_path"
  local rc=$?

  if [ "$rc" -eq 0 ] && [ -s "$report_path" ]; then
    ncu --import "$report_path" --page details > "$details_path" 2>> "$stderr_path"
    local import_rc=$?
    if [ "$import_rc" -eq 0 ] && grep -q "launch__registers_per_thread" "$details_path"; then
      echo "AVAILABLE,filtered_imported,$report_path,$stdout_path,$stderr_path,$details_path" > "$status_path"
    else
      echo "PARTIAL,report_without_expected_metrics,$report_path,$stdout_path,$stderr_path,$details_path" > "$status_path"
    fi
    return 0
  fi

  if grep -qi "ERR_NVGPUCTRPERM\\|permission\\|not supported\\|Unknown metric\\|No metrics to collect" "$stdout_path" "$stderr_path"; then
    echo "UNAVAILABLE,ncu_unavailable_or_bad_metric,$report_path,$stdout_path,$stderr_path,$details_path" > "$status_path"
  else
    echo "FAILED,ncu_failed,$report_path,$stdout_path,$stderr_path,$details_path" > "$status_path"
  fi
}

run_profile 100000 784 3 softMax3
run_profile 100000 784 4 softMax4
run_profile 100000 1024 3 softMax3
run_profile 100000 1024 4 softMax4
run_profile 50000 2048 3 softMax3
run_profile 50000 2048 4 softMax4

python3 - "$TRACK" "$RESULT" "$RAW" "$CSV" "${SLURM_JOB_ID:-unknown}" "$HOSTNAME_VALUE" "$GPU_NAME" "$CUDA_VERSION" "$NCU_VERSION" "$REPEAT_FOR_PROFILER" "$LAUNCH_SKIP" "$LAUNCH_COUNT" <<'PY'
import csv
import re
import sys
from pathlib import Path

track = Path(sys.argv[1])
result = Path(sys.argv[2])
raw = Path(sys.argv[3])
csv_path = Path(sys.argv[4])
job_id = sys.argv[5]
hostname = sys.argv[6]
gpu_name = sys.argv[7]
cuda_version = sys.argv[8]
ncu_version = sys.argv[9]
repeat = sys.argv[10]
launch_skip = sys.argv[11]
launch_count = sys.argv[12]

schema = [
    "benchmark", "mode", "analysis_track", "profiler_job_id", "sliceSize",
    "numSlice", "repeat", "impl", "kernel_filter", "launch_skip",
    "launch_count", "profiler_status", "ncu_version", "hostname", "gpu_name",
    "cuda_version", "report_path", "stdout_path", "stderr_path",
    "achieved_occupancy", "registers_per_thread",
    "static_shared_memory_bytes", "dynamic_shared_memory_bytes",
    "memory_throughput", "warp_execution_efficiency",
    "instruction_mix_summary", "math_special_function_summary",
    "profiler_timing_ms", "official_timing_used", "notes",
]

cases = [
    (100000, 784, 3, "softMax3"),
    (100000, 784, 4, "softMax4"),
    (100000, 1024, 3, "softMax3"),
    (100000, 1024, 4, "softMax4"),
    (50000, 2048, 3, "softMax3"),
    (50000, 2048, 4, "softMax4"),
]

def read_status(slice_size, impl):
    path = raw / f"slice_{slice_size}_impl_{impl}.status"
    if not path.exists():
        return ["FAILED", "missing_status", "n/a", "n/a", "n/a", "n/a"]
    parts = path.read_text(errors="replace").strip().split(",", 5)
    while len(parts) < 6:
        parts.append("n/a")
    return parts

def read(path):
    p = Path(path)
    return p.read_text(errors="replace") if p.exists() else ""

def metric(text, name):
    for line in text.splitlines():
        if line.strip().startswith(name):
            parts = line.split()
            if len(parts) >= 2:
                return " ".join(parts[1:])
    return "n/a"

rows = []
for num_slice, slice_size, impl, kernel in cases:
    status, note, report_path, stdout_path, stderr_path, details_path = read_status(slice_size, impl)
    details = read(details_path)
    stdout = read(stdout_path)
    kernel_seen = "n/a"
    m = re.search(r'Profiling "([^"]+)"', stdout)
    if m:
        kernel_seen = m.group(1)
    rows.append({
        "benchmark": "softmax-cuda",
        "mode": "Mode_C",
        "analysis_track": "analysis_track_1",
        "profiler_job_id": job_id,
        "sliceSize": str(slice_size),
        "numSlice": str(num_slice),
        "repeat": repeat,
        "impl": str(impl),
        "kernel_filter": kernel,
        "launch_skip": launch_skip,
        "launch_count": launch_count,
        "profiler_status": status,
        "ncu_version": ncu_version,
        "hostname": hostname,
        "gpu_name": gpu_name,
        "cuda_version": cuda_version,
        "report_path": report_path,
        "stdout_path": stdout_path,
        "stderr_path": stderr_path,
        "achieved_occupancy": metric(details, "launch__waves_per_multiprocessor"),
        "registers_per_thread": metric(details, "launch__registers_per_thread"),
        "static_shared_memory_bytes": metric(details, "launch__shared_mem_per_block_static"),
        "dynamic_shared_memory_bytes": metric(details, "launch__shared_mem_per_block_dynamic"),
        "memory_throughput": "n/a",
        "warp_execution_efficiency": "n/a",
        "instruction_mix_summary": "n/a",
        "math_special_function_summary": "n/a",
        "profiler_timing_ms": metric(details, "gpu__time_duration.sum"),
        "official_timing_used": "false",
        "notes": f"{note}; actual_kernel={kernel_seen}; metrics_import={details_path}",
    })

with csv_path.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=schema)
    writer.writeheader()
    writer.writerows(rows)

available = sum(1 for r in rows if r["profiler_status"] == "AVAILABLE")
partial = sum(1 for r in rows if r["profiler_status"] == "PARTIAL")
failed = len(rows) - available - partial

summary = [
    "# Profiler Summary Rerun",
    "",
    f"Profiler rerun job: `{job_id}` on `{hostname}`.",
    "",
    "This was an analysis-only profiler rerun. It does not count as Submission 2.",
    "",
    "Execution safeguards:",
    "",
    f"- repeat_for_profiler={repeat}",
    f"- launch_skip={launch_skip}",
    f"- launch_count={launch_count}",
    "- explicit metrics were requested",
    "- `.ncu-rep` files were imported with `ncu --import --page details`",
    "- official_timing_used=false for every row",
    "- no 128/256 profiling",
    "- profiler timing is not official speedup timing",
    "",
    "## Previous Failed/Not-Accepted Profiler Cases",
    "",
    "- Job `950683`: feasibility test failed before profiling because the existing binary was missing on the compute node. This was not a profiler availability result.",
    "- Job `950678`: profiler attached and wrote `.ncu-rep` files, but the direct `--set default` stdout reported no metrics in selected sections; the result was marked PARTIAL and not accepted as profiler evidence.",
    "",
    "## Rerun Status Table",
    "",
    "| slice | impl | status | registers/thread | dynamic shared mem | static shared mem | waves/SM | profiler duration |",
    "|---:|---:|---|---|---|---|---|---|",
]
for r in rows:
    summary.append(
        f"| {r['sliceSize']} | {r['impl']} | {r['profiler_status']} | {r['registers_per_thread']} | "
        f"{r['dynamic_shared_memory_bytes']} | {r['static_shared_memory_bytes']} | "
        f"{r['achieved_occupancy']} | {r['profiler_timing_ms']} |"
    )
summary.extend([
    "",
    "## Interpretation Limits",
    "",
    "- Metrics are profiler diagnostics only.",
    "- Profiler timing is not official timing.",
    "- Do not start Submission 2 until these artifacts are audited.",
])
(result / "profiler_summary.md").write_text("\n".join(summary) + "\n")

if available == 6:
    rec = "Submission 2 = reduction-structure ablation"
    reason = "Corrected profiler rerun produced resource metrics for all six large-slice impl=3/4 cases. The next safest evidence-building step is a reduction-structure ablation, unless human audit identifies a more specific resource-driven target."
elif available > 0:
    rec = "profiler inconclusive; use ablation fallback"
    reason = "Corrected profiler rerun produced only partial metrics. Avoid profiler-informed optimization and use ablation fallback if Submission 2 is approved."
else:
    rec = "profiler inconclusive; use ablation fallback"
    reason = "Corrected profiler rerun did not produce usable metrics."

decision = [
    "# Direction Decision After Profiler Rerun",
    "",
    f"Recommendation: `{rec}`",
    "",
    "Submission 2 remains deferred until human audit of profiler artifacts.",
    "",
    "## Evidence Status",
    "",
    f"- profiler rows: {len(rows)}",
    f"- AVAILABLE rows: {available}",
    f"- PARTIAL rows: {partial}",
    f"- FAILED/UNAVAILABLE rows: {failed}",
    "- official_timing_used=false for all rows",
    "",
    "## Previous Failure Recorded",
    "",
    "- The prior feasibility test failed because the binary was missing on the compute node.",
    "- The prior profiler job was PARTIAL because direct stdout did not expose metrics from the selected sections.",
    "",
    "## Rationale",
    "",
    reason,
    "",
    "## Guardrails",
    "",
    "- Do not use profiler timing as official timing.",
    "- Do not overclaim profiler-supported causality before human audit.",
    "- Do not start Submission 2 without explicit human approval.",
]
(track / "direction_decision_after_profiler.md").write_text("\n".join(decision) + "\n")
(track / "direction_decision_after_profiler_rerun.md").write_text("\n".join(decision) + "\n")
PY

echo "Profiler rerun complete. Summary: $CSV"
