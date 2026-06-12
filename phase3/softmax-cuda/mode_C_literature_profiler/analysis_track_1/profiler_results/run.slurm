#!/bin/bash
#SBATCH -A ACD115083
#SBATCH --job-name=softmax-c-prof
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:30:00
#SBATCH --output=/home/r14525078/HeCBench/phase3/softmax-cuda/mode_C_literature_profiler/analysis_track_1/profiler_results/result.out
#SBATCH --error=/home/r14525078/HeCBench/phase3/softmax-cuda/mode_C_literature_profiler/analysis_track_1/profiler_results/result.err

set -u
set -o pipefail

SRC=/home/r14525078/HeCBench/src/softmax-cuda
TRACK=/home/r14525078/HeCBench/phase3/softmax-cuda/mode_C_literature_profiler/analysis_track_1
RESULT="$TRACK/profiler_results"
RAW="$RESULT/profiler_raw"
CSV="$TRACK/profiler_summary.csv"
REPEAT_FOR_PROFILER=10
LAUNCH_SKIP=2
LAUNCH_COUNT=1

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
  echo "official_timing_used=false"
  echo "date=$(date -Is)"
  nvidia-smi || true
  nvcc --version || true
} | tee "$RESULT/environment.txt"

if [ ! -x "$SRC/main" ]; then
  echo "Existing binary $SRC/main is missing or not executable. Source was not modified and build was not attempted." | tee "$RESULT/binary_missing.txt"
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
  local txt_path="$RAW/${base}.txt"
  local status_path="$RAW/${base}.status"
  local fallback_report_path="$RAW/${base}_fallback.ncu-rep"
  local fallback_stdout_path="$RAW/${base}_fallback.stdout"
  local fallback_stderr_path="$RAW/${base}_fallback.stderr"

  if [ ! -x "$SRC/main" ]; then
    echo "FAILED,binary_missing,$report_path,$stdout_path,$stderr_path" > "$status_path"
    return 0
  fi

  ncu --target-processes all \
      --kernel-name-base function \
      --kernel-name "$kernel" \
      --launch-skip "$LAUNCH_SKIP" \
      --launch-count "$LAUNCH_COUNT" \
      --set default \
      --force-overwrite \
      --export "$report_path" \
      ./main "$num_slice" "$slice" "$impl" "$REPEAT_FOR_PROFILER" \
      > "$stdout_path" 2> "$stderr_path"
  local rc=$?
  cp "$stdout_path" "$txt_path" || true

  if [ "$rc" -eq 0 ] && [ -s "$report_path" ]; then
    echo "AVAILABLE,filtered,$report_path,$stdout_path,$stderr_path" > "$status_path"
    return 0
  fi

  if grep -qi "no kernels were profiled\\|no kernels matched\\|filter" "$stdout_path" "$stderr_path"; then
    ncu --target-processes all \
        --launch-skip "$LAUNCH_SKIP" \
        --launch-count "$LAUNCH_COUNT" \
        --set default \
        --force-overwrite \
        --export "$fallback_report_path" \
        ./main "$num_slice" "$slice" "$impl" "$REPEAT_FOR_PROFILER" \
        > "$fallback_stdout_path" 2> "$fallback_stderr_path"
    local fallback_rc=$?
    if [ "$fallback_rc" -eq 0 ] && [ -s "$fallback_report_path" ]; then
      cp "$fallback_stdout_path" "$txt_path" || true
      echo "PARTIAL,launch_limited_fallback,$fallback_report_path,$fallback_stdout_path,$fallback_stderr_path" > "$status_path"
      return 0
    fi
    echo "FAILED,fallback_failed,$fallback_report_path,$fallback_stdout_path,$fallback_stderr_path" > "$status_path"
    return 0
  fi

  if grep -qi "ERR_NVGPUCTRPERM\\|permission\\|not supported\\|Profiling is not supported" "$stdout_path" "$stderr_path"; then
    echo "UNAVAILABLE,permission_or_unsupported,$report_path,$stdout_path,$stderr_path" > "$status_path"
  else
    echo "FAILED,ncu_failed,$report_path,$stdout_path,$stderr_path" > "$status_path"
  fi
  return 0
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
    status_path = raw / f"slice_{slice_size}_impl_{impl}.status"
    if not status_path.exists():
        return "FAILED", "missing_status", str(raw / f"slice_{slice_size}_impl_{impl}.ncu-rep"), str(raw / f"slice_{slice_size}_impl_{impl}.stdout"), str(raw / f"slice_{slice_size}_impl_{impl}.stderr")
    parts = status_path.read_text(errors="replace").strip().split(",", 4)
    while len(parts) < 5:
        parts.append("n/a")
    return parts

def text_for(path):
    p = Path(path)
    if p.exists():
        return p.read_text(errors="replace")
    return ""

def find_metric(text, labels):
    for label in labels:
        pattern = re.compile(rf"{re.escape(label)}\\s+(?:%|byte|Kbyte|Mbyte|Gbyte|cycle|register/thread|msecond|usecond|nsecond|inst|warp|thread|block|sector|request)?\\s*([0-9][0-9,]*(?:\\.[0-9]+)?)", re.I)
        m = pattern.search(text)
        if m:
            return m.group(1).replace(",", "")
    return "n/a"

def summarize_instructions(text):
    lines = [ln.strip() for ln in text.splitlines() if "inst_" in ln or "sass__inst" in ln or "smsp__sass" in ln]
    return " | ".join(lines[:4]).replace(",", ";") if lines else "n/a"

def summarize_math(text):
    lines = [ln.strip() for ln in text.splitlines() if "special" in ln.lower() or "fma" in ln.lower() or "ffma" in ln.lower() or "ex2" in ln.lower()]
    return " | ".join(lines[:4]).replace(",", ";") if lines else "n/a"

rows = []
for num_slice, slice_size, impl, kernel in cases:
    status, note, report_path, stdout_path, stderr_path = read_status(slice_size, impl)
    text = text_for(stdout_path) + "\n" + text_for(stderr_path)
    rows.append({
        "benchmark": "softmax-cuda",
        "mode": "Mode_C",
        "analysis_track": "analysis_track_1",
        "profiler_job_id": job_id,
        "sliceSize": str(slice_size),
        "numSlice": str(num_slice),
        "repeat": repeat,
        "impl": str(impl),
        "kernel_filter": kernel if note == "filtered" else f"{kernel};fallback={note}",
        "launch_skip": launch_skip,
        "launch_count": launch_count,
        "profiler_status": status if status in {"NOT_RUN", "AVAILABLE", "UNAVAILABLE", "FAILED", "PARTIAL"} else "FAILED",
        "ncu_version": ncu_version,
        "hostname": hostname,
        "gpu_name": gpu_name,
        "cuda_version": cuda_version,
        "report_path": report_path,
        "stdout_path": stdout_path,
        "stderr_path": stderr_path,
        "achieved_occupancy": find_metric(text, ["Achieved Occupancy", "sm__warps_active.avg.pct_of_peak_sustained_active"]),
        "registers_per_thread": find_metric(text, ["Registers Per Thread", "launch__registers_per_thread"]),
        "static_shared_memory_bytes": find_metric(text, ["Static Shared Memory Per Block", "launch__shared_mem_per_block_static"]),
        "dynamic_shared_memory_bytes": find_metric(text, ["Dynamic Shared Memory Per Block", "launch__shared_mem_per_block_dynamic"]),
        "memory_throughput": find_metric(text, ["Memory Throughput", "DRAM Throughput", "dram__throughput.avg.pct_of_peak_sustained_elapsed"]),
        "warp_execution_efficiency": find_metric(text, ["Warp Execution Efficiency", "smsp__thread_inst_executed_per_inst_executed.ratio"]),
        "instruction_mix_summary": summarize_instructions(text),
        "math_special_function_summary": summarize_math(text),
        "profiler_timing_ms": find_metric(text, ["Duration", "gpu__time_duration.sum"]),
        "official_timing_used": "false",
        "notes": note,
    })

with csv_path.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=schema)
    writer.writeheader()
    writer.writerows(rows)

statuses = [r["profiler_status"] for r in rows]
available = sum(1 for s in statuses if s in {"AVAILABLE", "PARTIAL"})
failed = sum(1 for s in statuses if s in {"FAILED", "UNAVAILABLE"})

summary = [
    "# Profiler Summary",
    "",
    f"Profiler job: `{job_id}` on `{hostname}`.",
    "",
    "This was an analysis-only profiler run. It does not count as Submission 2.",
    "",
    "Execution safeguards:",
    "",
    f"- repeat_for_profiler={repeat}",
    f"- launch_skip={launch_skip}",
    f"- launch_count={launch_count}",
    "- official_timing_used=false for every row",
    "- no 128/256 profiling",
    "- profiler timing is not official speedup timing",
    "",
    "## Status Table",
    "",
    "| slice | impl | kernel_filter | profiler_status | report |",
    "|---:|---:|---|---|---|",
]
for r in rows:
    summary.append(f"| {r['sliceSize']} | {r['impl']} | {r['kernel_filter']} | {r['profiler_status']} | {r['report_path']} |")

summary.extend([
    "",
    "## Interpretation Limits",
    "",
    "- Use profiler metrics for explanation only.",
    "- Do not use profiler timing for official speedup.",
    "- Do not start Submission 2 until these artifacts are audited.",
])

(result / "profiler_summary.md").write_text("\n".join(summary) + "\n")

if available == 6:
    rec = "Submission 2 = reduction-structure ablation"
    reason = "Profiler data was collected for all representative large-slice impl=3/4 pairs. Use audited metrics to decide the exact ablation focus; reduction-structure ablation remains the lowest-risk explanatory next step unless profiler review identifies a clearer 2048-specific or block-size issue."
elif available > 0:
    rec = "profiler inconclusive; use ablation fallback"
    reason = "Profiler data is partial. Avoid profiler-informed optimization claims and use a reduction-structure ablation if Submission 2 is still desired."
else:
    rec = "profiler inconclusive; use ablation fallback"
    reason = "Profiler data was not collected. Use ablation fallback rather than profiler-informed optimization."

decision = [
    "# Direction Decision After Profiler",
    "",
    f"Recommendation: `{rec}`",
    "",
    "Submission 2 remains deferred until human audit of profiler artifacts.",
    "",
    "## Evidence Status",
    "",
    f"- profiler rows: {len(rows)}",
    f"- AVAILABLE/PARTIAL rows: {available}",
    f"- FAILED/UNAVAILABLE rows: {failed}",
    "- official_timing_used=false for all rows",
    "",
    "## Rationale",
    "",
    reason,
    "",
    "## Guardrails",
    "",
    "- Do not use profiler timing as official timing.",
    "- Do not claim profiler-supported causality until metrics are audited.",
    "- Do not start Submission 2 without explicit human approval.",
]
(track / "direction_decision_after_profiler.md").write_text("\n".join(decision) + "\n")
PY

echo "Profiler-only job complete. Summary: $CSV"
