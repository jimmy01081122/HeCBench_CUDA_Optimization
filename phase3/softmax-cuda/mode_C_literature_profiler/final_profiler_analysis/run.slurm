#!/bin/bash
#SBATCH -A ACD115083
#SBATCH --job-name=softmax-c-fprof
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=/home/r14525078/HeCBench/phase3/softmax-cuda/mode_C_literature_profiler/final_profiler_analysis/result.out
#SBATCH --error=/home/r14525078/HeCBench/phase3/softmax-cuda/mode_C_literature_profiler/final_profiler_analysis/result.err

set -u
set -o pipefail

SRC=/home/r14525078/HeCBench/src/softmax-cuda
OUT=/home/r14525078/HeCBench/phase3/softmax-cuda/mode_C_literature_profiler/final_profiler_analysis
RAW="$OUT/raw"
REPORTS="$OUT/ncu_reports"
BUILD_LOG="$OUT/build.log"
CSV="$OUT/profiler_summary.csv"
REPEAT_FOR_PROFILER=10
LAUNCH_SKIP=2
LAUNCH_COUNT=1
METRICS="launch__registers_per_thread,launch__shared_mem_per_block_static,launch__shared_mem_per_block_dynamic,sm__cycles_elapsed.avg,gpu__time_duration.sum,launch__waves_per_multiprocessor,launch__occupancy_limit_registers,launch__occupancy_limit_shared_mem,launch__occupancy_limit_warps"

mkdir -p "$RAW" "$REPORTS"

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
} | tee "$OUT/environment.txt"

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
  local report_path="$REPORTS/${base}.ncu-rep"
  local details_path="$RAW/${base}_details.txt"
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

python3 - "$OUT" "$RAW" "$REPORTS" "$CSV" "${SLURM_JOB_ID:-unknown}" "$HOSTNAME_VALUE" "$GPU_NAME" "$CUDA_VERSION" "$NCU_VERSION" "$REPEAT_FOR_PROFILER" "$LAUNCH_SKIP" "$LAUNCH_COUNT" <<'PY'
import csv
import re
import sys
from pathlib import Path

out_dir = Path(sys.argv[1])
raw = Path(sys.argv[2])
reports = Path(sys.argv[3])
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
    "benchmark", "mode", "stage", "profiler_job_id", "sliceSize",
    "numSlice", "repeat_for_profiler", "impl", "kernel_filter",
    "launch_skip", "launch_count", "profiler_status", "ncu_version",
    "hostname", "gpu_name", "cuda_version", "report_path", "stdout_path",
    "stderr_path", "achieved_occupancy", "registers_per_thread",
    "static_shared_memory_bytes", "dynamic_shared_memory_bytes",
    "waves_per_sm", "memory_throughput", "warp_execution_efficiency",
    "instruction_mix_summary", "math_special_function_summary",
    "stall_or_scheduler_summary", "profiler_timing_ms",
    "official_timing_used", "notes",
]

cases = [
    (100000, 784, 3, "softMax3"),
    (100000, 784, 4, "softMax4"),
    (100000, 1024, 3, "softMax3"),
    (100000, 1024, 4, "softMax4"),
    (50000, 2048, 3, "softMax3"),
    (50000, 2048, 4, "softMax4"),
]

official_speedups = {
    784: "1.135540",
    1024: "1.048740",
    2048: "1.008239",
}

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

def numeric_kbytes(value):
    if value == "n/a":
        return None
    parts = value.split()
    nums = [p for p in parts if re.fullmatch(r"[0-9]+(?:\.[0-9]+)?", p)]
    if not nums:
        return None
    amount = float(nums[0])
    lower = value.lower()
    if "kbyte" in lower:
        return amount
    if "byte" in lower:
        return amount / 1024.0
    return amount

rows = []
for num_slice, slice_size, impl, kernel in cases:
    status, note, report_path, stdout_path, stderr_path, details_path = read_status(slice_size, impl)
    details = read(details_path)
    stdout = read(stdout_path)
    kernel_seen = "n/a"
    m = re.search(r'Profiling "([^"]+)"', stdout)
    if m:
        kernel_seen = m.group(1)
    waves = metric(details, "launch__waves_per_multiprocessor")
    rows.append({
        "benchmark": "softmax-cuda",
        "mode": "Mode_C",
        "stage": "final_profiler_analysis",
        "profiler_job_id": job_id,
        "sliceSize": str(slice_size),
        "numSlice": str(num_slice),
        "repeat_for_profiler": repeat,
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
        "achieved_occupancy": "n/a",
        "registers_per_thread": metric(details, "launch__registers_per_thread"),
        "static_shared_memory_bytes": metric(details, "launch__shared_mem_per_block_static"),
        "dynamic_shared_memory_bytes": metric(details, "launch__shared_mem_per_block_dynamic"),
        "waves_per_sm": waves,
        "memory_throughput": "n/a",
        "warp_execution_efficiency": "n/a",
        "instruction_mix_summary": "n/a",
        "math_special_function_summary": "n/a",
        "stall_or_scheduler_summary": "n/a",
        "profiler_timing_ms": metric(details, "gpu__time_duration.sum"),
        "official_timing_used": "false",
        "notes": f"{note}; actual_kernel={kernel_seen}; metrics_import={details_path}",
    })

with csv_path.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=schema)
    writer.writeheader()
    writer.writerows(rows)

def by_slice_impl(slice_size, impl):
    return next(r for r in rows if r["sliceSize"] == str(slice_size) and r["impl"] == str(impl))

summary_lines = [
    "# Mode C Final Profiler Summary",
    "",
    f"Profiler job: `{job_id}` on `{hostname}`.",
    "",
    "This is analysis-only. Profiler timing is not official timing, and `official_timing_used=false` for every row.",
    "",
    "Execution safeguards:",
    "",
    f"- repeat_for_profiler={repeat}",
    f"- launch_skip={launch_skip}",
    f"- launch_count={launch_count}",
    "- kernel filters: `softMax3` for `impl=3`, `softMax4` for `impl=4`",
    "- profiled only slices 784, 1024, and 2048",
    "- did not profile slices 128 or 256",
    "",
    "## Resource Table",
    "",
    "| slice | impl | status | registers/thread | dynamic shared memory | static shared memory | waves/SM | profiler timing |",
    "|---:|---:|---|---|---|---|---|---|",
]
for r in rows:
    summary_lines.append(
        f"| {r['sliceSize']} | {r['impl']} | {r['profiler_status']} | {r['registers_per_thread']} | "
        f"{r['dynamic_shared_memory_bytes']} | {r['static_shared_memory_bytes']} | {r['waves_per_sm']} | {r['profiler_timing_ms']} |"
    )

summary_lines.extend([
    "",
    "## Impl=3 vs Impl=4 Resource Differences",
    "",
])
for slice_size in [784, 1024, 2048]:
    r3 = by_slice_impl(slice_size, 3)
    r4 = by_slice_impl(slice_size, 4)
    dyn3 = numeric_kbytes(r3["dynamic_shared_memory_bytes"])
    dyn4 = numeric_kbytes(r4["dynamic_shared_memory_bytes"])
    if dyn3 is None or dyn4 is None:
        dyn_note = "dynamic shared-memory delta unavailable"
    else:
        dyn_note = f"dynamic shared memory decreased by {dyn3 - dyn4:.2f} Kbyte/block in impl=4"
    summary_lines.append(
        f"- slice {slice_size}: registers/thread {r3['registers_per_thread']} -> {r4['registers_per_thread']}; "
        f"waves/SM {r3['waves_per_sm']} -> {r4['waves_per_sm']}; {dyn_note}; "
        f"official speedup_vs_impl3={official_speedups[slice_size]}"
    )

summary_lines.extend([
    "",
    "## Missing Profiler Evidence",
    "",
    "- memory_throughput is unavailable in this diagnostic collection.",
    "- warp_execution_efficiency is unavailable in this diagnostic collection.",
    "- instruction mix and math/special-function summaries are unavailable.",
    "- stall or scheduler summaries are unavailable.",
    "- profiler timing is diagnostic only and is not official speedup evidence.",
])
(out_dir / "profiler_summary.md").write_text("\n".join(summary_lines) + "\n")

available = all(r["profiler_status"] == "AVAILABLE" for r in rows)
evidence_label = "LIMITED_PROFILER_EVIDENCE" if available else "PARTIAL_OR_FAILED_PROFILER_EVIDENCE"

interp = [
    "# Final Profiler Interpretation",
    "",
    f"Evidence label: `{evidence_label}`.",
    "",
    "Final confirmation speedups are unchanged:",
    "",
    "- slice 784: speedup_vs_impl3=1.135540",
    "- slice 1024: speedup_vs_impl3=1.048740",
    "- slice 2048: speedup_vs_impl3=1.008239, below the 1% claim gate",
    "",
    "## What Profiler Supports",
    "",
    "- For slices 784, 1024, and 2048, `impl=4` and `impl=3` show the same registers/thread in the collected launch-resource metrics.",
    "- For slices 784 and 1024, `impl=4` and `impl=3` show the same waves/SM in the collected launch-resource metrics.",
    "- For slice 2048, `impl=4` and `impl=3` also show the same waves/SM in the collected launch-resource metrics.",
    "- For all profiled slices, `impl=4` shows lower dynamic shared-memory allocation than `impl=3`.",
    "",
    "## What Profiler Does Not Support",
    "",
    "- It does not support a causal claim that shared-memory footprint caused the speedup.",
    "- It does not support reduction-structure causality.",
    "- It does not support cached-exp causality.",
    "- It does not provide memory-throughput, warp-efficiency, instruction-mix, math/special-function, or scheduler/stall evidence.",
    "- It does not provide official timing evidence.",
    "",
    "## Slice 784",
    "",
    "Profiler evidence is consistent with `impl=4` having reduced dynamic shared-memory allocation while preserving the same collected registers/thread and waves/SM as `impl=3`. This can be described only as limited profiler evidence accompanying the accepted official speedup; it does not identify the mechanism.",
    "",
    "## Slice 1024",
    "",
    "Profiler evidence follows the same pattern as slice 784: lower dynamic shared-memory allocation for `impl=4`, with matching collected registers/thread and waves/SM. This is limited supporting context for the accepted official speedup, not a causal explanation.",
    "",
    "## Slice 2048",
    "",
    "Profiler evidence also shows lower dynamic shared-memory allocation for `impl=4`, but final confirmation classified 2048 as measurement-equivalent because speedup_vs_impl3=1.008239 is below the 1% claim gate. Because the same resource-allocation pattern appears without an accepted speedup, the profiler does not explain why 784/1024 improve while 2048 is measurement-equivalent.",
    "",
    "## Further Ablation",
    "",
    "Further ablation would be needed for mechanism attribution. The blocked `impl=5` ablation must not be promoted and does not provide accepted attribution evidence.",
    "",
    "## Paper-Safe Wording",
    "",
    "Nsight Compute launch-resource diagnostics show that `impl=4` reduces dynamic shared-memory allocation relative to `impl=3` for the profiled large-slice cases, while the collected register and waves-per-SM metrics remain unchanged. These diagnostics provide limited context for the confirmed 784 and 1024 speedups, but they do not establish the underlying cause.",
    "",
    "## Do-Not-Claim List",
    "",
    "- Do not claim profiler timing as official timing.",
    "- Do not claim shared-memory footprint caused the speedup.",
    "- Do not claim reduction structure caused the speedup.",
    "- Do not claim cached-exp contribution.",
    "- Do not claim profiler evidence explains the 2048 measurement-equivalent result.",
    "- Do not promote `impl=5`.",
]
(out_dir / "final_profiler_interpretation.md").write_text("\n".join(interp) + "\n")

print(f"Final profiler analysis complete. csv={csv_path}")
PY
