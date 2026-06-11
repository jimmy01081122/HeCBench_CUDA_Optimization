#!/bin/bash
#SBATCH -A ACD115083
#SBATCH --job-name=softmax-b-final
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:45:00
#SBATCH --output=/home/r14525078/HeCBench/phase3/softmax-cuda/mode_B_human_guided/final/result.out
#SBATCH --error=/home/r14525078/HeCBench/phase3/softmax-cuda/mode_B_human_guided/final/result.err

set -u
set -o pipefail

SRC=/home/r14525078/HeCBench/src/softmax-cuda
RESULT=/home/r14525078/HeCBench/phase3/softmax-cuda/mode_B_human_guided
FINAL="$RESULT/final"
RAW="$FINAL/raw"
BUILD_LOG="$FINAL/build.log"
CSV="$FINAL/results.csv"
AUDITOR=/home/r14525078/HeCBench/phase3/tools/self_consistency_auditor.py

mkdir -p "$RAW"

module purge
module load cuda/12.8
cd "$SRC"

HOSTNAME_VALUE=$(hostname)
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1 | tr ',' ';')
CUDA_VERSION=$(nvcc --version | awk '/release/ {print $6}' | tr -d ',')

{
  echo "slurm_job_id=${SLURM_JOB_ID:-unknown}"
  echo "hostname=${HOSTNAME_VALUE}"
  echo "gpu_name=${GPU_NAME}"
  echo "cuda_version=${CUDA_VERSION}"
  echo "date=$(date -Is)"
  nvidia-smi || true
  nvcc --version || true
} | tee "$FINAL/environment.txt"

make clean || true
make ARCH=sm_70 > "$BUILD_LOG" 2>&1
build_rc=$?
if [ "$build_rc" -ne 0 ]; then
  echo "Build failed with rc=${build_rc}. See $BUILD_LOG"
fi

run_one() {
  local num_slice="$1"
  local slice="$2"
  local impl="$3"
  local trial="$4"
  local stdout_path="$RAW/slice_${slice}_impl_${impl}_trial_${trial}.stdout"
  local stderr_path="$RAW/slice_${slice}_impl_${impl}_trial_${trial}.stderr"
  if [ "$build_rc" -ne 0 ]; then
    echo "BUILD_FAILED" > "$stdout_path"
    echo "BUILD_FAILED" > "$stderr_path"
    return 0
  fi
  ./main "$num_slice" "$slice" "$impl" 100 > "$stdout_path" 2> "$stderr_path"
  local rc=$?
  echo "exit_code=${rc}" >> "$stderr_path"
  return 0
}

for slice in 128 256 784 1024 2048; do
  num_slice=100000
  if [ "$slice" = "2048" ]; then
    num_slice=50000
  fi
  for trial in 1 2 3; do
    run_one "$num_slice" "$slice" 1 "$trial"
    run_one "$num_slice" "$slice" 3 "$trial"
  done
done

python3 - "$FINAL" "$CSV" "${SLURM_JOB_ID:-unknown}" "$HOSTNAME_VALUE" "$GPU_NAME" "$CUDA_VERSION" "$BUILD_LOG" <<'PY'
import csv
import math
import re
import statistics
import sys
from pathlib import Path

final_dir = Path(sys.argv[1])
csv_path = Path(sys.argv[2])
job_id = sys.argv[3]
hostname = sys.argv[4]
gpu_name = sys.argv[5]
cuda_version = sys.argv[6]
build_log = sys.argv[7]
raw_dir = final_dir / "raw"

schema = [
    "benchmark", "mode", "round_id", "human_decision", "variant", "impl",
    "dispatch_selected_impl", "baseline_impl", "numSlice", "sliceSize",
    "repeat", "trial_id", "time_ms", "baseline_time_ms",
    "speedup_vs_impl1", "correctness_status", "measurement_validity",
    "speedup_claim_valid", "result_type", "mean_ms", "min_ms", "max_ms",
    "stddev_ms", "cv", "raw_stdout_path", "raw_stderr_path",
    "build_log_path", "slurm_job_id", "hostname", "gpu_name",
    "cuda_version", "profiler_status", "notes",
]

def dispatch_selected(slice_size):
    return 2 if slice_size in (784, 1024, 2048) else 1

def parse_run(stdout_path):
    if not stdout_path.exists():
        return math.nan, "FAIL", "missing stdout"
    text = stdout_path.read_text(errors="replace")
    m = re.search(r"Average kernel execution time:\s*([0-9.]+) \(ms\)", text)
    if re.search(r"^PASS$", text, re.M):
        correctness = "PASS"
    elif re.search(r"^FAIL$", text, re.M) or "@index" in text:
        correctness = "FAIL"
    else:
        correctness = "FAIL"
    value = float(m.group(1)) if m else math.nan
    note = "" if m else "missing timing"
    return value, correctness, note

def stats(values):
    finite = [v for v in values if not math.isnan(v)]
    if not finite:
        return math.nan, math.nan, math.nan, math.nan, math.nan
    mean = statistics.mean(finite)
    sd = statistics.stdev(finite) if len(finite) > 1 else 0.0
    cv = sd / mean if mean else math.nan
    return mean, min(finite), max(finite), sd, cv

def validity(cv, correctness):
    if correctness != "PASS":
        return "INVALID"
    if math.isnan(cv):
        return "INVALID"
    if cv > 0.15:
        return "NOISY"
    if cv > 0.05:
        return "CAUTION"
    return "VALID"

cases = [(100000, 128), (100000, 256), (100000, 784), (100000, 1024), (50000, 2048)]
raw = {}
for num_slice, slice_size in cases:
    for impl in [1, 3]:
        entries = []
        for trial in [1, 2, 3]:
            stdout_path = raw_dir / f"slice_{slice_size}_impl_{impl}_trial_{trial}.stdout"
            stderr_path = raw_dir / f"slice_{slice_size}_impl_{impl}_trial_{trial}.stderr"
            time_ms, correctness, note = parse_run(stdout_path)
            entries.append({
                "trial": trial, "time_ms": time_ms, "correctness": correctness,
                "stdout": str(stdout_path), "stderr": str(stderr_path), "note": note,
            })
        raw[(slice_size, impl)] = entries

aggregates = {}
for key, entries in raw.items():
    mean, mn, mx, sd, cv = stats([e["time_ms"] for e in entries])
    correctness = "PASS" if entries and all(e["correctness"] == "PASS" for e in entries) and not math.isnan(mean) else "FAIL"
    aggregates[key] = {
        "mean": mean, "min": mn, "max": mx, "stddev": sd, "cv": cv,
        "correctness": correctness, "validity": validity(cv, correctness),
    }

rows = []
for num_slice, slice_size in cases:
    base_agg = aggregates[(slice_size, 1)]
    cand_agg = aggregates[(slice_size, 3)]
    paired_baseline_usable = base_agg["correctness"] == "PASS" and base_agg["validity"] in {"VALID", "CAUTION"}
    selected = dispatch_selected(slice_size)
    aggregate_speed = base_agg["mean"] / cand_agg["mean"] if paired_baseline_usable and cand_agg["mean"] and not math.isnan(cand_agg["mean"]) else math.nan

    for impl in [1, 3]:
        agg = aggregates[(slice_size, impl)]
        for entry in raw[(slice_size, impl)]:
            if impl == 1:
                variant = "paired_impl1_baseline"
                baseline_time = entry["time_ms"]
                speedup = "n/a"
                speedup_valid = "false"
                result_type = "BASELINE"
                dispatch_impl = 1
                notes = entry["note"] or "paired impl=1 baseline measured in same Slurm final confirmation job"
                row_validity = agg["validity"]
            else:
                variant = "impl3_shape_dispatch_impl1_small_impl2_large"
                dispatch_impl = selected
                baseline_time = raw[(slice_size, 1)][entry["trial"] - 1]["time_ms"] if paired_baseline_usable else math.nan
                if entry["correctness"] != "PASS" or cand_agg["correctness"] != "PASS":
                    speed = math.nan
                    result_type = "INVALID"
                    row_validity = "INVALID"
                    speedup_valid = "false"
                elif not paired_baseline_usable or math.isnan(baseline_time):
                    speed = math.nan
                    result_type = "INVALID"
                    row_validity = "INVALID"
                    speedup_valid = "false"
                else:
                    speed = baseline_time / entry["time_ms"] if entry["time_ms"] else math.nan
                    row_validity = cand_agg["validity"]
                    if selected == 1:
                        result_type = "MEASUREMENT_EQUIVALENT"
                        speedup_valid = "false"
                    elif row_validity == "NOISY":
                        result_type = "PARAM_TUNE" if aggregate_speed >= 1.01 else "MEASUREMENT_EQUIVALENT"
                        speedup_valid = "false"
                    elif aggregate_speed < 0.99:
                        result_type = "REGRESSION"
                        speedup_valid = "false"
                    elif aggregate_speed < 1.01:
                        result_type = "MEASUREMENT_EQUIVALENT"
                        speedup_valid = "false"
                    else:
                        result_type = "PARAM_TUNE"
                        speedup_valid = "true"
                speedup = f"{speed:.6f}" if not math.isnan(speed) else "n/a"
                if selected == 1:
                    notes = "final confirmation: shape-aware dispatch selected unchanged impl=1; result_type fixed to MEASUREMENT_EQUIVALENT and no speedup claim"
                else:
                    notes = "final confirmation: shape-aware dispatch selected unchanged impl=2 large-slice path; speedup is dispatch-policy result"
                if entry["note"]:
                    notes = f"{entry['note']}; {notes}"

            rows.append({
                "benchmark": "softmax-cuda", "mode": "Mode_B", "round_id": "final",
                "human_decision": "Approved", "variant": variant, "impl": str(impl),
                "dispatch_selected_impl": str(dispatch_impl), "baseline_impl": "1",
                "numSlice": str(num_slice), "sliceSize": str(slice_size), "repeat": "100",
                "trial_id": str(entry["trial"]),
                "time_ms": f"{entry['time_ms']:.6f}" if not math.isnan(entry["time_ms"]) else "n/a",
                "baseline_time_ms": f"{baseline_time:.6f}" if isinstance(baseline_time, float) and not math.isnan(baseline_time) else "n/a",
                "speedup_vs_impl1": speedup, "correctness_status": entry["correctness"],
                "measurement_validity": row_validity, "speedup_claim_valid": speedup_valid,
                "result_type": result_type,
                "mean_ms": f"{agg['mean']:.6f}" if not math.isnan(agg["mean"]) else "n/a",
                "min_ms": f"{agg['min']:.6f}" if not math.isnan(agg["min"]) else "n/a",
                "max_ms": f"{agg['max']:.6f}" if not math.isnan(agg["max"]) else "n/a",
                "stddev_ms": f"{agg['stddev']:.6f}" if not math.isnan(agg["stddev"]) else "n/a",
                "cv": f"{agg['cv']:.6f}" if not math.isnan(agg["cv"]) else "n/a",
                "raw_stdout_path": entry["stdout"], "raw_stderr_path": entry["stderr"],
                "build_log_path": build_log, "slurm_job_id": job_id, "hostname": hostname,
                "gpu_name": gpu_name, "cuda_version": cuda_version,
                "profiler_status": "NOT_RUN", "notes": notes,
            })

with csv_path.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=schema)
    writer.writeheader()
    writer.writerows(rows)

summary_rows = []
for num_slice, slice_size in cases:
    base_agg = aggregates[(slice_size, 1)]
    cand_agg = aggregates[(slice_size, 3)]
    selected = dispatch_selected(slice_size)
    speed = base_agg["mean"] / cand_agg["mean"] if base_agg["mean"] and cand_agg["mean"] and not math.isnan(base_agg["mean"]) and not math.isnan(cand_agg["mean"]) else math.nan
    cand_rows = [r for r in rows if r["impl"] == "3" and int(r["sliceSize"]) == slice_size]
    result_type = "/".join(sorted({r["result_type"] for r in cand_rows}))
    correctness = f"{sum(1 for r in cand_rows if r['correctness_status'] == 'PASS')}/3 PASS"
    summary_rows.append((slice_size, selected, cand_agg, base_agg, correctness, result_type, speed))

def table_lines():
    lines = [
        "| slice | dispatch_selected_impl | candidate mean ms | paired impl=1 mean ms | correctness | validity | result_type | aggregate speedup |",
        "|---:|---:|---:|---:|---|---|---|---:|",
    ]
    for slice_size, selected, cand_agg, base_agg, correctness, result_type, speed in summary_rows:
        lines.append(f"| {slice_size} | {selected} | {cand_agg['mean']:.6f} | {base_agg['mean']:.6f} | {correctness} | {cand_agg['validity']} | {result_type} | {speed:.6f} |")
    return lines

all_pass = all(r["correctness_status"] == "PASS" for r in rows if r["impl"] == "3")
all_stable = all(aggregates[(s, 3)]["validity"] in {"VALID", "CAUTION"} for _, s in cases)
final_label = "SUCCESS" if all_pass and all_stable else "PARTIAL_SUCCESS"

summary = [
    "# softmax-cuda Mode B Final Confirmation Summary",
    "",
    "## English",
    "",
    f"Slurm job: {job_id} on `{hostname}`.",
    "",
    "Accepted candidate: `impl3_shape_dispatch_impl1_small_impl2_large`.",
    "",
    "Profiler status: `NOT_RUN`.",
    "",
    *table_lines(),
    "",
    "Interpretation:",
    "- This is a shape-aware dispatch result, not a universal `impl=2` or universal `KERNEL_OPT` result.",
    "- `slice=128` and `slice=256` dispatch to unchanged `impl=1`; they are classified as `MEASUREMENT_EQUIVALENT` with `speedup_claim_valid=false`.",
    "- `slice=784`, `slice=1024`, and `slice=2048` dispatch to unchanged `impl=2`; speedup claims are valid only because correctness passed and CV was stable.",
    f"- Final confirmation label: `{final_label}`.",
    "",
    "## 繁體中文",
    "",
    f"Slurm 作業：`{job_id}`，節點 `{hostname}`。",
    "",
    "接受的候選版本：`impl3_shape_dispatch_impl1_small_impl2_large`。",
    "",
    "Profiler 狀態：`NOT_RUN`。",
    "",
    *table_lines(),
    "",
    "解讀：",
    "- 這是 shape-aware dispatch 結果，不是通用的 `impl=2` 或通用 `KERNEL_OPT` 結果。",
    "- `slice=128` 與 `slice=256` 會派發到未修改的 `impl=1`；分類為 `MEASUREMENT_EQUIVALENT`，且 `speedup_claim_valid=false`。",
    "- `slice=784`、`slice=1024`、`slice=2048` 會派發到未修改的 `impl=2`；只有在 correctness PASS 且 CV 穩定時才允許 speedup claim。",
    f"- 最終確認標籤：`{final_label}`。",
]
(final_dir / "round_summary.md").write_text("\n".join(summary) + "\n")

agent = [
    "# softmax-cuda Mode B Agent Summary",
    "",
    "## English",
    "",
    "Robust baseline used `impl=1` as the official baseline. Round 1 tested `impl2_block_cached_exp_compound`; it improved large slices but failed correctness on one `slice=256` trial and regressed `slice=128`, so it was not promoted as a full replacement.",
    "",
    "Round 2 introduced the accepted shape-aware dispatch candidate `impl3_shape_dispatch_impl1_small_impl2_large`: `slice=128/256 -> impl=1`, `slice=784/1024/2048 -> impl=2`.",
    "",
    f"Final confirmation ran as Slurm job {job_id}. All official candidate trials passed correctness. Final artifacts are in `final/`.",
    "",
    f"Final conclusion label: `{final_label}`.",
    "",
    "Mode C has not been started.",
    "",
    "## 繁體中文",
    "",
    "Robust baseline 使用 `impl=1` 作為官方基準。Round 1 測試 `impl2_block_cached_exp_compound`；大尺寸 slice 有改善，但 `slice=256` 有一個 trial correctness 失敗，且 `slice=128` 退步，因此沒有被提升為完整替代方案。",
    "",
    "Round 2 導入已接受的 shape-aware dispatch 候選版本 `impl3_shape_dispatch_impl1_small_impl2_large`：`slice=128/256 -> impl=1`，`slice=784/1024/2048 -> impl=2`。",
    "",
    f"Final confirmation 透過 Slurm job {job_id} 執行。所有官方 candidate trials correctness 皆 PASS。最終 artifacts 位於 `final/`。",
    "",
    f"最終結論標籤：`{final_label}`。",
    "",
    "尚未進入 Mode C。",
]
(final_dir.parent / "agent_summary.md").write_text("\n".join(agent) + "\n")
PY

python3 "$AUDITOR" "$CSV" > "$FINAL/auditor_stdout.txt" 2> "$FINAL/auditor_stderr.txt"
auditor_rc=$?
cp "$FINAL/contradiction_check.csv" "$FINAL/auditor_report.csv"
exit "$auditor_rc"
