#!/bin/bash
#SBATCH -A ACD115083
#SBATCH --job-name=softmax-c-sub2
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=03:00:00
#SBATCH --output=/home/r14525078/HeCBench/phase3/softmax-cuda/mode_C_literature_profiler/submission_2/result.out
#SBATCH --error=/home/r14525078/HeCBench/phase3/softmax-cuda/mode_C_literature_profiler/submission_2/result.err

set -u
set -o pipefail

SRC=/home/r14525078/HeCBench/src/softmax-cuda
SUB=/home/r14525078/HeCBench/phase3/softmax-cuda/mode_C_literature_profiler/submission_2
RAW="$SUB/raw"
BUILD_LOG="$SUB/build.log"
CSV="$SUB/results.csv"

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
} | tee "$SUB/environment.txt"

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
    run_one "$num_slice" "$slice" 4 "$trial"
    run_one "$num_slice" "$slice" 5 "$trial"
  done
done

python3 - "$SUB" "$CSV" "${SLURM_JOB_ID:-unknown}" "$HOSTNAME_VALUE" "$GPU_NAME" "$CUDA_VERSION" "$BUILD_LOG" "$build_rc" <<'PY'
import csv
import math
import re
import statistics
import sys
from pathlib import Path

sub_dir = Path(sys.argv[1])
csv_path = Path(sys.argv[2])
job_id = sys.argv[3]
hostname = sys.argv[4]
gpu_name = sys.argv[5]
cuda_version = sys.argv[6]
build_log = sys.argv[7]
build_rc = int(sys.argv[8])
raw_dir = sub_dir / "raw"

schema = [
    "benchmark", "mode", "stage", "submission_id", "variant", "impl",
    "baseline_impl", "mode_b_impl", "submission1_impl", "ablation_impl",
    "dispatch_selected_impl", "numSlice", "sliceSize", "repeat", "trial_id",
    "time_ms", "impl1_time_ms", "impl3_time_ms", "impl4_time_ms",
    "speedup_vs_impl1", "speedup_vs_impl3", "speedup_vs_impl4",
    "correctness_status", "measurement_validity", "speedup_claim_valid",
    "result_type", "mode_c_final_label", "mean_ms", "min_ms", "max_ms",
    "stddev_ms", "cv", "raw_stdout_path", "raw_stderr_path",
    "build_log_path", "slurm_job_id", "hostname", "gpu_name",
    "cuda_version", "profiler_status", "notes",
]

cases = [(100000, 128), (100000, 256), (100000, 784), (100000, 1024), (50000, 2048)]
impls = [1, 3, 4, 5]

def fmt(value):
    return f"{value:.6f}" if isinstance(value, float) and not math.isnan(value) else "n/a"

def dispatch_selected(impl, slice_size):
    if impl == 1:
        return 1
    if impl == 3:
        return 2 if slice_size in (784, 1024, 2048) else 1
    if impl == 4:
        return 4 if slice_size in (784, 1024, 2048) else 1
    if impl == 5:
        return 5 if slice_size in (784, 1024, 2048) else 1
    return impl

def parse_run(stdout_path):
    if build_rc != 0:
        return math.nan, "FAIL", "build failed"
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
    return float(m.group(1)) if m else math.nan, correctness, "" if m else "missing timing"

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

raw = {}
for num_slice, slice_size in cases:
    for impl in impls:
        entries = []
        for trial in [1, 2, 3]:
            stdout_path = raw_dir / f"slice_{slice_size}_impl_{impl}_trial_{trial}.stdout"
            stderr_path = raw_dir / f"slice_{slice_size}_impl_{impl}_trial_{trial}.stderr"
            time_ms, correctness, note = parse_run(stdout_path)
            entries.append({
                "trial": trial,
                "time_ms": time_ms,
                "correctness": correctness,
                "stdout": str(stdout_path),
                "stderr": str(stderr_path),
                "note": note,
            })
        raw[(slice_size, impl)] = entries

aggregates = {}
for key, entries in raw.items():
    mean, mn, mx, sd, cv = stats([e["time_ms"] for e in entries])
    correctness = "PASS" if entries and all(e["correctness"] == "PASS" for e in entries) and not math.isnan(mean) else "FAIL"
    aggregates[key] = {
        "mean": mean,
        "min": mn,
        "max": mx,
        "stddev": sd,
        "cv": cv,
        "correctness": correctness,
        "validity": validity(cv, correctness),
    }

def classify_impl5(slice_size, agg5, speed3, speed4):
    if agg5["correctness"] != "PASS":
        return "INVALID", "false"
    if agg5["validity"] == "NOISY":
        return "ABLATION_ONLY", "false"
    if slice_size in (128, 256):
        return "MEASUREMENT_EQUIVALENT", "false"
    if not math.isnan(speed3) and not math.isnan(speed4) and speed3 >= 1.01 and speed4 >= 1.01 and agg5["validity"] == "VALID":
        return "MODE_C_CANDIDATE", "true"
    return "ABLATION_ONLY", "false"

large_improve_both = []
large_regress_vs4 = []
rows = []
for num_slice, slice_size in cases:
    agg1 = aggregates[(slice_size, 1)]
    agg3 = aggregates[(slice_size, 3)]
    agg4 = aggregates[(slice_size, 4)]
    agg5 = aggregates[(slice_size, 5)]
    mean1, mean3, mean4, mean5 = agg1["mean"], agg3["mean"], agg4["mean"], agg5["mean"]
    agg_speed3 = mean3 / mean5 if mean3 and mean5 and not math.isnan(mean3) and not math.isnan(mean5) else math.nan
    agg_speed4 = mean4 / mean5 if mean4 and mean5 and not math.isnan(mean4) and not math.isnan(mean5) else math.nan
    if slice_size in (784, 1024, 2048):
        large_improve_both.append(agg5["correctness"] == "PASS" and agg5["validity"] == "VALID" and agg_speed3 >= 1.01 and agg_speed4 >= 1.01)
        large_regress_vs4.append(agg5["correctness"] == "PASS" and not math.isnan(agg_speed4) and agg_speed4 < 0.99)

    for impl in impls:
        agg = aggregates[(slice_size, impl)]
        for entry in raw[(slice_size, impl)]:
            trial = entry["trial"]
            time_ms = entry["time_ms"]
            impl1_time = raw[(slice_size, 1)][trial - 1]["time_ms"]
            impl3_time = raw[(slice_size, 3)][trial - 1]["time_ms"]
            impl4_time = raw[(slice_size, 4)][trial - 1]["time_ms"]
            speed1 = impl1_time / time_ms if impl1_time and time_ms and not math.isnan(impl1_time) and not math.isnan(time_ms) else math.nan
            speed3 = impl3_time / time_ms if impl3_time and time_ms and not math.isnan(impl3_time) and not math.isnan(time_ms) else math.nan
            speed4 = impl4_time / time_ms if impl4_time and time_ms and not math.isnan(impl4_time) and not math.isnan(time_ms) else math.nan
            if impl == 1:
                variant = "paired_impl1_baseline"
                result_type = "BASELINE"
                speedup_valid = "false"
                notes = entry["note"] or "paired impl=1 baseline"
            elif impl == 3:
                variant = "mode_b_impl3_shape_dispatch_impl1_small_impl2_large"
                result_type = "MODE_B_BASELINE"
                speedup_valid = "false"
                notes = entry["note"] or "Mode B accepted impl=3 baseline"
            elif impl == 4:
                variant = "impl4_shape_specialized_large_reduce"
                result_type = "MODE_C_CANDIDATE"
                speedup_valid = "false"
                notes = entry["note"] or "Submission 1 accepted candidate reference"
            else:
                variant = "impl5_reduction_structure_ablation"
                result_type, speedup_valid = classify_impl5(slice_size, agg5, agg_speed3, agg_speed4)
                if slice_size in (128, 256):
                    notes = entry["note"] or "guardrail row: impl=5 dispatches to unchanged impl=1; no small-slice speedup claim"
                else:
                    notes = entry["note"] or "partial reduction-structure ablation; strongest allowed wording is plausible contributor"
                    if not math.isnan(agg_speed4) and agg_speed4 < 0.99:
                        notes += "; aggregate regression vs impl=4"
            rows.append({
                "benchmark": "softmax-cuda",
                "mode": "Mode_C",
                "stage": "submission_2",
                "submission_id": "2",
                "variant": variant,
                "impl": str(impl),
                "baseline_impl": "1",
                "mode_b_impl": "3",
                "submission1_impl": "4",
                "ablation_impl": "5",
                "dispatch_selected_impl": str(dispatch_selected(impl, slice_size)),
                "numSlice": str(num_slice),
                "sliceSize": str(slice_size),
                "repeat": "100",
                "trial_id": str(trial),
                "time_ms": fmt(time_ms),
                "impl1_time_ms": fmt(impl1_time),
                "impl3_time_ms": fmt(impl3_time),
                "impl4_time_ms": fmt(impl4_time),
                "speedup_vs_impl1": fmt(speed1) if impl != 1 else "n/a",
                "speedup_vs_impl3": fmt(speed3) if impl in (4, 5) else "n/a",
                "speedup_vs_impl4": fmt(speed4) if impl == 5 else "n/a",
                "correctness_status": entry["correctness"],
                "measurement_validity": agg["validity"],
                "speedup_claim_valid": speedup_valid,
                "result_type": result_type,
                "mode_c_final_label": "PENDING",
                "mean_ms": fmt(agg["mean"]),
                "min_ms": fmt(agg["min"]),
                "max_ms": fmt(agg["max"]),
                "stddev_ms": fmt(agg["stddev"]),
                "cv": fmt(agg["cv"]),
                "raw_stdout_path": entry["stdout"],
                "raw_stderr_path": entry["stderr"],
                "build_log_path": build_log,
                "slurm_job_id": job_id,
                "hostname": hostname,
                "gpu_name": gpu_name,
                "cuda_version": cuda_version,
                "profiler_status": "NOT_RUN",
                "notes": notes,
            })

all_correct = all(aggregates[(s, i)]["correctness"] == "PASS" for _, s in cases for i in impls)
all_present = len(rows) == 60
missing_raw = [r for r in rows if not Path(r["raw_stdout_path"]).exists() or not Path(r["raw_stderr_path"]).exists()]
any_noisy_5 = any(aggregates[(s, 5)]["validity"] == "NOISY" for _, s in cases)
if build_rc != 0 or not all_present or missing_raw or not all_correct:
    final_label = "BLOCKED"
elif any(large_improve_both) and not any(large_regress_vs4):
    final_label = "SUCCESS_WITH_ADDITIONAL_SPEEDUP"
elif any_noisy_5:
    final_label = "INCONCLUSIVE"
else:
    final_label = "PARTIAL_SUCCESS"
for row in rows:
    row["mode_c_final_label"] = final_label

with csv_path.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=schema)
    writer.writeheader()
    writer.writerows(rows)

checks = []
expected = {(s, i) for s in (128, 256, 784, 1024, 2048) for i in impls}
present = {(int(r["sliceSize"]), int(r["impl"])) for r in rows}
checks.append(("all_official_cases_and_impls_present", "PASS" if present == expected and len(rows) == 60 else "FAIL", f"rows={len(rows)}"))
checks.append(("raw_paths_exist", "PASS" if not missing_raw else "FAIL", f"missing={len(missing_raw)}"))
bad_claim = [r for r in rows if r["correctness_status"] != "PASS" and r["speedup_claim_valid"] == "true"]
checks.append(("correctness_before_speedup_claim", "PASS" if not bad_claim else "FAIL", f"bad_rows={len(bad_claim)}"))
missing_speeds = [r for r in rows if r["impl"] == "5" and (r["speedup_vs_impl3"] == "n/a" or r["speedup_vs_impl4"] == "n/a")]
checks.append(("impl5_speedups_present", "PASS" if not missing_speeds else "FAIL", f"bad_rows={len(missing_speeds)}"))
small_claims = [r for r in rows if r["impl"] == "5" and r["sliceSize"] in {"128", "256"} and r["speedup_claim_valid"] == "true"]
checks.append(("small_slices_guardrail_only", "PASS" if not small_claims else "FAIL", f"bad_rows={len(small_claims)}"))
promoted = [r for r in rows if r["impl"] == "5" and r["result_type"] == "MODE_C_CANDIDATE"]
bad_promoted = [r for r in promoted if r["speedup_claim_valid"] != "true"]
checks.append(("impl5_promotion_rule", "PASS" if not bad_promoted else "FAIL", f"bad_rows={len(bad_promoted)}"))
prof_timing = [r for r in rows if r["profiler_status"] != "NOT_RUN"]
checks.append(("profiler_timing_not_used", "PASS" if not prof_timing else "FAIL", f"bad_rows={len(prof_timing)}"))
causal_notes = [r for r in rows if any(term in r["notes"].lower() for term in ("caused speedup", "proves", "proof of causality"))]
checks.append(("no_overclaim_causality", "PASS" if not causal_notes else "FAIL", f"bad_rows={len(causal_notes)}"))

with (sub_dir / "auditor_report.csv").open("w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["check", "status", "notes"])
    writer.writerows(checks)
with (sub_dir / "contradiction_check.csv").open("w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["check", "status", "notes"])
    writer.writerows(checks)

accepted = []
rejected = []
summary = [
    "# Mode C Submission 2 Summary",
    "",
    f"Slurm job: `{job_id}` on `{hostname}`.",
    "",
    "Candidate: `impl5_reduction_structure_ablation`.",
    "",
    "This is a partial reduction-structure ablation, not proof of causality.",
    "",
    "Profiler status for official timing rows: `NOT_RUN`.",
    "",
    "## Per-Slice Comparison",
    "",
    "| slice | impl=1 mean | impl=3 mean | impl=4 mean | impl=5 mean | speedup_vs_impl3 | speedup_vs_impl4 | correctness | CV | result_type |",
    "|---:|---:|---:|---:|---:|---:|---:|---|---:|---|",
]
for num_slice, slice_size in cases:
    agg1 = aggregates[(slice_size, 1)]
    agg3 = aggregates[(slice_size, 3)]
    agg4 = aggregates[(slice_size, 4)]
    agg5 = aggregates[(slice_size, 5)]
    speed3 = agg3["mean"] / agg5["mean"] if agg3["mean"] and agg5["mean"] and not math.isnan(agg3["mean"]) and not math.isnan(agg5["mean"]) else math.nan
    speed4 = agg4["mean"] / agg5["mean"] if agg4["mean"] and agg5["mean"] and not math.isnan(agg4["mean"]) and not math.isnan(agg5["mean"]) else math.nan
    result_type = next(r["result_type"] for r in rows if r["impl"] == "5" and r["sliceSize"] == str(slice_size))
    summary.append(f"| {slice_size} | {fmt(agg1['mean'])} | {fmt(agg3['mean'])} | {fmt(agg4['mean'])} | {fmt(agg5['mean'])} | {fmt(speed3)} | {fmt(speed4)} | {agg5['correctness']} | {fmt(agg5['cv'])} | {result_type} |")
    if result_type == "MODE_C_CANDIDATE":
        accepted.append(f"slice {slice_size}: impl=5 validly improves over impl=3 and impl=4")
    else:
        rejected.append(f"slice {slice_size}: no accepted impl=5 optimization speedup claim; result_type={result_type}")

summary.extend([
    "",
    f"Submission-level final label: `{final_label}`",
    "",
    "## Explicit Accepted Claims",
    "",
])
summary.extend([f"- {x}" for x in accepted] if accepted else ["- none"])
summary.extend([
    "",
    "## Explicit Rejected Claims",
    "",
])
summary.extend([f"- {x}" for x in rejected] if rejected else ["- none"])
summary.extend([
    "",
    "## Do-Not-Claim List",
    "",
    "- Do not claim `impl=5` proves reduction structure caused speedup.",
    "- Do not claim shared-memory footprint caused speedup.",
    "- Do not claim cached-exp contribution.",
    "- Do not use profiler timing as official timing.",
    "- Do not claim 128/256 small-slice speedup; these are guardrail rows.",
    "- Do not promote `impl=5` over `impl=4` unless the promotion rule is satisfied.",
])
(sub_dir / "summary.md").write_text("\n".join(summary) + "\n")
PY

echo "Submission 2 complete. results=$CSV"
