#!/bin/bash
#SBATCH -A ACD115083
#SBATCH --job-name=softmax-c-sub1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:45:00
#SBATCH --output=/home/r14525078/HeCBench/phase3/softmax-cuda/mode_C_literature_profiler/submission_1/result.out
#SBATCH --error=/home/r14525078/HeCBench/phase3/softmax-cuda/mode_C_literature_profiler/submission_1/result.err

set -u
set -o pipefail

SRC=/home/r14525078/HeCBench/src/softmax-cuda
SUB=/home/r14525078/HeCBench/phase3/softmax-cuda/mode_C_literature_profiler/submission_1
RAW="$SUB/raw"
BUILD_LOG="$SUB/build.log"
CSV="$SUB/results.csv"
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
    "baseline_impl", "mode_b_impl", "dispatch_selected_impl", "numSlice",
    "sliceSize", "repeat", "trial_id", "time_ms", "impl1_time_ms",
    "impl3_time_ms", "speedup", "speedup_vs_impl1", "speedup_vs_impl3",
    "correctness_status", "measurement_validity", "speedup_claim_valid",
    "result_type", "mode_c_final_label", "mean_ms", "min_ms", "max_ms",
    "stddev_ms", "cv", "raw_stdout_path", "raw_stderr_path",
    "build_log_path", "slurm_job_id", "hostname", "gpu_name",
    "cuda_version", "profiler_status", "notes",
]

cases = [(100000, 128), (100000, 256), (100000, 784), (100000, 1024), (50000, 2048)]
impls = [1, 3, 4]

def dispatch_selected(impl, slice_size):
    if impl == 1:
        return 1
    if impl == 3:
        return 2 if slice_size in (784, 1024, 2048) else 1
    if impl == 4:
        return 4 if slice_size in (784, 1024, 2048) else 1
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

def aggregate_validity(cv, correctness):
    if correctness != "PASS":
        return "INVALID"
    if math.isnan(cv):
        return "INVALID"
    if cv > 0.15:
        return "NOISY"
    if cv > 0.05:
        return "CAUTION"
    return "VALID"

def fmt(value):
    return f"{value:.6f}" if isinstance(value, float) and not math.isnan(value) else "n/a"

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
        "validity": aggregate_validity(cv, correctness),
    }

def classify_candidate(slice_size, impl, correctness, speed_vs_impl3, agg_validity):
    if correctness != "PASS":
        return "INVALID", "INVALID", "false"
    if agg_validity == "NOISY":
        if impl == 1:
            return "NOISY", "BASELINE", "false"
        return "NOISY", "NOISY", "false"
    if agg_validity == "CAUTION":
        speed_valid = "false"
    else:
        speed_valid = "true"

    if impl == 1:
        return agg_validity, "BASELINE", "false"
    if impl == 3:
        return agg_validity, "MODE_B_BASELINE", "false"
    if slice_size in (128, 256):
        return agg_validity, "MEASUREMENT_EQUIVALENT", "false"
    if math.isnan(speed_vs_impl3):
        return "INVALID", "INVALID", "false"
    if speed_vs_impl3 < 0.99:
        return agg_validity, "REGRESSION", "false"
    if speed_vs_impl3 < 1.01:
        return agg_validity, "MEASUREMENT_EQUIVALENT", "false"
    return agg_validity, "MODE_C_CANDIDATE", speed_valid

large_outcomes = []
rows = []
for num_slice, slice_size in cases:
    agg1 = aggregates[(slice_size, 1)]
    agg3 = aggregates[(slice_size, 3)]
    agg4 = aggregates[(slice_size, 4)]
    mean1 = agg1["mean"]
    mean3 = agg3["mean"]
    mean4 = agg4["mean"]
    aggregate_speed4_vs_3 = mean3 / mean4 if mean3 and mean4 and not math.isnan(mean3) and not math.isnan(mean4) else math.nan
    aggregate_speed4_vs_1 = mean1 / mean4 if mean1 and mean4 and not math.isnan(mean1) and not math.isnan(mean4) else math.nan
    valid_large_improvement = (
        slice_size in (784, 1024, 2048)
        and agg4["correctness"] == "PASS"
        and agg4["validity"] == "VALID"
        and aggregate_speed4_vs_3 >= 1.01
    )
    large_regression = (
        slice_size in (784, 1024, 2048)
        and agg4["correctness"] == "PASS"
        and not math.isnan(aggregate_speed4_vs_3)
        and aggregate_speed4_vs_3 < 0.99
    )
    if slice_size in (784, 1024, 2048):
        large_outcomes.append((slice_size, valid_large_improvement, large_regression))

    for impl in impls:
        agg = aggregates[(slice_size, impl)]
        for entry in raw[(slice_size, impl)]:
            trial = entry["trial"]
            impl1_time = raw[(slice_size, 1)][trial - 1]["time_ms"]
            impl3_time = raw[(slice_size, 3)][trial - 1]["time_ms"]
            time_ms = entry["time_ms"]
            if impl == 1:
                speed_vs_1 = math.nan
                speed_vs_3 = math.nan
                auditor_speed = math.nan
                validity, result_type, speedup_valid = classify_candidate(slice_size, impl, entry["correctness"], math.nan, agg["validity"])
                variant = "paired_impl1_baseline"
                notes = entry["note"] or "paired impl=1 baseline for Mode C Submission 1"
            elif impl == 3:
                speed_vs_1 = impl1_time / time_ms if impl1_time and time_ms and not math.isnan(impl1_time) and not math.isnan(time_ms) else math.nan
                speed_vs_3 = math.nan
                auditor_speed = math.nan
                validity, result_type, speedup_valid = classify_candidate(slice_size, impl, entry["correctness"], math.nan, agg["validity"])
                variant = "mode_b_impl3_shape_dispatch_impl1_small_impl2_large"
                notes = entry["note"] or "Mode B accepted impl=3 baseline measured in same Submission 1 Slurm job"
            else:
                speed_vs_1 = impl1_time / time_ms if impl1_time and time_ms and not math.isnan(impl1_time) and not math.isnan(time_ms) else math.nan
                speed_vs_3 = impl3_time / time_ms if impl3_time and time_ms and not math.isnan(impl3_time) and not math.isnan(time_ms) else math.nan
                validity, result_type, speedup_valid = classify_candidate(slice_size, impl, entry["correctness"], aggregate_speed4_vs_3, agg["validity"])
                variant = "impl4_shape_specialized_large_reduce"
                if slice_size in (128, 256):
                    auditor_speed = math.nan
                    notes = entry["note"] or "impl=4 dispatches to unchanged impl=1 for small official slice; no Mode C speedup claim"
                else:
                    auditor_speed = aggregate_speed4_vs_3
                    notes = entry["note"] or "impl=4 large-slice candidate; compare against paired impl=3 for Mode C claim"

            rows.append({
                "benchmark": "softmax-cuda",
                "mode": "Mode_C",
                "stage": "submission_1",
                "submission_id": "1",
                "variant": variant,
                "impl": str(impl),
                "baseline_impl": "1",
                "mode_b_impl": "3",
                "dispatch_selected_impl": str(dispatch_selected(impl, slice_size)),
                "numSlice": str(num_slice),
                "sliceSize": str(slice_size),
                "repeat": "100",
                "trial_id": str(trial),
                "time_ms": fmt(time_ms),
                "impl1_time_ms": fmt(impl1_time),
                "impl3_time_ms": fmt(impl3_time),
                "speedup": fmt(auditor_speed),
                "speedup_vs_impl1": fmt(speed_vs_1),
                "speedup_vs_impl3": fmt(speed_vs_3),
                "correctness_status": entry["correctness"],
                "measurement_validity": validity,
                "speedup_claim_valid": speedup_valid if result_type == "MODE_C_CANDIDATE" else "false",
                "result_type": result_type,
                "mode_c_final_label": "PENDING_HUMAN_AUDIT",
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

with csv_path.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=schema)
    writer.writeheader()
    writer.writerows(rows)

all_correct = all(aggregates[(s, impl)]["correctness"] == "PASS" for _, s in cases for impl in impls)
missing_raw = [r for r in rows if not Path(r["raw_stdout_path"]).exists() or not Path(r["raw_stderr_path"]).exists()]
any_large_improvement = any(x[1] for x in large_outcomes)
any_large_regression = any(x[2] for x in large_outcomes)
large_noisy = any(aggregates[(s, 4)]["validity"] == "NOISY" for s in (784, 1024, 2048))

if build_rc != 0 or missing_raw:
    final_label = "BLOCKED"
elif not all_correct:
    final_label = "BLOCKED"
elif any_large_improvement and not any_large_regression and not large_noisy:
    final_label = "SUCCESS_WITH_ADDITIONAL_SPEEDUP"
elif any_large_improvement:
    final_label = "PARTIAL_SUCCESS"
elif large_noisy:
    final_label = "INCONCLUSIVE"
else:
    final_label = "INCONCLUSIVE"

for row in rows:
    row["mode_c_final_label"] = final_label
with csv_path.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=schema)
    writer.writeheader()
    writer.writerows(rows)

auditor_checks = []
expected = {(128, 1), (128, 3), (128, 4), (256, 1), (256, 3), (256, 4),
            (784, 1), (784, 3), (784, 4), (1024, 1), (1024, 3), (1024, 4),
            (2048, 1), (2048, 3), (2048, 4)}
present = {(int(r["sliceSize"]), int(r["impl"])) for r in rows}
auditor_checks.append(("all_official_cases_present", "PASS" if present == expected and len(rows) == 45 else "FAIL", f"rows={len(rows)}"))
bad_correct_claim = [r for r in rows if r["correctness_status"] != "PASS" and r["speedup_claim_valid"] == "true"]
auditor_checks.append(("correctness_before_speedup_claim", "PASS" if not bad_correct_claim else "FAIL", f"bad_rows={len(bad_correct_claim)}"))
missing_speedup_vs_impl3 = [r for r in rows if r["impl"] == "4" and r["speedup_vs_impl3"] == "n/a"]
auditor_checks.append(("impl4_speedup_vs_impl3_present", "PASS" if not missing_speedup_vs_impl3 else "FAIL", f"bad_rows={len(missing_speedup_vs_impl3)}"))
small_claim = [r for r in rows if r["impl"] == "4" and r["sliceSize"] in {"128", "256"} and r["speedup_claim_valid"] == "true"]
auditor_checks.append(("small_slices_no_kernel_improvement_claim", "PASS" if not small_claim else "FAIL", f"bad_rows={len(small_claim)}"))
prof_claim = [r for r in rows if "profiler-supported" in r["notes"].lower() and r["profiler_status"] != "RUN"]
auditor_checks.append(("no_profiler_claim_without_data", "PASS" if not prof_claim else "FAIL", f"bad_rows={len(prof_claim)}"))
cached_claim = [r for r in rows if "cached-exp caused" in r["notes"].lower() or "cached exponentials caused" in r["notes"].lower()]
auditor_checks.append(("no_cached_exp_causality_without_ablation", "PASS" if not cached_claim else "FAIL", f"bad_rows={len(cached_claim)}"))
hidden_regression = [x for x in large_outcomes if x[2]]
auditor_checks.append(("no_aggregate_only_hidden_regression", "PASS" if not hidden_regression else "FAIL", f"large_regressions={[x[0] for x in hidden_regression]}"))

with (sub_dir / "auditor_report.csv").open("w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["check", "status", "notes"])
    writer.writerows(auditor_checks)

summary_lines = [
    "# Mode C Submission 1 Summary",
    "",
    f"Slurm job: {job_id} on `{hostname}`.",
    "",
    "Candidate: `impl4_shape_specialized_large_reduce`.",
    "",
    "Profiler status: `NOT_RUN`.",
    "",
    "## Per-Slice Results",
    "",
    "| slice | impl=1 mean ms | impl=3 mean ms | impl=4 mean ms | speedup_vs_impl3 | correctness | validity | result_type |",
    "|---:|---:|---:|---:|---:|---|---|---|",
]
for num_slice, slice_size in cases:
    agg1 = aggregates[(slice_size, 1)]
    agg3 = aggregates[(slice_size, 3)]
    agg4 = aggregates[(slice_size, 4)]
    speed43 = agg3["mean"] / agg4["mean"] if agg3["mean"] and agg4["mean"] and not math.isnan(agg3["mean"]) and not math.isnan(agg4["mean"]) else math.nan
    impl4_rows = [r for r in rows if r["impl"] == "4" and r["sliceSize"] == str(slice_size)]
    result_type = impl4_rows[0]["result_type"] if impl4_rows else "n/a"
    summary_lines.append(
        f"| {slice_size} | {fmt(agg1['mean'])} | {fmt(agg3['mean'])} | {fmt(agg4['mean'])} | "
        f"{fmt(speed43)} | {agg4['correctness']} | {agg4['validity']} | {result_type} |"
    )
summary_lines.extend([
    "",
    "## Interpretation",
    "",
    "- Primary Mode C comparison is `impl=4` vs `impl=3`.",
    "- Small slices 128 and 256 dispatch to unchanged `impl=1`; no Mode C optimization speedup is claimed for them.",
    "- Profiler was not run, so there is no profiler-supported bottleneck conclusion.",
    "- No cached-exp causality claim is made because no ablation was run.",
    f"- Submission 1 label before human audit: `{final_label}`.",
])
(sub_dir / "summary.md").write_text("\n".join(summary_lines) + "\n")
PY

python3 "$AUDITOR" "$CSV" > "$SUB/auditor_stdout.txt" 2> "$SUB/auditor_stderr.txt"
auditor_rc=$?
if [ "$auditor_rc" -ne 0 ]; then
  echo "self_consistency_auditor.py failed with rc=${auditor_rc}"
fi

echo "Submission 1 complete. results=$CSV"
