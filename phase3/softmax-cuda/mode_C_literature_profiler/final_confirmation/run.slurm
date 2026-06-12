#!/bin/bash
#SBATCH -A ACD115083
#SBATCH --job-name=softmax-c-final
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=03:00:00
#SBATCH --output=/home/r14525078/HeCBench/phase3/softmax-cuda/mode_C_literature_profiler/final_confirmation/result.out
#SBATCH --error=/home/r14525078/HeCBench/phase3/softmax-cuda/mode_C_literature_profiler/final_confirmation/result.err

set -u
set -o pipefail

SRC=/home/r14525078/HeCBench/src/softmax-cuda
CONF=/home/r14525078/HeCBench/phase3/softmax-cuda/mode_C_literature_profiler/final_confirmation
RAW="$CONF/raw"
BUILD_LOG="$CONF/build.log"
CSV="$CONF/results.csv"

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
} | tee "$CONF/environment.txt"

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

python3 - "$CONF" "$CSV" "${SLURM_JOB_ID:-unknown}" "$HOSTNAME_VALUE" "$GPU_NAME" "$CUDA_VERSION" "$BUILD_LOG" "$build_rc" <<'PY'
import csv
import math
import re
import statistics
import sys
from pathlib import Path

conf_dir = Path(sys.argv[1])
csv_path = Path(sys.argv[2])
job_id = sys.argv[3]
hostname = sys.argv[4]
gpu_name = sys.argv[5]
cuda_version = sys.argv[6]
build_log = sys.argv[7]
build_rc = int(sys.argv[8])
raw_dir = conf_dir / "raw"

schema = [
    "benchmark", "mode", "stage", "candidate", "variant", "impl",
    "baseline_impl", "mode_b_impl", "mode_c_impl", "dispatch_selected_impl",
    "numSlice", "sliceSize", "repeat", "trial_id", "time_ms",
    "impl1_time_ms", "impl3_time_ms", "impl4_time_ms",
    "speedup_vs_impl1", "speedup_vs_impl3", "correctness_status",
    "measurement_validity", "speedup_claim_valid", "result_type",
    "mode_c_final_label", "final_confirmation_status", "mean_ms", "min_ms",
    "max_ms", "stddev_ms", "cv", "raw_stdout_path", "raw_stderr_path",
    "build_log_path", "slurm_job_id", "hostname", "gpu_name",
    "cuda_version", "profiler_status", "official_timing_used", "notes",
]

cases = [(100000, 128), (100000, 256), (100000, 784), (100000, 1024), (50000, 2048)]
impls = [1, 3, 4]
large_slices = {784, 1024, 2048}
small_slices = {128, 256}

def fmt(value):
    return f"{value:.6f}" if isinstance(value, float) and not math.isnan(value) else "n/a"

def dispatch_selected(impl, slice_size):
    if impl == 1:
        return 1
    if impl == 3:
        return 2 if slice_size in large_slices else 1
    if impl == 4:
        return 4 if slice_size in large_slices else 1
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

def impl4_claim(slice_size):
    agg4 = aggregates[(slice_size, 4)]
    mean3 = aggregates[(slice_size, 3)]["mean"]
    mean4 = agg4["mean"]
    speed3 = mean3 / mean4 if mean3 and mean4 and not math.isnan(mean3) and not math.isnan(mean4) else math.nan
    if slice_size in small_slices:
        return False, "MEASUREMENT_EQUIVALENT", speed3
    if agg4["correctness"] == "PASS" and agg4["validity"] == "VALID" and not math.isnan(speed3) and speed3 >= 1.01:
        return True, "MODE_C_CANDIDATE", speed3
    return False, "MEASUREMENT_EQUIVALENT", speed3

rows = []
valid_large_claims = []
large_equiv_or_regressed = []
for num_slice, slice_size in cases:
    claim_valid, impl4_result_type, agg_speed3 = impl4_claim(slice_size)
    if slice_size in large_slices:
        valid_large_claims.append(claim_valid)
        large_equiv_or_regressed.append(not claim_valid)
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
            else:
                variant = "impl4_shape_specialized_large_reduce"
                result_type = impl4_result_type
                speedup_valid = "true" if claim_valid else "false"
                if slice_size in small_slices:
                    notes = entry["note"] or "guardrail row: impl=4 dispatches to unchanged impl=1; no small-slice speedup claim"
                elif claim_valid:
                    notes = entry["note"] or "valid large-slice Mode C speedup claim vs impl=3"
                else:
                    notes = entry["note"] or "large-slice row has no accepted additional speedup claim"
            rows.append({
                "benchmark": "softmax-cuda",
                "mode": "Mode_C",
                "stage": "final_confirmation",
                "candidate": "impl4_shape_specialized_large_reduce",
                "variant": variant,
                "impl": str(impl),
                "baseline_impl": "1",
                "mode_b_impl": "3",
                "mode_c_impl": "4",
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
                "speedup_vs_impl3": fmt(speed3) if impl != 3 else "n/a",
                "correctness_status": entry["correctness"],
                "measurement_validity": agg["validity"],
                "speedup_claim_valid": speedup_valid,
                "result_type": result_type,
                "mode_c_final_label": "",
                "final_confirmation_status": "",
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
                "official_timing_used": "true",
                "notes": notes,
            })

all_correct = all(aggregates[(s, i)]["correctness"] == "PASS" for _, s in cases for i in impls)
any_noisy = any(aggregates[(s, i)]["validity"] == "NOISY" for _, s in cases for i in impls)
missing_raw = any(not Path(e["stdout"]).exists() or not Path(e["stderr"]).exists() for entries in raw.values() for e in entries)
complete_rows = len(rows) == 45
any_valid_large = any(valid_large_claims)

if build_rc != 0 or not all_correct or missing_raw or not complete_rows:
    mode_c_final_label = "BLOCKED"
    final_confirmation_status = "BLOCKED"
elif any_noisy:
    mode_c_final_label = "INCONCLUSIVE"
    final_confirmation_status = "INCONCLUSIVE"
elif any_valid_large:
    mode_c_final_label = "SUCCESS_WITH_ADDITIONAL_SPEEDUP"
    final_confirmation_status = "CONFIRMED"
else:
    mode_c_final_label = "PARTIAL_SUCCESS"
    final_confirmation_status = "PARTIAL_CONFIRMATION"

for row in rows:
    row["mode_c_final_label"] = mode_c_final_label
    row["final_confirmation_status"] = final_confirmation_status

with csv_path.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=schema)
    writer.writeheader()
    writer.writerows(rows)

def auditor_checks(rows):
    checks = []
    expected = {(s, i) for _, s in cases for i in impls}
    seen = {(int(r["sliceSize"]), int(r["impl"])) for r in rows}
    checks.append(("all_official_cases_present", expected == seen, f"seen={len(seen)} expected={len(expected)}"))
    for impl in impls:
        ok = all((s, impl) in seen for _, s in cases)
        checks.append((f"impl_{impl}_rows_exist_for_every_slice", ok, "required official slices checked"))
    missing = [r for r in rows if not Path(r["raw_stdout_path"]).exists() or not Path(r["raw_stderr_path"]).exists()]
    checks.append(("raw_stdout_stderr_paths_exist", not missing, f"missing={len(missing)}"))
    bad_claims = [r for r in rows if r["speedup_claim_valid"] == "true" and r["correctness_status"] != "PASS"]
    checks.append(("correctness_pass_before_speedup_claim", not bad_claims, f"bad_rows={len(bad_claims)}"))
    impl4_missing_speed3 = [r for r in rows if r["impl"] == "4" and r["speedup_vs_impl3"] == "n/a"]
    checks.append(("speedup_vs_impl3_exists_for_impl4", not impl4_missing_speed3, f"bad_rows={len(impl4_missing_speed3)}"))
    small_claims = [r for r in rows if r["impl"] == "4" and r["sliceSize"] in ("128", "256") and (r["speedup_claim_valid"] != "false" or r["result_type"] != "MEASUREMENT_EQUIVALENT")]
    checks.append(("small_slices_128_256_not_claimed", not small_claims, f"bad_rows={len(small_claims)}"))
    speed1_metric_bad = [r for r in rows if r["impl"] == "4" and r["speedup_claim_valid"] == "true" and r["speedup_vs_impl3"] == "n/a"]
    checks.append(("speedup_vs_impl1_not_main_mode_c_metric", not speed1_metric_bad, f"bad_rows={len(speed1_metric_bad)}"))
    profiler_bad = [r for r in rows if r["profiler_status"] != "NOT_RUN"]
    checks.append(("profiler_status_not_run", not profiler_bad, f"bad_rows={len(profiler_bad)}"))
    official_bad = [r for r in rows if r["official_timing_used"] != "true"]
    checks.append(("official_timing_used_true", not official_bad, f"bad_rows={len(official_bad)}"))
    impl5_mentions = [r for r in rows if r["impl"] == "5" or "impl=5" in r["notes"] or "impl5" in r["variant"]]
    checks.append(("no_impl5_promotion", not impl5_mentions, f"bad_rows={len(impl5_mentions)}"))
    forbidden = ("caused", "causality", "cached-exp", "shared-memory-footprint", "reduction-structure")
    forbidden_rows = [r for r in rows if any(term in r["notes"] for term in forbidden)]
    checks.append(("no_profiler_or_mechanism_causality_claim", not forbidden_rows, f"bad_rows={len(forbidden_rows)}"))
    aggregate_hide = final_confirmation_status == "CONFIRMED" and not any_valid_large
    checks.append(("no_aggregate_only_success_hiding_per_slice_regression", not aggregate_hide, f"any_valid_large={any_valid_large}"))
    return checks

checks = auditor_checks(rows)

def write_checks(path):
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["check", "status", "notes"])
        for name, ok, notes in checks:
            writer.writerow([name, "PASS" if ok else "FAIL", notes])

write_checks(conf_dir / "auditor_report.csv")
write_checks(conf_dir / "contradiction_check.csv")

accepted = []
rejected = []
summary_rows = []
for _, slice_size in cases:
    agg1 = aggregates[(slice_size, 1)]
    agg3 = aggregates[(slice_size, 3)]
    agg4 = aggregates[(slice_size, 4)]
    claim_valid, result_type, speed3 = impl4_claim(slice_size)
    speed1 = agg1["mean"] / agg4["mean"] if agg1["mean"] and agg4["mean"] and not math.isnan(agg1["mean"]) and not math.isnan(agg4["mean"]) else math.nan
    summary_rows.append((slice_size, agg1["mean"], agg3["mean"], agg4["mean"], speed3, speed1, agg4["correctness"], agg4["cv"], result_type, claim_valid))
    if claim_valid:
        accepted.append(f"slice {slice_size}: impl=4 measured speedup_vs_impl3={speed3:.6f}")
    else:
        rejected.append(f"slice {slice_size}: no accepted impl=4 additional speedup claim; result_type={result_type}")

with (conf_dir / "summary.md").open("w") as f:
    f.write("# Mode C Final Confirmation Summary\n\n")
    f.write(f"Slurm job: `{job_id}` on `{hostname}`.\n\n")
    f.write("Candidate: `impl4_shape_specialized_large_reduce`.\n\n")
    f.write("`impl=5` is mentioned only as a blocked ablation artifact and was not run in final confirmation.\n\n")
    f.write("Profiler status for official timing rows: `NOT_RUN`.\n\n")
    f.write("## Per-Slice Table\n\n")
    f.write("| slice | impl=1 mean | impl=3 mean | impl=4 mean | speedup_vs_impl3 | speedup_vs_impl1 | correctness | CV | result_type |\n")
    f.write("|---:|---:|---:|---:|---:|---:|---|---:|---|\n")
    for slice_size, mean1, mean3, mean4, speed3, speed1, correctness, cv, result_type, claim_valid in summary_rows:
        f.write(f"| {slice_size} | {fmt(mean1)} | {fmt(mean3)} | {fmt(mean4)} | {fmt(speed3)} | {fmt(speed1)} | {correctness} | {fmt(cv)} | {result_type} |\n")
    f.write(f"\nMode C final label: `{mode_c_final_label}`\n\n")
    f.write(f"Final confirmation status: `{final_confirmation_status}`\n\n")
    f.write("## Accepted Claims\n\n")
    if accepted:
        for item in accepted:
            f.write(f"- {item}\n")
    else:
        f.write("- none\n")
    f.write("\n## Rejected Claims\n\n")
    if rejected:
        for item in rejected:
            f.write(f"- {item}\n")
    else:
        f.write("- none\n")
    f.write("\n## Do-Not-Claim List\n\n")
    f.write("- Do not claim small-slice Mode C speedup for 128 or 256.\n")
    f.write("- Do not use speedup_vs_impl1 as the main Mode C metric.\n")
    f.write("- Do not use profiler timing as official timing.\n")
    f.write("- Do not claim profiler-supported causality.\n")
    f.write("- Do not claim cached-exp contribution.\n")
    f.write("- Do not claim shared-memory-footprint causality.\n")
    f.write("- Do not claim reduction-structure causality.\n")
    f.write("- Do not promote `impl=5`; it remains a blocked ablation artifact.\n")
    f.write("- Do not hide per-slice regression or measurement equivalence behind aggregate success.\n")

print(f"Final confirmation complete. results={csv_path}")
PY
