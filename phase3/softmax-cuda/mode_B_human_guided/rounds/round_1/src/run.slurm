#!/bin/bash
#SBATCH -A ACD115083
#SBATCH --job-name=softmax-b-r1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:45:00
#SBATCH --output=/home/r14525078/HeCBench/phase3/softmax-cuda/mode_B_human_guided/rounds/round_1/result.out
#SBATCH --error=/home/r14525078/HeCBench/phase3/softmax-cuda/mode_B_human_guided/rounds/round_1/result.err

set -u
set -o pipefail

SRC=/home/r14525078/HeCBench/src/softmax-cuda
RESULT=/home/r14525078/HeCBench/phase3/softmax-cuda/mode_B_human_guided
ROUND="$RESULT/rounds/round_1"
RAW="$ROUND/raw"
BUILD_LOG="$ROUND/build.log"
CSV="$ROUND/results.csv"
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
} | tee "$ROUND/environment.txt"

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
    run_one "$num_slice" "$slice" 2 "$trial"
  done
done

python3 - "$ROUND" "$RESULT/robust_baseline/results.csv" "$CSV" "${SLURM_JOB_ID:-unknown}" "$HOSTNAME_VALUE" "$GPU_NAME" "$CUDA_VERSION" "$BUILD_LOG" <<'PY'
import csv
import math
import re
import statistics
import sys
from pathlib import Path

round_dir = Path(sys.argv[1])
robust_csv = Path(sys.argv[2])
csv_path = Path(sys.argv[3])
job_id = sys.argv[4]
hostname = sys.argv[5]
gpu_name = sys.argv[6]
cuda_version = sys.argv[7]
build_log = sys.argv[8]
raw_dir = round_dir / "raw"

schema = [
    "benchmark", "mode", "round_id", "human_decision", "variant", "impl",
    "baseline_impl", "numSlice", "sliceSize", "repeat", "trial_id",
    "time_ms", "baseline_time_ms", "speedup_vs_impl1",
    "correctness_status", "measurement_validity", "speedup_claim_valid",
    "result_type", "mean_ms", "min_ms", "max_ms", "stddev_ms", "cv",
    "raw_stdout_path", "raw_stderr_path", "build_log_path",
    "slurm_job_id", "hostname", "gpu_name", "cuda_version",
    "profiler_status", "notes",
]

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

def classify_cv(cv, correctness):
    if correctness != "PASS":
        return "INVALID"
    if math.isnan(cv):
        return "INVALID"
    if cv > 0.15:
        return "NOISY"
    if cv > 0.05:
        return "CAUTION"
    return "VALID"

def stats(values):
    finite = [v for v in values if not math.isnan(v)]
    if not finite:
        return math.nan, math.nan, math.nan, math.nan, math.nan
    mean = statistics.mean(finite)
    sd = statistics.stdev(finite) if len(finite) > 1 else 0.0
    cv = sd / mean if mean else math.nan
    return mean, min(finite), max(finite), sd, cv

robust = {}
if robust_csv.exists():
    with robust_csv.open(newline="") as f:
        for row in csv.DictReader(f):
            m = re.search(r"slice=(\d+);", row.get("case", ""))
            if m and row.get("correctness_status") == "PASS":
                try:
                    robust[int(m.group(1))] = float(row["mean"])
                except Exception:
                    pass

cases = [(100000, 128), (100000, 256), (100000, 784), (100000, 1024), (50000, 2048)]
raw = {}
for num_slice, slice_size in cases:
    for impl in [1, 2]:
        key = (slice_size, impl)
        raw[key] = []
        for trial in [1, 2, 3]:
            stdout_path = raw_dir / f"slice_{slice_size}_impl_{impl}_trial_{trial}.stdout"
            stderr_path = raw_dir / f"slice_{slice_size}_impl_{impl}_trial_{trial}.stderr"
            time_ms, correctness, note = parse_run(stdout_path)
            raw[key].append({
                "trial": trial,
                "time_ms": time_ms,
                "correctness": correctness,
                "stdout": str(stdout_path),
                "stderr": str(stderr_path),
                "note": note,
            })

aggregates = {}
for key, entries in raw.items():
    mean, mn, mx, sd, cv = stats([e["time_ms"] for e in entries])
    correctness = "PASS" if entries and all(e["correctness"] == "PASS" for e in entries) and not math.isnan(mean) else "FAIL"
    aggregates[key] = {
        "mean": mean, "min": mn, "max": mx, "stddev": sd, "cv": cv,
        "correctness": correctness,
        "validity": classify_cv(cv, correctness),
    }

rows = []
for num_slice, slice_size in cases:
    paired_base = aggregates[(slice_size, 1)]
    candidate = aggregates[(slice_size, 2)]
    paired_baseline_usable = paired_base["correctness"] == "PASS" and paired_base["validity"] in {"VALID", "CAUTION"}
    fallback_baseline = robust.get(slice_size, math.nan)
    aggregate_baseline = paired_base["mean"] if paired_baseline_usable else fallback_baseline
    fallback_note = "" if paired_baseline_usable else "paired impl=1 baseline invalid/unavailable; used robust baseline job 949640 as fallback" if not math.isnan(fallback_baseline) else "paired impl=1 baseline invalid/unavailable and robust fallback missing"

    for impl in [1, 2]:
        agg = aggregates[(slice_size, impl)]
        for entry in raw[(slice_size, impl)]:
            if impl == 1:
                baseline_time = entry["time_ms"]
                speedup = "n/a"
                speedup_valid = "false"
                result_type = "BASELINE"
                variant = "paired_impl1_baseline"
                notes = entry["note"] or "paired impl=1 baseline measured in same Slurm job"
            else:
                baseline_time = raw[(slice_size, 1)][entry["trial"] - 1]["time_ms"] if paired_baseline_usable else aggregate_baseline
                if entry["correctness"] != "PASS" or candidate["correctness"] != "PASS":
                    speed = math.nan
                    result_type = "INVALID"
                    validity = "INVALID"
                    speedup_valid = "false"
                elif math.isnan(baseline_time) or paired_base["correctness"] != "PASS" and math.isnan(fallback_baseline):
                    speed = math.nan
                    result_type = "INVALID"
                    validity = "INVALID"
                    speedup_valid = "false"
                else:
                    speed = baseline_time / entry["time_ms"] if entry["time_ms"] else math.nan
                    aggregate_speed = aggregate_baseline / candidate["mean"] if candidate["mean"] and not math.isnan(aggregate_baseline) else math.nan
                    validity = candidate["validity"]
                    if validity == "NOISY":
                        result_type = "KERNEL_OPT" if aggregate_speed >= 1.01 else "MEASUREMENT_EQUIVALENT"
                        speedup_valid = "false"
                    elif aggregate_speed < 1.0:
                        result_type = "REGRESSION"
                        speedup_valid = "false"
                    elif aggregate_speed < 1.01:
                        result_type = "MEASUREMENT_EQUIVALENT"
                        speedup_valid = "false"
                    else:
                        result_type = "KERNEL_OPT"
                        speedup_valid = "true"
                variant = "impl2_block_cached_exp_compound"
                notes = "; ".join(filter(None, [
                    entry["note"],
                    fallback_note,
                    "compound candidate: block-per-slice plus shared-memory cached exponentials; do not attribute speedup solely to cached exponentials",
                ]))
                baseline_time = baseline_time if not math.isnan(baseline_time) else math.nan
                speedup = f"{speed:.6f}" if not math.isnan(speed) else "n/a"
                agg = {**agg, "validity": validity}

            rows.append({
                "benchmark": "softmax-cuda",
                "mode": "Mode_B",
                "round_id": "1",
                "human_decision": "Approved",
                "variant": variant,
                "impl": str(impl),
                "baseline_impl": "1",
                "numSlice": str(num_slice),
                "sliceSize": str(slice_size),
                "repeat": "100",
                "trial_id": str(entry["trial"]),
                "time_ms": f"{entry['time_ms']:.6f}" if not math.isnan(entry["time_ms"]) else "n/a",
                "baseline_time_ms": f"{baseline_time:.6f}" if isinstance(baseline_time, float) and not math.isnan(baseline_time) else "n/a",
                "speedup_vs_impl1": speedup,
                "correctness_status": entry["correctness"],
                "measurement_validity": agg["validity"],
                "speedup_claim_valid": speedup_valid,
                "result_type": result_type,
                "mean_ms": f"{agg['mean']:.6f}" if not math.isnan(agg["mean"]) else "n/a",
                "min_ms": f"{agg['min']:.6f}" if not math.isnan(agg["min"]) else "n/a",
                "max_ms": f"{agg['max']:.6f}" if not math.isnan(agg["max"]) else "n/a",
                "stddev_ms": f"{agg['stddev']:.6f}" if not math.isnan(agg["stddev"]) else "n/a",
                "cv": f"{agg['cv']:.6f}" if not math.isnan(agg["cv"]) else "n/a",
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

candidate_rows = [r for r in rows if r["impl"] == "2"]
failed = [r for r in candidate_rows if r["correctness_status"] != "PASS"]
regressed = [r for r in candidate_rows if r["result_type"] == "REGRESSION"]
valid_speedups = [r for r in candidate_rows if r["speedup_claim_valid"] == "true"]
summary = [
    "# softmax-cuda Mode B Round 1 Summary",
    "",
    f"Slurm job: {job_id} on {hostname}",
    "",
    "Candidate: `impl2_block_cached_exp_compound`.",
    "",
    "Attribution: this is a compound candidate combining block-per-slice row parallelism and shared-memory cached exponentials. Results must not be attributed solely to cached exponentials.",
    "",
    f"Candidate failed trial rows: {len(failed)}",
    f"Candidate regression trial rows: {len(regressed)}",
    f"Candidate rows with valid speedup claims: {len(valid_speedups)}",
    "",
    "Per-slice and per-trial data are in `results.csv`; raw stdout/stderr are under `raw/`.",
    "",
    "Profiler status: NOT_RUN.",
]
if failed:
    summary.append("")
    summary.append("Candidate cannot be promoted as a full valid optimization because at least one official-slice trial failed correctness.")
elif regressed:
    summary.append("")
    summary.append("Candidate shows partial or negative behavior; regressions are preserved in `results.csv` and are not averaged away.")
elif valid_speedups:
    summary.append("")
    summary.append("At least one per-slice candidate result has a valid speedup claim, subject to auditor review.")
else:
    summary.append("")
    summary.append("No valid speedup claim was established.")
(round_dir / "round_summary.md").write_text("\n".join(summary) + "\n")
PY

python3 "$AUDITOR" "$CSV" > "$ROUND/auditor_stdout.txt" 2> "$ROUND/auditor_stderr.txt"
auditor_rc=$?
cp "$ROUND/contradiction_check.csv" "$ROUND/auditor_report.csv"
exit "$auditor_rc"
