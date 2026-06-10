#!/bin/bash
#SBATCH --job-name=prefetch-cuda
#SBATCH --output=result/prefetch-cuda-%j.out
#SBATCH --error=result/prefetch-cuda-%j.err
#SBATCH --gres=gpu:1
#SBATCH --account=ACD115083
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4

set -euo pipefail

BENCH_DIR="/home/r14525078/p1/HeCBench/src/prefetch-cuda"
RESULT_DIR="${BENCH_DIR}/result"
RESULT_FILE="${RESULT_DIR}/prefetch_cuda_result_${SLURM_JOB_ID}.txt"
ARCH="${ARCH:-sm_70}"
REPEAT="${REPEAT:-100}"

mkdir -p "${RESULT_DIR}"

module purge
module load cuda/12.8

{
  echo "benchmark=prefetch-cuda"
  echo "job_id=${SLURM_JOB_ID}"
  echo "node=$(hostname)"
  echo "bench_dir=${BENCH_DIR}"
  echo "arch=${ARCH}"
  echo "repeat=${REPEAT}"
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
  echo

  echo "=== Environment ==="
  module list
  which nvcc
  nvcc --version
  nvidia-smi -L
  echo

  echo "=== Build ==="
  cd "${BENCH_DIR}"
  make clean
  make ARCH="${ARCH}"
  echo

  echo "=== Run ==="
  output="$(./main "${REPEAT}")"
  printf '%s\n' "${output}"
  echo

  echo "mode,avg_ms,passes"
  with_avg="$(printf '%s\n' "${output}" | awk '
    /Concurrent managed access with prefetch/ {mode="with_prefetch"; next}
    /Concurrent managed access without prefetch/ {mode="without_prefetch"; next}
    /Average execution time:/ && mode=="with_prefetch" {sum += $4; count += 1}
    END {if (count > 0) printf "%.6f", sum / count}
  ')"
  without_avg="$(printf '%s\n' "${output}" | awk '
    /Concurrent managed access with prefetch/ {mode="with_prefetch"; next}
    /Concurrent managed access without prefetch/ {mode="without_prefetch"; next}
    /Average execution time:/ && mode=="without_prefetch" {sum += $4; count += 1}
    END {if (count > 0) printf "%.6f", sum / count}
  ')"
  with_passes="$(printf '%s\n' "${output}" | awk '
    /Concurrent managed access with prefetch/ {mode="with_prefetch"; next}
    /Concurrent managed access without prefetch/ {mode="without_prefetch"; next}
    /^PASS$/ && mode=="with_prefetch" {count += 1}
    END {print count + 0}
  ')"
  without_passes="$(printf '%s\n' "${output}" | awk '
    /Concurrent managed access with prefetch/ {mode="with_prefetch"; next}
    /Concurrent managed access without prefetch/ {mode="without_prefetch"; next}
    /^PASS$/ && mode=="without_prefetch" {count += 1}
    END {print count + 0}
  ')"
  printf 'with_prefetch,%s,%s/10\n' "${with_avg}" "${with_passes}"
  printf 'without_prefetch,%s,%s/10\n' "${without_avg}" "${without_passes}"
} 2>&1 | tee "${RESULT_FILE}"

echo "Result file: ${RESULT_FILE}"
