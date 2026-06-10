#!/bin/bash
#SBATCH --job-name=softmax-cuda
#SBATCH --output=result/softmax-cuda-%j.out
#SBATCH --error=result/softmax-cuda-%j.err
#SBATCH --gres=gpu:1
#SBATCH --account=ACD115083
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4

set -euo pipefail

BENCH_DIR="/home/r14525078/p1/HeCBench/src/softmax-cuda"
RESULT_DIR="${BENCH_DIR}/result"
RESULT_FILE="${RESULT_DIR}/softmax_cuda_result_${SLURM_JOB_ID}.txt"
ARCH="${ARCH:-sm_70}"
NUM_SLICE="${NUM_SLICE:-100000}"
REPEAT="${REPEAT:-100}"

mkdir -p "${RESULT_DIR}"

module purge
module load cuda/12.8

{
  echo "benchmark=softmax-cuda"
  echo "job_id=${SLURM_JOB_ID}"
  echo "node=$(hostname)"
  echo "bench_dir=${BENCH_DIR}"
  echo "arch=${ARCH}"
  echo "num_slice=${NUM_SLICE}"
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
  echo "slice_size,implementation,avg_ms,status"
  for slice_size in 16 32 64 128 256 512 784 1024; do
    for implementation in 0 1; do
      output="$(./main "${NUM_SLICE}" "${slice_size}" "${implementation}" "${REPEAT}")"
      avg_ms="$(printf '%s\n' "${output}" | awk '/Average kernel execution time/ {print $5}')"
      status="$(printf '%s\n' "${output}" | awk '/^(PASS|FAIL)$/ {print $1}')"
      printf '%s,%s,%s,%s\n' "${slice_size}" "${implementation}" "${avg_ms}" "${status}"
    done
  done
} 2>&1 | tee "${RESULT_FILE}"

echo "Result file: ${RESULT_FILE}"
