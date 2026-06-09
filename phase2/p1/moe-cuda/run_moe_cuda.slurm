#!/bin/bash
#SBATCH --job-name=moe-cuda
#SBATCH --output=result/moe-cuda-%j.out
#SBATCH --error=result/moe-cuda-%j.err
#SBATCH --gres=gpu:1
#SBATCH --account=ACD115083
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4

set -euo pipefail

BENCH_DIR="/home/r14525078/p1/HeCBench/src/moe-cuda"
RESULT_DIR="${BENCH_DIR}/result"
RESULT_FILE="${RESULT_DIR}/moe_cuda_result_${SLURM_JOB_ID}.txt"
ARCH="${ARCH:-sm_70}"

mkdir -p "${RESULT_DIR}"

module purge
module load cuda/12.8

{
  echo "benchmark=moe-cuda"
  echo "job_id=${SLURM_JOB_ID}"
  echo "node=$(hostname)"
  echo "bench_dir=${BENCH_DIR}"
  echo "arch=${ARCH}"
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
  make run
} 2>&1 | tee "${RESULT_FILE}"

echo "Result file: ${RESULT_FILE}"
