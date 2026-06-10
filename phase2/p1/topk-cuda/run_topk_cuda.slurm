#!/bin/bash
#SBATCH --job-name=topk-cuda
#SBATCH --output=result/topk-cuda-%j.out
#SBATCH --error=result/topk-cuda-%j.err
#SBATCH --gres=gpu:1
#SBATCH --account=ACD115083
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4

set -euo pipefail

BENCH_DIR="/home/r14525078/p1/HeCBench/src/topk-cuda"
RESULT_DIR="${BENCH_DIR}/result"
RESULT_FILE="${RESULT_DIR}/topk_cuda_result_${SLURM_JOB_ID}.txt"
ARCH="${ARCH:-sm_70}"
BATCH_SIZE="${BATCH_SIZE:-3072}"
REPEAT="${REPEAT:-1000}"

mkdir -p "${RESULT_DIR}"

module purge
module load cuda/12.8

{
  echo "benchmark=topk-cuda"
  echo "job_id=${SLURM_JOB_ID}"
  echo "node=$(hostname)"
  echo "bench_dir=${BENCH_DIR}"
  echo "arch=${ARCH}"
  echo "batch_size=${BATCH_SIZE}"
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
  ./main "${BATCH_SIZE}" "${REPEAT}"
} 2>&1 | tee "${RESULT_FILE}"

echo "Result file: ${RESULT_FILE}"
