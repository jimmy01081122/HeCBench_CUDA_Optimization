#!/bin/bash
#SBATCH -J simpleMultiDevice_cuda
#SBATCH -A ACD115083
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH -t 00:10:00
#SBATCH -o result/simpleMultiDevice_cuda_%j.out
#SBATCH -e result/simpleMultiDevice_cuda_%j.err

set -euo pipefail

BENCH_DIR="/home/r14525078/p1/HeCBench/src/simpleMultiDevice-cuda"
RESULT_DIR="${BENCH_DIR}/result"
RESULT_FILE="${RESULT_DIR}/simpleMultiDevice_cuda_result_${SLURM_JOB_ID}.txt"
ARCH="${ARCH:-sm_70}"
REPEAT="${REPEAT:-1000}"
MAX_GPU="${MAX_GPU:-4}"

mkdir -p "${RESULT_DIR}"

module purge
module load cuda/12.8

{
  echo "benchmark=simpleMultiDevice-cuda"
  echo "job_id=${SLURM_JOB_ID}"
  echo "node=$(hostname)"
  echo "bench_dir=${BENCH_DIR}"
  echo "arch=${ARCH}"
  echo "repeat=${REPEAT}"
  echo "max_gpu=${MAX_GPU}"
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
  make ARCH="${ARCH}" MAX_GPU="${MAX_GPU}"
  echo

  echo "=== Run ==="
  ./main "${REPEAT}"
} 2>&1 | tee "${RESULT_FILE}"

echo "Result file: ${RESULT_FILE}"
