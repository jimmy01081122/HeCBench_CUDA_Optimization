#!/bin/bash
#SBATCH -J shmembench_cuda
#SBATCH -A ACD115083
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH -t 00:10:00
#SBATCH -o result/shmembench_cuda_%j.out
#SBATCH -e result/shmembench_cuda_%j.err

set -euo pipefail

mkdir -p result
RESULT_FILE="result/shmembench_cuda_result_${SLURM_JOB_ID}.txt"
CSV_FILE="result/shmembench_results_${SLURM_JOB_ID}.csv"

{
  module purge
  module load cuda/12.8

  echo "=== Environment ==="
  module list
  echo "which nvcc: $(which nvcc)"
  nvcc --version
  echo "hostname: $(hostname)"
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
  nvidia-smi -L
  nvidia-smi

  echo "=== Build ==="
  make clean || true
  make ARCH=sm_70

  echo "=== Run ==="
  ./main 1000

  echo "=== CSV parse ==="
  python3 parse_shmembench_results.py "${RESULT_FILE}" "${CSV_FILE}" "${SLURM_JOB_ID}" "$(hostname)"
  echo "CSV_FILE=${CSV_FILE}"
} 2>&1 | tee "${RESULT_FILE}"
