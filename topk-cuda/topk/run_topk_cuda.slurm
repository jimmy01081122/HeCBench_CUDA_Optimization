#!/bin/bash
#SBATCH -J topk_cuda
#SBATCH -A ACD115083
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH -t 00:20:00
#SBATCH -o result/topk_cuda_%j.out
#SBATCH -e result/topk_cuda_%j.err

set -euo pipefail

mkdir -p result
RESULT_FILE="result/topk_cuda_result_${SLURM_JOB_ID}.txt"
CSV_FILE="result/topk_cuda_result_${SLURM_JOB_ID}.csv"
TOPK_BATCH_SIZE="${TOPK_BATCH_SIZE:-3072}"
TOPK_REPEAT="${TOPK_REPEAT:-100}"
TOPK_VARIANT="${TOPK_VARIANT:-cached_workspace_async}"

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
  echo "Variant: ${TOPK_VARIANT}"
  echo "Command: ./main ${TOPK_BATCH_SIZE} ${TOPK_REPEAT}"
  ./main "${TOPK_BATCH_SIZE}" "${TOPK_REPEAT}"
} 2>&1 | tee "${RESULT_FILE}"

python3 parse_topk_results.py "${RESULT_FILE}" "${CSV_FILE}" "${SLURM_JOB_ID}" "${TOPK_VARIANT}"
