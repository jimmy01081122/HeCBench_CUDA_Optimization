#!/bin/bash
#SBATCH --job-name=moe-cuda
#SBATCH --account=ACD115083
#SBATCH --partition=gp1d
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:20:00
#SBATCH --output=/home/r14525078/p2/HeCBench/src/moe-cuda/result/%x-%j.out
#SBATCH --error=/home/r14525078/p2/HeCBench/src/moe-cuda/result/%x-%j.err

set -euo pipefail

cd /home/r14525078/p2/HeCBench/src/moe-cuda
mkdir -p result

MODE="${MODE:-baseline}"
RESULT_FILE="result/${MODE}_${SLURM_JOB_ID}.txt"

{
  echo "mode=${MODE}"
  echo "job_id=${SLURM_JOB_ID}"
  echo "node=${SLURMD_NODENAME:-unknown}"
  echo "date=$(date -Is)"
  echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true
  module purge
  module load cuda/12.8
  nvcc --version
  make clean || true
  make ARCH=sm_70
  ./main 32768 384 1 1000
  ./main 32768 384 2 1000
  ./main 32768 384 4 1000
  ./main 32768 384 8 1000
} 2>&1 | tee "${RESULT_FILE}"
