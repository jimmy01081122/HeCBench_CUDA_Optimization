#!/bin/bash
#SBATCH --job-name=shmembench-cuda
#SBATCH --account=ACD115083
#SBATCH --partition=gp1d
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:10:00
#SBATCH --output=/home/r14525078/p2/HeCBench/src/shmembench-cuda/result/%x-%j.out
#SBATCH --error=/home/r14525078/p2/HeCBench/src/shmembench-cuda/result/%x-%j.err

set -euo pipefail

BENCH_DIR=/home/r14525078/p2/HeCBench/src/shmembench-cuda
RESULT_DIR=${BENCH_DIR}/result
RESULT_TXT=${RESULT_DIR}/shmembench_cuda_${SLURM_JOB_ID}.txt

cd "${BENCH_DIR}"
mkdir -p "${RESULT_DIR}"

{
  echo "job_id=${SLURM_JOB_ID}"
  echo "node=${SLURMD_NODENAME:-$(hostname)}"
  echo "date=$(date -Is)"
  echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true
  module purge
  module load cuda/12.8
  nvcc --version
  make clean || true
  make ARCH=sm_70
  ./main 1000
} 2>&1 | tee "${RESULT_TXT}"
