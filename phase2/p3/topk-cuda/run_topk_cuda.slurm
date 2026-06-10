#!/bin/bash
#SBATCH --job-name=topk-cuda
#SBATCH --account=ACD115083
#SBATCH --partition=gp2d
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=/home/r14525078/p3/HeCBench/src/topk-cuda/result/topk-cuda_%j.out
#SBATCH --error=/home/r14525078/p3/HeCBench/src/topk-cuda/result/topk-cuda_%j.err

set -euo pipefail

BENCH_DIR=/home/r14525078/p3/HeCBench/src/topk-cuda
RESULT_DIR=${BENCH_DIR}/result
VARIANT=${TOPK_VARIANT:-attempt1_cached_workspace}
RESULT_TXT=${RESULT_DIR}/topk-cuda_result_${VARIANT}_${SLURM_JOB_ID}.txt

mkdir -p "${RESULT_DIR}"
cd "${BENCH_DIR}"

{
  echo "job_id=${SLURM_JOB_ID}"
  echo "node=${SLURMD_NODENAME:-$(hostname)}"
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
  echo "benchmark_command=./main 3072 100"
  echo "variant=${VARIANT}"
  echo "date=$(date -Is)"
  echo "pwd=$(pwd)"
  echo
  module purge
  module load cuda/12.8
  echo "loaded_modules:"
  module list 2>&1
  echo
  echo "nvcc_version:"
  nvcc --version
  echo
  echo "build_command=make clean || true && make ARCH=sm_70"
  make clean || true
  make ARCH=sm_70
  echo
  echo "run:"
  ./main 3072 100
} 2>&1 | tee "${RESULT_TXT}"
