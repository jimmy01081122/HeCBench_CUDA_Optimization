#!/bin/bash
#SBATCH --job-name=moe-cuda
#SBATCH --account=ACD115083
#SBATCH --partition=gp1d
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --output=/home/r14525078/p3/HeCBench/src/moe-cuda/result/moe-cuda_%j.out
#SBATCH --error=/home/r14525078/p3/HeCBench/src/moe-cuda/result/moe-cuda_%j.err
#SBATCH --time=00:20:00

set -euo pipefail

BENCH_DIR=/home/r14525078/p3/HeCBench/src/moe-cuda
RESULT_DIR=${BENCH_DIR}/result
RESULT_FILE=${RESULT_DIR}/moe-cuda_result_${SLURM_JOB_ID}.txt

mkdir -p "${RESULT_DIR}"
cd "${BENCH_DIR}"

module purge
module load cuda/12.8

{
  echo "job_id=${SLURM_JOB_ID}"
  echo "node=$(hostname)"
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
  echo "date=$(date -Is)"
  echo "pwd=$(pwd)"
  echo "modules:"
  module list 2>&1 || true
  echo "nvcc:"
  nvcc --version
  echo "benchmark commands:"
  echo "make clean || true && make ARCH=sm_70"
  echo "./main 32768 384 1 1000"
  echo "./main 32768 384 2 1000"
  echo "./main 32768 384 4 1000"
  echo "./main 32768 384 8 1000"
  echo "trials=${TRIALS:-1}"
} | tee "${RESULT_FILE}"

{
  make clean || true
  make ARCH=sm_70
  for trial in $(seq 1 "${TRIALS:-1}"); do
    echo "TRIAL ${trial}"
    ./main 32768 384 1 1000
    ./main 32768 384 2 1000
    ./main 32768 384 4 1000
    ./main 32768 384 8 1000
  done
} 2>&1 | tee -a "${RESULT_FILE}"
