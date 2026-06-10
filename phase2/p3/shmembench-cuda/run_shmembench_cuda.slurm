#!/bin/bash
#SBATCH --job-name=shmembench-cuda
#SBATCH --account=ACD115083
#SBATCH --partition=gp1d
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --output=/home/r14525078/p3/HeCBench/src/shmembench-cuda/result/shmembench-cuda_%j.out
#SBATCH --error=/home/r14525078/p3/HeCBench/src/shmembench-cuda/result/shmembench-cuda_%j.err

set -euo pipefail

BENCH_DIR=/home/r14525078/p3/HeCBench/src/shmembench-cuda
RESULT_DIR="${BENCH_DIR}/result"
mkdir -p "${RESULT_DIR}"

cd "${BENCH_DIR}"

echo "job_id=${SLURM_JOB_ID:-unknown}"
echo "node=$(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "date=$(date -Is)"

module purge
module load cuda/12.8

echo "loaded_modules:"
module list 2>&1
echo "nvcc_version:"
nvcc --version
echo "benchmark_command=./main 1000"

make clean || true
make ARCH=sm_70

./main 1000 2>&1 | tee "${RESULT_DIR}/shmembench-cuda_result_${SLURM_JOB_ID}.txt"
