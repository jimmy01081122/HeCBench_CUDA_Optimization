#!/bin/bash
#SBATCH --job-name=simpleMultiDevice-cuda
#SBATCH --account=ACD115083
#SBATCH --partition=gp2d
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:10:00
#SBATCH --output=/home/r14525078/p3/HeCBench/src/simpleMultiDevice-cuda/result/simpleMultiDevice-cuda_%j.out
#SBATCH --error=/home/r14525078/p3/HeCBench/src/simpleMultiDevice-cuda/result/simpleMultiDevice-cuda_%j.err

set -euo pipefail

BENCH_DIR=/home/r14525078/p3/HeCBench/src/simpleMultiDevice-cuda
RESULT_DIR=${BENCH_DIR}/result
RESULT_TXT=${RESULT_DIR}/simpleMultiDevice-cuda_result_${SLURM_JOB_ID}.txt

mkdir -p "${RESULT_DIR}"
cd "${BENCH_DIR}"

{
  echo "job_id=${SLURM_JOB_ID}"
  echo "node=${SLURMD_NODENAME:-$(hostname)}"
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
  echo "benchmark_command=./main 1000"
  echo "date=$(date -Is)"
  echo
  module purge
  module load cuda/12.8
  echo "loaded_modules:"
  module list 2>&1
  echo
  nvcc --version
  echo
  nvidia-smi
  echo
  make clean || true
  make ARCH=sm_70 MAX_GPU=4
  echo
  ./main 1000
} 2>&1 | tee "${RESULT_TXT}"
