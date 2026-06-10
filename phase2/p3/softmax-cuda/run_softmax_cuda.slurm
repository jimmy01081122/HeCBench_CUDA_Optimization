#!/bin/bash
#SBATCH --job-name=softmax-cuda
#SBATCH --account=ACD115083
#SBATCH --partition=gp1d
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:20:00
#SBATCH --output=/home/r14525078/p3/HeCBench/src/softmax-cuda/result/softmax-cuda_%j.out
#SBATCH --error=/home/r14525078/p3/HeCBench/src/softmax-cuda/result/softmax-cuda_%j.err

set -euo pipefail

BENCH_DIR=/home/r14525078/p3/HeCBench/src/softmax-cuda
RESULT_DIR=${BENCH_DIR}/result
RESULT_FILE=${RESULT_DIR}/softmax-cuda_result_${SLURM_JOB_ID}.txt
TRIALS=${TRIALS:-1}

mkdir -p "${RESULT_DIR}"
cd "${BENCH_DIR}"

{
  echo "job_id=${SLURM_JOB_ID}"
  echo "node=${SLURMD_NODENAME:-unknown}"
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
  echo "date=$(date -Is)"
  echo "pwd=$(pwd)"
  echo "module purge"
  module purge
  echo "module load cuda/12.8"
  module load cuda/12.8
  echo "trials=${TRIALS}"
  echo "loaded_modules:"
  module list 2>&1
  echo "nvcc_version:"
  nvcc --version
  echo "nvidia_smi:"
  nvidia-smi || true

  echo "build_command=make clean || true && make ARCH=sm_70"
  make clean || true
  make ARCH=sm_70

  for trial in $(seq 1 "${TRIALS}"); do
    echo "trial=${trial}"
    for slice in 64 128 256 512 784 1024 2048; do
      for impl in 0 1; do
        echo "benchmark_command=./main 100000 ${slice} ${impl} 100"
        ./main 100000 "${slice}" "${impl}" 100
      done
    done
  done
} 2>&1 | tee "${RESULT_FILE}"
