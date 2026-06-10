#!/bin/bash
#SBATCH --job-name=prefetch-cuda
#SBATCH --account=ACD115083
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:10:00
#SBATCH --output=/home/r14525078/p2/HeCBench/src/prefetch-cuda/result/slurm_%j.out
#SBATCH --error=/home/r14525078/p2/HeCBench/src/prefetch-cuda/result/slurm_%j.err

set -euo pipefail

cd /home/r14525078/p2/HeCBench/src/prefetch-cuda
mkdir -p result

RESULT_FILE="result/${RUN_LABEL:-run}_${SLURM_JOB_ID}.txt"

{
  echo "label=${RUN_LABEL:-run}"
  echo "job_id=${SLURM_JOB_ID}"
  echo "node=${SLURMD_NODENAME:-$(hostname)}"
  echo "date=$(date -Is)"
  echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
  echo "module_list_begin"
  module list 2>&1 || true
  echo "module_list_end"

  module purge
  module load cuda/12.8

  echo "nvcc_version_begin"
  nvcc --version
  echo "nvcc_version_end"

  make clean || true
  make ARCH=sm_70

  echo "run_repeat_10_begin"
  ./main 10
  echo "run_repeat_10_end"

  echo "run_repeat_100_begin"
  ./main 100
  echo "run_repeat_100_end"
} 2>&1 | tee "$RESULT_FILE"
