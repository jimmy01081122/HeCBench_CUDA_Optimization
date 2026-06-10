#!/bin/bash
#SBATCH --job-name=topk-cuda
#SBATCH --account=ACD115083
#SBATCH --partition=gp1d
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output=/home/r14525078/p2/HeCBench/src/topk-cuda/result/%x-%j.out
#SBATCH --error=/home/r14525078/p2/HeCBench/src/topk-cuda/result/%x-%j.err

set -euo pipefail

BENCH_DIR=/home/r14525078/p2/HeCBench/src/topk-cuda
RESULT_DIR=${BENCH_DIR}/result
RUN_TAG=${RUN_TAG:-run}
RESULT_TXT=${RESULT_DIR}/${RUN_TAG}-${SLURM_JOB_ID}.txt

cd "${BENCH_DIR}"
mkdir -p "${RESULT_DIR}"

module purge
module load cuda/12.8

{
  echo "run_tag=${RUN_TAG}"
  echo "job_id=${SLURM_JOB_ID}"
  echo "node=${SLURMD_NODENAME:-$(hostname)}"
  echo "date=$(date -Is)"
  echo "cuda_arch=sm_70"
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true
  nvcc --version
  make clean || true
  make ARCH=sm_70
  ./main 3072 100
} 2>&1 | tee "${RESULT_TXT}"
