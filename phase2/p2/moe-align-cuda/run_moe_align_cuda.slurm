#!/bin/bash
#SBATCH --job-name=moe-align-cuda
#SBATCH --account=ACD115083
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:20:00
#SBATCH --output=/home/r14525078/p2/HeCBench/src/moe-align-cuda/result/%x-%j.out
#SBATCH --error=/home/r14525078/p2/HeCBench/src/moe-align-cuda/result/%x-%j.err

set -euo pipefail

BENCH_DIR=/home/r14525078/p2/HeCBench/src/moe-align-cuda
RESULT_DIR=${BENCH_DIR}/result
TAG=${MOE_ALIGN_TAG:-baseline}
REPEAT=${MOE_ALIGN_REPEAT:-100000}
RESULT_TXT=${RESULT_DIR}/${TAG}-${SLURM_JOB_ID}.txt

mkdir -p "${RESULT_DIR}"
cd "${BENCH_DIR}"

{
  echo "tag=${TAG}"
  echo "job_id=${SLURM_JOB_ID}"
  echo "node=${SLURMD_NODENAME:-unknown}"
  echo "date=$(date -Is)"
  echo "repeat=${REPEAT}"
  echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
  echo "hostname=$(hostname)"
} | tee "${RESULT_TXT}"

module purge
module load cuda/12.8

{
  echo "nvcc=$(command -v nvcc)"
  nvcc --version
  nvidia-smi
  make clean || true
  make ARCH=sm_70
  ./main "${REPEAT}"
} 2>&1 | tee -a "${RESULT_TXT}"
