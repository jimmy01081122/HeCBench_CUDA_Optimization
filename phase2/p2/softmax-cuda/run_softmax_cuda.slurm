#!/bin/bash
#SBATCH --job-name=softmax-cuda
#SBATCH --account=ACD115083
#SBATCH --partition=gp1d
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:10:00
#SBATCH --output=/home/r14525078/p2/HeCBench/src/softmax-cuda/result/%x-%j.out
#SBATCH --error=/home/r14525078/p2/HeCBench/src/softmax-cuda/result/%x-%j.err

set -euo pipefail

BENCH_DIR=/home/r14525078/p2/HeCBench/src/softmax-cuda
RESULT_DIR=${BENCH_DIR}/result
RESULT_FILE=${RESULT_DIR}/softmax_cuda_${SLURM_JOB_ID}.txt

cd "${BENCH_DIR}"
mkdir -p "${RESULT_DIR}"

echo "job_id=${SLURM_JOB_ID}"
echo "node=${SLURMD_NODENAME:-unknown}"
echo "date=$(date -Is)"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"

module purge
module load cuda/12.8

nvcc --version
nvidia-smi

make clean || true
make ARCH=sm_70

{
  echo "case numSlice=100000 sliceSize=784 implementation=0 repeat=100"
  ./main 100000 784 0 100
  echo "case numSlice=100000 sliceSize=784 implementation=1 repeat=100"
  ./main 100000 784 1 100
} | tee "${RESULT_FILE}"
