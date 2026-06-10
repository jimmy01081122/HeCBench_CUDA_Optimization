#!/usr/bin/env bash
#SBATCH --job-name=simpleMultiDevice-cuda
#SBATCH --account=ACD115083
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:4
#SBATCH --time=00:10:00
#SBATCH --output=/home/r14525078/p2/HeCBench/src/simpleMultiDevice-cuda/result/%x-%j.out
#SBATCH --error=/home/r14525078/p2/HeCBench/src/simpleMultiDevice-cuda/result/%x-%j.err

set -euo pipefail

cd /home/r14525078/p2/HeCBench/src/simpleMultiDevice-cuda
mkdir -p result

echo "job_id=${SLURM_JOB_ID:-unknown}"
echo "node=${SLURMD_NODENAME:-${SLURM_NODELIST:-unknown}}"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
echo "date=$(date -Is)"

module purge
module load cuda/12.8

echo "nvcc=$(command -v nvcc)"
nvcc --version
nvidia-smi

make clean || true
make ARCH=sm_70 MAX_GPU=4

tag="${SIMPLE_MULTI_TAG:-baseline}"
result_file="result/${tag}_${SLURM_JOB_ID:-manual}.txt"
echo "tag=${tag}"
./main 1000 | tee "${result_file}"
echo "result_file=${result_file}"
