#!/bin/bash
#SBATCH -J topk_cuda
#SBATCH -A ACD115083
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH -t 00:20:00
#SBATCH -o result/topk_cuda_%j.out
#SBATCH -e result/topk_cuda_%j.err

mkdir -p result

echo "=== Loading modules ==="
module purge
module load cuda/12.8
module list

echo "=== Environment ==="
which nvcc
nvcc --version
hostname
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
nvidia-smi -L
nvidia-smi

echo "=== Build ==="
make clean || true
make ARCH=sm_70 2>&1

echo "=== Run ==="
./main 3072 100 2>&1 | tee result/topk_cuda_result_${SLURM_JOB_ID}.txt
