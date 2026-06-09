#!/bin/bash
#SBATCH -J simpleMultiDevice_cuda
#SBATCH -A ACD115083
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH -t 00:10:00
#SBATCH -o result/simpleMultiDevice_cuda_%j.out
#SBATCH -e result/simpleMultiDevice_cuda_%j.err

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

echo "=== Build ==="
make clean || true
make ARCH=sm_70 MAX_GPU=4

echo "=== Run 1 GPU ==="
CUDA_VISIBLE_DEVICES=0 ./main 1000 2>&1 | tee result/simpleMultiDevice_cuda_result_1gpu_${SLURM_JOB_ID}.txt

echo "=== Run 2 GPUs ==="
CUDA_VISIBLE_DEVICES=0,1 ./main 1000 2>&1 | tee result/simpleMultiDevice_cuda_result_2gpu_${SLURM_JOB_ID}.txt

echo "=== Run 4 GPUs ==="
CUDA_VISIBLE_DEVICES=0,1,2,3 ./main 1000 2>&1 | tee result/simpleMultiDevice_cuda_result_4gpu_${SLURM_JOB_ID}.txt

echo "=== Processing Results ==="
cat result/simpleMultiDevice_cuda_result_1gpu_${SLURM_JOB_ID}.txt \
    result/simpleMultiDevice_cuda_result_2gpu_${SLURM_JOB_ID}.txt \
    result/simpleMultiDevice_cuda_result_4gpu_${SLURM_JOB_ID}.txt \
    > result/simpleMultiDevice_cuda_combined_${SLURM_JOB_ID}.txt

python3 parse_simpleMultiDevice_results.py \
    result/simpleMultiDevice_cuda_combined_${SLURM_JOB_ID}.txt \
    result/simpleMultiDevice_results_${SLURM_JOB_ID}.csv \
    ${SLURM_JOB_ID} \
    $(hostname)

echo "=== CSV Content ==="
cat result/simpleMultiDevice_results_${SLURM_JOB_ID}.csv
