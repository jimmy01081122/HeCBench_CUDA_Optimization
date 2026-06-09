#!/bin/bash
#SBATCH --job-name=pingpong-cuda
#SBATCH --account=ACD115083
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --gres=gpu:2
#SBATCH --time=00:20:00
#SBATCH --output=/home/r14525078/p2/HeCBench/src/pingpong-cuda/result/%x-%j.out
#SBATCH --error=/home/r14525078/p2/HeCBench/src/pingpong-cuda/result/%x-%j.err

set -euo pipefail

BENCH_DIR=/home/r14525078/p2/HeCBench/src/pingpong-cuda
RESULT_DIR="${BENCH_DIR}/result"
RESULT_FILE="${RESULT_DIR}/pingpong-cuda-${SLURM_JOB_ID}.txt"

mkdir -p "${RESULT_DIR}"
cd "${BENCH_DIR}"

module purge
module load nvhpc-24.11_hpcx-2.20_cuda-12.6

{
  echo "job_id=${SLURM_JOB_ID}"
  echo "node=${SLURMD_NODENAME:-unknown}"
  echo "date=$(date -Is)"
  export UCX_TLS=self,shm,cuda_copy,cuda_ipc
  echo "UCX_TLS=${UCX_TLS}"
  echo "NCCL_DEBUG=${NCCL_DEBUG:-unset}"
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
  nvidia-smi -L
  nvidia-smi --query-gpu=name,driver_version --format=csv
  make clean || true
  make ARCH=sm_70 main-mpi main-nccl
  mpirun -x UCX_TLS --mca coll ^hcoll,ucc --mca pml ucx -n 2 ./main-mpi
  mpirun -x UCX_TLS --mca coll ^hcoll,ucc --mca pml ucx -n 2 ./main-nccl
} 2>&1 | tee "${RESULT_FILE}"
