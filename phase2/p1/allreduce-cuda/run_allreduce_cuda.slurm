#!/bin/bash
#SBATCH -J allreduce_cuda
#SBATCH -A ACD115083
#SBATCH -N 1
#SBATCH --ntasks-per-node=2
#SBATCH --gpus-per-node=2
#SBATCH -t 00:15:00
#SBATCH -o result/allreduce_cuda_%j.out
#SBATCH -e result/allreduce_cuda_%j.err

set -euo pipefail

BENCH_DIR="/home/r14525078/p1/HeCBench/src/allreduce-cuda"
RESULT_DIR="${BENCH_DIR}/result"
RESULT_FILE="${RESULT_DIR}/allreduce_cuda_result_${SLURM_JOB_ID}.txt"
ARCH="sm_70"

mkdir -p "${RESULT_DIR}"

module purge
module load nvhpc-24.11_hpcx-2.20_cuda-12.6

MPI_ROOT="${MPI_ROOT:-$(dirname "$(dirname "$(which mpicc)")")}"
NCCL_ROOT="${NCCL_ROOT:-/work/HPC_SYS/twnia2/pkg-rocky8/nvidia/hpc_sdk/Linux_x86_64/24.11/comm_libs/12.6/nccl}"
export UCX_TLS=self,shm,cuda_copy,cuda_ipc

{
  echo "benchmark=allreduce-cuda"
  echo "job_id=${SLURM_JOB_ID}"
  echo "node=$(hostname)"
  echo "bench_dir=${BENCH_DIR}"
  echo "arch=${ARCH}"
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
  echo "MPI_ROOT=${MPI_ROOT}"
  echo "NCCL_ROOT=${NCCL_ROOT}"
  echo "UCX_TLS=${UCX_TLS}"
  echo

  echo "=== Environment ==="
  module list
  which nvcc
  nvcc --version
  which mpicc
  which mpirun
  mpirun --version
  nvidia-smi -L
  echo

  echo "=== Build ==="
  cd "${BENCH_DIR}"
  make clean || true
  make ARCH="${ARCH}" MPI_ROOT="${MPI_ROOT}" NCCL_ROOT="${NCCL_ROOT}"
  echo

  echo "=== Run ==="
  mpirun -x UCX_TLS --mca coll ^hcoll,ucc --mca pml ucx -n 2 ./main
} 2>&1 | tee "${RESULT_FILE}"

echo "Result file: ${RESULT_FILE}"
