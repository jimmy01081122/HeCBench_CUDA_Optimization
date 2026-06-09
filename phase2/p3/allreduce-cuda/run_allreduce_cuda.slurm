#!/bin/bash
#SBATCH --job-name=allreduce-cuda
#SBATCH --account=ACD115083
#SBATCH --partition=gp2d
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --time=00:30:00
#SBATCH --output=/home/r14525078/p3/HeCBench/src/allreduce-cuda/result/allreduce-cuda_%j.out
#SBATCH --error=/home/r14525078/p3/HeCBench/src/allreduce-cuda/result/allreduce-cuda_%j.err

set -euo pipefail

BENCH_DIR=/home/r14525078/p3/HeCBench/src/allreduce-cuda
RESULT_DIR=${BENCH_DIR}/result
RESULT_TXT=${RESULT_DIR}/allreduce-cuda_result_${SLURM_JOB_ID}.txt

cd "${BENCH_DIR}"
mkdir -p "${RESULT_DIR}"

module purge
module load nvhpc-24.11_hpcx-2.20_cuda-12.6

{
  echo "job_id=${SLURM_JOB_ID}"
  echo "node=${SLURMD_NODENAME:-$(hostname)}"
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
  echo "benchmark_command=UCX_TLS=self,shm,cuda_copy,cuda_ipc mpirun -x UCX_TLS --mca coll ^hcoll,ucc --mca pml ucx -n 2 ./main"
  echo "date=$(date -Iseconds)"
  echo "nvcc_version_begin"
  nvcc --version
  echo "nvcc_version_end"
  echo "modules_begin"
  module list
  echo "modules_end"
  echo "make_begin"
  make clean || true
  make ARCH=sm_70
  echo "make_end"
  echo "benchmark_begin"
  UCX_TLS=self,shm,cuda_copy,cuda_ipc mpirun -x UCX_TLS --mca coll ^hcoll,ucc --mca pml ucx -n 2 ./main
  echo "benchmark_end"
} 2>&1 | tee "${RESULT_TXT}"
