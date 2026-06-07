#!/bin/bash
#SBATCH -A ACD115083
#SBATCH -N 1
#SBATCH --ntasks-per-node=2
#SBATCH --gpus-per-node=2
#SBATCH -t 00:15:00
#SBATCH -o result/pingpong_cuda_%j.out
#SBATCH -e result/pingpong_cuda_%j.err

set -euo pipefail

mkdir -p result

module purge
module load nvhpc-24.11_hpcx-2.20_cuda-12.6

MPI_ROOT="${MPI_ROOT:-$(dirname "$(dirname "$(which mpicc)")")}"
export UCX_TLS=self,shm,cuda_copy,cuda_ipc
RESULT_FILE="result/pingpong_cuda_result_${SLURM_JOB_ID}.txt"
MPI_RAW="result/pingpong_cuda_mpi_raw_${SLURM_JOB_ID}.txt"
NCCL_RAW="result/pingpong_cuda_nccl_raw_${SLURM_JOB_ID}.txt"
CSV_FILE="result/pingpong_results_${SLURM_JOB_ID}.csv"

append_result_rows() {
  local backend="$1"
  local raw_file="$2"
  perl -ne 'if (/^'"${backend}"'\s*:\s*Transfer size \(B\):\s*([0-9]+),\s*Transfer Time \(s\):\s*([0-9.]+),\s*Bandwidth \(GB\/s\):\s*([0-9.]+)/) { print "RESULT,backend='"${backend}"',size_bytes=$1,time_s=$2,gbps=$3\n"; }' "${raw_file}"
}

{
  echo "=== Environment ==="
  module list
  echo "which nvcc: $(which nvcc)"
  nvcc --version
  echo "which mpicc: $(which mpicc)"
  echo "which mpirun: $(which mpirun)"
  mpirun --version
  nvidia-smi -L
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
  echo "MPI_ROOT=${MPI_ROOT}"
  echo "UCX_TLS=${UCX_TLS}"
  echo "Launcher: mpirun -x UCX_TLS --mca coll ^hcoll,ucc --mca pml ucx -n 2"

  echo "=== Build ==="
  make clean || true
  make ARCH=sm_70 MPI_ROOT="${MPI_ROOT}"

  echo "=== MPI baseline ==="
  mpirun -x UCX_TLS --mca coll ^hcoll,ucc --mca pml ucx -n 2 ./main-mpi 2>&1 | tee "${MPI_RAW}"
  append_result_rows MPI "${MPI_RAW}"

  echo "=== NCCL baseline ==="
  mpirun -x UCX_TLS --mca coll ^hcoll,ucc --mca pml ucx -n 2 ./main-nccl 2>&1 | tee "${NCCL_RAW}"
  append_result_rows NCCL "${NCCL_RAW}"

  echo "=== CSV parse ==="
  python3 parse_pingpong_results.py "${RESULT_FILE}" "${CSV_FILE}" "${SLURM_JOB_ID}" "${SLURM_NODELIST}"
  echo "CSV_FILE=${CSV_FILE}"
} 2>&1 | tee "${RESULT_FILE}"
