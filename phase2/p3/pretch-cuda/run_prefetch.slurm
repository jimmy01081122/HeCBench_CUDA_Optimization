#!/bin/bash
#SBATCH --job-name=prefetch-cuda
#SBATCH --account=ACD115083
#SBATCH --partition=gp1d
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:20:00
#SBATCH --output=/home/r14525078/p3/HeCBench/src/prefetch-cuda/result/prefetch-cuda_%j.out
#SBATCH --error=/home/r14525078/p3/HeCBench/src/prefetch-cuda/result/prefetch-cuda_%j.err

set -uo pipefail

BENCH_DIR=/home/r14525078/p3/HeCBench/src/prefetch-cuda
RESULT_DIR="${BENCH_DIR}/result"
RESULT_TXT="${RESULT_DIR}/prefetch-cuda_result_${SLURM_JOB_ID}.txt"

mkdir -p "${RESULT_DIR}"
cd "${BENCH_DIR}" || exit 1

{
  echo "job_id=${SLURM_JOB_ID}"
  echo "node=${SLURMD_NODENAME:-$(hostname)}"
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
  echo "date=$(date -Is)"
  echo "pwd=$(pwd)"
  echo "benchmark_commands=make clean || true && make ARCH=sm_70; ./main 10; ./main 100"
  echo
  echo "=== Modules ==="
  module purge
  module load cuda/12.8
  module list 2>&1
  echo
  echo "=== nvcc --version ==="
  nvcc --version
  echo
  echo "=== nvidia-smi ==="
  nvidia-smi
  echo
  echo "=== Build ==="
  make clean || true
  make ARCH=sm_70
  build_status=$?
  echo "build_status=${build_status}"
  if [ "${build_status}" -ne 0 ]; then
    exit "${build_status}"
  fi
  echo
  echo "=== Run ./main 10 ==="
  ./main 10
  run10_status=$?
  echo "run10_status=${run10_status}"
  echo
  echo "=== Run ./main 100 ==="
  ./main 100
  run100_status=$?
  echo "run100_status=${run100_status}"
  exit $((run10_status != 0 || run100_status != 0))
} 2>&1 | tee "${RESULT_TXT}"

exit "${PIPESTATUS[0]}"
