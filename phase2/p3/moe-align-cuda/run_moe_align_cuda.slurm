#!/bin/bash
#SBATCH --job-name=moe-align
#SBATCH --account=ACD115083
#SBATCH --partition=gp2d
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:20:00
#SBATCH --output=/home/r14525078/p3/HeCBench/src/moe-align-cuda/result/moe-align_%j.out
#SBATCH --error=/home/r14525078/p3/HeCBench/src/moe-align-cuda/result/moe-align_%j.err

set -u

BENCH_DIR=/home/r14525078/p3/HeCBench/src/moe-align-cuda
RESULT_DIR="${BENCH_DIR}/result"
RESULT_TXT="${RESULT_DIR}/moe-align_result_${SLURM_JOB_ID}.txt"

mkdir -p "${RESULT_DIR}"
cd "${BENCH_DIR}" || exit 1

{
  echo "job_id=${SLURM_JOB_ID}"
  echo "node=${SLURMD_NODENAME:-unknown}"
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
  echo "benchmark_command=./main 100000"
  echo "date_start=$(date -Is)"
  echo "modules_before="
  module list 2>&1 || true
  module purge
  module load cuda/12.8
  echo "modules_after="
  module list 2>&1 || true
  nvcc --version
  echo "build_command=make clean || true && make ARCH=sm_70"
  make clean || true
  make ARCH=sm_70
  build_status=$?
  echo "build_status=${build_status}"
  if [ "${build_status}" -ne 0 ]; then
    echo "status=BUILD_FAIL"
    exit "${build_status}"
  fi
  echo "run_start=$(date -Is)"
  ./main 100000
  run_status=$?
  echo "run_status=${run_status}"
  echo "date_end=$(date -Is)"
  exit "${run_status}"
} 2>&1 | tee "${RESULT_TXT}"
