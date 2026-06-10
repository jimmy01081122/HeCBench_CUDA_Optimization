#!/bin/bash
#SBATCH --job-name=prefetch-prof
#SBATCH --account=ACD115083
#SBATCH --partition=gp1d
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:20:00
#SBATCH --output=/home/r14525078/p3/HeCBench/src/prefetch-cuda/result/prefetch-cuda_profile_%j.out
#SBATCH --error=/home/r14525078/p3/HeCBench/src/prefetch-cuda/result/prefetch-cuda_profile_%j.err

set -euo pipefail

BENCH_DIR=/home/r14525078/p3/HeCBench/src/prefetch-cuda
RESULT_DIR="${BENCH_DIR}/result"
RESULT_TXT="${RESULT_DIR}/prefetch-cuda_profile_result_${SLURM_JOB_ID}.txt"
NSYS_BASE="${RESULT_DIR}/prefetch-cuda_nsys_${SLURM_JOB_ID}"

mkdir -p "${RESULT_DIR}"
cd "${BENCH_DIR}"

module purge
module load cuda/12.8

{
  echo "job_id=${SLURM_JOB_ID}"
  echo "node=${SLURMD_NODENAME:-$(hostname)}"
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
  echo "date=$(date -Is)"
  echo "modules:"
  module list 2>&1
  echo "profilers:"
  command -v nsys || true
  command -v nvprof || true
  command -v ncu || true
  echo "build_command=make clean || true && make ARCH=sm_70"
  make clean || true
  make ARCH=sm_70

  if command -v nsys >/dev/null 2>&1; then
    echo "profile_command=nsys profile --trace=cuda --stats=true --force-overwrite=true --output=${NSYS_BASE} ./main 10"
    nsys profile --trace=cuda --stats=true --force-overwrite=true --output="${NSYS_BASE}" ./main 10
  elif command -v nvprof >/dev/null 2>&1; then
    echo "profile_command=nvprof --unified-memory-profiling per-process-device ./main 10"
    nvprof --unified-memory-profiling per-process-device ./main 10
  else
    echo "profile_status=unavailable"
  fi
} 2>&1 | tee "${RESULT_TXT}"
