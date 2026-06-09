#!/bin/bash
#SBATCH --job-name=moe-align-prof
#SBATCH --account=ACD115083
#SBATCH --partition=gp2d
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:20:00
#SBATCH --output=/home/r14525078/p3/HeCBench/src/moe-align-cuda/result/moe-align_profile_%j.out
#SBATCH --error=/home/r14525078/p3/HeCBench/src/moe-align-cuda/result/moe-align_profile_%j.err

set -u

BENCH_DIR=/home/r14525078/p3/HeCBench/src/moe-align-cuda
RESULT_DIR="${BENCH_DIR}/result"
PROFILE_TXT="${RESULT_DIR}/moe-align_profile_${SLURM_JOB_ID}.txt"

mkdir -p "${RESULT_DIR}"
cd "${BENCH_DIR}" || exit 1

{
  module purge
  module load cuda/12.8
  echo "job_id=${SLURM_JOB_ID}"
  echo "node=${SLURMD_NODENAME:-unknown}"
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
  echo "profiler_command=ncu --metrics sm__warps_active.avg.pct_of_peak_sustained_active,dram__throughput.avg.pct_of_peak_sustained_elapsed --target-processes all --launch-count 12 ./main 1"
  nvcc --version
  module list 2>&1 || true
  make clean || true
  make ARCH=sm_70
  if command -v ncu >/dev/null 2>&1; then
    ncu --metrics sm__warps_active.avg.pct_of_peak_sustained_active,dram__throughput.avg.pct_of_peak_sustained_elapsed --target-processes all --launch-count 12 ./main 1
  else
    echo "ncu_unavailable"
  fi
} 2>&1 | tee "${PROFILE_TXT}"
