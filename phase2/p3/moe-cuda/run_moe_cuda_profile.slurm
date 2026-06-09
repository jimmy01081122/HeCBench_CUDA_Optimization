#!/bin/bash
#SBATCH --job-name=moe-cuda-prof
#SBATCH --account=ACD115083
#SBATCH --partition=gp1d
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --output=/home/r14525078/p3/HeCBench/src/moe-cuda/result/moe-cuda_profile_%j.out
#SBATCH --error=/home/r14525078/p3/HeCBench/src/moe-cuda/result/moe-cuda_profile_%j.err
#SBATCH --time=00:30:00

set -euo pipefail

BENCH_DIR=/home/r14525078/p3/HeCBench/src/moe-cuda
RESULT_DIR=${BENCH_DIR}/result
PROFILE_FILE=${RESULT_DIR}/moe-cuda_profile_${SLURM_JOB_ID}.txt

mkdir -p "${RESULT_DIR}"
cd "${BENCH_DIR}"

module purge
module load cuda/12.8

{
  echo "job_id=${SLURM_JOB_ID}"
  echo "node=$(hostname)"
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
  echo "date=$(date -Is)"
  echo "modules:"
  module list 2>&1 || true
  echo "ncu_path=$(command -v ncu || true)"
  echo "nvprof_path=$(command -v nvprof || true)"
  echo "nsys_path=$(command -v nsys || true)"
} | tee "${PROFILE_FILE}"

make clean || true 2>&1 | tee -a "${PROFILE_FILE}"
make ARCH=sm_70 2>&1 | tee -a "${PROFILE_FILE}"

if command -v ncu >/dev/null 2>&1; then
  METRICS="sm__warps_active.avg.pct_of_peak_sustained_active,dram__bytes.sum,lts__t_bytes.sum,l1tex__data_pipe_lsu_wavefronts_mem_shared.sum"
  for topk in 1 2 4 8; do
    echo "NCU topk=${topk}" | tee -a "${PROFILE_FILE}"
    ncu --target-processes all --metrics "${METRICS}" --csv ./main 32768 384 "${topk}" 1 2>&1 | tee -a "${PROFILE_FILE}"
  done
elif command -v nsys >/dev/null 2>&1; then
  for topk in 1 2 4 8; do
    echo "NSYS topk=${topk}" | tee -a "${PROFILE_FILE}"
    nsys profile --stats=true --force-overwrite=true -o "${RESULT_DIR}/moe-cuda_topk${topk}_${SLURM_JOB_ID}" ./main 32768 384 "${topk}" 1 2>&1 | tee -a "${PROFILE_FILE}"
  done
else
  echo "No ncu/nsys profiler found in job environment." | tee -a "${PROFILE_FILE}"
fi
