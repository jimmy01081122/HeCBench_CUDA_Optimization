#!/bin/bash
#SBATCH -J p2p_cuda
#SBATCH -A ACD115083
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=2
#SBATCH -t 00:10:00
#SBATCH -o result/p2p_cuda_%j.out
#SBATCH -e result/p2p_cuda_%j.err

set -euo pipefail

stage() {
    echo
    echo "============================================================"
    echo "[STAGE] $1"
    echo "[TIME ] $(date '+%Y-%m-%d %H:%M:%S')"
    echo "============================================================"
}

fail_report() {
    echo
    echo "============================================================"
    echo "[ERROR] Script failed"
    echo "[TIME ] $(date '+%Y-%m-%d %H:%M:%S')"
    echo "[LINE ] ${BASH_LINENO[0]}"
    echo "[CMD  ] ${BASH_COMMAND}"
    echo "============================================================"
}

trap fail_report ERR

BENCH_DIR="/home/r14525078/HeCBench/src/p2p-cuda"
RESULT_DIR="${BENCH_DIR}/result"
RESULT_FILE="${RESULT_DIR}/p2p_cuda_result_${SLURM_JOB_ID}.txt"
REPEAT=2000
ARCH=sm_70

mkdir -p "${RESULT_DIR}"

stage "Job info"

echo "job_id=${SLURM_JOB_ID}"
echo "node=$(hostname)"
echo "submit_dir=${SLURM_SUBMIT_DIR}"
echo "bench_dir=${BENCH_DIR}"
echo "result_dir=${RESULT_DIR}"
echo "result_file=${RESULT_FILE}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "repeat=${REPEAT}"
echo "arch=${ARCH}"

stage "Load CUDA"

module purge
module load cuda/12.8

which nvcc
nvcc --version

stage "GPU info"

nvidia-smi -L
nvidia-smi

stage "Enter benchmark directory"

cd "${BENCH_DIR}"
pwd
ls -la

stage "Build p2p-cuda"

make clean || true
make ARCH="${ARCH}"

if [[ ! -x ./main ]]; then
    echo "[ERROR] ./main not found or not executable"
    exit 1
fi

stage "Run p2p-cuda"

{
    echo "benchmark=p2p-cuda"
    echo "job_id=${SLURM_JOB_ID}"
    echo "node=$(hostname)"
    echo "bench_dir=${BENCH_DIR}"
    echo "arch=${ARCH}"
    echo "repeat=${REPEAT}"
    echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
    echo "start_time=$(date '+%Y-%m-%d %H:%M:%S')"
    echo

    echo "[CMD] ./main ${REPEAT}"
    ./main "${REPEAT}"

    echo
    echo "end_time=$(date '+%Y-%m-%d %H:%M:%S')"
} | tee "${RESULT_FILE}"

stage "Done"

echo "Result file: ${RESULT_FILE}"
