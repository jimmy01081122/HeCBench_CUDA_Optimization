#!/bin/bash
#SBATCH -J hecbench_moe_align
#SBATCH -A ACD115083
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH -t 00:10:00
#SBATCH -o logs/moe_align_%j.out
#SBATCH -e logs/moe_align_%j.err

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

mkdir -p logs results

stage "Job started"

echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Submit dir: ${SLURM_SUBMIT_DIR}"
echo "Current dir: $(pwd)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"

stage "Load CUDA module"

module load cuda

echo "nvcc path:"
which nvcc

echo "nvcc version:"
nvcc --version

stage "Check GPU information"

nvidia-smi

stage "Set benchmark configuration"

BENCH_DIR="${SLURM_SUBMIT_DIR}"
ARCH=sm_70
REPEAT=100
RUNS=10

RESULT_FILE="${BENCH_DIR}/results/moe_align_result_${SLURM_JOB_ID}.txt"

echo "BENCH_DIR=${BENCH_DIR}"
echo "RESULT_FILE=${RESULT_FILE}"
echo "ARCH=${ARCH}"
echo "REPEAT=${REPEAT}"
echo "RUNS=${RUNS}"

stage "Enter benchmark directory"

cd "${BENCH_DIR}"

echo "Current directory: $(pwd)"
echo "Files before build:"
ls -la

stage "Check Makefile architecture settings"

grep -n "ARCH\|arch\|sm_" Makefile || true

stage "Check main argument usage"

grep -n "Usage\|argc\|argv\|atoi" main.cu || true

stage "Clean previous build"

make clean || true
rm -f main *.o

stage "Build moe-align-cuda for V100 sm_70"

make ARCH="${ARCH}"

stage "Check build output"

ls -lh ./main

if [[ ! -x ./main ]]; then
    echo "[ERROR] ./main not found or not executable"
    exit 1
fi

stage "Start benchmark runs"

{
    echo "benchmark=moe-align-cuda"
    echo "job_id=${SLURM_JOB_ID}"
    echo "node=$(hostname)"
    echo "submit_dir=${SLURM_SUBMIT_DIR}"
    echo "bench_dir=${BENCH_DIR}"
    echo "arch=${ARCH}"
    echo "repeat=${REPEAT}"
    echo "runs=${RUNS}"
    echo "note=main argument is repeat, not input size"
    echo "start_time=$(date '+%Y-%m-%d %H:%M:%S')"
    echo

    for i in $(seq 1 "${RUNS}"); do
        echo "------------------------------------------------------------"
        echo "[RUN ${i}] start: $(date '+%Y-%m-%d %H:%M:%S')"
        echo "[RUN ${i}] command: timeout 120s ./main ${REPEAT}"
        echo "------------------------------------------------------------"

        timeout 120s ./main "${REPEAT}"

        echo "------------------------------------------------------------"
        echo "[RUN ${i}] end: $(date '+%Y-%m-%d %H:%M:%S')"
        echo "------------------------------------------------------------"
        echo
    done

    echo "end_time=$(date '+%Y-%m-%d %H:%M:%S')"
} | tee "${RESULT_FILE}"

stage "Benchmark completed"

echo "Result file: ${RESULT_FILE}"

stage "Job finished successfully"
