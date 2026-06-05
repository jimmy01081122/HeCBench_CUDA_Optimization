#!/bin/bash
#SBATCH -J allreduce_cuda_agent
#SBATCH -A ACD115083
#SBATCH -N 1
#SBATCH --ntasks-per-node=2
#SBATCH --gpus-per-node=2
#SBATCH -t 00:10:00
#SBATCH -o result/allreduce_cuda_%j.out
#SBATCH -e result/allreduce_cuda_%j.err

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

BENCH_DIR="/home/r14525078/HeCBench/src/allreduce-cuda"
RESULT_DIR="${BENCH_DIR}/result"
RESULT_FILE="${RESULT_DIR}/allreduce_cuda_result_${SLURM_JOB_ID}.txt"
RC_FILE="${RESULT_DIR}/allreduce_cuda_rc_${SLURM_JOB_ID}.txt"
ARCH="sm_70"

mkdir -p "${RESULT_DIR}"

stage "Job info"
echo "job_id=${SLURM_JOB_ID}"
echo "node=$(hostname)"
echo "submit_dir=${SLURM_SUBMIT_DIR}"
echo "bench_dir=${BENCH_DIR}"
echo "result_dir=${RESULT_DIR}"
echo "result_file=${RESULT_FILE}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "arch=${ARCH}"

stage "Load modules"
module purge
module load nvhpc-24.11_hpcx-2.20_cuda-12.6
module list

stage "Compiler and MPI info"
which nvcc
nvcc --version
which mpicc
which mpirun
mpirun --version
MPI_ROOT="$(dirname "$(dirname "$(which mpicc)")")"
echo "MPI_ROOT=${MPI_ROOT}"

stage "GPU info"
nvidia-smi -L
nvidia-smi
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

stage "Enter benchmark directory"
cd "${BENCH_DIR}"
pwd
ls -la

stage "Build allreduce-cuda"
make clean || true
make ARCH="${ARCH}" MPI_ROOT="${MPI_ROOT}"
nvcc -std=c++17 -I"${MPI_ROOT}/include" -DOMPI_SKIP_MPICXX= \
    -Xcompiler -Wall -arch="${ARCH}" -O3 \
    mpi_cuda_allreduce_sanity.cu -o mpi_cuda_allreduce_sanity \
    -L"${MPI_ROOT}/lib" -lmpi -DOMPI_SKIP_MPICXX=

if [[ ! -x ./main ]]; then
    echo "[ERROR] ./main not found or not executable"
    exit 1
fi
if [[ ! -x ./mpi_cuda_allreduce_sanity ]]; then
    echo "[ERROR] ./mpi_cuda_allreduce_sanity not found or not executable"
    exit 1
fi

stage "Run allreduce-cuda"
: > "${RC_FILE}"
MPI_LAUNCH=(mpirun -x UCX_TLS=self,shm,cuda_copy,cuda_ipc --mca coll ^hcoll,ucc --mca pml ucx -n 2)
{
    echo "benchmark=allreduce-cuda"
    echo "job_id=${SLURM_JOB_ID}"
    echo "node=$(hostname)"
    echo "bench_dir=${BENCH_DIR}"
    echo "arch=${ARCH}"
    echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
    echo "MPI_ROOT=${MPI_ROOT}"
    echo "start_time=$(date '+%Y-%m-%d %H:%M:%S')"
    echo

    echo "UCX_TLS=self,shm,cuda_copy,cuda_ipc"
    echo "MPI launcher: ${MPI_LAUNCH[*]}"
    echo

    echo "[CMD] ${MPI_LAUNCH[*]} ./mpi_cuda_allreduce_sanity"
    set +e
    "${MPI_LAUNCH[@]}" ./mpi_cuda_allreduce_sanity
    sanity_rc=$?
    set -e
    echo "sanity_exit_code=${sanity_rc}"
    echo "sanity_exit_code=${sanity_rc}" >> "${RC_FILE}"
    echo

    echo "[CMD] ${MPI_LAUNCH[*]} ./main"
    set +e
    "${MPI_LAUNCH[@]}" ./main
    main_rc=$?
    set -e
    echo "ring_allreduce_exit_code=${main_rc}"
    echo "ring_allreduce_exit_code=${main_rc}" >> "${RC_FILE}"

    echo
    echo "end_time=$(date '+%Y-%m-%d %H:%M:%S')"
} | tee "${RESULT_FILE}"

sanity_rc="$(awk -F= '/sanity_exit_code/ {print $2}' "${RC_FILE}")"
main_rc="$(awk -F= '/ring_allreduce_exit_code/ {print $2}' "${RC_FILE}")"
if [[ "${sanity_rc}" != "0" || "${main_rc}" != "0" ]]; then
    echo "[ERROR] sanity_exit_code=${sanity_rc} ring_allreduce_exit_code=${main_rc}"
    exit 1
fi

stage "Done"
echo "Result file: ${RESULT_FILE}"
