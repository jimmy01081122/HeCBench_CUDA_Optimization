#!/bin/bash
#SBATCH --job-name=allreduce-cuda
#SBATCH --account=ACD115083
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --gres=gpu:2
#SBATCH --time=00:20:00
#SBATCH --output=/home/r14525078/p2/HeCBench/src/allreduce-cuda/result/allreduce_cuda_%j.out
#SBATCH --error=/home/r14525078/p2/HeCBench/src/allreduce-cuda/result/allreduce_cuda_%j.err

set -euo pipefail

BENCH_DIR=/home/r14525078/p2/HeCBench/src/allreduce-cuda
RESULT_DIR=${BENCH_DIR}/result
RESULT_TXT=${RESULT_DIR}/allreduce_cuda_${SLURM_JOB_ID}.txt

cd "${BENCH_DIR}"
mkdir -p "${RESULT_DIR}"

echo "job_id=${SLURM_JOB_ID}"
echo "node_list=${SLURM_JOB_NODELIST:-unknown}"
echo "submit_dir=${SLURM_SUBMIT_DIR:-unknown}"
echo "date=$(date -Is)"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"

module purge
module load nvhpc-24.11_hpcx-2.20_cuda-12.6
module list

make clean || true
make ARCH=sm_70

UCX_TLS=self,shm,cuda_copy,cuda_ipc \
mpirun -x UCX_TLS --mca coll ^hcoll,ucc --mca pml ucx -n 2 ./main \
    2>&1 | tee "${RESULT_TXT}"
