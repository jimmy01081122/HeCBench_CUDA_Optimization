#!/bin/bash
#SBATCH --job-name=moe-align-cuda
#SBATCH --output=result/moe-align-cuda-%j.out
#SBATCH --error=result/moe-align-cuda-%j.err
#SBATCH --gres=gpu:1
#SBATCH --account=ACD115083
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4

set -euo pipefail

cd /home/r14525078/p1/HeCBench/src/moe-align-cuda
mkdir -p result

module purge
module load cuda/12.8

make clean
make ARCH=sm_70
./main 100000
