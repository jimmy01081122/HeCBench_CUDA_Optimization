# Input Prompt

Benchmark: `adam-cuda`
Category: ML kernel

## Source prompt files
N/A

## Default recorded prompt
```text
更改並優化此原始代碼
硬體與環境：
- GPU: NVIDIA Tesla V100-SXM2-32GB
- CUDA arch: sm_70
- Scheduler: Slurm
- Account: ACD115083
- CUDA module:
  module purge
  module load cuda/12.8
- 優先使用單 GPU
- 不得在 login node 直接執行 GPU benchmark
- 必須使用 sbatch
Slurm 腳本要求：
  #SBATCH -J topk_cuda
  #SBATCH -A ACD115083
  #SBATCH -N 1
  #SBATCH --ntasks-per-node=1
  #SBATCH --gpus-per-node=1
  #SBATCH -t 00:20:00
  #SBATCH -o result/topk_cuda_%j.out
  #SBATCH -e result/topk_cuda_%j.err
```
