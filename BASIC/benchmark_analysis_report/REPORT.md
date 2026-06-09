# HeCBench AI 輔助程式碼優化結果總結與統計分析

本報告由 `/home/a/PP` 既有結果檔、CSV、agent summary 與 prompt 彙整產生。重點是比較各 AI 輔助版本在 correctness 有效前提下的效能變化，並指出 prompt 與後續優化可改善的地方。

## 產出檔案

- `data/benchmark_summary.csv`: 跨 benchmark 統一摘要表。
- `prompts/prompt_inventory.csv`: prompt 規格盤點。
- `REPORT.md`: 本中文總報告。
- `generate_report.py`: 可重跑的整理腳本。

## 整體結論

- 共彙整 10 個 benchmark 類別、25 筆可比較摘要。
- 可計算 speedup 的案例平均為 10.838x；但不同 benchmark 的 metric 不同，這個數字只能視為方向性統計。
- 7/9 份 prompt 明確包含 baseline、correctness、提交限制、raw output 或備份等防偽加速約束。
- 最可信的優化通常不是單純要求「更快」，而是讓 AI 在固定測資、固定 repeat、完整 correctness、完整 raw log 與有限提交次數下逐步假設驗證。

## Benchmark 摘要

### allreduce-cuda
- `CODEX` `2 ranks / 2 GPUs`: metric baseline run failed after size 0 -> all tested nonzero sizes PASS correctness/timing, speedup `n/a`, correctness `PASS after tuned UCX launcher`。策略: avoid broken GDRCopy path with UCX_TLS=self,shm,cuda_copy,cuda_ipc。

### moe-align
- `CG_vs_GM_V1` `all parameter combinations`: metric GM mean latency -> CG mean latency ratio, speedup `1.300`, correctness `not shown in comparison csv`。策略: winner count: {'CG': 28, 'GM': 2}; mean CG advantage 29.95%。
- `CG_vs_GM_V2` `all parameter combinations`: metric GM mean latency -> CG mean latency ratio, speedup `1.376`, correctness `not shown in comparison csv`。策略: winner count: {'GM': 11, 'CG': 19}; mean CG advantage 37.56%。
- `CG_vs_GM_V3` `all parameter combinations`: metric GM mean latency -> CG mean latency ratio, speedup `1.087`, correctness `not shown in comparison csv`。策略: winner count: {'GM_V3': 14, 'CG_V3': 16}; mean CG advantage 8.74%。

### moe-cuda
- `CG` `topk=1`: metric 311.860419 -> 168.573779 us, speedup `1.850`, correctness `PASS`。策略: dedicated top-k=1 softmax probability kernel。
- `CG` `topk=2`: metric 395.990085 -> 344.903119 us, speedup `1.148`, correctness `PASS`。策略: fused softmax + top-k。
- `CG` `topk=4`: metric 599.585193 -> 561.396344 us, speedup `1.068`, correctness `PASS`。策略: fused softmax + top-k。
- `CG` `topk=8`: metric 1112.569775 -> 949.431598 us, speedup `1.172`, correctness `PASS`。策略: original two-kernel path。

### p2p-cuda
- `CODEX` `4-GPU all-pair sweep`: metric 48.24 GB/s previous best -> 48.4455 GB/s GB/s, speedup `1.004`, correctness `144 sweep points PASS; 12/12 pair checks PASS`。策略: topology-aware peer copy sweep; best gain is measurement-equivalent。

### pingpong-cuda
- `CODEX` `2 ranks / 2 GPUs final sweep`: metric NCCL 22.898 GB/s at 1073741824 bytes -> MPI 24.248 GB/s at 1073741824 bytes GB/s, speedup `1.059`, correctness `MPI PASS; NCCL PASS`。策略: tuned CUDA-aware MPI/UCX path was fastest for two-rank ping-pong。

### prefetch-cuda
- `GM` `repeat=10, with_prefetch`: metric 6.695 -> 6.682 ms, speedup `1.002`, correctness `PASS`。策略: vectorized/tuned grid-stride loops; prefetch overhead dominates when prefetch is already used。
- `GM` `repeat=10, without_prefetch`: metric 12.850 -> 8.398 ms, speedup `1.530`, correctness `PASS`。策略: vectorized/tuned grid-stride loops; prefetch overhead dominates when prefetch is already used。
- `GM` `repeat=100, with_prefetch`: metric 1.751 -> 1.697 ms, speedup `1.032`, correctness `PASS`。策略: vectorized/tuned grid-stride loops; prefetch overhead dominates when prefetch is already used。
- `GM` `repeat=100, without_prefetch`: metric 2.181 -> 1.729 ms, speedup `1.261`, correctness `PASS`。策略: vectorized/tuned grid-stride loops; prefetch overhead dominates when prefetch is already used。

### shmembench-cuda
- `CG` `best block=256`: metric baseline 13121.63 GB/s -> 13203.950032 GB/s, speedup `1.006`, correctness `PASS`。策略: barriered float4 shared-memory swap microbenchmark。
- `GM` `best block=512`: metric original block 256 -> 13917.41 GB/s, speedup `n/a`, correctness `PASS`。策略: dynamic_shared_memory_volta_opt。

### simpleMultiDevice-cuda
- `GM` `2 GPUs vs 1 GPU`: metric 12552.438572 -> 11185.031777 us, speedup `1.122`, correctness `PASS`。策略: multi-GPU partitioned reduction; end-to-end limited by H2D copy。
- `GM` `4 GPUs vs 1 GPU`: metric 12552.438572 -> 5623.295548 us, speedup `2.232`, correctness `PASS`。策略: multi-GPU partitioned reduction; end-to-end limited by H2D copy。

### softmax-cuda
- `GM` `slice=128`: metric 4.535808 -> 0.129300 ms, speedup `35.080`, correctness `PASS`。策略: optimized (warp-level + SM cached expf)。
- `GM` `slice=256`: metric 15.743037 -> 0.306514 ms, speedup `51.362`, correctness `PASS`。策略: optimized (block-level + SM cached expf)。
- `GM` `slice=784`: metric 55.378628 -> 0.929280 ms, speedup `59.593`, correctness `PASS`。策略: optimized (block-level + SM cached expf)。
- `GM` `slice=1024`: metric 65.308907 -> 1.184686 ms, speedup `55.128`, correctness `PASS`。策略: optimized (block-level + SM cached expf)。
- `GM` `slice=2048`: metric 42.144474 -> 1.679667 ms, speedup `25.091`, correctness `PASS`。策略: optimized (block-level + SM cached expf)。

### topk-cuda
- `GM` `14 hidden_size/topk combinations`: metric baseline radix selection -> workspace_reuse_block512 us, speedup `1.442`, correctness `PASS`。策略: reuse CUB workspace and tune block size to 512。
- `CG` `14 hidden_size/topk combinations`: metric cuda_event_instrumented -> cached_workspace_async us, speedup `1.326`, correctness `PASS`。策略: cache workspace and remove repeated allocation/synchronization。

## 最佳加速案例

| Benchmark | Agent | Case | Speedup | 策略 |
|---|---|---|---:|---|
| softmax-cuda | GM | slice=784 | 59.593x | optimized (block-level + SM cached expf) |
| softmax-cuda | GM | slice=1024 | 55.128x | optimized (block-level + SM cached expf) |
| softmax-cuda | GM | slice=256 | 51.362x | optimized (block-level + SM cached expf) |
| softmax-cuda | GM | slice=128 | 35.080x | optimized (warp-level + SM cached expf) |
| softmax-cuda | GM | slice=2048 | 25.091x | optimized (block-level + SM cached expf) |
| simpleMultiDevice-cuda | GM | 4 GPUs vs 1 GPU | 2.232x | multi-GPU partitioned reduction; end-to-end limited by H2D copy |
| moe-cuda | CG | topk=1 | 1.850x | dedicated top-k=1 softmax probability kernel |
| prefetch-cuda | GM | repeat=10, without_prefetch | 1.530x | vectorized/tuned grid-stride loops; prefetch overhead dominates when prefetch is already used |

## Prompt 比較

多數 prompt 的優點是明確規定不得刪 correctness、不得縮小輸入、不得把 FAIL 當成功，並要求保留 raw output。這些限制讓 AI 輔助優化比較像可審核實驗，而不是只產生漂亮但不可驗證的敘事。

主要差異如下：

- `moe_prompt.md` 較短，聚焦在固定測資與三次提交；適合快速比較兩個 agent，但缺少 profiler、統計變異、baseline 定義細節。
- `pingpong`、`topk`、`shmembench`、`softmax` 等 prompt 較完整，明確指定 sbatch、環境、讀取 `.out/.err/.txt`、不得跳測、備份檔案與 submission limit。
- `moe/CLOUD` 的結果顯示 prompt 若沒有強制「矛盾檢查」與「baseline 必須實測」，agent 仍可能在報告中同時宣稱失敗與全通過，或使用 estimated baseline 做過度結論。

## 後續程式優化建議

- 對 `moe-cuda`: 採用 CG V3 hybrid dispatch 作為主線；再用 Nsight Compute 驗證 top-k 8 原始 path 的 global memory traffic 與 top-k reduction 成本，避免憑直覺融合。
- 對 `topk-cuda`: workspace reuse 已證明有效，下一步應量測 CUB temporary storage、radix pass 次數、occupancy/register pressure；block size 512 可作為 V100 預設，但應保留自動 sweep。
- 對 `softmax-cuda`: block-level cached expf 在 slice 784/1024 最佳，應針對 slice size 建立 dispatch policy；小 slice 128 仍以 warp cached 為佳。
- 對 `shmembench-cuda`: GM 版本的 block-size dependent checksum 讓 sweep 成為有效結果，建議加入 per-block analytical checksum 或 reference kernel，並用 profiler 量 shared bank conflict。
- 對 `pingpong/allreduce`: launcher 是效能與 correctness 的一部分，應把 UCX/NCCL/MPI transport 設定納入 benchmark metadata，避免把環境修復誤判成 kernel 優化。
- 對 `simpleMultiDevice`: 端到端受 H2D copy 主導，後續應拆分 kernel-only、copy-only、overlap copy/compute 三種模式，否則 GPU 數量擴展會被傳輸掩蓋。

## Prompt 改善建議

建議把後續 prompt 統一成以下規格：

1. 明確定義 baseline 必須是實測結果，不得使用估計值替代正式比較。
2. 要求每次優化都輸出 machine-readable `RESULT` 或 CSV，欄位至少包含 job id、node、case、metric、correctness、status、variant。
3. 要求報告自動檢查矛盾：若任一 case FAIL，不得在總結寫 all tests PASS。
4. 要求至少 3 次 trial 或提供 stddev/CV；若 submission limit 太少，至少 final confirmation 要重複量測。
5. 要求區分「環境修復」、「量測修復」、「實際 kernel 優化」，避免把可執行性修復當成演算法加速。
6. 要求列出無效嘗試與拒採理由，像 `moe` 的 top-k 8 full fusion regression、`shmembench` 的 checksum failure 都應保留。
7. 對 GPU kernel 題目加入 profiler 指標：occupancy、register、shared bank conflict、dram throughput、kernel launch count。

## 資料限制

本報告只使用目前 `/home/a/PP` 中已存在的結果檔；沒有重新提交 Slurm job，也沒有重新跑 GPU benchmark。不同 benchmark 的硬體 node、CUDA 版本、metric 單位與 agent 版本不完全一致，因此跨 benchmark 排名只能做研究管理上的比較，不能當作單一效能排行榜。

