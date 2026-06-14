| Benchmark           | Baseline | Optimized | Env-Optimized |
| ------------------- | -------- | --------- | ------------- |
| cc-cuda             | 0.0034   | fail      | fail          |
| floydwarshall-cuda  | 0.107097 | 0.000024  | 0.00024       |
| floydwarshall2-cuda | 0.000851 | 0.098891  | 0.09133       |
| gc-cuda             | 0.000048 | 0.000285  | fail          |
| mis-cuda            | 0.00136  | 0.002057  | fail          |
| merge-cuda          | 17.03105 | 13.7232   | 16.6688       |
| quicksort-cuda      | 46.1346  | 45.8452   | fail          |
| sortKV-cuda         | 88.1414  | 72.9803   | 76.29895      |
| bitonic-sort-cuda   | 70.13246 | 33.50338  | 34.68863      |
| split-cuda          | 3423.724 | 3023.754  | 3569.766      |


### Prompts

- optimized:
  
    請幫我優化這份程式碼，需輸出相同的資料，請給我完整的程式碼。
- env-optimized:
    
    以下是我的環境，請再進行優化，給我完整的程式碼。
    - GPU: NVIDIA Tesla V100-SXM2-32GB
    - CUDA arch: sm_70
    - Scheduler: Slurm 

### 結果分析

本實驗呈現以下幾個明顯趨勢：
- 部分規則型問題（Sorting）能有效加速
- 大量程式出現正確性錯誤（FAIL）
- 部分加速結果不合理（極端加速）
- 環境優化（V100 tuning）未必提升效能

顯示自動化優化在 GPU 程式中仍存在顯著限制。

### 正確性分析

## Graph 類演算法（cc, gc, mis）

全部在優化後出現 FAIL
原始版本正常

可能原因：

- 破壞 iteration/convergence 邏輯
- race condition（同步不足）
- atomic 操作錯誤
- frontier propagation 改變

結論：

AI 優化對於「不規則、迭代式演算法」極易破壞正確性

### Floyd-Warshall 異常加速

    0.107 → 0.000024（~4000×）
此結果在計算複雜度 O(N³) 下不合理。
推測原因：

- kernel 未完整執行
- iteration 未完成
- 記憶體存取錯誤
- 提前終止

結論：

極端加速通常代表「錯誤計算」，而非真實優化

### 部分程式直接 FAIL
包含：

- cc
- gc
- mis
- quicksort（env）

顯示：

優化改動已影響核心邏輯

# 效能分析
## 成功
| Benchmark    | Speedup |
| ------------ | ------- |
| bitonic-sort | \~2.1×  |
| sortKV       | \~1.2×  |
| merge        | \~1.2×  |

特性：

- 高規律性 memory access
- 無 iteration dependency
- 高平行性

AI 能有效優化：

- memory coalescing
- block size 調整
- global memory 減少

## 退化
#### floydwarshall2-cuda

    0.00085 → 0.098（約慢 100倍）

原因：

- shared memory 使用不當
- synchronization 過多
- occupancy 降低


#### split-cuda（env版本退化）
    3023 → 3569

表示：

- 過度 tuning occupancy
- 忽略 memory-bound 特性

# 環境最佳化分析（V100 sm_70）
Env-Optimized 並未穩定提升效能，甚至出現退化或錯誤。
原因包括：
- 誤判 kernel bottleneck
    + 計算密集 vs 記憶體密集未區分
- 未有效利用 V100 特性
    + warp-level primitive（如 __shfl_sync）未使用
    + shared memory 利用不佳
- 錯誤調整 occupancy
    + 並非所有 kernel 都適合高 occupancy

# AI優化適用問題
| 類型                | 表現     |
| ----------------- | ------ |
| Sorting / 規則型問題   | 良好   |
| Dense matrix（部分）  | 不穩定 |
| Graph / Irregular | 高風險  |


