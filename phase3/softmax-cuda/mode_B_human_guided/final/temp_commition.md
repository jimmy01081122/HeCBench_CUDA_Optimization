# Mode B 目前進度報告

## 0. 總體狀態

目前 **Phase 3 Mode B：Human-in-the-loop Guided Optimization** 已完成：

```text
1. 三個 benchmark 的 robust baseline phase
2. softmax-cuda 的 Mode B Round 1
3. softmax-cuda 的 Mode B Round 2
4. softmax-cuda 的 final confirmation
```

目前狀態：

```text
softmax-cuda: Mode B 完成，SUCCESS
topk-cuda: robust baseline 完成，尚未開始 Round 1 optimization
shmembench-cuda: robust baseline 完成，尚未開始 Round 1 optimization
Mode C: 尚未開始
```

主規劃器判定：

```text
Mode B 已成功完成第一個 benchmark：softmax-cuda。
下一步應進入 topk-cuda Mode B Round 1 proposal。
```

***

# 1. Mode B Robust Baseline 階段

## 1.1 完成項目

server 執行端已完成三個 benchmark 的 robust baseline，且在進入優化前停止，符合規範。

### softmax-cuda

```text
狀態：SUCCESS
官方 cases：5 個 impl=1 official cases
correctness：全部 PASS
measurement_validity：全部 VALID
```

### topk-cuda

```text
狀態：SUCCESS
官方 cases：14 cases
trials：7 trials
correctness：全部 PASS
measurement_validity：12 VALID，2 CAUTION，0 NOISY
```

### shmembench-cuda

```text
狀態：SUCCESS for official comparison
official case：block_size=256
diagnostics：block_size=128/512/1024 保留
```

## 1.2 工程基礎修正

server 也完成：

```text
1. 正確命名 self_consistency_auditor.py
2. 新增三個 Mode B robust-baseline Slurm scripts
3. 清理 build products
4. 確認無 job running
```

## 1.3 主規劃器判定

robust baseline 階段通過。  
可以進入 Mode B optimization，但必須逐題進行，不可三題同時開跑。

***

# 2. softmax-cuda Mode B 進度

## 2.1 Round 1 Proposal

Round 1 初始方向：

```text
新增 impl=2：
block-per-slice + shared-memory cached exp compound candidate
```

初始 proposal 被判定為：

```text
NEEDS_REVISION
```

主要問題：

```text
1. impl=2 是 compound candidate，不是單一最小修改。
2. 原 proposal 缺 paired impl=1 baseline。
3. CSV schema 不完整。
4. raw stdout/stderr 保存規則不足。
5. result_type / speedup_claim_valid / measurement_validity 規則不足。
6. profiler fallback 欄位不足。
```

修正後，proposal 通過審查並批准執行。

***

## 2.2 Round 1 執行結果

Round 1 candidate：

```text
impl2_block_cached_exp_compound
```

結果摘要：

| slice | impl=1 mean ms | impl=2 mean ms | correctness        | result                |
| ----: | -------------: | -------------: | ------------------ | --------------------- |
|   128 |       0.135152 |       0.554750 | PASS 3/3           | REGRESSION            |
|   256 |       0.323384 |       0.594147 | PASS 2/3, FAIL 1/3 | INVALID               |
|   784 |       1.434026 |       1.108087 | PASS 3/3           | KERNEL\_OPT per-slice |
|  1024 |       2.068956 |       1.300902 | PASS 3/3           | KERNEL\_OPT per-slice |
|  2048 |       2.212359 |       1.680560 | PASS 3/3           | KERNEL\_OPT per-slice |

## 2.3 Round 1 判定

主規劃器判定：

```text
Round 1 = PARTIAL_SUCCESS
Full replacement = REJECTED
Large-slice candidate = KEEP
```

原因：

```text
1. slice=128 嚴重 regression。
2. slice=256 有 correctness failure。
3. slice=784/1024/2048 有有效改善。
4. 結果不能 promoted 為 full optimization。
```

Round 1 的研究價值：

```text
impl=2 不適合作為 universal replacement。
impl=2 對 large slices 有價值。
後續應建立 shape-aware dispatch。
```

***

## 2.4 Round 2 Proposal

Round 2 目標：

```text
建立 shape-aware dispatch candidate
```

Candidate：

```text
impl3_shape_dispatch_impl1_small_impl2_large
```

Dispatch policy：

| slice | selected impl | reason                             |
| ----: | ------------: | ---------------------------------- |
|   128 |        impl=1 | Round 1 impl=2 regression          |
|   256 |        impl=1 | Round 1 impl=2 correctness failure |
|   784 |        impl=2 | Round 1 impl=2 PASS and faster     |
|  1024 |        impl=2 | Round 1 impl=2 PASS and faster     |
|  2048 |        impl=2 | Round 1 impl=2 PASS and faster     |

Round 2 proposal 符合要求，因此批准執行。

***

## 2.5 Round 2 執行結果

Round 2 job：

```text
Slurm job: 949703
Candidate: impl3_shape_dispatch_impl1_small_impl2_large
```

結果：

| slice | selected impl | candidate mean ms | paired impl=1 mean ms | correctness | result\_type            |  speedup |
| ----: | ------------: | ----------------: | --------------------: | ----------- | ----------------------- | -------: |
|   128 |             1 |          0.135674 |              0.144732 | PASS        | MEASUREMENT\_EQUIVALENT | 1.066763 |
|   256 |             1 |          0.306408 |              0.305251 | PASS        | MEASUREMENT\_EQUIVALENT | 0.996224 |
|   784 |             2 |          1.107988 |              1.437362 | PASS        | PARAM\_TUNE             | 1.297273 |
|  1024 |             2 |          1.300765 |              2.082344 | PASS        | PARAM\_TUNE             | 1.600861 |
|  2048 |             2 |          1.670514 |              2.213330 | PASS        | PARAM\_TUNE             | 1.324940 |

主規劃器判定：

```text
Round 2 = ACCEPT
Result type = PARAM_TUNE / SHAPE_AWARE_DISPATCH
```

原因：

```text
1. 全部 official slices correctness PASS。
2. small slices 使用 impl=1，避免 regression。
3. large slices 使用 impl=2，保留有效改善。
4. 不再宣稱 impl=2 是 universal replacement。
```

***

## 2.6 softmax-cuda Final Confirmation

Final confirmation job：

```text
Slurm job: 949717
Node: gn1228.twcc.ai
Candidate: impl3_shape_dispatch_impl1_small_impl2_large
Profiler status: NOT_RUN
```

Final confirmation 結果：

| slice | selected impl | candidate mean ms | paired impl=1 mean ms | correctness | result\_type            |  speedup |
| ----: | ------------: | ----------------: | --------------------: | ----------- | ----------------------- | -------: |
|   128 |             1 |          0.134574 |              0.134869 | 3/3 PASS    | MEASUREMENT\_EQUIVALENT | 1.002197 |
|   256 |             1 |          0.321505 |              0.321793 | 3/3 PASS    | MEASUREMENT\_EQUIVALENT | 1.000895 |
|   784 |             2 |          1.036402 |              1.442716 | 3/3 PASS    | PARAM\_TUNE             | 1.392043 |
|  1024 |             2 |          1.238443 |              2.104045 | 3/3 PASS    | PARAM\_TUNE             | 1.698944 |
|  2048 |             2 |          1.672904 |              2.237452 | 3/3 PASS    | PARAM\_TUNE             | 1.337466 |

## 2.7 softmax-cuda Mode B 最終判定

```text
softmax-cuda Mode B: SUCCESS
Accepted candidate: impl3_shape_dispatch_impl1_small_impl2_large
Result type: PARAM_TUNE / SHAPE_AWARE_DISPATCH
```

正式解讀：

```text
這不是 universal KERNEL_OPT。
這不是 impl=2 全面替代。
這不是 impl=0 → impl=1 speedup。
這是根據 slice size 選擇既有 optimized path 的 shape-aware dispatch policy。
```

有效改善：

```text
slice=784: 1.392x
slice=1024: 1.699x
slice=2048: 1.337x
```

measurement-equivalent：

```text
slice=128
slice=256
```

***

# 3. topk-cuda Mode B 目前狀態

## 3.1 已完成

```text
robust baseline completed
14 cases PASS
7 trials
12 VALID
2 CAUTION
0 NOISY
auditor PASS
```

## 3.2 尚未開始

```text
Mode B Round 1 optimization 尚未開始
```

## 3.3 下一步

下一步應要求 server 產出：

```text
topk-cuda Mode B Round 1 proposal
```

但不能直接 sbatch。

proposal 必須先回答：

```text
1. 2 個 CAUTION cases 是哪些？
2. 是否需要再重測 CAUTION cases？
3. Round 1 應該先做 remeasurement 還是進入 optimization？
4. 若優化，單一 hypothesis 是什麼？
5. 是 workspace reuse、block size tuning、還是 dispatch policy？
```

***

# 4. shmembench-cuda Mode B 目前狀態

## 4.1 已完成

```text
official validated comparison:
  block_size=256
  correctness PASS

diagnostic cases:
  block_size=128/512/1024 preserved
```

## 4.2 尚未開始

```text
Mode B Round 1 optimization 尚未開始
```

## 4.3 後續策略

shmembench 不應以追 speedup 為主。

優先目標：

```text
1. 確認 block_size=256 是否已接近上限
2. 分析 128/512 correctness failure
3. 分析 1024 shared memory compile failure
4. 若做 padded/vectorized，必須只和 block_size=256 original 比較
```

***

# 5. Mode B 目前總結

## 已完成

```text
[完成] 三題 robust baseline
[完成] softmax Round 1 proposal/revision/execution
[完成] softmax Round 2 proposal/execution
[完成] softmax final confirmation
[完成] softmax Mode B SUCCESS
```

## 未完成

```text
[未完成] topk Mode B Round 1
[未完成] shmembench Mode B Round 1
[未完成] Mode B total report
[未完成] Mode C
```

***

# 6. 當前主規劃器決策

目前不應再跑 softmax Mode B。

```text
softmax-cuda Mode B 已完成。
```

現在應進入：

```text
topk-cuda Mode B Round 1 proposal
```

***

# 7. 下一步給 server 的指令

可以直接貼：

```text
softmax-cuda Mode B is complete and accepted as SUCCESS.

Do not run additional softmax Mode B rounds.

Proceed to prepare topk-cuda Mode B Round 1 proposal ONLY.

Do not submit sbatch yet.

Use the robust baseline results:
- 14 official cases
- 7 trials
- 12 VALID
- 2 CAUTION
- 0 NOISY

The proposal must first analyze:
1. Which 2 cases are CAUTION.
2. Why they are CAUTION.
3. Whether they require remeasurement before optimization.
4. Whether Round 1 should focus on:
   - remeasurement/stability,
   - workspace reuse,
   - block-size tuning,
   - dispatch policy,
   - or another single hypothesis.

Return:
1. robust baseline summary
2. CAUTION case analysis
3. bottleneck hypothesis
4. proposed single change
5. expected improvement and risk
6. validation plan
7. required CSV fields
8. human approval checkpoint

Do not modify source.
Do not submit sbatch.
Wait for human approval.
```

***

# 8. 論文中目前可以寫的 Mode B 中期結論

目前可以保守寫：

```text
In Phase 3 Mode B, softmax-cuda demonstrates the value of human-guided optimization. The agent’s first candidate, a compound block-level cached-exp kernel, improved larger slices but regressed or failed on smaller slices. Human review rejected it as a full replacement and redirected the process toward a shape-aware dispatch policy. The final accepted dispatcher preserved the existing optimized impl=1 path for slices 128 and 256 while selecting the impl=2 path for slices 784, 1024, and 2048. Final confirmation passed correctness for all official cases and achieved valid large-slice improvements of 1.392x, 1.699x, and 1.337x. This result is classified as PARAM_TUNE / SHAPE_AWARE_DISPATCH rather than a universal kernel optimization.
```

中文：

```text
在 Phase 3 Mode B 中，softmax-cuda 展現了人機協作優化的價值。Agent 的第一個候選版本是一個 compound block-level cached-exp kernel，它在較大 slice 上有效提升效能，但在較小 slice 上發生 regression 或 correctness failure。人類審查因此拒絕將其作為全面替代方案，並將後續方向調整為 shape-aware dispatch policy。最終接受的 dispatcher 對 slice=128 與 256 保留既有 impl=1 optimized path，對 slice=784、1024、2048 選擇 impl=2 path。Final confirmation 中所有 official cases 均通過 correctness，且大 slice 分別取得 1.392x、1.699x 與 1.337x 的有效改善。本結果應分類為 PARAM_TUNE / SHAPE_AWARE_DISPATCH，而非通用 kernel optimization。
```

***

# 9. 簡短狀態

```text
Mode B overall progress:
  1/3 benchmark completed

Completed:
  softmax-cuda

Next:
  topk-cuda Round 1 proposal

Pending:
  shmembench-cuda
  Mode B final report
  Mode C
```
