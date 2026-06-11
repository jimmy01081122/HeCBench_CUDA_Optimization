# AI 輔助 HeCBench Benchmark 優化與人機協作工作流研究報告

## Phase 3 Softmax-focused Human-AI Collaborative Optimization Report

***

## 摘要

本研究以 HeCBench CUDA benchmark 為實驗平台，探討 AI agent 在既有 GPU benchmark 上進行程式優化時的能力、限制與可審核性。研究核心不是單純追求最高 speedup，而是分析 AI 在不同 benchmark 類型中何時能成功優化、何時失敗，以及失敗原因是否來自硬體瓶頸、測量噪音、correctness violation、baseline 無效、prompt 約束不足或結果分類錯誤。

本研究前期以多個 HeCBench CUDA benchmark 建立 benchmark taxonomy 與 prompt 約束分層。Phase 2 結果顯示，不同 benchmark 的 AI 輔助成果性質差異很大：`softmax-cuda` 屬於較明確的 kernel / AI primitive 優化案例；`topk-cuda` 的有效策略主要與 workspace reuse、block-size tuning 及 selection primitive 有關；`p2p-cuda` 屬 topology-aware measurement；`allreduce-cuda` 則主要是 launcher / UCX / environment fix，而非 kernel optimization。這說明若不區分 `KERNEL_OPT`、`PARAM_TUNE`、`ENV_FIX`、`MEASURE_FIX` 與 `TOPOLOGY_MEASURE`，AI 結果容易被過度宣稱或錯誤分類。 [\[arxiv.org\]](https://arxiv.org/abs/2510.00555), [\[arxiv.org\]](https://arxiv.org/abs/2403.17134)

Phase 2 也顯示，較完整的 prompt 通常包含 baseline、correctness、sbatch、submission limit、raw output 等條款；這些條款能降低偽加速與不可審核結果的風險。不過，prompt inventory 也指出多數 prompt 仍缺少 profiler 指標與重複統計要求，這成為 Phase 3 強化的核心方向。 [\[arxiv.org\]](https://arxiv.org/html/2603.07169v1)

Phase 3 原始設計選擇三個代表性 benchmark：`softmax-cuda`、`topk-cuda` 與 `shmembench-cuda`。其中 `softmax-cuda` 目前已完成 Mode B human-in-the-loop guided optimization，並完成 final confirmation；`topk-cuda` 與 `shmembench-cuda` 則完成 robust baseline，但尚未進入 Mode B optimization，因此在本報告中僅作為 optional supporting evidence。Mode C 尚未執行，本報告不對 Mode C 的最佳協作方式與加速程度下結論。

本報告最主要成果是：`softmax-cuda` Mode B 證明，人類審查能將 AI agent 產生的 partial candidate 收斂為可驗證、可審核、可解釋的 shape-aware dispatch policy。Agent 在 Round 1 提出 `impl=2` compound block-level cached-exp candidate，該候選版本在 large slices 上有效，但在 small slices 上出現 regression 或 correctness failure。人類審查拒絕將其作為 universal replacement，並要求建立 shape-aware dispatch。Round 2 產生 `impl=3` dispatcher，對 `slice=128/256` 保留既有 `impl=1` optimized path，對 `slice=784/1024/2048` 選擇 `impl=2` path。Final confirmation 中所有 official cases 均 3/3 correctness PASS，大 slice 分別取得 1.392x、1.699x 與 1.337x 的有效改善。本結果應分類為 `PARAM_TUNE / SHAPE_AWARE_DISPATCH`，而非 universal `KERNEL_OPT`。

***

## 1. 研究動機與核心問題

現有 GPU benchmark 通常涵蓋大量不同類型的 CUDA kernel，例如 memory bandwidth、shared memory、reduction、Top-K selection、softmax、multi-GPU communication 等。這些 benchmark 原本用於測量系統或程式效能，但同時也可作為 AI 輔助程式優化的實驗平台。

本研究的起點是：若讓 AI agent 嘗試優化一組既有 benchmark，例如從 HeCBench 中選取多個 CUDA benchmark，AI 可能會在部分題目取得加速，也可能在部分題目失敗。然而，單純統計「幾題成功、幾題失敗」不足以形成有價值的研究。更重要的是回答：

```text
1. AI 成功優化的 benchmark 有什麼共同特徵？
2. AI 失敗的 benchmark 是因為硬體瓶頸、測量不穩定、correctness 限制，還是 prompt 約束不足？
3. AI 是否會產生偽加速，例如刪除慢 case、使用 invalid baseline、忽略 correctness failure？
4. 人類操作者能否透過 prompt、baseline、correctness gate、variance check 與審查流程，將 AI 的 partial result 收斂為可用策略？
```

因此，本研究不只關心 AI 能否產生 speedup，更關心 AI 產生的結果是否可驗證、可重現、可審核。

***

## 2. 研究問題

本研究圍繞以下問題展開：

```text
RQ1: AI agent 在既有 CUDA benchmark 上能否產生有效優化？
RQ2: 成功與失敗的 benchmark 有什麼差異？
RQ3: AI 產生的 speedup 是否一定可信？
RQ4: Prompt 約束是否能降低偽加速與不可審核結果？
RQ5: 人類操作者能否將 AI 的 partial optimization 轉化為可用策略？
RQ6: 哪些成果應分類為 kernel optimization，哪些只是 measurement fix、environment fix 或 parameter tuning？
```

本報告目前主要回答其中一個具體子問題：

```text
在 softmax-cuda 這個高優化空間 benchmark 上，人機協作是否能將 AI agent 的 partial candidate 收斂成可驗證的優化策略？
```

目前結果支持肯定答案，但限定條件是：

```text
1. 使用 robust / paired baseline。
2. 所有 official cases 保留。
3. correctness gate 嚴格執行。
4. regression 與 failure 不被隱藏。
5. 結果類型正確分類為 shape-aware dispatch，而非 universal kernel optimization。
```

***

## 3. 研究流程總覽

本研究分為三個階段。

### 3.1 Phase 1：Benchmark taxonomy

Phase 1 將 benchmark 分類為：

```text
AI Primitive / Kernel Optimization:
  softmax-cuda
  topk-cuda
  moe-cuda
  moe-align-cuda

Memory-System / Measurement Benchmark:
  prefetch-cuda
  shmembench-cuda
  p2p-cuda

Multi-GPU / Communication / Environment:
  allreduce-cuda
  pingpong-cuda
  simpleMultiDevice-cuda
```

### 3.2 Phase 2：Prompt 約束分層

Phase 2 比較 P1 / P2 / P3 三種 prompt 約束強度：

```text
P1: 弱約束
P2: 中約束
P3: 強約束
```

Phase 2 的核心發現是：P3 不一定帶來最高 speedup，但能顯著提升結果的可審核性，尤其是在 baseline validity、correctness gate、raw output preservation、submission limit、CSV schema 與 contradiction check 方面。 [\[arxiv.org\]](https://arxiv.org/html/2603.07169v1), [\[arxiv.org\]](https://arxiv.org/abs/2403.17134)

### 3.3 Phase 3：人機協作式優化

Phase 3 原始設計選擇三個代表性 benchmark：

```text
softmax-cuda: 高優化空間
topk-cuda: 中優化空間
shmembench-cuda: 低優化空間 / shared memory microbenchmark
```

目前 Phase 3 已完成：

```text
1. 三題 robust baseline。
2. softmax-cuda Mode B Round 1。
3. softmax-cuda Mode B Round 2。
4. softmax-cuda final confirmation。
```

目前尚未完成：

```text
1. topk-cuda Mode B optimization。
2. shmembench-cuda Mode B optimization。
3. Mode C。
```

因此，本報告收斂為 softmax-focused report。

***

## 4. 專案與 Artifact 狀態

### 4.1 目前專案狀態

| Benchmark         | Mode A                        | Mode B         | Mode C                    | 目前角色                         |
| ----------------- | ----------------------------- | -------------- | ------------------------- | ---------------------------- |
| `softmax-cuda`    | SUCCESS                       | SUCCESS        | NOT\_STARTED / NOT\_FOUND | 主成果                          |
| `topk-cuda`       | SUCCESS                       | BASELINE\_ONLY | NOT\_STARTED / NOT\_FOUND | optional supporting evidence |
| `shmembench-cuda` | INVALID / diagnostic findings | BASELINE\_ONLY | NOT\_STARTED / NOT\_FOUND | optional supporting evidence |

### 4.2 Authoritative files

本報告應以以下檔案作為主要證據來源：

```text
phase3/softmax-cuda/mode_B_human_guided/final/results.csv
phase3/softmax-cuda/mode_B_human_guided/final/round_summary.md
phase3/softmax-cuda/mode_B_human_guided/final/main.cu
phase3/softmax-cuda/mode_B_human_guided/final/auditor_report.csv
phase3/softmax-cuda/mode_B_human_guided/final/contradiction_check.csv
phase3/softmax-cuda/mode_B_human_guided/final/state_report.md
phase3/softmax-cuda/mode_B_human_guided/final/temp_commition.md
phase3/softmax-cuda/mode_B_human_guided/rounds/round_1/round_summary.md
phase3/softmax-cuda/mode_B_human_guided/rounds/round_1/patch_summary.md
phase3/softmax-cuda/mode_B_human_guided/rounds/round_2/round_summary.md
phase3/softmax-cuda/mode_B_human_guided/rounds/round_2/patch_summary.md
phase3/reports/mode_A_report.md
```

### 4.3 Stale / superseded / do-not-cite files

以下檔案目前不應作為正式報告證據：

```text
phase3/softmax-cuda/mode_B_human_guided/agent_summary.md
phase3/softmax-cuda/mode_B_human_guided/results.csv
phase3/reports/mode_B_report.md
phase3/reports/MODE_B_REPORT.md
phase3/mode_A_agent_only
```

原因：

```text
1. parent-level agent_summary.md 與 results.csv 仍停留在 robust baseline 或舊 schema。
2. mode_B_report.md / MODE_B_REPORT.md 未反映 softmax Round 1、Round 2 與 final confirmation。
3. phase3/mode_A_agent_only 是 topk Mode A 的重複目錄。
```

### 4.4 Hostname conflict

先前摘要曾出現 `gn1288.twcc.ai`，經 artifact sync 檢查後確認這是文字摘要 typo。正式報告應採用：

```text
Round 1 / Round 2: gn1221.twcc.ai
Final confirmation: gn1228.twcc.ai
```

***

## 5. 實驗規則與可審核性要求

Phase 3 結果依下列規則審查：

```text
1. 所有 GPU benchmark execution 必須透過 sbatch。
2. 禁止 login node 直接執行 ./main 或任何 GPU benchmark binary。
3. correctness FAIL → result invalid。
4. baseline invalid → speedup=n/a。
5. improvement <1% → MEASUREMENT_EQUIVALENT，不得宣稱顯著加速。
6. high CV → speedup claim invalid unless remeasured。
7. ENV_FIX / MEASURE_FIX / TOPOLOGY_MEASURE 不得報告為 KERNEL_OPT。
8. profiler unavailable 是 limitation，不是 experiment failure。
9. P1/P2/P3 或 Mode A/B/C 比較必須使用相同 official cases。
10. 不得刪除 slow、failing 或 regressing cases。
```

### 5.1 Result type classification

本報告使用以下分類：

```text
BASELINE
BASELINE_COMPARISON
KERNEL_OPT
PARAM_TUNE
SHAPE_AWARE_DISPATCH
MEASURE_FIX
ENV_FIX
TOPOLOGY_MEASURE
MEASUREMENT_EQUIVALENT
REGRESSION
INVALID
INCONCLUSIVE
```

### 5.2 Correctness 與 measurement validity 解耦

```text
correctness_status:
  PASS / FAIL / PARTIAL / NOT_PROVIDED

measurement_validity:
  VALID / CAUTION / NOISY / INVALID / LIMITED / DIAGNOSTIC_FAIL

speedup_claim_valid:
  true / false
```

此設計避免把 correctness PASS 直接等同於 speedup 有效。

***

## 6. 成功、失敗與 Partial Success 的分類邏輯

AI 輔助 benchmark 優化不能只用「有沒有變快」判斷成功。本研究將結果分為：

```text
1. 完整成功：
   所有 official cases correctness PASS，且相對有效 baseline 有穩定提升。

2. Partial success：
   部分 cases 有效改善，但另一些 cases regression 或 invalid。
   例如 softmax Round 1 的 impl=2。

3. Measurement-equivalent：
   效能差異小於 1%，不可宣稱顯著加速。
   例如 softmax final 中 slice=128 / 256。

4. Invalid：
   correctness FAIL、baseline invalid、raw output 缺失或 official cases 不完整。

5. Environment / measurement fix：
   問題解決來自 launcher、UCX、NCCL、Slurm、timer 或測量方式修正，不應寫成 kernel optimization。

6. Diagnostic failure：
   該 case 失敗本身具有研究價值，但不參與 official speedup。
   例如 shmembench-cuda 中 block_size=128/512/1024。
```

這個分類使研究能分析：

```text
為什麼 AI 在某些 benchmark 成功？
為什麼 AI 在某些 benchmark 失敗？
失敗是否揭露了 benchmark 的硬體限制、correctness 限制或測量問題？
```

***

## 7. Phase 2 結果與啟示

Phase 2 結果顯示，AI agent 的優化成果必須依 benchmark 類型與結果性質分類。例如：

```text
softmax-cuda:
  明確 AI primitive / kernel-oriented optimization case。

topk-cuda:
  workspace reuse、block size tuning、radix selection 類問題。

p2p-cuda:
  topology-aware measurement，不應宣稱為顯著 kernel speedup。

allreduce-cuda:
  launcher / UCX / environment fix，不應寫成 kernel optimization。
```

這些分類說明，AI-assisted optimization 研究不能只比較 speedup 數值，還必須區分：

```text
1. 修改了什麼？
2. 是否保持 correctness？
3. 是否使用有效 baseline？
4. 是否只是 environment / measurement 修復？
5. 是否可能是 measurement noise？
```

Phase 3 的 Mode B 設計即是將 Phase 2 的防偽加速規則帶入人機協作流程。

***

## 8. Phase 3 Mode A：Agent-only Baseline 與風險揭露

Mode A 的主要價值不是優化，而是測量基準與風險揭露。

### 8.1 softmax-cuda Mode A

Mode A 建立 `impl=0` naive reference 與 `impl=1` existing optimized baseline 的差異。`impl=1` 明顯優於 `impl=0`，但這是既有 implementation 差異，不是 Phase 3 agent speedup。

因此後續規則明確定義：

```text
impl=0 = naive reference
impl=1 = existing optimized baseline
Mode B/C 的 baseline 必須是 impl=1
不得將 impl=0 → impl=1 計為 Phase 3 speedup
```

### 8.2 topk-cuda Mode A

Mode A 顯示 `topk-cuda` 可能在沒有程式修改的情況下，因 baseline CV 過高產生約 1.4x 表面加速。這是 pseudo-speedup，不是有效優化。

因此 topk 進入 Mode B 前必須先做 robust baseline remeasurement。

### 8.3 shmembench-cuda Mode A

Mode A 顯示 `shmembench-cuda` 的原始 block-size sweep 不完全可用。`block_size=256, variant=original` 是 official validated comparison；`128/512/1024` 應保留為 diagnostic failures，不納入 official speedup。

***

## 9. Mode B Robust Baseline

### 9.1 softmax-cuda robust baseline

```text
official cases:
  slice=128,256,784,1024,2048

baseline implementation:
  impl=1

correctness:
  all PASS

measurement_validity:
  all VALID
```

### 9.2 topk-cuda robust baseline

```text
official cases:
  14 cases

trials:
  7

correctness:
  all PASS

measurement_validity:
  12 VALID, 2 CAUTION, 0 NOISY
```

`topk-cuda` 尚未進入 Mode B optimization，因此不得宣稱 topk Mode B speedup。

### 9.3 shmembench-cuda robust baseline

```text
official validated comparison:
  block_size=256, variant=original

diagnostic sweep:
  block_size=128,512,1024
```

`shmembench-cuda` 尚未進入 Mode B optimization，因此不得宣稱 shmembench Mode B speedup。

***

## 10. softmax-cuda Mode B Round 1：Partial Candidate

### 10.1 Candidate

Round 1 candidate：

```text
impl2_block_cached_exp_compound
```

該 candidate 是 compound candidate，同時改變：

```text
1. row parallelism:
   warp-per-slice → block-per-slice

2. computation / memory strategy:
   recompute exp → shared-memory cached exp
```

因此，任何效能改善不能單獨歸因於 cached exponentials。

### 10.2 Round 1 結果

| slice | paired impl=1 mean ms | impl=2 mean ms | correctness        | result                        |
| ----: | --------------------: | -------------: | ------------------ | ----------------------------- |
|   128 |              0.135152 |       0.554750 | PASS 3/3           | REGRESSION                    |
|   256 |              0.323384 |       0.594147 | PASS 2/3, FAIL 1/3 | INVALID                       |
|   784 |              1.434026 |       1.108087 | PASS 3/3           | valid large-slice improvement |
|  1024 |              2.068956 |       1.300902 | PASS 3/3           | valid large-slice improvement |
|  2048 |              2.212359 |       1.680560 | PASS 3/3           | valid large-slice improvement |

### 10.3 Round 1 判定

```text
Round 1 = PARTIAL_SUCCESS
Full replacement = REJECTED
Large-slice candidate = KEEP
```

Round 1 證明：

```text
impl=2 不適合作為 universal replacement。
impl=2 對 large slices 有價值。
```

這一步也證明 human review 的重要性。若只看 large-slice speedup，agent 可能把 candidate 錯誤包裝為 full success。但人類審查拒絕 universal replacement，並要求建立 shape-aware dispatch。

***

## 11. softmax-cuda Mode B Round 2：Shape-aware Dispatch

### 11.1 Human-guided correction

Round 2 根據 Round 1 evidence 建立 dispatch policy：

| slice | selected impl | reason                             |
| ----: | ------------: | ---------------------------------- |
|   128 |        impl=1 | Round 1 impl=2 regression          |
|   256 |        impl=1 | Round 1 impl=2 correctness failure |
|   784 |        impl=2 | Round 1 impl=2 PASS and faster     |
|  1024 |        impl=2 | Round 1 impl=2 PASS and faster     |
|  2048 |        impl=2 | Round 1 impl=2 PASS and faster     |

Candidate：

```text
impl3_shape_dispatch_impl1_small_impl2_large
```

### 11.2 Round 2 結果

| slice | selected impl | candidate mean ms | paired impl=1 mean ms | correctness | result\_type            |  speedup |
| ----: | ------------: | ----------------: | --------------------: | ----------- | ----------------------- | -------: |
|   128 |             1 |          0.135674 |              0.144732 | PASS        | MEASUREMENT\_EQUIVALENT | 1.066763 |
|   256 |             1 |          0.306408 |              0.305251 | PASS        | MEASUREMENT\_EQUIVALENT | 0.996224 |
|   784 |             2 |          1.107988 |              1.437362 | PASS        | PARAM\_TUNE             | 1.297273 |
|  1024 |             2 |          1.300765 |              2.082344 | PASS        | PARAM\_TUNE             | 1.600861 |
|  2048 |             2 |          1.670514 |              2.213330 | PASS        | PARAM\_TUNE             | 1.324940 |

### 11.3 Round 2 判定

```text
Round 2 = ACCEPT
result_type = PARAM_TUNE / SHAPE_AWARE_DISPATCH
```

Round 2 解決了 Round 1 的問題：

```text
1. small slices 不再使用 impl=2，避免 regression / invalid。
2. large slices 保留 impl=2 的有效改善。
3. 所有 official cases correctness PASS。
```

***

## 12. softmax-cuda Final Confirmation

### 12.1 Final 設定

```text
Slurm job:
  949717

Node:
  gn1228.twcc.ai

Candidate:
  impl3_shape_dispatch_impl1_small_impl2_large

Profiler status:
  NOT_RUN
```

### 12.2 Final result table

| slice | dispatch | baseline mean ms | candidate mean ms |   speedup | correctness | result\_type            | speedup\_claim\_valid |
| ----: | -------: | ---------------: | ----------------: | --------: | ----------- | ----------------------- | --------------------- |
|   128 |   impl=1 |         0.134869 |          0.134574 | 1.002197x | PASS        | MEASUREMENT\_EQUIVALENT | false                 |
|   256 |   impl=1 |         0.321793 |          0.321505 | 1.000895x | PASS        | MEASUREMENT\_EQUIVALENT | false                 |
|   784 |   impl=2 |         1.442716 |          1.036402 | 1.392043x | PASS        | PARAM\_TUNE             | true                  |
|  1024 |   impl=2 |         2.104045 |          1.238443 | 1.698944x | PASS        | PARAM\_TUNE             | true                  |
|  2048 |   impl=2 |         2.237452 |          1.672904 | 1.337466x | PASS        | PARAM\_TUNE             | true                  |

### 12.3 Final interpretation

Final confirmation 顯示：

```text
1. 五個 official slices 均 3/3 correctness PASS。
2. slice=128/256 dispatch 到 impl=1，結果為 measurement-equivalent。
3. slice=784/1024/2048 dispatch 到 impl=2，保留有效 large-slice improvement。
4. auditor / contradiction check 通過。
```

因此：

```text
softmax-cuda Mode B = SUCCESS
result_type = PARAM_TUNE / SHAPE_AWARE_DISPATCH
```

***

## 13. topk-cuda Optional Supporting Evidence

`topk-cuda` 在本報告中不是 Mode B optimization 成果，而是 optional supporting evidence。

目前狀態：

```text
Mode B = BASELINE_ONLY
robust baseline completed
14 cases PASS
7 trials
12 VALID
2 CAUTION
0 NOISY
optimization rounds not started
```

其研究價值在於：

```text
1. Mode A 曾揭露 high-CV pseudo-speedup 風險。
2. Mode B robust baseline 以 7 trials 改善測量穩定性。
3. 這支持本研究對 variance / CV / paired baseline 的重視。
```

不可宣稱：

```text
topk-cuda Mode B 已完成 optimization
topk-cuda Mode B 有 speedup
```

***

## 14. shmembench-cuda Optional Supporting Evidence

`shmembench-cuda` 同樣不是 Mode B optimization 成果，而是 optional supporting evidence。

目前狀態：

```text
Mode B = BASELINE_ONLY
official validated comparison = block_size=256, variant=original
diagnostic failures = block_size=128,512,1024
optimization rounds not started
```

其研究價值在於：

```text
1. low-optimization-space benchmark 需要先確認 correctness-valid comparison。
2. diagnostic failures 不應被刪除，也不應納入 official speedup。
3. 這證明 official sweep 必須依 correctness 收斂，而不能盲目追求完整 sweep。
```

不可宣稱：

```text
shmembench-cuda Mode B 已完成 optimization
diagnostic failures 代表整個 benchmark official failure
```

***

## 15. Mode C

### 15.1 Artifact status

```text
Mode C: NOT_STARTED / NOT_FOUND
```

目前沒有 Mode C artifact。

### 15.2 最佳協作方式

留白。

```text
尚未執行 Mode C，因此不對 Mode C 的最佳協作方式下結論。
```

### 15.3 加速程度

留白。

```text
尚未執行 Mode C，因此不對 Mode C 的加速程度下結論。
```

***

## 16. 成功與失敗原因討論

### 16.1 softmax 成功原因

`softmax-cuda` 成功不是因為 AI 一次產生完美 kernel，而是因為人機協作流程能處理 partial success：

```text
1. AI agent 產生 impl=2 compound candidate。
2. impl=2 在 large slices 有效，但在 small slices 失敗。
3. human review 拒絕 universal replacement。
4. human review 要求 shape-aware dispatch。
5. impl=3 將 partial success 轉成全 official sweep correctness PASS 的策略。
```

### 16.2 topk 暫緩原因

`topk-cuda` 暫未進入 optimization，原因是 Mode A 曾揭露 high-CV pseudo-speedup 風險。它需要 robust baseline 與 variance filter，不能直接從表面 speedup 判斷 AI 是否成功。

### 16.3 shmembench 暫緩原因

`shmembench-cuda` 暫未進入 optimization，原因是 official comparison 需先依 correctness 收斂到 `block_size=256`。其他 block sizes 應作 diagnostic failures，而非納入 official speedup。

### 16.4 人類審查角色

本研究顯示，人類操作者不是只負責看結果，而是負責：

```text
1. 判斷 partial result 的有效邊界。
2. 防止錯誤歸因。
3. 防止偽加速。
4. 阻止 universal replacement 的過度宣稱。
5. 將 regression / invalid case 轉化為下一輪策略。
```

***

## 17. Threats to Validity

### 17.1 單一主案例限制

目前完整 Mode B optimization 僅完成 `softmax-cuda`。因此不能外推為：

```text
Mode B 對所有 benchmark 都有效。
```

更精確的結論是：

```text
在 softmax-cuda 這個高優化空間 benchmark 上，Mode B human-in-the-loop workflow 有效。
```

### 17.2 Profiler 未執行

softmax Mode B final confirmation 中：

```text
profiler_status = NOT_RUN
```

因此不得宣稱 profiler-supported bottleneck conclusion。

### 17.3 `impl=2` 是 compound candidate

`impl=2` 同時改變 row parallelism 與 exp caching，因此不能把 large-slice improvement 單獨歸因於 cached exponentials。

### 17.4 Optional benchmarks 未完成 optimization

`topk-cuda` 與 `shmembench-cuda` 目前只完成 robust baseline，因此不能作為 Mode B speedup 主結果。

### 17.5 Mode C 尚未開始

Mode C 的最佳協作方式與加速程度尚無資料。

### 17.6 Stale summaries

部分 parent-level summary 檔案過時，不應引用。

***

## 18. Do-Not-Claim List

正式報告不得宣稱：

```text
1. Mode C has started.
2. Mode C is effective.
3. topk-cuda Mode B optimization has begun or succeeded.
4. shmembench-cuda Mode B optimization has begun or succeeded.
5. impl=3 is a universal kernel optimization.
6. impl=2 is a universal replacement for all softmax shape sizes.
7. slice=128 or slice=256 has valid optimization speedup under impl=3.
8. profiler-supported bottleneck conclusions for Mode B.
9. shared-memory cached exponentials alone caused the large-slice speedup.
10. impl=0 → impl=1 is Phase 3 speedup.
```

***

## 19. 論文可用核心結論

可寫入論文或報告的核心結論如下：

```text
softmax-cuda 的 Mode B 實驗證明，人機協作流程能將 AI agent 產生的 partial optimization 轉化為可驗證的 shape-aware dispatch policy。Agent 在 Round 1 中提出的 compound block-level cached-exp candidate 對 large slices 有效，但在 small slices 上 regression 或 correctness failure。人類審查拒絕其作為 universal replacement，並引導 agent 在 Round 2 中建立 shape-aware dispatcher。Final confirmation 顯示，該 dispatcher 保留 impl=1 給 slice=128/256，並選擇 impl=2 給 slice=784/1024/2048，所有 official cases correctness PASS，large slices 分別取得 1.392x、1.699x 與 1.337x 有效改善。此結果證明，Mode B 的主要價值在於將 trial-and-error 的 AI 候選優化轉化為 correctness-gated、shape-aware、可審核的工程策略。
```

***

## 20. 總結

本研究整體目標是利用 AI 協助優化既有 benchmark，並分析 AI 成功與失敗的原因。Phase 3 目前已完成 `softmax-cuda` 這一個代表性高優化空間 benchmark 的 Mode B 人機協作實驗。

本階段最重要的結論不是「AI 產生了一個完美 kernel」，而是：

```text
AI 產生的 partial candidate 需要人類審查、baseline 對照、correctness gate、per-case 分析與 artifact auditing，才能收斂成可驗證、可解釋、可放入論文的優化策略。
```

在 `softmax-cuda` 中，這個過程具體表現為：

```text
Round 1:
  AI 提出 impl=2 compound candidate。
  large slices 有效，但 small slices 失敗。

Human review:
  拒絕 universal replacement。
  要求 shape-aware dispatch。

Round 2:
  建立 impl=3 dispatcher。
  small slices 使用 impl=1。
  large slices 使用 impl=2。

Final:
  全 official cases correctness PASS。
  large slices 取得 1.392x、1.699x 與 1.337x 有效改善。
```

因此，目前可正式標記：

```text
softmax-cuda Mode B:
  SUCCESS

result_type:
  PARAM_TUNE / SHAPE_AWARE_DISPATCH

Mode C:
  NOT_STARTED / 留白

topk-cuda:
  optional supporting evidence only

shmembench-cuda:
  optional supporting evidence only
```

本報告的中心主張是：

```text
人機協作的價值不在於讓 AI 一次產生完美解，而在於讓 AI 產生的候選方向，經由人類審查與可審核實驗流程，轉化為穩定、可驗證、可解釋的工程策略。
```
