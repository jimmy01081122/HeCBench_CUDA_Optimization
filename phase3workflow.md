# 第三階段實驗計畫：人機協作式 AI 程式優化流程

## 0. 第三階段定位

第三階段不再擴大 benchmark 數量，而是從第二階段結果中選出三個代表性 benchmark，深入研究：

```text
AI agent 在不同優化空間下，經由人類操作者、文獻查詢、profiler、prompt 約束與自適應工作流輔助後，是否能產生更可信、更有效率、更可解釋的程式優化結果。
```

第二階段已經顯示，不同 benchmark 的 AI 優化成果本質不同：`softmax-cuda` 屬於明確 kernel optimization，`topk-cuda` 屬 workspace reuse / radix selection 優化，`shmembench-cuda` 則屬於 shared memory microbenchmark 且提升幅度較小，需要 profiler 進一步驗證。 [\[arxiv.org\]](https://arxiv.org/abs/2510.00555), [\[arxiv.org\]](https://arxiv.org/abs/2403.17134)

***

# 1. 第三階段研究目標

## 1.1 核心研究問題

第三階段建議聚焦以下問題：

```text
RQ3-1:
在 P3 強約束基礎上加入 human-in-the-loop 後，是否能超越純 agent 的 Phase 2 P3 結果？

RQ3-2:
人類介入最有效的位置是 bottleneck diagnosis、文獻查詢、實驗設計、還是結果審核？

RQ3-3:
加入 profiler 與文獻查詢後，agent 是否能從 trial-and-error 轉為 hypothesis-driven optimization？

RQ3-4:
低、中、高優化空間 benchmark 中，人機協作的邊際收益是否不同？
```

***

# 2. 第三階段 Benchmark 選擇

## 2.1 低、中、高三類代表案例

建議選：

```text
低優化空間：shmembench-cuda
中優化空間：topk-cuda
高優化空間：softmax-cuda
```

選擇理由如下。

| 優化空間 | Benchmark         | 第二階段觀察                                                                                                                                                                                                                     | 第三階段研究價值                                                                         |
| ---- | ----------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------- |
| 低    | `shmembench-cuda` | P3 speedup 約 1.0293x，屬小幅提升，需 profiler 確認是否有實質改善。 [\[arxiv.org\]](https://arxiv.org/abs/2510.00555), [\[arxiv.org\]](https://arxiv.org/abs/2403.17134)                                   | 測試 AI 是否能辨識硬體上限，避免過度宣稱微小提升。                                                      |
| 中    | `topk-cuda`       | P3 speedup 約 1.1995x；有效策略為 workspace reuse、radix selection 與 block size tuning。 [\[arxiv.org\]](https://arxiv.org/abs/2510.00555)                                                           | 測試人類是否能引導 agent 從 workspace tuning 走向 shape-aware dispatch 或更合理的 radix strategy。 |
| 高    | `softmax-cuda`    | P3 normalized speedup 約 1.4575x；BASIC/GM 探索性實驗曾在 slice=784 達到 59.593x，但兩者基準不同不可混用。 [\[arxiv.org\]](https://arxiv.org/abs/2510.00555), [\[arxiv.org\]](https://arxiv.org/abs/2403.17134) | 測試人機協作是否能建立可解釋、跨 shape 的 softmax dispatch policy。                                |

***

# 3. 第三階段總體實驗設計

## 3.1 實驗模式

每個 benchmark 執行三種模式：

```text
Mode A: P3 Agent-only baseline
Mode B: Human-in-the-loop guided optimization
Mode C: Literature + profiler augmented adaptive workflow
```

***

## 3.2 模式定義

### Mode A：P3 Agent-only baseline

目的：建立「強約束但無中途人類介入」的對照組。

```text
輸入：Phase 2 P3 prompt
限制：baseline + 最多 5 次 optimization submission
人類角色：只提供 prompt，不中途指導
產出：agent_summary.md、CSV、raw output
```

### Mode B：Human-in-the-loop guided optimization

目的：測試人類在瓶頸診斷、實驗方向與結果審核中的價值。

```text
輸入：Phase 2 P3 結果 + 人類審查 checkpoint
限制：baseline + 最多 6 次 optimization submission + 1 次 final confirmation
人類角色：批准或否決每輪假設與修改
產出：decision_log.md、human_intervention_log.md、agent_summary.md
```

### Mode C：Literature + profiler augmented adaptive workflow

目的：測試加入文獻查詢與 profiler 後，agent 是否能形成 evidence-driven optimization。

```text
輸入：Phase 2 P3 結果 + profiler + 文獻/文件摘要
限制：baseline + 最多 6 次 optimization submission + 1 次 final confirmation
人類角色：確認文獻適用性與 profiler 解釋是否合理
產出：profiler_summary.md、literature_notes.md、decision_log.md
```

***

# 4. 自適應工作流程

每一輪優化必須依照固定流程執行，不允許 agent 直接「改完就跑」。

## 4.1 每輪流程

```text
Step 1: Observe
讀取 Phase 2 P3 baseline、source code、raw output、profiler 或 timing 資料。

Step 2: Diagnose
提出單一 bottleneck hypothesis。

Step 3: Retrieve
查詢 CUDA 文件、論文、既有優化案例或本專案先前結果。

Step 4: Plan
提出一個最小修改，說明預期改善與風險。

Step 5: Human Checkpoint
人類批准、要求修改或拒絕該輪計畫。

Step 6: Execute
執行 sbatch，保存 raw output、stderr、CSV。

Step 7: Validate
檢查 correctness、performance、variance、profiler 指標與 contradiction。

Step 8: Decide
接受、拒絕、rollback 或停止。
```

***

## 4.2 每輪必填紀錄

每輪必須寫入 `decision_log.md`：

```markdown
## Round N

### Observation
- baseline metric:
- current best metric:
- correctness:
- profiler observation:

### Hypothesis
- bottleneck hypothesis:
- evidence:

### Proposed Change
- files to modify:
- expected improvement:
- risk:

### Human Decision
- approved / rejected / revise
- reason:

### Result
- job id:
- correctness:
- metric:
- speedup:
- accepted / rejected:
- reason:
```

***

# 5. 通用硬性規則

第三階段所有實驗共用以下規則。

## 5.1 正確性規則

```text
1. correctness FAIL → result invalid
2. partial PASS → invalid unless explicitly marked partial
3. changed input size → invalid for official comparison
4. changed correctness tolerance → must be justified and separately reported
5. approximate algorithm cannot replace exact algorithm unless task explicitly changes
```

## 5.2 效能與次數規則

```text
1. speedup 必須相對 Phase 2 P3 valid baseline 或是本階段重測的 baseline
2. BASIC exploratory result 可作參考，但不得混入 Phase 3 normalized comparison
3. improvement < 1% → MEASUREMENT_EQUIVALENT 或是 marginal
4. 對於正式的加速比計算，Baseline 與 Final Candidate 皆必須執行至少 3 次 trials。若 Baseline 最初僅測得 1 次，應於報告最終加速比前重新進行多次測量確認。
5. final result 必須報 mean / min / max / stddev / CV
```

## 5.3 結果分類

每一輪結果必須標為：

```text
BASELINE
KERNEL_OPT
PARAM_TUNE
MEASURE_FIX
PROFILER_EXPLANATION
NO_EFFECT
REGRESSION
MEASUREMENT_EQUIVALENT
INVALID
```

## 5.4 停止條件

任一條成立即停止該 benchmark：

```text
1. 連續 2 輪有效修改 speedup < 1%
2. profiler 顯示已接近瓶頸且無合理修改方向
3. correctness 連續失敗且無明確修復路徑
4. human reviewer 判定下一步會改變 benchmark 語意
5. 達到 submission limit
```

## 5.5 第三階段額外硬性規則 (Additional Hard Rules for Phase 3)

1. **CSV Schema 統一**：Mode A、Mode B 和 Mode C 的 CSV schema 必須完全一致。
   對於 Mode A：
   - `mode = Mode_A`
   - `round = submission index`
   - `human_decision = None_Agent_Only`
   
2. **禁止 login node 直接執行**：禁止在 login node 直接執行 `./main` 或任何 GPU benchmark binary。
   所有 correctness validation、timing、profiler、MPI/NCCL 以及 GPU 執行步驟必須透過 sbatch 進行。
   
3. **Profiler 容錯與 fallback**：Nsight Compute (`ncu`) 為建議而非阻塞條件。
   若 ncu 不可用或發生權限錯誤（如 hardware counter 被拒）：
   - 設置 `profiler_available = False`
   - 在 `profiler_summary.md` 記錄原因，並在 final report 寫入 limitations
   - 繼續使用 timing + correctness + variance 做最低限度分析，不重複盲目嘗試
   
4. **固定 Sweep 規格**：官方測試 cases 必須完全固定，不得為提升平均加速比而刪除慢 case 或失敗 case。
   - `softmax-cuda`: `slice_size = 128, 256, 784, 1024, 2048`
   - `topk-cuda`: `hidden_size = 3072, 4096, 8192, 16384, 32768, 65536, 131072` 且 `topk = 1024, 2048`（共 14 cases）
   - `shmembench-cuda`: `block_size = 128, 256, 512, 1024`，且新實驗需標註 `variant`
   
5. **執行一致性審計 (Self-Consistency Auditor)**：每個 Mode 完成後，必須自動執行一致性檢查。
   若偵測到矛盾，在修正前該結果不得晉升為最終報告，且不得手動覆寫 `contradiction_check.csv`。
   審計器應強制執行以下細部檢驗規則：
   - **Rule V1**：若變異數 `CV > 15%`，將 `measurement_validity` 設為 `NOISY`。
   - **Rule V2**：若 `CV > 15%` 且加速比 `speedup > 1.05`，將 `speedup_claim_valid` 設為 `false`。
   - **Rule V2b**：若 `speedup < 1.01`，設定 `result_type = MEASUREMENT_EQUIVALENT` 且 `speedup_claim_valid = false`。
   - **Rule V2c**：若該輪次未發生任何原始碼/程式修改，`speedup_claim_valid` 必須設為 `false`（除非明確標記為 `BASELINE_REMEASUREMENT` 或 `BASELINE_COMPARISON` 以供測量穩定性分析使用）。
   - **Rule V3**：若 `benchmark=softmax-cuda` 且比較組為 `impl0_to_impl1`，應將 `result_type` 設為 `BASELINE_COMPARISON`，而非 `AGENT_OPT`。
   - **Rule V4**：若 `benchmark=shmembench-cuda`、`block_size != 256` 且 `correctness != PASS`，標記為 `DIAGNOSTIC_FAIL`，而非判定整題失敗。
   - **Rule V5**：若官方驗證的 baseline 缺失，`speedup` 必須設為 `n/a`。
   - **Rule V6**：若可選的變體取代了原始 Naive Baseline 行，則判定該行結果 `INVALID`。
   - **Rule V7**：若 `correctness_status != PASS`，則 `speedup` 必須設為 `n/a` 且 `speedup_claim_valid = false`。
   - **Rule V8**：若為 Mode A 且無原始碼修改，則 `result_type` 必須被歸類為 `BASELINE_REMEASUREMENT`、`BASELINE_COMPARISON`、`MEASUREMENT_EQUIVALENT`、`REGRESSION` 或 `NOISY_MEASUREMENT`，不得歸類為 `KERNEL_OPT`。
   - **Rule V9**：若加速比小於 1%（speedup < 1.01），設置 `result_type = MEASUREMENT_EQUIVALENT` 且 `speedup_claim_valid = false`。
   - **Rule V10**：若加速比小於 1.0（speedup < 1.0），設置 `result_type = REGRESSION` 且 `speedup_claim_valid = false`。

   **全域執行規則**：Mode B/C 不得直接沿用 Mode A 中由 repeated measurement 產生的 speedup 作為優化基準。Mode A 的加速比數值僅視為測量穩定性指標而非優化基準。Mode B/C 開始前，必須針對對應配置重新建立一次 correctness-gated 且重複測量的 Baseline，再以此計算後續的優化加速比。
   
6. **無效 Baseline 處理**：若 baseline 無效，則 `speedup` 必須填為 `n/a`。
   
7. **加速比判定**：若加速比低於 1% (speedup < 1.01)，必須分類為 `MEASUREMENT_EQUIVALENT` 或邊際（marginal）提升，不得宣稱為 significant。
   
8. **分類嚴謹性**：`ENV_FIX`、`MEASURE_FIX` 與 `TOPOLOGY_MEASURE` 不得被描述為 kernel 層級優化（kernel-level optimization）。

***

# 6. 實驗目錄結構

建議第三階段所有資料放在：

```text
/home/a/PP/phase3
```

目錄：

```text
phase3/
├── README.md
├── metadata/
│   ├── phase3_benchmarks.csv
│   ├── phase3_protocol.csv
│   └── result_schema.csv
├── prompts/
│   ├── softmax-cuda/
│   ├── topk-cuda/
│   └── shmembench-cuda/
├── softmax-cuda/
│   ├── baseline/
│   ├── mode_A_agent_only/
│   ├── mode_B_human_guided/
│   ├── mode_C_literature_profiler/
│   └── final_summary.md
├── topk-cuda/
│   ├── baseline/
│   ├── mode_A_agent_only/
│   ├── mode_B_human_guided/
│   ├── mode_C_literature_profiler/
│   └── final_summary.md
├── shmembench-cuda/
│   ├── baseline/
│   ├── mode_A_agent_only/
│   ├── mode_B_human_guided/
│   ├── mode_C_literature_profiler/
│   └── final_summary.md
└── reports/
    ├── PHASE3_REPORT.md
    ├── PHASE3_TABLES.md
    └── PHASE3_THREATS_TO_VALIDITY.md
```

***

# 7. 三個 Benchmark 的具體計畫

***

## 7.1 `softmax-cuda`：高優化空間案例

### 7.1.1 研究目標

```text
建立 shape-aware softmax dispatch policy，檢查人機協作是否能超越 Phase 2 P3 的 normalized result。
```

### 7.1.2 Phase 2 參考結果

```text
Phase 2 P3 speedup: 1.4575x
BASIC exploratory best: slice=784 達 59.593x
注意：兩者 baseline 與測量範圍不同，不可直接混用。
```

第二階段資料明確指出，`softmax-cuda` 是實質 kernel optimization 案例，但 BASIC/GM 的 59.593x 與 Phase 2 P3 的 1.4575x 屬不同基準。 [\[arxiv.org\]](https://arxiv.org/abs/2510.00555), [\[arxiv.org\]](https://arxiv.org/abs/2403.17134)

### 7.1.3 必測 shapes

```text
slice = 128
slice = 256
slice = 784
slice = 1024
slice = 2048
```

建議：

```text
batch = 100000 for slice <= 1024
batch = 50000 for slice = 2048
repeat = 根據 runtime 設定，但每組至少 3 trials
```

### 7.1.4 優化方向

```text
1. warp-level softmax for small slice
2. block-level softmax for medium slice
3. multi-warp / block reduction for larger slice
4. cache expf result if beneficial
5. shape-aware dispatch
6. optional fast_math comparison, but correctness must remain PASS
```

### 7.1.5 Profiler 指標

```text
dram throughput
achieved occupancy
warp execution efficiency
registers per thread
kernel duration
special function unit pressure if available
```

### 7.1.6 Human Checkpoints

人類必須審查：

```text
1. 是否過度特化 single slice
2. 是否放寬 tolerance
3. fast_math 是否造成數值風險
4. dispatch policy 是否合理
```

### 7.1.7 成功標準

```text
1. 所有 official shape correctness PASS
2. final mean speedup > Phase 2 P3 baseline
3. 若只提升單一 shape，必須標 partial success
4. profiler 支持瓶頸改善
```

***

## 7.2 `topk-cuda`：中優化空間案例

### 7.2.1 研究目標

```text
從 workspace reuse 延伸到 shape-aware radix top-k 策略，檢查 human guidance 是否能超越 agent-only P3。
```

### 7.2.2 Phase 2 參考結果

```text
P3 speedup: 1.1995x
GM early result: 14 hidden_size/topk combinations 平均 speedup 1.442x
主要策略：workspace reuse、block size tuning、減少 repeated allocation / synchronization
```

第二階段資料顯示 `topk-cuda` 的有效策略是 workspace reuse、cached workspace 與 block size tuning。 [\[arxiv.org\]](https://arxiv.org/abs/2510.00555)

### 7.2.3 必測 cases

```text
hidden_size = 3072, 4096, 8192, 16384, 32768, 65536, 131072
topk = 1024, 2048
```

共：

```text
14 cases
```

### 7.2.4 優化方向

```text
1. CUB workspace reuse
2. block size 256 / 512 / 1024 sweep
3. reduce synchronization
4. shape-aware dispatch
5. buffer reuse
6. avoid repeated cudaMalloc / cudaFree in timed loop
```

### 7.2.5 Profiler 指標

```text
kernel launch count
cudaMalloc / cudaFree count
workspace allocation overhead
dram throughput
occupancy
register pressure
temporary storage size
```

### 7.2.6 Human Checkpoints

人類必須審查：

```text
1. 是否仍是 exact top-k
2. 是否改變排序或 tie-breaking 語意
3. 是否只優化某一 hidden_size
4. 是否跳過慢 case
```

### 7.2.7 成功標準

```text
1. 14 cases 全部 correctness PASS
2. final geometric mean speedup > Phase 2 P3 baseline
3. 若某些 case regression，必須報告
4. shape-aware policy 可解釋
```

***

## 7.3 `shmembench-cuda`：低優化空間案例

### 7.3.1 研究目標

```text
驗證 AI 是否能辨識硬體限制與 measurement-equivalent result，而不是過度宣稱微小提升。
```

### 7.3.2 Phase 2 參考結果

```text
P3 speedup: 約 1.0293x
性質：shared memory microbenchmark
後續需求：profiler bank conflict / shared throughput
```

第二階段資料指出，`shmembench-cuda` 屬於小幅改善案例，需 profiler 驗證 shared memory / bank conflict 相關解釋。 [\[arxiv.org\]](https://arxiv.org/abs/2510.00555), [\[arxiv.org\]](https://arxiv.org/abs/2403.17134)

### 7.3.3 必測 cases

```text
block size = 128, 256, 512, 1024
access pattern = original, padded, vectorized if supported
repeat = fixed
trials >= 3
```

### 7.3.4 優化方向

```text
1. bank conflict reduction
2. padding
3. vectorized shared memory access
4. block size tuning
5. reduce unnecessary __syncthreads
```

### 7.3.5 Profiler 指標

```text
shared load throughput
shared store throughput
shared bank conflicts
achieved occupancy
block duration
instruction throughput
```

### 7.3.6 Human Checkpoints

人類必須審查：

```text
1. bandwidth 計算公式是否被改變
2. checksum / correctness 是否完整
3. 是否少算資料量
4. 是否只是 measurement artifact
```

### 7.3.7 成功標準

```text
1. correctness PASS
2. 若 speedup < 1%，標 measurement-equivalent
3. 若 speedup 1%~5%，需 profiler 支持
4. 若無有效加速，但能證明硬體限制，也視為研究成功
```

***

# 8. 統一 Result Schema

建立：

```text
phase3/metadata/result_schema.csv
```

欄位：

```csv
benchmark,mode,round,job_id,node,case,variant,metric_name,metric_value,unit,baseline_metric,speedup,correctness,status,result_type,mean,min,max,stddev,cv,profiler_available,human_decision,correctness_status,measurement_validity,speedup_claim_valid,notes
```

## 8.1 Mode A 固定填值規則

```text
mode = Mode_A
round = submission index
human_decision = None_Agent_Only
profiler_available = True / False
```

## 8.2 Mode B / C 固定填值規則

```text
mode = Mode_B 或 Mode_C
round = adaptive round index
human_decision = Approved / Rejected / Revise / Stop
profiler_available = True / False
```

## 8.3 加入 Phase 3 Prompt 的規則

```text
All modes must write CSV rows using the same result_schema.csv.

For baseline CSV rows:
- mode = Mode_A
- round = baseline
- human_decision = None_Agent_Only
- baseline_metric = n/a
- speedup = n/a
- result_type = BASELINE

For optimization rows:
- round = 1, 2, 3, ... (corresponding to optimization round index)

For final confirmation rows:
- round = final

Do not omit human_decision just because no human was involved; set it strictly to None_Agent_Only for Mode A.
```

***

# 9. Human Intervention Log Schema

建立：

```text
human_intervention_log.md
```

模板：

```markdown
# Human Intervention Log

## Intervention N

- benchmark:
- mode:
- round:
- trigger:
- human instruction:
- agent response:
- accepted:
- reason:
- effect on result:
```

Trigger 類型：

```text
correctness_fail
suspected_pseudo_speedup
profiler_contradiction
semantic_change_risk
measurement_scope_change
insufficient_evidence
stop_condition
```

***

# 10. Phase 3 Prompt 模板

每個 benchmark 都用同一骨架，再填入 benchmark-specific section。

```markdown
# Phase 3 Human-AI Collaborative Optimization Prompt

You are a CUDA performance engineer collaborating with a human researcher.

## Benchmark

- benchmark:
- path:
- category:
- Phase 2 P3 baseline:
- Phase 2 result type:
- known limitations:

## Research Goal

Improve beyond Phase 2 P3 if possible.
If not possible, produce a profiler-backed explanation.

## Required Workflow

For every optimization round:

1. Observe
2. Diagnose
3. Retrieve
4. Plan
5. Human checkpoint
6. Execute
7. Validate
8. Decide

## Hard Rules

- Do not remove correctness.
- Do not reduce official input size.
- Do not change benchmark semantics.
- If correctness FAIL, result invalid.
- If speedup < 1%, mark measurement-equivalent.
- If profiler contradicts hypothesis, reject modification.
- If improvement depends on changing measurement scope, reject modification.
- Always preserve raw output.

## Required Outputs

- raw .out / .err
- result CSV
- decision_log.md
- human_intervention_log.md
- profiler_summary.md
- agent_summary.md
```

***

# 11. 執行順序

建議使用以下 11 步整合執行順序，以確保跨模式（Mode A/B/C）的測試規格被嚴格控制，且每一步都有自動審計把關：

```text
Step 0: Create phase3 directory and schema files
Step 1: Lock official sweep specifications
Step 2: Run Mode A for softmax/topk/shmembench
Step 3: Run self-consistency auditor for Mode A
Step 4: Human review Mode A results
Step 5: Run Mode B human-guided workflow
Step 6: Run self-consistency auditor for Mode B
Step 7: Run Mode C literature/profiler workflow
Step 8: Run self-consistency auditor for Mode C
Step 9: Merge Mode A/B/C CSVs
Step 10: Generate PHASE3_REPORT.md
```

***

# 12. Phase 3 最終報告結構

建立：

```text
phase3/reports/PHASE3_REPORT.md
```

內容：

```markdown
# Phase 3 Human-AI Collaborative Optimization Report

## 1. Objective

## 2. Benchmark Selection

## 3. Experimental Modes

## 4. Workflow Design

## 5. Softmax Results

## 6. TopK Results

## 7. Shmembench Results

## 8. Human Intervention Analysis

## 9. Profiler-supported Bottleneck Analysis

## 10. Comparison Against Phase 2 P3

## 11. Collaboration Gain

## 12. Failure Modes

## 13. Threats to Validity

## 14. Conclusion
```

***

# 13. 核心評估指標

## 13.1 效能

```text
speedup over Phase 2 P3
mean / stddev / CV
per-case regression count
```

## 13.2 協作效率

```text
rounds to first valid improvement
human interventions count
accepted modifications / proposed modifications
rollback count
```

## 13.3 嚴謹性

```text
correctness pass rate
invalid result count
contradiction count
profiler-supported claims count
```

## 13.4 人機協作收益

```text
collaboration_gain = Phase3_HITL_best / Phase2_P3_best
```

僅當：

```text
same benchmark
same case
same metric
same correctness rule
```

才可計算。

***

# 14. 最終建議

第三階段應避免再追求「benchmark 數量」。你要證明的是：

```text
人機協作如何改變 AI agent 的優化品質。
```

因此第三階段的重點應是：

```text
1. 少量 benchmark
2. 深度 workflow
3. 強 correctness
4. profiler-supported conclusion
5. decision log
6. human intervention analysis
```

這樣最後論文主張會更穩：

```text
AI agent 能產生程式優化，但其科學價值取決於 prompt 約束、人類審查與可驗證工作流。人類的角色不是取代 AI 寫程式，而是定義有效結果、控制實驗邊界、檢查假設與防止偽加速。
```

***

# 15. 建議補充的檔案

第三階段應新增以下檔案與結構以落實規範：

```text
phase3/metadata/official_sweeps.yaml
phase3/metadata/result_schema.csv
phase3/tools/self_consistency_auditor.py
phase3/templates/human_intervention_log.md
phase3/templates/profiler_summary.md
phase3/templates/decision_log.md
```

其中 `official_sweeps.yaml` 的內容如下，定義了固定的 Sweep 規格：

```yaml
softmax-cuda:
  official_cases:
    - batch_size: 100000
      slice_size: 128
    - batch_size: 100000
      slice_size: 256
    - batch_size: 100000
      slice_size: 784
    - batch_size: 100000
      slice_size: 1024
    - batch_size: 50000
      slice_size: 2048

topk-cuda:
  hidden_size: [3072, 4096, 8192, 16384, 32768, 65536, 131072]
  topk: [1024, 2048]

shmembench-cuda:
  block_size: [128, 256, 512, 1024]
  variants:
    - original
    - padded
    - vectorized
```
