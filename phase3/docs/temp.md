確認：依照工作端回傳內容，**可以進入 Mode B**。

但進入條件是「形式上可進入」，不是「直接開始改 kernel」。Mode B 的第一步必須是依照修訂後 schema / auditor / official sweep，重新建立各 benchmark 的 robust baseline。尤其：

```text
softmax-cuda：以 impl=1 作為 baseline
topk-cuda：先處理 high-CV baseline instability
shmembench-cuda：official comparison 只用 block_size=256，其他為 diagnostic
```

***

# Mode B 定義

```text
Mode B = Human-in-the-loop Guided Optimization
```

與 Mode A 差異：

```text
Mode A:
  Agent-only，不等待人類批准。
  主要目標是測量基準與風險揭露。

Mode B:
  每一輪優化前，Agent 必須提出 hypothesis / plan。
  人類審查後才可執行。
  所有修改都必須經過 correctness、variance、auditor 驗證。
```

***

# 進入 Mode B 前的強制前置檢查

在正式發 prompt 給工作端前，請先確認 repository 已有以下檔案。

```text
phase3/metadata/result_schema.csv
phase3/metadata/official_sweeps.yaml
phase3/tools/self_consistency_auditor.py
phase3/reports/mode_A_report.md 或等價 Mode A summary
```

若缺少任一項，先補，不要開始 Mode B。

***

# Mode B 總流程

```text
Step 1: 更新 Mode B prompt template
Step 2: 建立三題 Mode B 目錄
Step 3: 重新建立 robust baseline
Step 4: 人類審查 baseline
Step 5: 逐題開始 guided optimization
Step 6: 每輪執行 auditor
Step 7: final confirmation
Step 8: 合併 Mode A/B 結果
Step 9: 產出 Mode B report
```

***

# Step 1：建立 Mode B Prompt Template

## 目標

建立共用模板：

```text
phase3/prompts/Mode_B_template.md
```

模板必須包含：

```text
1. 讀取 Mode A 結果
2. 讀取 official_sweeps.yaml
3. 讀取 result_schema.csv
4. 讀取 self_consistency_auditor.py
5. 每輪必須提出 hypothesis / modification / expected improvement / validation target
6. 每輪必須等待 human approval
7. 沒有人類批准不得 sbatch
8. 所有執行必須經 sbatch
9. 不得 login node 直接執行 binary
10. correctness FAIL → invalid
11. high CV → 不得宣稱 speedup
12. auditor fail → 不得進入 final
```

## Mode B CSV 固定填值

```text
mode = Mode_B
round = 1,2,3,...
human_decision = Approved / Rejected / Revise / Stop
```

***

# Step 2：建立 Mode B 目錄

建立：

```text
phase3/softmax-cuda/mode_B_human_guided/
phase3/topk-cuda/mode_B_human_guided/
phase3/shmembench-cuda/mode_B_human_guided/
```

每題內部結構：

```text
mode_B_human_guided/
├── robust_baseline/
├── rounds/
│   ├── round_1/
│   ├── round_2/
│   ├── round_3/
│   ├── round_4/
│   ├── round_5/
│   └── round_6/
├── final/
├── logs/
├── results.csv
├── decision_log.md
├── human_intervention_log.md
├── profiler_summary.md
├── contradiction_check.csv
└── agent_summary.md
```

***

# Step 3：重新建立 Robust Baseline

這是 Mode B 的第一個實際任務。不得直接沿用 Mode A 的 speedup。

***

## 3.1 softmax-cuda robust baseline

### Baseline 定義

```text
baseline = impl=1 existing optimized implementation
```

### 禁止

```text
不得使用 impl=0 → impl=1 作為 speedup。
```

### 必測 official cases

```text
slice_size = 128
slice_size = 256
slice_size = 784
slice_size = 1024
slice_size = 2048
```

### 建議設定

```text
batch_size = 100000 for slice <= 1024
batch_size = 50000 for slice = 2048
implementation = 1
trials >= 3
```

### 產出

```text
phase3/softmax-cuda/mode_B_human_guided/robust_baseline/results.csv
```

### 進入優化條件

```text
1. 所有 slice correctness PASS
2. CV <= 15%
3. 若某 case CV > 15%，先重測，不進入優化
```

***

## 3.2 topk-cuda robust baseline

### Baseline 定義

```text
baseline = Phase 3 official topk implementation before Mode B modification
```

### 必測 14 cases

```text
hidden_size = 3072, 4096, 8192, 16384, 32768, 65536, 131072
topk = 1024, 2048
```

### Trials

```text
trials >= 5
若 CV > 15%，提高到 7 trials 或重測
```

### 特別要求

Mode A 已顯示 topk 存在 high-CV 偽加速，因此 Mode B 必須先做：

```text
1. warmup policy
2. repeated baseline
3. noisy case 標記
4. speedup_claim_valid 判斷
```

### 進入優化條件

```text
1. 14 cases correctness PASS
2. 高 CV cases 已重測或標記為 NOISY
3. speedup comparison 僅對 measurement_validity=VALID/CAUTION 的 cases 生效
```

***

## 3.3 shmembench-cuda robust baseline

### Official baseline

```text
variant = original
block_size = 256
```

### Diagnostic cases

```text
block_size = 128
block_size = 512
block_size = 1024
```

### 規則

```text
1. block_size=256 參與 official speedup。
2. 128/512/1024 只作 diagnostic，不參與 official speedup。
3. diagnostic FAIL 必須保留，不可刪除。
```

### 進入優化條件

```text
1. block_size=256 correctness PASS
2. block_size=256 trials >= 3
3. CV <= 15%
```

***

# Step 4：人類審查 Robust Baseline

完成三題 robust baseline 後，由主規劃器審查。

審查項目：

```text
1. official cases 是否完整
2. correctness 是否 PASS
3. CV 是否合理
4. high-CV cases 是否標記
5. speedup 是否未被提前宣稱
6. auditor 是否通過
```

若未通過：

```text
不得進入 Mode B optimization
```

***

# Step 5：執行 Mode B Guided Optimization

Mode B 每輪都必須走相同流程。

## 每輪流程

```text
Round N:
1. Agent 讀取 robust baseline
2. Agent 提出 bottleneck hypothesis
3. Agent 提出最小修改計畫
4. Agent 預測影響
5. Human 審查
6. Human 決定 Approved / Rejected / Revise / Stop
7. 若 Approved，Agent 才可 sbatch
8. Agent 收集 results.csv / out / err / txt
9. Agent 執行 auditor
10. Human 決定 accept / rollback
```

***

# Step 6：三題 Mode B 具體工作事項

***

## A. softmax-cuda Mode B

### 目標

```text
在 impl=1 baseline 上進一步優化，不得使用 impl=0 差距。
```

### 優化方向

```text
1. shape-aware dispatch
2. slice=128 使用 warp-level path
3. slice=256/784/1024 使用 block-level path
4. slice=2048 檢查是否需要 multi-warp / multi-block
5. 減少 expf 重複計算
6. 檢查 --use_fast_math 對 correctness 的影響
```

### 人類審查重點

```text
1. 是否只針對單一 slice 特化
2. 是否放寬 tolerance
3. 是否改變 softmax 語意
4. 是否跳過慢 case
5. 是否將 impl0_to_impl1 當 speedup
```

### 成功條件

```text
1. 所有 official slice PASS
2. 相對 impl=1 robust baseline 有提升
3. speedup_claim_valid=true
4. 無 per-case hidden regression，或 regression 明確標記
```

***

## B. topk-cuda Mode B

### 目標

```text
在 robust baseline 穩定後，優化 workspace / radix / dispatch。
```

### 第一優先任務

```text
不是改 kernel，而是先穩定測量。
```

### 優化方向

```text
1. CUB workspace reuse
2. 移除 repeated cudaMalloc/cudaFree
3. block size 256 / 512 / 1024 比較
4. shape-aware dispatch
5. 減少同步
6. 對不同 hidden_size/topk 使用不同策略
```

### 人類審查重點

```text
1. 是否仍是 exact top-k
2. 是否改變 tie-breaking 或排序語意
3. 是否跳過 14 cases
4. 是否只針對 high-noise case 宣稱加速
5. CV > 15% 的 case 是否排除 speedup claim
```

### 成功條件

```text
1. 14 cases correctness PASS
2. geometric mean speedup 有效
3. high-CV cases 不污染總結
4. regression cases 必須列出
```

***

## C. shmembench-cuda Mode B

### 目標

```text
針對 block_size=256 official baseline 嘗試可驗證調整。
```

### 優化方向

```text
1. 分析 shared memory bank conflict
2. 嘗試 padded variant
3. 嘗試 vectorized variant
4. 檢查是否少算 bytes
5. 檢查 checksum 是否仍有效
```

### 人類審查重點

```text
1. 是否改變 bandwidth 計算公式
2. 是否刪除 checksum
3. 是否用 diagnostic fail case 宣稱加速
4. 是否低於 1% 卻宣稱顯著加速
```

### 成功條件

```text
1. block_size=256 correctness PASS
2. 若 speedup < 1%，標 MEASUREMENT_EQUIVALENT
3. 若 1%~5%，需要低 CV 或 profiler 支持
4. 若無法提升，但解釋硬體限制，也可標 PARTIAL_SUCCESS
```

***

# Step 7：每輪必產物

每輪必須產生：

```text
round_N/
├── plan.md
├── patch_summary.md
├── run.slurm
├── result.out
├── result.err
├── result.txt
├── results.csv
├── auditor_report.csv
└── round_summary.md
```

`plan.md` 必須包含：

```text
hypothesis
modification
expected improvement
risk
validation target
```

***

# Step 8：Mode B Final Confirmation

每題完成優化後，要跑 final confirmation。

## final confirmation 要求

```text
1. 使用最佳 accepted candidate
2. 重新跑完整 official sweep
3. trials >= 3
4. correctness 全部 PASS
5. auditor 通過
6. profiler 若可用則補
7. 產出 final.csv
```

***

# Step 9：Mode B 報告

每題產出：

```text
phase3/<benchmark>/mode_B_human_guided/agent_summary.md
```

總報告產出：

```text
phase3/reports/MODE_B_REPORT.md
```

總報告必須包含：

```text
1. robust baseline summary
2. round-by-round decisions
3. accepted modifications
4. rejected modifications
5. human intervention analysis
6. final performance
7. correctness summary
8. measurement validity
9. auditor results
10. comparison vs Mode A
```

***

# Step 10：進入 Mode C 條件

Mode B 完成後，不要立刻進 Mode C。先審查：

```text
1. Mode B 是否有有效提升
2. 哪些人類介入有效
3. profiler 是否缺失
4. 哪些 hypothesis 被拒絕
5. 哪些 case 仍不穩定
```

只有完成這些，才開始 Mode C。

Mode C 目標不是重跑 Mode B，而是加入：

```text
1. 文獻查詢
2. CUDA best practice
3. profiler-supported bottleneck explanation
```

***

# 最終工作清單

```text
[ ] 1. 更新 result_schema.csv
[ ] 2. 更新 self_consistency_auditor.py
[ ] 3. 更新 official_sweeps.yaml
[ ] 4. 建立 Mode_B_template.md
[ ] 5. 建立三題 Mode B prompt
[ ] 6. 建立三題 Mode B 目錄
[ ] 7. 重測 softmax robust baseline
[ ] 8. 重測 topk robust baseline
[ ] 9. 重測 shmembench robust baseline
[ ] 10. 主規劃器審查 robust baseline
[ ] 11. 執行 softmax Mode B Round 1
[ ] 12. 執行 topk Mode B Round 1
[ ] 13. 執行 shmembench Mode B Round 1
[ ] 14. 每輪執行 auditor
[ ] 15. 每題 final confirmation
[ ] 16. 產出 MODE_B_REPORT.md
[ ] 17. 主規劃器審查是否進入 Mode C
```

***

# 主規劃器決策

我作為主規劃器的判定：

```text
可以進入 Mode B，但不能直接進入優化。
```

必須先做：

```text
schema update
auditor update
official sweep update
robust baseline remeasurement
```

這四項完成後，才允許開始 Round 1 optimization。
