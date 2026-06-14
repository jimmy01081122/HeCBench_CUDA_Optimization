以下開始規劃 **Phase 2：Prompt 規格設計與實驗操作流程**。  
Phase 2 的核心目標是把「給 AI agent 的 prompt」從臨時指令，標準化成可比較、可重現、可審核的實驗條件。

你目前已有 `prompt_inventory`，其中多數 prompt 已包含 correctness、baseline、sbatch、submission limit、raw output 等欄位，但仍普遍缺少 profiler 指標與變異統計要求。這代表 Phase 2 的重點不是重新發明 prompt，而是建立一套分層 prompt 規格與驗證制度。 [\[prompt_inventory 1 \| Excel\]](https://ntums365-my.sharepoint.com/personal/r14525078_m365_ntu_edu_tw/_layouts/15/Doc.aspx?sourcedoc=%7B7D3DA484-20A1-40A8-AAB4-88ECAA36DEA8%7D&file=prompt_inventory%201.csv&action=default&mobileredirect=true)

***

# Phase 2：Prompt 規格設計計畫書

## 1. Phase 2 研究目的

Phase 2 要回答以下問題：

```text
RQ2-1: prompt 約束強度是否影響 AI agent 的正確性、效能與可審核性？
RQ2-2: 強約束 prompt 是否能降低偽加速、錯誤報告與不可重現結果？
RQ2-3: 哪些 prompt 條款最關鍵？
RQ2-4: prompt.md 是否比一般網頁對話更適合作為工程協作介面？
```

目前報告已指出，缺少 baseline、矛盾檢查與 raw output 的情境下，agent 可能出現「失敗與全通過同時宣稱」或「使用 estimated baseline 過度推論」等問題。 [\[ntums365-m...epoint.com\]](https://ntums365-my.sharepoint.com/personal/r14525078_m365_ntu_edu_tw/Documents/Microsoft%20Copilot%20Chat%20%E6%AA%94%E6%A1%88/REPORT.md)

***

# 2. Phase 2 輸入資料

Phase 1 已建立 benchmark taxonomy，本階段使用以下 10 個 benchmark：

```text
p2p-cuda
shmembench-cuda
topk-cuda
allreduce-cuda
moe-cuda
pingpong-cuda
simpleMultiDevice-cuda
moe-align
prefetch-cuda
softmax-cuda
```

這些 benchmark 已涵蓋 AI primitive、memory system、multi-GPU communication、environment repair、measurement characterization 等類型。`softmax-cuda` 是典型 kernel optimization 案例，`allreduce-cuda` 是 environment / launcher repair 案例，`p2p-cuda` 則是 topology-aware measurement 案例。 [\[benchmark_summary \| Excel\]](https://ntums365-my.sharepoint.com/personal/r14525078_m365_ntu_edu_tw/_layouts/15/Doc.aspx?sourcedoc=%7B186B7123-75D1-472D-B5D3-A69B0A4CAA82%7D&file=benchmark_summary.csv&action=default&mobileredirect=true), [\[ntums365-m...epoint.com\]](https://ntums365-my.sharepoint.com/personal/r14525078_m365_ntu_edu_tw/Documents/Microsoft%20Copilot%20Chat%20%E6%AA%94%E6%A1%88/REPORT.md)

***

# 3. Phase 2 產出目標

Phase 2 最終應產生以下檔案：

```text
phase2/
├── README.md
├── prompt_schema.md
├── prompt_quality_rubric.md
├── prompt_templates/
│   ├── P1_weak_prompt_template.md
│   ├── P2_medium_prompt_template.md
│   └── P3_strong_prompt_template.md
├── prompts/
│   ├── p2p-cuda/
│   │   ├── P1_prompt.md
│   │   ├── P2_prompt.md
│   │   └── P3_prompt.md
│   ├── softmax-cuda/
│   │   ├── P1_prompt.md
│   │   ├── P2_prompt.md
│   │   └── P3_prompt.md
│   └── ...
├── metadata/
│   ├── benchmark_taxonomy.csv
│   ├── prompt_inventory_phase2.csv
│   └── prompt_assignment_matrix.csv
├── scripts/
│   ├── check_prompt_requirements.py
│   ├── score_prompt_quality.py
│   └── generate_prompt_inventory.py
└── reports/
    ├── PHASE2_PLAN.md
    ├── PHASE2_ACCEPTANCE.md
    └── PHASE2_SUMMARY.md
```

***

# 4. Prompt 分層設計

Phase 2 將 prompt 分成三個等級。

***

## 4.1 P1：弱約束 prompt

### 目的

模擬一般網頁對話或簡單請求。

### 特徵

```text
1. 只描述任務目標
2. 不強制 baseline
3. 不強制 raw output
4. 不強制 submission limit
5. 不強制 correctness gate
6. 不強制產出 CSV
```

### 用途

用來觀察 AI agent 在低約束下是否會：

```text
1. 忽略 correctness
2. 擅自縮小測資
3. 宣稱未驗證的 speedup
4. 不保存 raw output
5. 產生不可重現結果
```

***

## 4.2 P2：中約束 prompt

### 目的

模擬有經驗工程師給 agent 的基本規格。

### 必須包含

```text
1. benchmark path
2. baseline 必須執行
3. correctness 必須 PASS
4. 最多提交次數
5. raw output 保存
6. result summary
7. 不得刪除 correctness
```

### 用途

用來測試基本工程約束是否足以避免偽加速。

***

## 4.3 P3：強約束 prompt

### 目的

作為論文實驗中的標準化高嚴謹 prompt。

### 必須包含

```text
1. benchmark path
2. result path
3. 硬體與軟體環境
4. Slurm 腳本要求
5. baseline 不算提交次數
6. optimization submissions <= N
7. 每次提交前必須說明假設
8. 每次提交後必須讀取 out/err/result
9. correctness gate
10. raw output preservation
11. CSV result schema
12. agent_summary.md
13. contradiction check
14. invalid result rule
15. result_type classification
16. profiler requirement
17. variance / repeated trials requirement
18. final conclusion label
```

目前 prompt inventory 顯示，多數較完整 prompt 已包含 baseline、correctness、sbatch、submission limit、raw output，但缺少 profiler 與變異統計要求，因此 P3 必須補上這兩項。 [\[prompt_inventory 1 \| Excel\]](https://ntums365-my.sharepoint.com/personal/r14525078_m365_ntu_edu_tw/_layouts/15/Doc.aspx?sourcedoc=%7B7D3DA484-20A1-40A8-AAB4-88ECAA36DEA8%7D&file=prompt_inventory%201.csv&action=default&mobileredirect=true)

***

# 5. Prompt Schema 設計

每份 prompt 必須符合以下 schema。

## 5.1 Metadata 區塊

```markdown
# Prompt Metadata

- benchmark:
- canonical_name:
- benchmark_category:
- prompt_level: P1 / P2 / P3
- target_agent:
- submission_limit:
- baseline_counts_as_submission: false
- required_gpus:
- requires_mpi:
- requires_nccl:
- expected_metric:
- correctness_required: true
```

***

## 5.2 Environment 區塊

```markdown
# Environment

- GPU: NVIDIA Tesla V100-SXM2-32GB
- CUDA arch: sm_70
- Scheduler: Slurm
- Account: ACD115083
- Module:
  module purge
  module load cuda/12.8
```

對 `allreduce-cuda` 與 `pingpong-cuda`，需改用 NVHPC / HPC-X：

```markdown
module purge
module load nvhpc-24.11_hpcx-2.20_cuda-12.6
```

`allreduce-cuda` 的有效 launcher 是透過 `UCX_TLS=self,shm,cuda_copy,cuda_ipc` 避開 broken GDRCopy path；這類 launcher 必須成為 benchmark metadata，而不能隱藏在 prompt 裡。 [\[benchmark_summary \| Excel\]](https://ntums365-my.sharepoint.com/personal/r14525078_m365_ntu_edu_tw/_layouts/15/Doc.aspx?sourcedoc=%7B186B7123-75D1-472D-B5D3-A69B0A4CAA82%7D&file=benchmark_summary.csv&action=default&mobileredirect=true), [\[ntums365-m...epoint.com\]](https://ntums365-my.sharepoint.com/personal/r14525078_m365_ntu_edu_tw/Documents/Microsoft%20Copilot%20Chat%20%E6%AA%94%E6%A1%88/REPORT.md)

***

## 5.3 Baseline 區塊

```markdown
# Baseline Requirements

1. Do not modify source code before baseline.
2. Build and run baseline.
3. Save:
   - result/<benchmark>_<jobid>.out
   - result/<benchmark>_<jobid>.err
   - result/<benchmark>_result_<jobid>.txt
4. Record:
   - job id
   - node
   - CUDA_VISIBLE_DEVICES
   - nvcc version
   - benchmark command
   - correctness
   - metric
5. If baseline fails, classify failure.
6. If baseline has no valid nonzero result, do not compute speedup.
```

***

## 5.4 Optimization Submission 區塊

```markdown
# Optimization Submission Rules

After baseline, at most N optimization sbatch submissions are allowed.

Before each submission, state:
1. modification
2. hypothesis
3. expected improvement
4. validation target

After each submission, read:
1. .out
2. .err
3. result .txt
4. CSV if generated

Then classify:
- PASS / FAIL
- accepted / rejected
- reason
```

***

## 5.5 Correctness Gate

```markdown
# Correctness Gate

A result is valid only if correctness PASS.

Invalid cases:
- correctness FAIL
- only size 0 PASS
- skipped / waived
- output missing
- stderr fatal error
- benchmark changed semantics
```

此規則很重要，因為 `allreduce-cuda` baseline 曾只通過 size 0，但非零 size 失敗，因此 size 0 不可視為完整 correctness PASS。 [\[benchmark_summary \| Excel\]](https://ntums365-my.sharepoint.com/personal/r14525078_m365_ntu_edu_tw/_layouts/15/Doc.aspx?sourcedoc=%7B186B7123-75D1-472D-B5D3-A69B0A4CAA82%7D&file=benchmark_summary.csv&action=default&mobileredirect=true), [\[ntums365-m...epoint.com\]](https://ntums365-my.sharepoint.com/personal/r14525078_m365_ntu_edu_tw/Documents/Microsoft%20Copilot%20Chat%20%E6%AA%94%E6%A1%88/REPORT.md)

***

## 5.6 Result Type 區塊

每次結果必須分類：

```text
KERNEL_OPT
PARAM_TUNE
MEASURE_FIX
BUILD_FIX
ENV_FIX
CORRECT_FIX
TOPOLOGY_MEASURE
NO_EFFECT
REGRESSION
```

例如：

```text
softmax-cuda → KERNEL_OPT
allreduce-cuda → ENV_FIX
p2p-cuda → TOPOLOGY_MEASURE / MEASUREMENT_EQUIVALENT
prefetch-cuda → MEMORY_MIGRATION / PARAM_TUNE
```

既有 benchmark summary 顯示，不同 benchmark 的優化本質不同，若不分類會把 environment repair 誤寫成 kernel speedup。 [\[benchmark_summary \| Excel\]](https://ntums365-my.sharepoint.com/personal/r14525078_m365_ntu_edu_tw/_layouts/15/Doc.aspx?sourcedoc=%7B186B7123-75D1-472D-B5D3-A69B0A4CAA82%7D&file=benchmark_summary.csv&action=default&mobileredirect=true)

***

# 6. Prompt Quality Rubric

Phase 2 需要建立 prompt 評分表。

總分 100 分。

## 6.1 評分項目

```text
A. Task clarity: 10
B. Environment specificity: 10
C. Baseline requirement: 10
D. Correctness gate: 15
E. Raw output requirement: 10
F. Submission limit: 10
G. CSV / machine-readable output: 10
H. Contradiction check: 10
I. Variance / repeated trials: 10
J. Profiler requirement: 5
```

## 6.2 評分標準

### A. Task clarity

```text
0: 任務模糊
5: 有 benchmark path，但目標不明
10: 有 benchmark path、metric、success criteria
```

### D. Correctness gate

```text
0: 未提 correctness
5: 提到 correctness，但未說 FAIL invalid
10: 有 PASS/FAIL 規則
15: 有 invalid result rule + contradiction rule
```

### I. Variance / repeated trials

```text
0: 未要求重複測試
5: 要求 final repeat
10: 要求 trial >= 3 並輸出 stddev/CV
```

目前 prompt inventory 顯示，變異與重複統計是主要缺口之一。 [\[prompt_inventory 1 \| Excel\]](https://ntums365-my.sharepoint.com/personal/r14525078_m365_ntu_edu_tw/_layouts/15/Doc.aspx?sourcedoc=%7B7D3DA484-20A1-40A8-AAB4-88ECAA36DEA8%7D&file=prompt_inventory%201.csv&action=default&mobileredirect=true)

***

# 7. Phase 2 實驗矩陣

## 7.1 Prompt assignment matrix

每個 benchmark 至少產生三份 prompt：

```text
P1_prompt.md
P2_prompt.md
P3_prompt.md
```

矩陣如下：

```csv
benchmark,P1,P2,P3
p2p-cuda,yes,yes,yes
shmembench-cuda,yes,yes,yes
topk-cuda,yes,yes,yes
allreduce-cuda,yes,yes,yes
moe-cuda,yes,yes,yes
pingpong-cuda,yes,yes,yes
simpleMultiDevice-cuda,yes,yes,yes
moe-align,yes,yes,yes
prefetch-cuda,yes,yes,yes
softmax-cuda,yes,yes,yes
```

總數：

```text
10 benchmarks × 3 prompt levels = 30 prompts
```

***

# 8. 操作指引

## Step 1：建立 Phase 2 目錄

```bash
cd /home/r14525078/HeCBench
mkdir -p phase2/{prompt_templates,prompts,metadata,scripts,reports}
```

***

## Step 2：建立 prompt template

```bash
touch phase2/prompt_templates/P1_weak_prompt_template.md
touch phase2/prompt_templates/P2_medium_prompt_template.md
touch phase2/prompt_templates/P3_strong_prompt_template.md
```

***

## Step 3：建立 benchmark prompt 目錄

```bash
cd /home/r14525078/HeCBench/phase2/prompts

for b in \
  p2p-cuda \
  shmembench-cuda \
  topk-cuda \
  allreduce-cuda \
  moe-cuda \
  pingpong-cuda \
  simpleMultiDevice-cuda \
  moe-align \
  prefetch-cuda \
  softmax-cuda
do
  mkdir -p "$b"
  touch "$b/P1_prompt.md"
  touch "$b/P2_prompt.md"
  touch "$b/P3_prompt.md"
done
```

***

## Step 4：建立 prompt inventory schema

```bash
cat > /home/r14525078/HeCBench/phase2/metadata/prompt_inventory_phase2.csv <<'EOF'
benchmark,prompt_level,prompt_file,lines,chars,mentions_correctness,mentions_baseline,mentions_sbatch,mentions_submission_limit,mentions_raw_output,mentions_csv,mentions_contradiction_check,mentions_variance,mentions_profiler,prompt_score,main_strength,main_gap
EOF
```

***

## Step 5：建立 prompt assignment matrix

```bash
cat > /home/r14525078/HeCBench/phase2/metadata/prompt_assignment_matrix.csv <<'EOF'
benchmark,category,P1,P2,P3,notes
p2p-cuda,multi_gpu_interconnect,yes,yes,yes,topology-aware measurement
shmembench-cuda,shared_memory,yes,yes,yes,requires profiler bank conflict metrics
topk-cuda,ai_topk,yes,yes,yes,workspace reuse and radix selection
allreduce-cuda,mpi_collective,yes,yes,yes,requires tuned UCX metadata
moe-cuda,moe_inference,yes,yes,yes,topk-dependent optimization
pingpong-cuda,mpi_nccl_pingpong,yes,yes,yes,MPI/NCCL comparison
simpleMultiDevice-cuda,multi_gpu_scaling,yes,yes,yes,H2D-copy-limited
moe-align,moe_alignment,yes,yes,yes,correctness field needs explicit handling
prefetch-cuda,unified_memory,yes,yes,yes,managed memory migration
softmax-cuda,softmax_kernel,yes,yes,yes,large kernel-level speedup case
EOF
```

***

# 9. Prompt Template 內容

## 9.1 P1 weak prompt template

```markdown
# P1 Weak Prompt

Please optimize the CUDA benchmark at:

<benchmark_path>

Goal:
- Improve performance.
- Keep correctness.

Please inspect the code, make changes, run the benchmark, and report the result.
```

***

## 9.2 P2 medium prompt template

```markdown
# P2 Medium Prompt

You are a CUDA performance engineer.

Benchmark path:
<benchmark_path>

Result path:
<result_path>

Goal:
- Establish baseline.
- Improve performance.
- Keep correctness PASS.

Rules:
1. Run baseline before modifying source.
2. Save raw output.
3. Do not remove correctness checks.
4. At most <N> optimization submissions after baseline.
5. Report job id, correctness, metric, and speedup.

Environment:
- GPU: NVIDIA Tesla V100-SXM2-32GB
- CUDA arch: sm_70
- Scheduler: Slurm
- Account: ACD115083

Final output:
- agent_summary.md
```

***

## 9.3 P3 strong prompt template

```markdown
# P3 Strong Prompt

You are a CUDA performance engineer conducting a reproducible optimization experiment.

## Benchmark Metadata

- benchmark:
- benchmark_path:
- result_path:
- category:
- required_gpus:
- requires_mpi:
- requires_nccl:
- expected_metric:
- correctness_required: true

## Environment

- GPU: NVIDIA Tesla V100-SXM2-32GB
- CUDA arch: sm_70
- Scheduler: Slurm
- Account: ACD115083

## Hard Rules

1. Baseline does not count toward submission limit.
2. After baseline, at most <N> optimization sbatch submissions.
3. Before each submission, state:
   - modification
   - hypothesis
   - expected improvement
   - validation target
4. After each submission, read:
   - .out
   - .err
   - result .txt
   - CSV if generated
5. Do not delete correctness checks.
6. Do not shrink input to fake speedup.
7. If correctness FAIL, metric is invalid.
8. If only size 0 PASS, result is invalid.
9. If improvement < 1%, mark as measurement-equivalent.
10. Classify result type:
    - BUILD_FIX
    - ENV_FIX
    - MEASURE_FIX
    - CORRECT_FIX
    - KERNEL_OPT
    - PARAM_TUNE
    - TOPOLOGY_MEASURE
    - NO_EFFECT
    - REGRESSION

## Baseline

Run baseline without source modification.

Save:
- result/<benchmark>_<jobid>.out
- result/<benchmark>_<jobid>.err
- result/<benchmark>_result_<jobid>.txt

Record:
- job_id
- node
- CUDA_VISIBLE_DEVICES
- nvcc version
- command
- correctness
- metric

## Optimization Plan

Submission 1:
- fix build / run / output reproducibility

Submission 2:
- improve timing and RESULT CSV

Submission 3:
- small parameter sweep

Submission 4:
- kernel or configuration optimization

Submission 5:
- final confirmation with repeated trials

## Statistics

Final result must include:
- trials >= 3 if feasible
- mean
- min
- max
- stddev
- CV

## Profiler Requirement

If this is a CUDA kernel benchmark, collect or plan to collect:
- dram throughput
- shared throughput
- achieved occupancy
- register usage
- shared memory usage
- bank conflict if relevant

If profiler cannot be run due to submission limit, mark it as a limitation.

## Final Report

Create:
<result_path>/agent_summary.md

Must include:
1. environment
2. baseline
3. submission history
4. correctness summary
5. performance table
6. result type classification
7. contradiction check
8. limitations
9. final conclusion

Final conclusion must be one of:
- SUCCESS
- PARTIAL
- ENVIRONMENT ISSUE
- CODE ISSUE
- INCONCLUSIVE
- MEASUREMENT-EQUIVALENT
```

***

# 10. Phase 2 驗收標準

Phase 2 完成必須滿足：

````text
1. 10 個 benchmark 全部有 P1/P2/P3 prompt
2. 每份 prompt 都通過 schema check
3. 每份 prompt 都有 inventory entry
4. P3 prompt score >= 85/100
5.# Phase 2：Prompt 規格設計與人機協作實驗計畫書

本階段目標是把 Phase 1 已分類的 benchmark 資料集，轉換成一套**可比較、可重現、可審核**的 prompt 實驗系統。  
Phase 2 不直接追求效能最佳化，而是建立不同 prompt 設計方式對 AI agent 行為、結果品質、效能提升、正確性與可重現性的影響模型。

Phase 1 已整理的資料集包含：`p2p-cuda`、`shmembench-cuda`、`topk-cuda`、`allreduce-cuda`、`moe-cuda`、`pingpong-cuda`、`simpleMultiDevice-cuda`、`moe-align`、`prefetch-cuda`、`softmax-cuda`。其中 `softmax-cuda` 代表實質 kernel optimization，`allreduce-cuda` 代表 environment / launcher repair，`p2p-cuda` 代表 topology-aware measurement，`topk-cuda` 與 `moe-cuda` 代表 AI inference primitive optimization。這些分類可支撐 Phase 2 的 prompt 分層實驗設計。

---

# 1. Phase 2 研究目標

## 1.1 核心問題

Phase 2 要回答：

```text
RQ2-1: prompt.md 的約束強度是否會影響 AI agent 的 correctness、可重現性與效能結果？
RQ2-2: 強約束 prompt 是否能降低偽加速、錯誤報告與矛盾結論？
RQ2-3: 哪些 prompt 元素最能提升人機協作效率？
RQ2-4: prompt 應如何標準化，才能支援後續 Phase 3 的 controlled optimization experiment？
````

目前已有 prompt inventory 顯示，多數完整 prompt 已包含 correctness、baseline、sbatch、submission limit、raw output 等欄位；較短 prompt 則缺少 profiler 指標與變異統計要求。這正是 Phase 2 要補強與實驗化的重點。 [\[prompt_inventory 1 \| Excel\]](https://ntums365-my.sharepoint.com/personal/r14525078_m365_ntu_edu_tw/_layouts/15/Doc.aspx?sourcedoc=%7B7D3DA484-20A1-40A8-AAB4-88ECAA36DEA8%7D&file=prompt_inventory%201.csv&action=default&mobileredirect=true)

***

# 2. Phase 2 產出項目

Phase 2 完成後，應產出以下檔案與資料夾。

```text
metadata/
  benchmark_taxonomy.csv
  prompt_taxonomy.csv
  prompt_quality_rubric.md

prompts/
  P1_weak/
    <benchmark>/prompt.md
  P2_medium/
    <benchmark>/prompt.md
  P3_strong/
    <benchmark>/prompt.md

schemas/
  result_schema.md
  prompt_schema.md
  agent_summary_schema.md
  contradiction_rules.md

tools/
  prompt_inventory.py
  validate_prompt.py
  contradiction_checker.py

phase2/
  PHASE2_PLAN.md
  PHASE2_EXECUTION_GUIDE.md
  PHASE2_ACCEPTANCE_REPORT.md
```

***

# 3. Phase 2 實驗對象

## 3.1 Benchmark 分組

Phase 2 不需要所有 benchmark 都立即進行實測，但每個 benchmark 都需要產生 prompt 規格。

## A. AI primitive 類

```text
softmax-cuda
topk-cuda
moe-cuda
moe-align
```

用途：

```text
測試 agent 是否能做 kernel-level optimization、workspace reuse、softmax/top-k fusion、routing primitive 優化。
```

其中 `softmax-cuda` 已有顯著加速案例，slice=784 speedup 約 59.593x；`topk-cuda` 的 workspace reuse 與 block size tuning 也已有有效結果。 [\[benchmark_summary \| Excel\]](https://ntums365-my.sharepoint.com/personal/r14525078_m365_ntu_edu_tw/_layouts/15/Doc.aspx?sourcedoc=%7B186B7123-75D1-472D-B5D3-A69B0A4CAA82%7D&file=benchmark_summary.csv&action=default&mobileredirect=true)

## B. Memory system 類

```text
prefetch-cuda
shmembench-cuda
p2p-cuda
```

用途：

```text
測試 memory-bound 題目中 AI 的實際優化上限、measurement repair、topology-aware analysis。
```

`p2p-cuda` 的 best gain 約 1.004x，應視為 measurement-equivalent；`shmembench-cuda` 與 `prefetch-cuda` 則適合觀察 memory-bound 任務是否主要受硬體限制。 [\[benchmark_summary \| Excel\]](https://ntums365-my.sharepoint.com/personal/r14525078_m365_ntu_edu_tw/_layouts/15/Doc.aspx?sourcedoc=%7B186B7123-75D1-472D-B5D3-A69B0A4CAA82%7D&file=benchmark_summary.csv&action=default&mobileredirect=true)

## C. Communication / multi-GPU 類

```text
allreduce-cuda
pingpong-cuda
simpleMultiDevice-cuda
p2p-cuda
```

用途：

```text
測試 agent 是否能處理 MPI / NCCL / UCX / Slurm / launcher / GPU allocation 等非純程式碼問題。
```

`allreduce-cuda` 是典型 environment / launcher repair 案例；`pingpong-cuda` 顯示 tuned CUDA-aware MPI 在 two-rank ping-pong 下比 NCCL 更快。 [\[benchmark_summary \| Excel\]](https://ntums365-my.sharepoint.com/personal/r14525078_m365_ntu_edu_tw/_layouts/15/Doc.aspx?sourcedoc=%7B186B7123-75D1-472D-B5D3-A69B0A4CAA82%7D&file=benchmark_summary.csv&action=default&mobileredirect=true), [\[ntums365-m...epoint.com\]](https://ntums365-my.sharepoint.com/personal/r14525078_m365_ntu_edu_tw/Documents/Microsoft%20Copilot%20Chat%20%E6%AA%94%E6%A1%88/REPORT.md)

***

# 4. Prompt 分層設計

Phase 2 的核心是建立三種 prompt 強度。

***

## 4.1 P1：弱約束 Prompt

### 目的

測試一般使用者在網頁對話或簡短 CLI prompt 中會得到什麼結果。

### 特徵

```text
1. 只說明 benchmark path
2. 只要求提升效能
3. 簡單要求保持 correctness
4. 不明確限制 submission 次數
5. 不強制 raw output
6. 不強制 CSV
7. 不強制矛盾檢查
```

### 預期風險

```text
1. agent 可能跳過 baseline
2. agent 可能只報告成功結果
3. agent 可能把部分 PASS 說成全部 PASS
4. agent 可能使用 estimated baseline
5. agent 可能沒有保留 stderr / raw log
```

既有報告已指出，若缺少 baseline 與矛盾檢查，agent 可能出現同時宣稱失敗與全通過、或使用 estimated baseline 做過度結論的問題。 [\[ntums365-m...epoint.com\]](https://ntums365-my.sharepoint.com/personal/r14525078_m365_ntu_edu_tw/Documents/Microsoft%20Copilot%20Chat%20%E6%AA%94%E6%A1%88/REPORT.md)

### P1 Prompt 模板

```markdown
# Task

You are a CUDA performance engineer. Optimize the benchmark below.

Benchmark path:
`/home/r14525078/HeCBench/src/<benchmark>`

Goal:
Improve performance while preserving correctness.

Please build, run, optimize, and summarize the result.
```

***

## 4.2 P2：中約束 Prompt

### 目的

測試「基本研究規格」是否足以防止常見偽加速。

### 必須包含

```text
1. baseline 必須先跑
2. correctness 不得刪除
3. raw output 必須保留
4. sbatch 必須使用
5. submission limit
6. 每次提交需記錄 job id
7. 最終需產出 summary
```

### P2 Prompt 模板

```markdown
# Task

You are a CUDA performance engineer. Optimize the benchmark below with reproducible validation.

Benchmark path:
`/home/r14525078/HeCBench/src/<benchmark>`

Result path:
`/home/r14525078/HeCBench/src/<benchmark>/result`

Environment:
- GPU: NVIDIA Tesla V100-SXM2-32GB
- CUDA arch: sm_70
- Scheduler: Slurm
- Account: ACD115083

Rules:
1. Baseline does not count toward the optimization submission limit.
2. After baseline, you may submit at most 5 sbatch jobs.
3. Do not delete or weaken correctness checks.
4. Do not treat FAIL as PASS.
5. Do not reduce input size to fake speedup.
6. Save raw output:
   - result/<benchmark>_<jobid>.out
   - result/<benchmark>_<jobid>.err
   - result/<benchmark>_result_<jobid>.txt
7. Final summary must include:
   - baseline result
   - each job id
   - modifications
   - correctness
   - performance
   - final conclusion
```

***

## 4.3 P3：強約束 Prompt

### 目的

建立論文級 prompt 規格，用於可審核 AI-assisted optimization。

### 必須包含

```text
1. baseline 真實執行
2. correctness gate
3. submission limit
4. raw output preservation
5. CSV machine-readable output
6. contradiction checker
7. profiler requirement
8. result type classification
9. invalid result policy
10. final acceptance criteria
```

目前完整 prompt 通常包含 baseline、correctness、sbatch、submission limit 與 raw output，但普遍缺少 profiler 指標與統計變異要求。因此 P3 必須補上這兩項。 [\[prompt_inventory 1 \| Excel\]](https://ntums365-my.sharepoint.com/personal/r14525078_m365_ntu_edu_tw/_layouts/15/Doc.aspx?sourcedoc=%7B7D3DA484-20A1-40A8-AAB4-88ECAA36DEA8%7D&file=prompt_inventory%201.csv&action=default&mobileredirect=true)

### P3 Prompt 必備章節

```text
1. Role
2. Benchmark path
3. Environment
4. Known benchmark behavior
5. Hard constraints
6. Stage 0 baseline
7. Optimization submissions
8. Correctness rules
9. Measurement rules
10. CSV output schema
11. Contradiction rules
12. Profiler requirements
13. Final report schema
14. Acceptance criteria
```

***

# 5. Phase 2 實驗矩陣

## 5.1 全量 Prompt 產生矩陣

10 個 benchmark × 3 種 prompt 強度：

```text
10 × 3 = 30 prompts
```

目錄：

```text
prompts/P1_weak/<benchmark>/prompt.md
prompts/P2_medium/<benchmark>/prompt.md
prompts/P3_strong/<benchmark>/prompt.md
```

***

## 5.2 Phase 2 實測子集

Phase 2 不一定要立刻實測 30 組。建議先挑 3 題做 pilot。

```text
softmax-cuda
prefetch-cuda
allreduce-cuda
```

理由：

```text
softmax-cuda    代表實質 kernel optimization
prefetch-cuda   代表 memory-bound / limited improvement
allreduce-cuda  代表 environment / launcher repair
```

這三題剛好覆蓋三種不同 AI 協作情境。`softmax-cuda` 有極高 speedup，`prefetch-cuda` 有有限改善，`allreduce-cuda` 則是環境修復案例。 [\[benchmark_summary \| Excel\]](https://ntums365-my.sharepoint.com/personal/r14525078_m365_ntu_edu_tw/_layouts/15/Doc.aspx?sourcedoc=%7B186B7123-75D1-472D-B5D3-A69B0A4CAA82%7D&file=benchmark_summary.csv&action=default&mobileredirect=true)

***

# 6. Prompt 品質評分規則

建立：

```text
metadata/prompt_quality_rubric.md
```

每份 prompt 以 100 分評分。

## 6.1 評分表

```text
A. Baseline clarity: 10
B. Correctness protection: 15
C. Submission limit: 10
D. Raw output requirement: 10
E. Environment reproducibility: 10
F. Machine-readable result: 10
G. Contradiction checking: 10
H. Profiler / variance requirement: 10
I. Failure classification: 10
J. Final report schema: 5
```

## 6.2 分級

```text
0–39: weak prompt
40–69: medium prompt
70–100: strong prompt
```

***

# 7. 統一 Prompt Schema

建立：

```text
schemas/prompt_schema.md
```

內容如下。

```markdown
# Prompt Schema

## 1. Metadata
- benchmark
- prompt_level
- author
- date
- expected_gpu_count
- expected_runtime

## 2. Environment
- GPU
- CUDA version
- Slurm account
- required modules

## 3. Benchmark Knowledge
- expected input
- expected output
- correctness rule
- metric

## 4. Baseline Rule
- baseline command
- baseline output path
- valid baseline definition

## 5. Submission Rule
- max submissions
- baseline excluded or included
- required pre-submission explanation

## 6. Correctness Rule
- what counts as PASS
- what invalidates result
- tolerance rule

## 7. Measurement Rule
- repeat count
- warmup
- timing method
- unit

## 8. Raw Output Rule
- .out
- .err
- result.txt
- CSV

## 9. Invalid Result Rule
- FAIL
- timeout
- partial pass
- skipped
- no metric

## 10. Final Report Rule
- summary path
- required sections
```

***

# 8. 統一 Result Schema

建立：

```text
schemas/result_schema.md
```

CSV 欄位：

```csv
benchmark,prompt_level,agent,job_id,node,gpu_model,cuda_version,case,variant,metric_name,metric_value,unit,correctness,status,result_type,notes
```

## result\_type 定義

```text
KERNEL_OPT
ENV_FIX
MEASURE_FIX
BUILD_FIX
CORRECT_FIX
PARAM_TUNE
NO_EFFECT
REGRESSION
MEASUREMENT_EQUIVALENT
INCONCLUSIVE
```

***

# 9. 矛盾檢查規則

建立：

```text
schemas/contradiction_rules.md
```

## 9.1 必查規則

```text
Rule 1:
If any official case has correctness=FAIL,
summary must not say "all tests PASS".

Rule 2:
If baseline is invalid,
speedup must be marked as n/a.

Rule 3:
If only size 0 passed,
correctness must not be full PASS.

Rule 4:
If improvement < 1%,
result_type must be MEASUREMENT_EQUIVALENT.

Rule 5:
If result_type is ENV_FIX,
summary must not call it kernel optimization.

Rule 6:
If stderr has fatal error,
status must not be SUCCESS.

Rule 7:
If benchmark skipped,
status must not be SUCCESS.
```

這些規則直接對應既有報告指出的問題：AI 可能在報告中同時宣稱失敗與全通過，或把環境修復誤判成演算法優化。 [\[ntums365-m...epoint.com\]](https://ntums365-my.sharepoint.com/personal/r14525078_m365_ntu_edu_tw/Documents/Microsoft%20Copilot%20Chat%20%E6%AA%94%E6%A1%88/REPORT.md)

***

# 10. Phase 2 操作流程

## Step 1：建立目錄

```bash
cd /home/r14525078/HeCBench

mkdir -p metadata
mkdir -p schemas
mkdir -p tools
mkdir -p phase2

mkdir -p prompts/P1_weak
mkdir -p prompts/P2_medium
mkdir -p prompts/P3_strong
```

***

## Step 2：建立 benchmark 清單

```bash
cat > metadata/phase2_benchmarks.txt <<'EOF'
p2p-cuda
shmembench-cuda
topk-cuda
allreduce-cuda
moe-cuda
pingpong-cuda
simpleMultiDevice-cuda
moe-align
prefetch-cuda
softmax-cuda
EOF
```

***

## Step 3：建立 prompt 目錄

```bash
while read -r b; do
  mkdir -p "prompts/P1_weak/${b}"
  mkdir -p "prompts/P2_medium/${b}"
  mkdir -p "prompts/P3_strong/${b}"
done < metadata/phase2_benchmarks.txt
```

***

## Step 4：建立 prompt\_quality\_rubric.md

```bash
cat > metadata/prompt_quality_rubric.md <<'EOF'
# Prompt Quality Rubric

Total: 100 points

## A. Baseline clarity: 10
Requires real measured baseline, not estimated baseline.

## B. Correctness protection: 15
Explicitly prohibits deleting or weakening correctness checks.

## C. Submission limit: 10
Defines maximum optimization submissions.

## D. Raw output requirement: 10
Requires .out, .err, and result txt.

## E. Environment reproducibility: 10
Requires module list, nvcc version, GPU info, Slurm settings.

## F. Machine-readable result: 10
Requires RESULT lines or CSV.

## G. Contradiction checking: 10
Requires final report consistency checks.

## H. Profiler / variance requirement: 10
Requires repeated trials, stddev/CV, or profiler metrics.

## I. Failure classification: 10
Requires categorizing build/runtime/correctness/environment failure.

## J. Final report schema: 5
Requires structured agent_summary.md.
EOF
```

***

## Step 5：建立 P1 / P2 / P3 prompt 模板

### P1 模板

```bash
cat > prompts/P1_weak/TEMPLATE.md <<'EOF'
# Task

You are a CUDA performance engineer.

Optimize the benchmark below:

Benchmark path:
`/home/r14525078/HeCBench/src/<BENCHMARK>`

Goal:
Improve performance while preserving correctness.

Please build, run, optimize, and summarize the result.
EOF
```

***

### P2 模板

```bash
cat > prompts/P2_medium/TEMPLATE.md <<'EOF'
# Task

You are a CUDA performance engineer. Optimize this benchmark with reproducible validation.

Benchmark path:
`/home/r14525078/HeCBench/src/<BENCHMARK>`

Result path:
`/home/r14525078/HeCBench/src/<BENCHMARK>/result`

Environment:
- GPU: NVIDIA Tesla V100-SXM2-32GB
- CUDA arch: sm_70
- Scheduler: Slurm
- Account: ACD115083

Rules:
1. Baseline does not count toward the optimization submission limit.
2. After baseline, you may submit at most 5 sbatch jobs.
3. Do not delete or weaken correctness checks.
4. Do not treat FAIL as PASS.
5. Do not reduce input size to fake speedup.
6. Save raw output:
   - result/<BENCHMARK>_<jobid>.out
   - result/<BENCHMARK>_<jobid>.err
   - result/<BENCHMARK>_result_<jobid>.txt
7. Final summary must include:
   - baseline result
   - job IDs
   - modifications
   - correctness
   - performance
   - final conclusion
EOF
```

***

### P3 模板

```bash
cat > prompts/P3_strong/TEMPLATE.md <<'EOF'
# Task

You are a CUDA performance engineer conducting a reproducible, correctness-gated optimization experiment.

Benchmark path:
`/home/r14525078/HeCBench/src/<BENCHMARK>`

Result path:
`/home/r14525078/HeCBench/src/<BENCHMARK>/result`

Environment:
- GPU: NVIDIA Tesla V100-SXM2-32GB
- CUDA arch: sm_70
- Scheduler: Slurm
- Account: ACD115083

Hard Rules:
1. Baseline must be measured, not estimated.
2. Baseline does not count toward the optimization submission limit.
3. After baseline, at most 5 sbatch submissions are allowed.
4. Before each submission, explain:
   - modification
   - hypothesis
   - expected improvement
   - validation target
5. After each submission, read:
   - .out
   - .err
   - result txt
6. Do not delete or weaken correctness checks.
7. Do not treat FAIL, skipped, timeout, or partial PASS as SUCCESS.
8. Do not reduce official input size to fake speedup.
9. Preserve raw output.
10. Produce machine-readable RESULT rows or CSV.
11. Classify each result as:
    - KERNEL_OPT
    - ENV_FIX
    - MEASURE_FIX
    - BUILD_FIX
    - CORRECT_FIX
    - PARAM_TUNE
    - NO_EFFECT
    - REGRESSION
    - MEASUREMENT_EQUIVALENT
    - INCONCLUSIVE
12. If improvement < 1%, mark as MEASUREMENT_EQUIVALENT.
13. If baseline is invalid, speedup must be n/a.
14. If any case FAILS, summary must not say all tests PASS.
15. Generate:
    `result/agent_summary.md`

Final Report Required Sections:
1. Environment
2. Benchmark characterization
3. Baseline
4. Submission history
5. Correctness summary
6. Performance table
7. Optimization analysis
8. Limitations
9. Final conclusion
EOF
```

***

# 11. 產生 30 份初始 prompt

```bash
while read -r b; do
  sed "s/<BENCHMARK>/${b}/g" prompts/P1_weak/TEMPLATE.md > "prompts/P1_weak/${b}/prompt.md"
  sed "s/<BENCHMARK>/${b}/g" prompts/P2_medium/TEMPLATE.md > "prompts/P2_medium/${b}/prompt.md"
  sed "s/<BENCHMARK>/${b}/g" prompts/P3_strong/TEMPLATE.md > "prompts/P3_strong/${b}/prompt.md"
done < metadata/phase2_benchmarks.txt
```

***

# 12. Prompt Inventory 工具規格

建立：

```text
tools/prompt_inventory.py
```

功能：

```text
1. 掃描 prompts/**/*.md
2. 統計 lines / chars
3. 檢查是否包含：
   correctness
   baseline
   sbatch
   submission limit
   raw output
   CSV
   contradiction
   profiler
   variance
4. 輸出 metadata/prompt_inventory_phase2.csv
```

輸出欄位：

```csv
prompt_file,level,benchmark,lines,chars,mentions_correctness,mentions_baseline,mentions_sbatch,mentions_submission_limit,mentions_raw_output,mentions_csv,mentions_contradiction,mentions_profiler,mentions_variance,score
```

***

# 13. Phase 2 Pilot 實驗設計

## 13.1 Pilot benchmark

```text
softmax-cuda
prefetch-cuda
allreduce-cuda
```

## 13.2 Pilot matrix

```text
3 benchmarks × 3 prompt levels = 9 experiments
```

## 13.3 每組實驗限制

```text
baseline + 5 optimization submissions
```

## 13.4 觀察項目

```text
1. build 是否成功
2. correctness 是否成功
3. speedup
4. 無效提交比例
5. 是否產出 CSV
6. 是否產出 agent_summary.md
7. 是否有 contradiction
8. result_type 分布
```

***

# 14. Phase 2 Acceptance Criteria

Phase 2 完成條件：

```text
1. 10 個 benchmark 均有 P1 / P2 / P3 prompt
2. prompt_quality_rubric.md 完成
3. prompt_schema.md 完成
4. result_schema.md 完成
5. contradiction_rules.md 完成
6. prompt_inventory.py 完成
7. 至少 9 組 pilot experiment 規劃完成
8. 每份 prompt 可追溯 benchmark、level、rules
9. 每份 prompt 都明確標示 baseline / correctness / raw output policy
```

***

# 15. Phase 2 最終報告格式

建立：

```text
phase2/PHASE2_ACCEPTANCE_REPORT.md
```

內容：

```markdown
# Phase 2 Acceptance Report

## 1. Objective
## 2. Benchmark Set
## 3. Prompt Levels
## 4. Generated Files
## 5. Prompt Inventory Summary
## 6. Pilot Experiment Plan
## 7. Known Risks
## 8. Acceptance Checklist
## 9. Next Phase Recommendation
```

***

# 16. 下一步 Phase 3 預告

Phase 3 應執行：

```text
P1 / P2 / P3 prompt 實測比較
```

推薦先跑：

```text
softmax-cuda
prefetch-cuda
allreduce-cuda
```

原因：

```text
softmax-cuda:
  觀察強 prompt 是否仍能導出巨大 kernel optimization

prefetch-cuda:
  觀察強 prompt 是否能避免過度宣稱 memory-bound 題目

allreduce-cuda:
  觀察強 prompt 是否能正確分類 environment fix
```

***

# 17. 本階段核心結論

Phase 2 的重點不是讓 AI 更快，而是把 AI 輔助程式優化變成一套可驗證流程。

具體來說：

```text
P1 測試一般使用者風險
P2 測試基本工程約束是否足夠
P3 測試論文級 reproducibility prompt 是否能最大化可信度
```

你目前已有的結果顯示：強約束 prompt 能避免偽加速、要求 raw output、保留 correctness，且較適合用於科學實驗；但仍需補上 profiler 與統計變異要求。 [\[prompt_inventory 1 \| Excel\]](https://ntums365-my.sharepoint.com/personal/r14525078_m365_ntu_edu_tw/_layouts/15/Doc.aspx?sourcedoc=%7B7D3DA484-20A1-40A8-AAB4-88ECAA36DEA8%7D&file=prompt_inventory%201.csv&action=default&mobileredirect=true), [\[ntums365-m...epoint.com\]](https://ntums365-my.sharepoint.com/personal/r14525078_m365_ntu_edu_tw/Documents/Microsoft%20Copilot%20Chat%20%E6%AA%94%E6%A1%88/REPORT.md)
