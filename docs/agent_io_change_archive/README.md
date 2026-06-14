# Agent I/O and Change Archive

本資料夾整理「input prompt / system prompt 對應 output，以及對 agent 做的更改」相關文件。所有檔案皆由原專案複製而來，並盡量保留原始相對路徑，方便回溯來源。

## 建議主入口

請優先閱讀：

- `benchmark_view/README.md`

`benchmark_view/` 依照論文 `/home/a/paper.tex` 內紀錄的 29 個 benchmark 建立一對一資料夾。每個 benchmark 皆包含：

- `input_prompt.md`
- `output.md`
- `agent_changes.md`

若原始資料夾中沒有 `prompt.md`，該 benchmark 的 `input_prompt.md` 會填入指定的預設 prompt：

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

## 目錄說明

### 00_system_and_rules

放置專案層級的 system/rule/protocol 類文件，例如：

- `docs/rule.md`
- `docs/prompt.md`
- `phase2/prompt_schema.md`
- `phase2/prompt_quality_rubric.md`
- `phase2/prompt_templates/`
- `phase3/prompts/`
- `phase3/metadata/result_schema.csv`
- `phase3/metadata/official_sweeps.yaml`

注意：repo 內未保存完整聊天或 API 層級的原始 system message；此處保存的是專案實驗中可追溯的 system-like rules、prompt templates 與 protocol 文件。

### 01_input_prompts

放置 agent 實際接收或由實驗流程產生的 input prompt，包括：

- BASIC 探索階段的 `prompt.md`
- Phase 2 的 P1/P2/P3 prompt
- Phase 2 prompt templates
- Phase 3 Mode A / Mode B prompt
- Mode C submission / analysis prompt
- 外部基本測試 `external_basic_tests/prompts.md`

此資料夾用於回答「agent 收到什麼輸入」。

### 02_corresponding_outputs

放置 prompt 對應的 output 與結果紀錄，包括：

- `agent_summary.md`
- `summary.md`
- Slurm `.out`
- `results.csv`
- `contradiction_check.csv`
- profiler summary / report
- evaluation summary
- 外部基本測試 `external_basic_tests/data.md` 與 `output_mapping.md`

此資料夾用於回答「對應 prompt 產生了什麼輸出與實驗結果」。

### 03_agent_changes

放置 agent 修改行為的證據，包括：

- `patch_summary.md`
- `decision_log.md`
- `human_intervention_log.md`
- `*.bak_agent`
- Phase 3 final / round source snapshots
- 外部基本測試 `external_basic_tests/agent_changes.md`

此資料夾用於回答「agent 對程式、執行腳本或實驗流程做了哪些更改」。其中 `.bak_agent` 通常代表 agent 修改前保存的備份，可與同一路徑的修改後檔案交叉比對。
外部基本測試未提供 source diff 或 patch summary，因此該部分以 `N/A` 標記缺失欄位。

### 04_schemas_and_summaries

放置跨實驗彙整與查核用表格，包括：

- `phase2_level_summary.csv`
- `prompt_assignment_matrix.csv`
- `prompt_inventory_phase2.csv`
- `benchmark_summary_used.csv`
- `invalid_results.csv`
- `contradiction_check.csv`
- `SUMMARY_TABLES.md`
- `CHINESE_REPORT.md`
- `AI_AGENT_PARALLEL_OPTIMIZATION_REPORT_ZH.md`
- `rest.md`
- `phase2/rest_large_optimization/README.md`
- `phase2/rest_large_optimization/rest_phase2_summary.csv`
- `phase2/rest_large_optimization/REST_PHASE2_ORGANIZED_SUMMARY_ZH.md`
- `paper.tex`
- `paper_zh.md`

此資料夾用於快速理解整體實驗設計、結果分類與資料有效性。

## 建議閱讀順序

1. 先讀 `benchmark_view/README.md`，逐一對照 29 個 benchmark 的 prompt、output、change。
2. 讀 `04_schemas_and_summaries/paper.tex` 與 `04_schemas_and_summaries/paper_zh.md` 理解壓縮論文版。
3. 讀 `04_schemas_and_summaries/docs/AI_AGENT_PARALLEL_OPTIMIZATION_REPORT_ZH.md` 理解完整中文研究報告。
4. 再讀 `00_system_and_rules/phase2/prompt_templates/` 與 `00_system_and_rules/phase3/prompts/` 理解規則與 prompt 約束。
5. 若需原始檔案脈絡，再到 `01_input_prompts/`、`02_corresponding_outputs/`、`03_agent_changes/` 查原始分散資料。

## 檔案數量

建立時的分類檔案數：

- `00_system_and_rules`: 20
- `01_input_prompts`: 102
- `02_corresponding_outputs`: 499
- `03_agent_changes`: 72
- `04_schemas_and_summaries`: 15
- `benchmark_view`: 88
- `README.md`: 1

合計：797 個檔案。
