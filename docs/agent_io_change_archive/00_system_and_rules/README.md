# HeCBench CUDA Optimization (AI-Assisted Code Optimization & Prompt Auditability Evaluation)

本專案基於 [ORNL HeCBench](https://github.com/ORNL/HeCBench.git)，旨在研究如何利用 AI 輔助進行 GPU CUDA 核函數與 MoE (Mixture-of-Experts) 算子的效能優化，並系統性評估不同 Prompt 約束層級（P1 弱約束、P2 中約束、P3 強約束）對 AI Agent 優化行為可重現性與可審核性的影響。

---

## 專案目錄結構 (Folder Structure)

- `BASIC/`：早期 AI Agent 優化嘗試的程式碼與結果。
  - 包含個別測試子資料夾 (`softmax-cuda`, `moe-cuda` 等) 及彙整工具 `benchmark_analysis_report/`。
- `phase2/`：正規化 Prompt 約束層級實驗專案包。
  - `prompt_templates/`：通用 P1/P2/P3 模板。
  - `prompts/`：30 個針對各 Benchmark 的約束引導 Prompt。
  - `p1/`, `p2/`, `p3/`：不同 Prompt 強度下運行的實驗原始碼、日誌與 CSV 統計。
  - `reports/`：運作計畫、驗證指標與分析報告。
- [evaluation_summary/](file:///home/a/PP/evaluation_summary/)：**最終評估與學術審核彙整目錄**（本專案之核心產出）。
  - [CHINESE_REPORT.md](file:///home/a/PP/evaluation_summary/CHINESE_REPORT.md)：中文總評估報告，探討 Prompt 強度如何防範 AI 偽加速，並回答 RQ1~RQ7。
  - [SUMMARY_TABLES.md](file:///home/a/PP/evaluation_summary/SUMMARY_TABLES.md)與 `data/`：統計圖表與數據，包含 [invalid_results.csv](file:///home/a/PP/evaluation_summary/data/invalid_results.csv) 與 [contradiction_check.csv](file:///home/a/PP/evaluation_summary/data/contradiction_check.csv)。
  - `benchmarks/`：10 個標準化 Benchmark 的個別約束分析檔。

---

## 快速開始與生成彙整 (Quick Start)

專案提供自動化矛盾檢驗與報告生成腳本：

```bash
python3 phase2/scripts/generate_evaluation_summary.py
```

執行後會自動比對原始實驗數據、判定 invalid 案例與邏輯衝突，並將更新後的報告寫入 [evaluation_summary/](file:///home/a/PP/evaluation_summary/)。

---

## 核心結論 (Core Takeaways)

- **Prompt 約束直接影響學術嚴謹度**：P1（弱約束）因缺少 baseline 與重試，容易產生包含測量範圍變更在內的「偽加速」宣稱；而 P3（強約束）透過 `correctness gate` 與 `contradiction check`，在資料集內達成了矛盾發生率為 0 的高審核性。
- **優化類型精確分類**：研究成功將優化成果區分為實質核心優化（`KERNEL_OPT`）、超參調校（`PARAM_TUNE`）、環境修復（`ENV_FIX`）與多 GPU 擴展限制（`MULTI_GPU_SCALING`），避免將 launcher 設置等系統庫修復誤宣稱為算法加速。

## TODO 
detail in [abstract.md]((file:///home/a/PP/abstract.md))

探討在程式優化領域中，如何透過 **prompt.md、AI agent、操作者介入、一般網頁對話、CLI agent、自動化實驗流程** 等不同人機協作形式，最大化 AI 與人類協作的效率、效能與結果可信度。

研究的核心不是單純回答「AI 能否讓程式變快」，而是進一步追問：

```text
1. 什麼樣的 prompt 設計能讓 AI 產生可驗證的優化結果？
2. AI agent 在程式優化中容易犯哪些錯？
3. 人類操作者應該在哪些節點介入？
4. 如何避免偽加速、錯誤 baseline、correctness 缺失與不可重現結果？
5. 如何把 AI 輔助優化從一次性嘗試，轉化為論文級可審核實驗流程？
```

***

### Benchmark 


```text
1. softmax-cuda
2. topk-cuda
3. moe-cuda
4. moe-align-cuda
5. prefetch-cuda
6. shmembench-cuda
7. p2p-cuda
8. allreduce-cuda
9. pingpong-cuda
10. simpleMultiDevice-cuda
```

## NEXT：人機協作優化


```text
低優化空間：shmembench-cuda
中優化空間：topk-cuda
高優化空間：softmax-cuda
```

### 1. `shmembench-cuda`：低優化空間

研究重點：

```text
AI 是否能判斷已接近硬體限制？
是否會將微小提升過度宣稱？
人類如何用 profiler 與 measurement-equivalent 規則限制 agent？
```

### 2. `topk-cuda`：中優化空間

研究重點：

```text
AI 是否能辨識 workspace allocation / synchronization overhead？
人類如何引導 agent 從 trial-and-error 走向 shape-aware dispatch 或 radix strategy？
```

### 3. `softmax-cuda`：高優化空間

研究重點：

```text
AI 是否能進一步設計 shape-aware softmax dispatch？
人類如何防止過度特化單一 slice？
如何使用 profiler 驗證 expf、reduction、memory traffic 的瓶頸？
```

***

## Workflow

Phase 3 不應再只是「給 prompt → agent 跑完」。應採用自適應工作流：

```text
1. Observe
   讀取 baseline、raw log、profiler 結果

2. Diagnose
   形成 bottleneck hypothesis

3. Retrieve
   查詢論文、CUDA 文件、既有優化案例

4. Plan
   提出單一可驗證修改

5. Human checkpoint
   人類審查是否改變 benchmark 語意

6. Execute
   sbatch 執行，保存 raw output

7. Validate
   correctness、performance、variance、profiler 一起驗證

8. Decide
   accept / reject / rollback / stop
```

這個流程的重點是讓 AI 從 trial-and-error 轉為 evidence-driven optimization。

