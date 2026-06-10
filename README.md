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
