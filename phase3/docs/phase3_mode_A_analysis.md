# Phase 3 Mode A (Agent-Only Baseline) 實驗結果分析報告

> [!NOTE]
> Mode A 的主要成果不是程式優化，而是測量基準與風險揭露。Mode A 結果顯示，agent-only 流程在未進行程式修改時仍可能產生表面 speedup，尤其在 `topk-cuda` 中，高 CV 導致的 baseline outlier 會造成 1.4x 以上的偽加速。因此，後續 Mode B/C 必須將 variance filter 與 `speedup_claim_valid` 欄位納入 schema。

---

## 1. 實驗環境與執行概述
- **硬體平台**：NVIDIA Tesla V100-SXM2-32GB
- **軟體環境**：CUDA 12.8 (sm_70)
- **排程器**：Slurm 排程執行 (Job ID: 949514, 949515, 949516)
- **測試模式**：Mode A (Agent-only baseline，無人類中途批准，共 5 次優化提交額度)。
- **核心特徵**：在 Mode A 中，Agent 未進行實質的原始碼級優化，而是直接重測 Naive/Optimized 基準代碼以檢驗測量穩定性與環境噪音。其主要功能為建立測量基準、噪音分析與揭露官方 Sweep 的可行性邊界。

---

## 2. Benchmark 數據深度分析

### 2.1 shmembench-cuda (低優化空間)
[results.csv](file:///home/a/PP/phase3/shmembench-cuda/mode_A_agent_only/results.csv) / [agent_summary.md](file:///home/a/PP/phase3/shmembench-cuda/mode_A_agent_only/agent_summary.md)

#### 測試結果摘要
| block_size | variant | correctness_status | measurement_validity | metric_value (GB/s) | speedup | notes |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 128 | original | **FAIL** | DIAGNOSTIC_FAIL | 13368.04 | 1.001x | 校驗錯誤 |
| 256 | original | **PASS** | VALID | 13273.14 | 1.001x | 校驗通過 |
| 512 | original | **FAIL** | DIAGNOSTIC_FAIL | 13930.17 | 0.999x | 校驗錯誤 |
| 1024 | original | **FAIL** | DIAGNOSTIC_FAIL | n/a | n/a | 共享記憶體超出硬體上限，編譯失敗 |

#### 關鍵發現與原因分析
1. **正確性失效與 block_size 硬性耦合**：
   - 僅有 `block_size=256` 通過校驗，而 `128` 與 `512` 均回報 `FAIL`。這揭露了 `shmembench` 原始 correctness 驗證機制或共享記憶體配置與 block_size 有著硬性耦合，隨意調整會破壞數據流而導致校驗失敗。
2. **靜態共享記憶體限制**：
   - `block_size=1024` 時編譯失敗（`shared memory exceeds limit`）。V100 每個 block 的 static shared memory 上限為 48KB/96KB，該 kernel 在 1024 執行緒配置下超出了此硬體上限。
3. **實驗結論與 Sweep 收縮建議**：
   - **不建議完全刪除** block_size=128/512/1024，而應將其改為 **Diagnostic Sweep (診斷 Sweep)**，其結果 FAIL 不代表整個 benchmark 失效；而將 `block_size=256, variant=original` 設為唯一的 **Official Validated Comparison (官方驗證比對組)**。

---

### 2.2 softmax-cuda (高優化空間)
[results.csv](file:///home/a/PP/phase3/softmax-cuda/mode_A_agent_only/results.csv) / [agent_summary.md](file:///home/a/PP/phase3/softmax-cuda/mode_A_agent_only/agent_summary.md)

#### 測試結果摘要
| slice_size | implementation | correctness_status | measurement_validity | baseline_metric (ms) | metric_value (ms) | speedup |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 128 | impl=0 (naive) | **PASS** | VALID | 4.394 | 4.410 | 0.996x |
| 128 | impl=1 (opt baseline) | **PASS** | VALID | 0.135 | 0.135 | 1.000x |
| 256 | impl=0 (naive) | **PASS** | VALID | 15.664 | 15.660 | 1.000x |
| 256 | impl=1 (opt baseline) | **PASS** | VALID | 0.311 | 0.305 | 1.021x |
| 784 | impl=0 (naive) | **PASS** | VALID | 54.549 | 54.510 | 1.001x |
| 784 | impl=1 (opt baseline) | **PASS** | VALID | 1.452 | 1.451 | 1.001x |
| 1024 | impl=0 (naive) | **PASS** | VALID | 64.488 | 64.549 | 0.999x |
| 1024 | impl=1 (opt baseline) | **PASS** | VALID | 2.108 | 2.119 | 0.995x |
| 2048 | impl=0 (naive) | **PASS** | VALID | 41.672 | 41.941 | 0.993x |
| 2048 | impl=1 (opt baseline) | **PASS** | VALID | 2.242 | 2.237 | 1.002x |

#### 關鍵發現與原因分析
1. **優化基線與 Naive 的差距**：
   - 預載的優化實現 (`impl=1`) 相比 Naive 實現 (`impl=0`) 效能極高（例如 `slice=784` 時加速比達 **~37.5倍**）。**但這並非 Mode A Agent 產生的新優化，而是既有的基準。**
2. **高測量穩定性**：
   - Mode A Final 與 Baseline 對比，加速比均精準保持在 `1.00x`，CV 普遍低於 0.8%，證明在 sbatch 獨佔節點上的測量噪音極低，結果具備高再現性。
3. **實驗結論與後續起點**：
   - 確認 `impl=1` 為穩定的既有優化基準（**existing optimized baseline**）。Mode B/C 必須以 `impl=1` 作為起點，所有新優化加速比必須計算為 `new_candidate / impl=1_baseline`，**不得將 impl=0 -> impl=1 的差距計為優化成果**。

---

### 2.3 topk-cuda (中優化空間)
[results.csv](file:///home/a/PP/phase3/topk-cuda/mode_A_agent_only/results.csv) / [agent_summary.md](file:///home/a/PP/phase3/topk-cuda/mode_A_agent_only/agent_summary.md)

#### 測試結果摘要 (部分高波動形狀與大形狀)
| hidden_size | topk | correctness_status | measurement_validity | baseline_metric (us) | metric_value (us) | speedup | baseline_CV | final_CV | speedup_claim_valid |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 3072 | 2048 | **PASS** | VALID | 795.93 | 785.08 | 1.014x | 5.9% | 7.2% | true |
| 3072 | 1024 | **PASS** | CAUTION | 708.57 | 773.07 | 0.917x | 0.2% | **12.5%** | true |
| 4096 | 2048 | **PASS** | **NOISY** | 1196.41 | 845.63 | **1.415x** | **54.6%** | 11.2% | **false** |
| 8192 | 2048 | **PASS** | **NOISY** | 1545.94 | 1073.88 | **1.440x** | **43.2%** | 7.9% | **false** |
| 131072 | 1024 | **PASS** | VALID | 13465.55 | 12394.57 | 1.086x | 0.5% | 1.4% | true |

#### 關鍵發現與原因分析
1. **嚴重的測量變異性與「偽加速」陷阱**：
   - 原始碼未做任何修改，但在 `hidden_size=4096, topk=2048` 卻出現了 **1.415x** 的「加速」。這是由於 Baseline 測試的變異數 (CV) 達到了驚人的 **54.6%**（存在極端離群值 1951 us），而 Final 測試環境恢復穩定，因而產生不實的表面加速。
2. **解耦狀態判定**：
   - Correctness 角度為 SUCCESS，但測量有效性為 **NOISY**。這證明了單一 `SUCCESS` 標籤會掩蓋嚴重的測量風險，必須將正確性狀態與測量有效性狀態進行欄位解耦。

---

## 3. 建議回傳給主規劃器的正式修訂建議

### 3.1 新增 Measurement Validity 欄位
建議在結果 schema 中增加正確性與測量有效性解耦欄位：
`correctness_status,measurement_validity,speedup_claim_valid`
- `correctness_status`：`PASS / FAIL / NOT_PROVIDED`
- `measurement_validity`：`VALID / CAUTION / NOISY / INVALID`
- `speedup_claim_valid`：`true / false`

### 3.2 建立變異數審計器 (Variance Filter)
自動審計器 `self_consistency_auditor.py` 應加入以下規則：
- `CV <= 5%` $\rightarrow$ `measurement_validity = VALID`
- `5% < CV <= 15%` $\rightarrow$ `measurement_validity = CAUTION`
- `CV > 15%` $\rightarrow$ `measurement_validity = NOISY`
- `CV > 15%` 且 `speedup > 1.05` $\rightarrow$ `speedup_claim_valid = false`（加速判定無效，除非重測確認）
- `CV > 30%` $\rightarrow$ `require_remeasurement = true`（強制重測）

### 3.3 修正 shmembench-cuda 官方 Sweep 規格
將 shmembench 的 Sweep 區分為：
- **Official Validated Comparison**：`block_size=256, variant=original`。只有此組失敗，整題才算 invalid。
- **Diagnostic Sweep**：`block_size=128, 512, 1024`。必須保留其 FAIL 或編譯失敗結果，但不參與 official speedup 幾何平均。

### 3.4 修正 softmax-cuda Mode A Baseline 定義
- `impl=0` 為 naive reference，`impl=1` 為 existing optimized baseline。
- Mode B/C 必須以 `impl=1` 作為起點，不得將 `impl=0` $\rightarrow$ `impl=1` 計為優化成果。

### 3.5 Mode A 結論分類修正
- **shmembench-cuda**：
  - `correctness_status = PARTIAL`
  - `measurement_validity = LIMITED`
  - `conclusion`：官方對比僅使用 `block_size=256`；其他為診斷性失敗 (diagnostic failures)。
- **softmax-cuda**：
  - `correctness_status = PASS`
  - `measurement_validity = VALID`
  - `conclusion`：已建立穩定的 optimized baseline，無新 agent 優化。
- **topk-cuda**：
  - `correctness_status = PASS`
  - `measurement_validity = NOISY` (部分中小形狀)
  - `conclusion`：必須在 Mode B/C 進行測量穩定化與變異數過濾。

---

## 4. 三個 Benchmark 的 Mode B/C 具體執行策略

### 4.1 shmembench-cuda
1. **穩定與驗證**：確認 `block_size=256` 的 correctness 與 variance。
2. **瓶頸剖析**：分析 `128` 和 `512` 造成 checksum FAIL 的具體原因，以及 `1024` 共享記憶體編譯失敗。
3. **優化比對**：若開發 `padded` 或 `vectorized` 變體，必須與 `block_size=256, variant=original` baseline 做對照。
4. **驗收標準**：
   - 保持 correctness PASS。
   - 解釋 block-size 限制。
   - 優化提升需具備低 CV（$\le 5\%$ 或至少 $\le 15\%$），否則不計。提升 $<1\%$ 標為 `measurement-equivalent`。

### 4.2 softmax-cuda
1. **起點防線**：嚴格禁止以 `impl=0` $\rightarrow$ `impl=1` 宣稱優化。
2. **基線對照**：Baseline = `impl=1`，Candidate = 優化後的 `impl=1` 或新寫的 `impl=2`。
3. **全面 Sweep**：必須測滿 `slice=128, 256, 784, 1024, 2048`。
4. **驗收標準**：
   - 所有 slice correctness PASS。
   - 相對 `impl=1` 基準有實質且穩定的提升。
   - 禁止只對單一形狀（如 `784`）特化；若僅改善部分 slice，標為 `PARTIAL_SUCCESS`。

### 4.3 topk-cuda
1. **穩定化測量**：首要任務是改善測量穩定性。
   - 增加 trials 至 5 次或 7 次。
   - 增加 warmup 輪次。
   - 嘗試鎖定 GPU 時脈（若權限允許）；否則在 report limitations 中明記。
   - 重新測量 CV > 15% 的 cases，建立 robust baseline。
2. **優化開發**：在測量穩定後，才進行 workspace reuse、block size 與 dispatch policy 的優化。
3. **驗收標準**：
   - 14 個 cases 全部 correctness PASS。
   - 使用穩定且重測後的 baseline 計算加速比，高 CV 的 case 排除在 speedup 宣稱之外，並如實報告 per-case regression。

---

## 5. 建議加入 Auditor 的新規則
直接在 `self_consistency_auditor.py` 中部署：
- **Rule V1**: If CV > 15%, set `measurement_validity=NOISY`.
- **Rule V2**: If CV > 15% and speedup > 1.05, set `speedup_claim_valid=false`.
- **Rule V3**: If benchmark=softmax-cuda and comparison is impl0_to_impl1, set `result_type=BASELINE_COMPARISON`, not `AGENT_OPT`.
- **Rule V4**: If benchmark=shmembench-cuda and block_size != 256 and correctness != PASS, mark as `DIAGNOSTIC_FAIL`, not final benchmark failure.
- **Rule V5**: If official validated baseline is missing, `speedup=n/a`.
- **Rule V6**: If optional variant replaces original baseline, mark `INVALID`.
