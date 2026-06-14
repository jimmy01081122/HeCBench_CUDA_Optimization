# HeCBench AI 輔助程式優化評估報告

## 摘要
本研究系統性比較了 P1（弱約束）、P2（中約束）、P3（強約束）三種 prompt 約束層級在 HeCBench CUDA 效能優化任務中的表現。研究涵蓋了 10 個標準化 CUDA 基準測試，共計 30 筆 Phase 2 核心數據及先前 BASIC 實驗成果。本研究顯示，弱約束 prompt 產生的結果雖可能包含有效加速，但因缺少 baseline 實測、CSV、raw output 與矛盾檢查，其可審核性不足。在 10 個 P1 結果中，至少 4 個存在 baseline 缺失、資料殘缺或測量範圍改變等無效性問題；另有 1 個存在邏輯矛盾。相較之下，P3 強約束 prompt 在本資料集內 contradiction check 為 0，能完整標記 invalid baseline、measurement-equivalent result 與 environment fix，避免將可執行性修復誤宣稱為 kernel optimization。P3 的 2 筆 invalid 數據（allreduce-cuda 與 pingpong-cuda）並非 agent 錯誤宣稱，而是因 baseline 無效而被規則正確排除

---

## 1. 專案掃描與資料來源
以下檔案與目錄的狀態：
- `/home/a/PP/benchmark_summary.csv`：`DATA_MISSING` (根目錄缺失，本報告已引用備份於 `/home/a/PP/BASIC/benchmark_analysis_report/data/benchmark_summary.csv` 之資料)。
- `/home/a/PP/phase2/reports/phase2_level_summary.csv`：`DATA_FOUND` (完整存在，包含 P1/P2/P3 三層級對照數據共 30 筆)。
- `/home/a/PP/BASIC/`：`DATA_FOUND` (包含早期 AI Agent 優化嘗試之 summary.md 與 raw logs)。

### 資料來源優先順序
本研究報告採用以下資料來源優先順序：
1. `phase2_level_summary.csv` (結構化跨層級摘要)
2. 各 benchmark 專屬 raw CSV 檔案 (如 `topk-cuda_results.csv`)
3. `agent_summary.md` (Agent 手動總結報告)
4. `BASIC/benchmark_analysis_report/data/benchmark_summary.csv` (早期匯總表)
5. Markdown 報告本文敘事

若不同來源數字衝突，以結構化 CSV 優先；若 CSV 缺 correctness 欄位，則不將結果標為完整 PASS。

---

## 2. Benchmark 分類
本研究的 10 個標準化 HeCBench 測試案例依其特性與硬體開銷，劃分為三大類：
1. **AI Primitive / Kernel Optimization (AI 算子與核心優化)**:
   - `softmax-cuda` (Softmax 計算特化)
   - `topk-cuda` (Radix Selection 排序與篩選)
   - `moe-cuda` (門控計算與分派)
   - `moe-align-cuda` (MoE 專家序列對齊)
2. **Memory-System / Measurement Benchmark (記憶體系統與測量基準)**:
   - `prefetch-cuda` (統一記憶體預取與分頁)
   - `shmembench-cuda` (共享記憶體交換微基準)
   - `p2p-cuda` (Peer-to-Peer 拓撲頻寬測試)
3. **Multi-GPU / Communication / Environment (多 GPU 通訊與環境配置)**:
   - `allreduce-cuda` (環狀歸約通訊)
   - `pingpong-cuda` (點對點乒乓延遲測試)
   - `simpleMultiDevice-cuda` (多 GPU Element-wise 歸約，擴展性受 PCIe 傳輸主導)

---

## 3. Prompt 層級設計
不同層級的 prompt.md 明確規定了 AI Agent 的行為邊界與約束強度：
- **P1 弱約束 (Weak Prompt)**：提供 benchmark path、基本執行目標與最少量環境提示，但不強制 baseline、CSV、raw output、contradiction check 或 repeated trials。
- **P2 中約束 (Medium Prompt)**：增加角色設定（CUDA performance engineer），規定必須先實測 baseline、保存原始輸出、設定最多嘗試次數限制、並要求輸出 `agent_summary.md` 報告。
- **P3 強約束 (Strong Prompt)**：在 P2 基礎上，強制要求嚴格的 Correctness Gate、Variance/Trials（重複三次試驗）、Profiler 數據記載、標準 CSV 輸出格式，並加入 Contradiction Check（矛盾自我審查）與 Result Type 分類。

---

## 4. Prompt 約束條款分析
我們對 Prompt 的各項關鍵條款進行了質與量化分析：
- **4.1 baseline**：P1 未明確禁止 estimated baseline，也未強制 baseline 必須為實測結果，導致部分案例缺少有效 baseline，如 `moe-align-cuda` 等案例無從計算 speedup。P3 強制實測 baseline，確保了 speedup 的可計算性。
- **4.2 correctness gate**：P1 缺少 machine-readable correctness gate，因此 correctness 多仰賴 agent 摘要敘述，審核成本較高，且容易漏掉 `shmembench-cuda` 等 checksum failed 的潛在錯誤代碼。P3 強制要求比對 correctness 欄位，杜絕了錯誤計算。
- **4.3 raw output**：P1 沒有保留原始日誌的要求，導致 `p2p-cuda` 僅保留了部分 GPU pairs，數據不完整。P3 強制要求備份所有 `.out` 與 `.err`。
- **4.4 submission limit**：P2/P3 的限制促使 Agent 在前幾次優化失敗後，主動回退代碼或進行調整，避免了無限迴圈。
- **4.5 CSV schema**：P3 規定的 CSV schema 強迫 Agent 輸出結構化資料，降低了解析日誌時的整理誤差。
- **4.6 contradiction check**：自我審查有效制止了「對 correctness FAIL 的優化版本進行宣稱」，如 `pingpong-cuda` 在 baseline 缺失時，主動標記為不計算 speedup。
- **4.7 variance / profiler**：重複 3 次試驗能提供初步變異估計，使研究者能辨識小幅提升是否落在測量雜訊內。對於低於 1% 的提升，本研究將其標記為 measurement-equivalent。

---

## 5. 各 Benchmark 結果總覽
(詳細數據見 [SUMMARY_TABLES.md](file:///home/a/PP/evaluation_summary/SUMMARY_TABLES.md))
- **softmax-cuda**：在 P3 中獲得 **1.4575x** 的加速。需要特別注意的是，早期 BASIC/GM 探索性實驗中，在大 slice=784 下曾達到 **59.593x** 的最高加速；這是由於 BASIC 使用了特定 slice 大小與針對性重寫優化，而 P3 的 1.4575x 是在 Phase 2 正規化比較（跨多個 slice 分佈）下的結果。兩者基準不同，不可直接混用。
- **topk-cuda**：透過 CUB temporary workspace reuse 移除 timed allocation，獲得穩定 **1.1995x (P3)** 的加速。
- **allreduce-cuda**：被判定為 `ENV_FIX`。Best Speedup 為 `n/a`，因為其實質成果是避開 GDRCopy 錯誤的 launcher 修復，並非程式優化。
- **moe-align-cuda**：在 P3 下獲得 **1.1504x** 加速，但在比較 CSV 中 correctness 狀態為 `NOT_EXPLICIT_IN_COMPARISON_CSV`。

---

## 6. P1 / P2 / P3 對比分析
- **數據完整度**：P1 的 CSV 記錄率為 0%；P2 為 0%（僅有 markdown 報告）；P3 達到 100%。
- **無效數據率 (Invalid Rate)**：
  - P1 無效/缺乏完整資料的個數為 **4** 個（`moe-align-cuda` 缺 baseline、`p2p-cuda` 資料殘缺、`prefetch-cuda` 缺對照、`simpleMultiDevice-cuda` 改變測量範圍）。
  - P2 為 **0** 個。
  - P3 為 **2** 個（`allreduce-cuda` 與 `pingpong-cuda` 由於 baseline 缺失或無效，被規則正確排除於加速比計算之外）。
- **矛盾發生數 (Contradictions)**：
  - P1 有 **1** 處矛盾（`allreduce-cuda` 在 baseline 失敗下依然回報 1.1635x，且將 launcher 修復歸類為 kernel opt）。
  - P2 有 **1** 處（`allreduce-cuda` 在 baseline 失敗下依然回報 2.7280x 加速比）。
  - P3 有 **0** 處矛盾，所有 invalid cases 都被正確判定與標註，未將環境修復誤宣告為加速。

---

## 7. 有效與無效優化分類
本研究將 AI 輔助成果細分為以下四類，以精確界定其價值：
1. **實質 kernel / algorithm optimization (代碼/算法優化)**：包含 `softmax-cuda`、`topk-cuda`、`moe-cuda`。
2. **Parameter / strategy tuning (參數/快取調校)**：包含 `moe-align-cuda`、`prefetch-cuda`、`shmembench-cuda`。
3. **Multi-GPU scaling / topology characterization (多 GPU 擴展與拓撲掃描)**：包含 `simpleMultiDevice-cuda`、`p2p-cuda`。例如 `simpleMultiDevice-cuda` 實測主要受 PCIe 傳輸 (H2D) 主導，其優化受傳輸瓶頸限制，僅獲得 1.2% 的邊際加速。
4. **Environment / communication repair (環境與通訊啟動修復)**：包含 `allreduce-cuda`、`pingpong-cuda`。這些成果在於修復 UCX 傳輸鏈接或 NCCL launcher。

---

## 8. 人機協作模式分析
雖然 AI Agent 在優化 kernel、尋找環境變數配置上展現了極高的自動化能力，但在以下情境中**人類操作者依然不可取代**：
1. **研究方案設計與約束制定**：AI 無法自主設計 P3 這樣嚴謹的對照組實驗，必須由人類設計 prompt 模板與 correctness validation。
2. **根因診斷的最終確認**：如在 `allreduce-cuda` 中，AI 發現了 GDRCopy symbol error，但仍需人類確認環境中的 `nvhpc` 套件衝突並指定排除路徑。
3. **邊界與學術定義**：AI 傾向於將任何能縮短時間的修改（包括修改 timer 範圍）都宣稱為 speedup。必須由人類設定「改變測量範圍 = 無效優化」的紅線。

---

## 9. Threats to Validity
1. **硬體與環境不一致性**：BASIC 與 Phase 2 運行的 GPU 節點（如 `gn1222` vs `gn1224`）以及 module 版本存在微小差異，跨 benchmark 的平均 speedup 不能做直接數值比較。
2. **P1 數據之解析誤差**：P1 缺少結構化 summary，部分數據是由 AI 通過 raw logs 反向提取，存在記錄偏差。

---

## 10. 後續實驗建議
1. **統一 Result Type 的強約束**：將後續優化實驗的 Result Type 分類設為強約束，禁止 Agent 在非 `KERNEL_OPT` 分類下宣稱 speedup。
2. **細化傳輸與計算時間**：對 multi-GPU 及傳輸限制型題目，要求單獨輸出 `kernel_time`、`copy_time`、`overlap_ratio`，不允許只回報 `total_time`。
3. **引入自動矛盾檢查器**：在 Agent 運行完畢後，由外部 Python 腳本（如本次的 `generate_evaluation_summary.py`）進行自動審計，拒絕任何 correctness 缺失或 baseline 無效的宣稱。

---

## 11. 結論
本研究證明，prompt 的約束強度不只影響 AI agent 的輸出格式，也直接影響結果是否能被科學審核。P1 約束過弱，容易產生包含偽加速與資訊殘缺的無效結果。P3 prompt 的主要價值不是保證最高 speedup，而是強制建立 baseline、correctness、raw log、CSV、variance 與 contradiction check，使 AI 輔助程式優化從一次性嘗試轉化為可重現、可科學審核的實驗流程。

---

## 論文問題解答 (RQ Answers)

### RQ1: 哪些 benchmark 獲得實質 kernel speedup？
答：`softmax-cuda` (1.4575x)、`topk-cuda` (1.1995x)、`moe-cuda` (1.0778x)。其中 `softmax-cuda` 是最明確的 kernel-level speedup 案例；`topk-cuda` 屬 workspace / radix selection optimization；`moe-cuda` 屬較小幅但有效的 AI primitive optimization。

### RQ2: 哪些結果只是 environment fix？
答：`allreduce-cuda` (修復 UCX/GDRCopy 啟動引數)、`pingpong-cuda` (屬 transport comparison / measurement repair；最終結果顯示 tuned CUDA-aware MPI 在 two-rank ping-pong 下優於 NCCL，但這不代表 NCCL 在 collective 類 workload 中較差)。

### RQ3: 邊際或測量等價的結果 (measurement-equivalent) 包含哪些？
答：`p2p-cuda` (1.0022x，小於 1% 屬 measurement-equivalent)、`simpleMultiDevice-cuda` (1.0121x，受傳輸瓶頸主導的邊際加速)、`shmembench-cuda` (1.0293x，微幅但可量測，需 profiler 進一步確認)。

### RQ4: P3 是否比 P1 / P2 更能防止偽加速？
答：是。P3 通過 baseline 實測要求、重複 3 次 trial、CSV 格式約束與矛盾自我審查，成功過濾了 P1 中出現的「改變 timer 範圍 (simpleMultiDevice-cuda)」以及「使用 invalid baseline 宣稱加速 (moe-align-cuda, prefetch-cuda)」等偽加速現象。

### RQ5: 核心 prompt 條款中哪些最重要？
答：最關鍵的條款為 **Correctness Gate** (禁止 correctness 缺失)、**Measured Baseline Requirements** (禁止估算 baseline)、**Variance/Repeated Trials** (排除雜訊) 與 **Contradiction Check** (拒絕 logical contradiction)。

### RQ6: 人類操作者在哪些情境下仍不可取代？
答：人類在「設計約束協定（如 P3 規則）」、「判定加速本質與進行學術防偽（如劃定環境修復與程式優化邊界）」以及「排查深層系統庫鏈接衝突（如 GDRCopy symbol error）」時仍不可取代。

### RQ7: 哪些結果還不足以支撐論文主張？
答：P1 結果不應作為核心論文證據；若要使用，必須回溯 raw logs 並重新通過 P3 等級的 validation，因為它們缺乏 raw output、CSV 或 valid baseline 對照，審核軌跡不完整。
