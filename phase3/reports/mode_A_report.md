# Phase 3 Mode A (Agent-Only Baseline) 實驗結果分析報告與後續要求

> [!NOTE]
> Mode A 的主要成果不是程式優化，而是測量基準與風險揭露。Mode A 結果顯示，agent-only 流程在未進行程式修改時仍可能產生表面 speedup，尤其在 `topk-cuda` 中，高 CV 導致的 baseline outlier 會造成 1.4x 以上的偽加速。因此，後續 Mode B/C 必須將 variance filter 與 `speedup_claim_valid` 欄位納入 schema。
> **因此，Mode A 中出現的任何 speedup 都應優先被解讀為測量穩定性訊號，而非 agent optimization 成果，除非有實際程式修改與穩定重測支持。**

---

## 1. 實驗環境與執行概述
- **硬體平台**：NVIDIA Tesla V100-SXM2-32GB
- **軟體環境**：CUDA 12.8 (sm_70)
- **排程器**：Slurm 排程執行 (Job ID: 949514, 949515, 949516)
- **測試模式**：Mode A (Agent-only baseline，無人類中途批准，共 5 次優化提交額度)。
- **核心特徵**：在 Mode A 中，Agent 未進行實質的原始碼級優化，而是直接重測 Naive/Optimized 基準代碼以檢驗測量穩定性與環境噪音。其主要功能為建立測量基準、噪音分析與揭露官方 Sweep 的可行性邊界。

---

## 2. Benchmark 數據與 Mode A 實測結果

### 2.1 shmembench-cuda (低優化空間)
[results.csv](file:///home/a/PP/phase3/shmembench-cuda/mode_A_agent_only/results.csv) / [agent_summary.md](file:///home/a/PP/phase3/shmembench-cuda/mode_A_agent_only/agent_summary.md)

#### 測試結果摘要
| block_size | variant | correctness_status | measurement_validity | metric_value (GB/s) | speedup | speedup_claim_valid | notes |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 128 | original | **FAIL** | DIAGNOSTIC_FAIL | 13368.04 | n/a | false | 校驗錯誤 |
| 256 | original | **PASS** | VALID | 13273.14 | 1.001x | false (measurement-equivalent) | 校驗通過 |
| 512 | original | **FAIL** | DIAGNOSTIC_FAIL | 13930.17 | n/a | false | 校驗錯誤 |
| 1024 | original | **FAIL** | DIAGNOSTIC_FAIL | n/a | n/a | false | 共享記憶體超出限制，編譯失敗 |

#### 關鍵發現與原因分析
1. **正確性失效與 block_size 硬性耦合**：
   - 僅有 `block_size=256` 通過校驗，而 `128` 與 `512` 均回報 `FAIL`。這揭露了 `shmembench` 原始 correctness 驗證機制或共享記憶體配置與 block_size 有著硬性耦合，隨意調整會破壞數據流而導致校驗失敗。
   - **block_size=128 與 512 的 GB/s 僅作為診斷性觀察，不可用於 speedup claim；其 speedup 欄位已標為 n/a**。
2. **靜態共享記憶體限制**：
   - `block_size=1024` 時編譯失敗（`shared memory exceeds limit`）。V100 每個 block 的 static shared memory 上限為 48KB/96KB，該 kernel 在 1024 執行緒配置下超出了此硬體上限。

---

### 2.2 softmax-cuda (高優化空間)
[results.csv](file:///home/a/PP/phase3/softmax-cuda/mode_A_agent_only/results.csv) / [agent_summary.md](file:///home/a/PP/phase3/softmax-cuda/mode_A_agent_only/agent_summary.md)

#### 測試結果摘要
| slice_size | implementation | correctness_status | measurement_validity | result_type | baseline_metric (ms) | metric_value (ms) | speedup | speedup_claim_valid |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 128 | impl=0 (naive) | **PASS** | VALID | NAIVE_REFERENCE | 4.394 | 4.410 | 0.996x | false |
| 128 | impl=1 (opt) | **PASS** | VALID | EXISTING_OPTIMIZED_BASELINE | 0.135 | 0.135 | 1.000x | false |
| 256 | impl=0 (naive) | **PASS** | VALID | NAIVE_REFERENCE | 15.664 | 15.660 | 1.000x | false |
| 256 | impl=1 (opt) | **PASS** | VALID | EXISTING_OPTIMIZED_BASELINE | 0.311 | 0.305 | 1.021x | false (comparison only) |
| 784 | impl=0 (naive) | **PASS** | VALID | NAIVE_REFERENCE | 54.549 | 54.510 | 1.001x | false |
| 784 | impl=1 (opt) | **PASS** | VALID | EXISTING_OPTIMIZED_BASELINE | 1.452 | 1.451 | 1.001x | false |
| 1024 | impl=0 (naive) | **PASS** | VALID | NAIVE_REFERENCE | 64.488 | 64.549 | 0.999x | false |
| 1024 | impl=1 (opt) | **PASS** | VALID | EXISTING_OPTIMIZED_BASELINE | 2.108 | 2.119 | 0.995x | false |
| 2048 | impl=0 (naive) | **PASS** | VALID | NAIVE_REFERENCE | 41.672 | 41.941 | 0.993x | false |
| 2048 | impl=1 (opt) | **PASS** | VALID | EXISTING_OPTIMIZED_BASELINE | 2.242 | 2.237 | 1.002x | false |

#### 關鍵發現與原因分析
1. **優化基線與 Naive 的差距**：
   - 預載的優化實現 (`impl=1`) 相比 Naive 實現 (`impl=0`) 效能極高（例如 `slice=784` 時加速比達 **~37.5倍**）。**但這並非 Mode A Agent 產生的新優化，而是既有的基準。** 其兩者對比之 `result_type` 為 `BASELINE_COMPARISON` 而非 `AGENT_OPT`。
2. **高測量穩定性**：
   - Mode A Final 與 Baseline 對比，加速比均精準保持在 `1.00x`，CV 普遍低於 0.8%，證明在 sbatch 獨佔節點上的測量噪音極低，結果具備高再現性。

---

### 2.3 topk-cuda (中優化空間)
[results.csv](file:///home/a/PP/phase3/topk-cuda/mode_A_agent_only/results.csv) / [agent_summary.md](file:///home/a/PP/phase3/topk-cuda/mode_A_agent_only/agent_summary.md)

#### 測試結果摘要 (部分高波動形狀與大形狀)
| hidden_size | topk | correctness_status | measurement_validity | baseline_metric (us) | metric_value (us) | speedup | baseline_CV | final_CV | speedup_claim_valid | result_type |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 3072 | 2048 | **PASS** | VALID | 795.93 | 785.08 | 1.014x | 5.9% | 7.2% | **false** | MEASUREMENT_EQUIVALENT (low speedup) |
| 3072 | 1024 | **PASS** | **CAUTION** | 708.57 | 773.07 | 0.917x | 0.2% | 12.5% | **false** | REGRESSION |
| 4096 | 2048 | **PASS** | **NOISY** | 1196.41 | 845.63 | 1.415x | 54.6% | 11.2% | **false** | NOISY_MEASUREMENT (pseudo-speedup) |
| 8192 | 2048 | **PASS** | **NOISY** | 1545.94 | 1073.88 | 1.440x | 43.2% | 7.9% | **false** | NOISY_MEASUREMENT (pseudo-speedup) |
| 131072 | 1024 | **PASS** | VALID | 13465.55 | 12394.57 | 1.086x | 0.5% | 1.4% | **true** | BASELINE_REMEASUREMENT (stable diff observed) |

#### 關鍵發現與原因分析
1. **嚴重的測量變異性與「偽加速」陷阱**：
   - 原始碼未做任何修改，但在 `hidden_size=4096, topk=2048` 卻出現了 **1.415x** 的「加速」。這是由於 Baseline 測試的變異數 (CV) 達到了驚人的 **54.6%**（存在極端離群值 1951 us），而 Final 測試環境恢復穩定，因而產生不實的表面加速。
2. **小幅優化與 Regression 判定**：
   - 對於 `3072, 1024` 的加速比為 `0.917x`，應被正確判定為 `REGRESSION`，且 `speedup_claim_valid` 設為 `false`。
   - 對於 `3072, 2048` 雖然 CV 偏低，但由於加速比僅 1.4%，低於統計顯著，且本輪未改代碼，因此判定為 `MEASUREMENT_EQUIVALENT` 且 `speedup_claim_valid=false`。
   - 對於 `131072, 1024` 雖然 CV 極低且測得 8.6% 的偏差，但由於 Mode A 並無代碼修改，此現象被分類為 `BASELINE_REMEASUREMENT`（測量穩定差異），而非 Agent 的優化成果。

---

## 3. 可回傳給主規劃器的 Mode A 結論與後續要求

Mode A 完成後，三個 benchmark 呈現出不同的基準特性。Mode A 的主要價值不是產生新優化，而是建立 agent-only 測量基準、揭露測量噪音與確認後續 Mode B/C 的可比較條件。

### 3.1 softmax-cuda：穩定 existing optimized baseline

`softmax-cuda` 在 Mode A 中建立了穩定的 existing optimized baseline。後續 Mode B / Mode C 必須以 `impl=1` 作為正式基準。

明確規則如下：

```text
impl=0 = naive reference
impl=1 = existing optimized baseline
Mode B/C candidate = modified impl=1 或新增 impl=2
```

因此，後續不得把：

```text
impl=0 → impl=1
```

的差距計為 Phase 3 agent speedup。這只是既有 naive 與既有 optimized implementation 的基準差異，不是 Mode B/C 產生的新優化。

### 3.2 topk-cuda：需先重建 robust baseline

`topk-cuda` 在 Mode A 中暴露出明顯測量不穩定性。部分 cases 雖然 correctness PASS，但 baseline CV 高達約 40%–50%，造成約 1.4x 的表面加速。由於 Mode A 並未進行實質原始碼修改，這類 speedup 應判定為 measurement artifact，而不是 agent optimization。

後續 Mode B / Mode C 在進行 workspace reuse、block-size tuning 或 dispatch policy 優化前，必須先完成：

```text
1. robust baseline remeasurement
2. 高 CV cases 重測
3. variance filter
4. speedup_claim_valid 判定
```

建議規則：

```text
CV > 15% → measurement_validity=NOISY
CV > 15% 且 speedup > 1.05 → speedup_claim_valid=false
CV > 30% → require_remeasurement=true
```

### 3.3 shmembench-cuda：official comparison 與 diagnostic sweep 必須分離

`shmembench-cuda` 顯示原本的 block-size sweep 不完全可用。Mode A 結果顯示：

```text
block_size=256, variant=original
```

是唯一通過 correctness 的 official validated comparison。

其餘：

```text
block_size=128
block_size=512
block_size=1024
```

應保留為 diagnostic failures，而不是直接刪除，也不得納入 official speedup 計算。

建議後續規則：

```text
official validated comparison:
  block_size=256, variant=original

diagnostic sweep:
  block_size=128, 512, 1024
```

只有 official validated comparison 失敗時，才判定整題 official comparison invalid。Diagnostic failures 應保留於報告中，用於解釋 benchmark 的 block-size 耦合、checksum 限制或 shared memory allocation 邊界。
**Mode B/C 不應再以 block size sweep 作為主要優化方向，除非先修正 checksum / validation 與 shared-memory allocation 對 block size 的耦合。**

---

## 4. 進入 Mode B/C 前必須完成的修訂

因此，進入 Mode B / Mode C 之前，必須先更新以下三個核心元件：

### A. 更新 result schema

需要新增或確認以下欄位：

```csv
correctness_status,measurement_validity,speedup_claim_valid,require_remeasurement
```

建議語意如下：

```text
correctness_status:
  PASS / FAIL / NOT_PROVIDED / PARTIAL

measurement_validity:
  VALID / CAUTION / NOISY / INVALID / LIMITED / DIAGNOSTIC_FAIL

speedup_claim_valid:
  true / false

require_remeasurement:
  true / false
```

### B. 更新 self-consistency auditor

Auditor 必須加入以下規則：

```text
1. correctness_status != PASS → speedup=n/a, speedup_claim_valid=false
2. CV > 15% → measurement_validity=NOISY
3. CV > 15% and speedup > 1.05 → speedup_claim_valid=false
4. CV > 30% → require_remeasurement=true
5. Mode_A 且無 source change → 不得標記 KERNEL_OPT
6. softmax-cuda 的 impl0_to_impl1 comparison → BASELINE_COMPARISON, not AGENT_OPT
7. shmembench-cuda 中 block_size != 256 且 correctness FAIL → DIAGNOSTIC_FAIL
8. optional variant 取代 original baseline → INVALID
9. If speedup < 1.01, set result_type=MEASUREMENT_EQUIVALENT and speedup_claim_valid=false.
10. If speedup < 1.0, set result_type=REGRESSION and speedup_claim_valid=false.
```

### C. 更新 official_sweeps

`official_sweeps.yaml` 必須明確鎖定三題的官方比較規格：

```yaml
softmax-cuda:
  official_baseline:
    implementation: 1
    description: existing optimized baseline
  official_cases:
    - slice_size: 128
    - slice_size: 256
    - slice_size: 784
    - slice_size: 1024
    - slice_size: 2048
  forbidden_comparison:
    - impl0_to_impl1_as_agent_speedup

topk-cuda:
  official_cases:
    hidden_size: [3072, 4096, 8192, 16384, 32768, 65536, 131072]
    topk: [1024, 2048]
  robust_baseline_required: true
  noisy_case_policy:
    cv_over_15_percent: exclude_from_speedup_claim_until_remeasured

shmembench-cuda:
  official_validated_comparison:
    variant: original
    block_size: 256
  diagnostic_sweep:
    - variant: original
      block_size: 128
    - variant: original
      block_size: 512
    - variant: original
      block_size: 1024
  optional_variants:
    - padded
    - vectorized
```

---

## 5. 最終主規劃器決策建議

在完成上述 schema、auditor 與 official sweep 修訂前，不應直接啟動 Mode B / Mode C。否則後續人機協作結果會受到以下問題污染：

```text
1. softmax-cuda 可能把 impl=0 → impl=1 誤當 agent speedup
2. topk-cuda 可能把 high-CV noise 誤當有效優化
3. shmembench-cuda 可能把 diagnostic failure 誤當 official benchmark failure
4. Mode B/C 可能與 Mode A 使用 different 測試範圍，導致 speedup 不可比較
```

建議執行順序：

```text
1. 更新 result_schema.csv
2. 更新 self_consistency_auditor.py
3. 建立或修正 official_sweeps.yaml
4. 重測 robust baselines
5. 通過 auditor 後再啟動 Mode B
6. Mode B 完成後再啟動 Mode C
```

總結一句話：

```text
Mode A 已完成三項任務：softmax 建立穩定 optimized baseline，topk 揭露測量噪音，shmembench 確認 official comparison 邊界。進入 Mode B/C 前，必須先把這些發現轉化為 schema、auditor 與 official sweep 規則，否則後續人機協作結果將不可比較。
```
