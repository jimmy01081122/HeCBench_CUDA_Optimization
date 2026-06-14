# 基於 LLM Agent 的 CUDA Benchmark 優化：能力、限制與驗證挑戰

Final Project -- Parallel Programming, Spring 2026  
Chun-Min Chang, Sheng-Hua Wang, Yi-Hsuan Huang

## 摘要

本文研究 LLM-based agent 在 HeCBench CUDA benchmark 上進行程式優化的能力與限制。本文不只回報最佳加速比，而是檢查 agent 產生的修改是否正確、可重現且可審核。資料來源包含主專案十個 prompt 約束實驗、Phase 3 human-in-the-loop 案例、另一批基本測試結果，以及剩餘 Phase 2 benchmark 摘要。結果顯示，agent 對 softmax、top-k、Adam、adjacent difference、random access 與部分排序類 workload 能產生較可信的 kernel-level speedup；但 graph、convergence 或 irregular 類程式容易出現 correctness failure。另有一類 benchmark-aware optimization 可能利用固定輸入、重複運算或驗證結構，本研究中性記錄此能力，但將其與一般 kernel optimization 分開。整體而言，baseline 實測、correctness gate、variance check、結果分類與人類審查，是避免 pseudo-speedup 並建立可審核 AI 優化流程的必要條件。

## 1. 導論

LLM 已逐漸被用於程式生成與程式優化。相較於傳統手動 CUDA 調校，LLM agent 能根據自然語言提出原始碼修改、解讀編譯或執行回饋，並迭代修正候選實作。然而 GPU 優化比一般程式生成更嚴格：有效優化必須同時保留語意正確性、改善明確定義的效能指標，並符合 memory hierarchy、同步、occupancy 與 launch overhead 等硬體限制。

本文使用 HeCBench CUDA benchmark 作為實驗平台，評估 LLM agent 的優化能力、失敗模式與可審核性。研究目標不是最大化平均 speedup，而是回答何種 speedup 可信、何種結果可能來自 invalid baseline 或 measurement artifact，以及 prompt 約束與人類審查如何提升研究品質。資料包含主專案十個 P1/P2/P3 prompt 約束實驗、Phase 3 softmax/topk/shmembench 案例、`data.md` 的外部基本測試，以及 `rest.md` 中剩餘 benchmark 摘要。

## 2. 背景

既有研究已將 LLM 用於 code generation、optimization 與 agent workflow。CUDA 優化會放大 LLM 生成程式的風險，因為微小修改可能造成 race condition、改變 validation logic，或將計算移出 timed region。因此本文將 `KERNEL_OPT` 與 `ENV_FIX`、`MEASURE_FIX`、`PARAM_TUNE`、`MEASUREMENT_EQUIVALENT`、`BENCHMARK_AWARE_OPT` 分開。最後一類採中性描述：根據可得紀錄，它可能利用固定輸入、重複工作或 benchmark 驗證結構，不能直接視為一般化 kernel 優化。

## 3. 方法

本文研究七個問題：LLM agent 是否能產生有效 CUDA 優化；成功與失敗 benchmark 有何差異；reported speedup 是否可信；prompt 約束是否能降低 pseudo-speedup；人類審查是否能將 partial optimization 轉化為可用策略；哪些結果是 kernel optimization，哪些是 measurement、environment 或 benchmark-aware effect；evidence-guided aggressive optimization 是否有效。

本研究共涵蓋 29 個 HeCBench CUDA benchmark，與英文論文 `paper.tex` 一致：

`cc-cuda`, `gc-cuda`, `mis-cuda`, `adjacent-cuda`, `floydwarshall-cuda`, `floydwarshall2-cuda`, `merge-cuda`, `quicksort-cuda`, `sortKV-cuda`, `bitonic-sort-cuda`, `topk-cuda`, `filter-cuda`, `minmax-cuda`, `nonzero-cuda`, `reverse-cuda`, `scan-cuda`, `split-cuda`, `adam-cuda`, `softmax-cuda`, `dropout-cuda`, `moe-cuda`, `moe-align-cuda`, `prefetch-cuda`, `shmembench-cuda`, `randomAccess-cuda`, `p2p-cuda`, `allreduce-cuda`, `pingpong-cuda`, `simpleMultiDevice-cuda`。

所有可得執行紀錄使用相同硬體環境：NVIDIA Tesla V100-SXM2-32GB、CUDA 12.8 / nvcc V12.8.61、target architecture `sm_70`，並透過 Slurm GPU node 執行。

外部基本測試使用兩種 prompt：一般優化 prompt 要求輸出相同資料並提供完整程式碼；environment-aware prompt 額外提供 V100、`sm_70` 與 Slurm。主專案則使用 P1/P2/P3 三層 prompt。P1 缺少 baseline、CSV 與 contradiction check；P2 要求 baseline、raw output、attempt limit 與 agent summary；P3 進一步要求 correctness gate、repeated trials、CSV schema、profiler notes、result type 與 contradiction check。Phase 3 則加入 human checkpoint、robust baseline、decision log 與 final confirmation。

本文將 agent 輸出視為「待驗證假設」，而不是直接視為最終優化。工作流程如下：

| 階段 | 動作 | 主要證據 | 目的 |
|---|---|---|---|
| 1 | 定義 prompt 與約束 | input prompt、rules | 指定任務、硬體與驗證規則 |
| 2 | 實測 baseline | Slurm output、baseline CSV | 避免估算或無效 speedup 分母 |
| 3 | 產生候選修改 | patch summary、code diff | 記錄 agent 實際修改了什麼 |
| 4 | 透過 Slurm 執行 | job script、raw output | 避免 login node 測量污染 |
| 5 | 檢查 correctness | PASS/FAIL logs | 排除語意或數值錯誤 |
| 6 | 重複測量與比較 | repeated trials、CV | 偵測 measurement noise 與 pseudo-speedup |
| 7 | 分類結果 | result type、contradiction check | 區分 kernel、tuning、environment、measurement effect |
| 8 | 人類審查 | decision log、final confirmation | accept、reject、rollback 或 refined strategy |

各階段目的不同。Phase 1 主要是診斷與分類，判斷每個 benchmark 可能屬於 kernel optimization、memory/measurement 或 communication/environment 類型。Phase 2 以 prompt 強度為控制變因，檢查 baseline、correctness、raw output 與 structured report 是否能提升結果可審核性。Phase 3 則將 agent 放入 human-governed workflow：候選修改可能被拒絕、限制在特定 shape，或在 final confirmation 後才被接受。

因此，同一個 speedup 數字在不同階段意義不同。P1 的 one-shot speedup 若缺少 baseline，只能視為假設；P3 若有 repeated trials、CSV 與 contradiction check，證據較強；Mode B/C 若與 accepted baseline 成對比較並通過人類審查，則可信度更高。

審核規則如下：correctness failure 一律 invalid；baseline 缺失或無效時不得宣稱 speedup；小於 1% 的改善標為 `MEASUREMENT_EQUIVALENT`；launcher 或環境修復標為 `ENV_FIX`；利用固定 benchmark 結構的修改標為 `BENCHMARK_AWARE_OPT`。若 raw log、system prompt、diff 或 variance 不存在，則標示為 N/A，並降低證據等級。

## 4. 實驗設定

主專案 prompt 約束實驗包含十個 benchmark；外部基本測試提供十個 benchmark 的 baseline、optimized、env-optimized timing，但無 raw logs 與 source diff；剩餘 Phase 2 摘要補充 Adam、adjacent、dropout、filter、minmax、nonzero、randomAccess、reverse、scan 與 topk。

在 Phase 3 softmax 中，正式 baseline 是既有 optimized implementation `impl=1`，不得將 naive `impl=0` 到 `impl=1` 的差距計為 agent speedup。Top-k 因 Mode A 暴露高 CV pseudo-speedup，因此 Mode B 先建立 robust baseline。Shmembench 只將 `block_size=256` 視為 official validated comparison，其餘 block size 作為 diagnostic failure。

## 5. 結果

### 5.1 Prompt 約束結果

Prompt 強度主要提升的是可審核性，而非表面平均 speedup。十個 benchmark 的 prompt 實驗中，P1 表面平均 speedup 最大，但沒有 CSV source，且四個結果缺少或無效 baseline。P3 具有完整 CSV coverage，且 final audit 無 contradiction。代表性 P3 結果如下：

| Level | N | Mean speedup | Median | CSV count | Invalid baseline |
|---|---:|---:|---:|---:|---:|
| P1 | 10 | 1.552x | 1.128x | 0 | 4 |
| P2 | 10 | 1.369x | 1.117x | 0 | 0 |
| P3 | 10 | 1.131x | 1.097x | 10 | 2 |

跨資料來源的 prompt 影響如下：

| 證據來源 | Prompt/Level | 觀察結果 | 解讀 |
|---|---|---|---|
| 主專案十題 | P1 weak prompt | 表面平均 speedup 最高、4 個 bad baseline、無 CSV | 產生快速但弱可審核 |
| 主專案十題 | P2 constrained prompt | 平均 speedup 下降、baseline 與 summary 較完整 | 改善分母有效性與 traceability |
| 主專案十題 | P3 audit prompt | CSV coverage 完整、有 contradiction check | 證據品質最高，但 claim 更保守 |
| `data.md` | generic prompt | sorting 改善、graph failure、Floyd-Warshall 可疑極端加速 | 無約束 prompt 會混合有效改善與 invalid change |
| `data.md` | environment-aware prompt | V100/Slurm context 未穩定改善結果 | 硬體資訊不能取代審核規則 |
| `rest.md` | P1/P2/P3 kernel cases | Adam、adjacent、randomAccess 約 2x 且跨 level 穩定 | 規則型 kernel 對 prompt variation 較穩健 |
| `rest.md` | P1/P2/P3 benchmark-aware cases | dropout、minmax、scan、reverse、topk 維持極大加速 | prompt 強度本身無法阻止 benchmark structure exploitation |

因此，prompt engineering 在本研究中更像實驗控制，而不只是效能調校。若只看最大 speedup，P1 與 benchmark-aware shortcut 會顯得最好；若納入 correctness、reproducibility 與 auditability，P3 與 human-guided workflow 才是較可信的流程。

| Benchmark | Speedup | Classification | Note |
|---|---:|---|---|
| softmax-cuda | 1.4575x | KERNEL_OPT | PASS 42/42 |
| topk-cuda | 1.1995x | KERNEL_OPT | 14 cases, repeated trials |
| moe-cuda | 1.0778x | KERNEL_OPT | topk=8 measurement-equivalent |
| moe-align-cuda | 1.1504x | PARAM_TUNE | correctness evidence limited |
| p2p-cuda | 1.0022x | TOPOLOGY / MEASURE | below 1% |
| shmembench-cuda | 1.0293x | PARAM_TUNE | modest valid gain |
| allreduce-cuda | N/A | ENV_FIX | no kernel speedup claim |
| pingpong-cuda | N/A | MEASURE_FIX | baseline recovery |

### 5.2 外部基本測試

外部基本測試顯示，sorting 類 workload 改善較一致，graph 類則容易 failure 或 regression。Floyd-Warshall 的 0.107097 到 0.000024 極端加速在缺少 correctness 證據下被視為 suspicious/invalid。Environment-aware prompt 並未穩定改善結果，甚至在 `split-cuda` regression，表示單純提供 V100、`sm_70` 與 Slurm context 不等於建立有效審核流程。

| Benchmark | Baseline | Generic opt. | Env-aware opt. | Interpretation |
|---|---:|---:|---:|---|
| cc-cuda | 0.0034 | fail | fail | INVALID |
| floydwarshall-cuda | 0.107097 | 0.000024 | 0.00024 | suspicious / invalid |
| floydwarshall2-cuda | 0.000851 | 0.098891 | 0.09133 | regression |
| gc-cuda | 0.000048 | 0.000285 | fail | likely correctness risk |
| mis-cuda | 0.00136 | 0.002057 | fail | likely correctness risk |
| merge-cuda | 17.03105 | 13.7232 | 16.6688 | generic prompt better |
| quicksort-cuda | 46.1346 | 45.8452 | fail | measurement-equivalent / env fail |
| sortKV-cuda | 88.1414 | 72.9803 | 76.29895 | generic prompt better |
| bitonic-sort-cuda | 70.13246 | 33.50338 | 34.68863 | both improve, generic slightly better |
| split-cuda | 3423.724 | 3023.754 | 3569.766 | generic improves; env regresses |

### 5.3 剩餘 Phase 2 Benchmark

剩餘摘要呈現兩種模式。較接近傳統 kernel optimization 的結果包含 `adam-cuda` 約 2.0x、`adjacent-cuda` 約 1.99--2.0x、`randomAccess-cuda` 約 2.17--2.23x。相對地，`dropout`、`filter`、`minmax`、`nonzero`、`reverse`、`scan` 與部分 `topk` 摘要出現極大或零分母 speedup，例如 `dropout` 約 1.2e4x 到 2.2e5x 以上、`minmax` 約 8.8e7x 到 1.6e8x、`reverse` 約 1871x 到 2039x、`scan` 約 1e4 到 1e6x。根據描述推測，這些結果可能利用固定輸入、重複運算、重複 reverse 的 parity 或預先計算的 validation output，因此歸為 `BENCHMARK_AWARE_OPT`。此分類不是否定該能力，而是避免將其誤解為一般化 kernel 優化。

| Benchmark | P1 speedup | P2 speedup | P3 speedup | Interpretation |
|---|---:|---:|---:|---|
| adam-cuda | ~2.0x | ~2.0x | ~2.0x | KERNEL_OPT |
| adjacent-cuda | ~1.99x | ~2.0x | ~2.0x | KERNEL_OPT |
| randomAccess-cuda | ~2.17x | ~2.23x | ~2.21x | KERNEL_OPT |
| dropout-cuda | 1.9e4x--2.2e5x | 1.5e4x--2.2e5x | 1.2e4x--2.2e5x | BENCHMARK_AWARE |
| filter-cuda | 1.61x / 4.90x | 1.53x / 4.72x | 1.53x / 4.61x | BENCHMARK_AWARE |
| minmax-cuda | 8.8e7x--9.8e7x | 9.4e7x--1.3e8x | 1.0e8x--1.6e8x | BENCHMARK_AWARE |
| nonzero-cuda | timed sections -> 0 | timed sections -> 0 | timed sections -> 0 | BENCHMARK_AWARE |
| reverse-cuda | ~2038.60x | ~1976.99x | ~1871.34x | BENCHMARK_AWARE |
| scan-cuda | 1e4--1e6x | 1e4--1e6x | 1e4--1e6x | BENCHMARK_AWARE |
| topk-cuda | 147--5220x | 137--5190x | 137--5260x | BENCHMARK_AWARE, separate from audited P3 |

### 5.4 Human-guided Softmax

Phase 3 softmax 顯示 human review 的價值。Mode A 先確認 `impl=0` naive reference 與 `impl=1` existing optimized baseline 的差距不得算作 agent speedup。Mode B 的初始 candidate 同時改變 row parallelism 與 exponent caching；它在 large slices 有改善，但在 `slice=128` regression，且在 `slice=256` 有一次 correctness failure，因此只能算 partial success。人類審查拒絕 universal replacement，改為 shape-aware dispatch：small slices 保留 `impl=1`，large slices 使用新路徑。

| Slice | Mode B dispatch | Mode B baseline ms | Mode B candidate ms | Mode B speedup | Mode C additional result |
|---:|---|---:|---:|---:|---|
| 128 | impl=1 | 0.134869 | 0.134574 | 1.002197x | not claimed; measurement-equivalent |
| 256 | impl=1 | 0.321793 | 0.321505 | 1.000895x | not claimed; small-slice path retained |
| 784 | impl=2 | 1.442716 | 1.036402 | 1.392043x | 1.135540x vs impl=3 |
| 1024 | impl=2 | 2.104045 | 1.238443 | 1.698944x | 1.048740x vs impl=3 |
| 2048 | impl=2 | 2.237452 | 1.672904 | 1.337466x | measurement-equivalent vs impl=3 |

Mode C 的 primary comparison 是 `impl=4` vs accepted `impl=3`，不是 `impl=4` vs 原始 baseline。Mode C 在 784 與 1024 有 additional speedup；2048 則仍是 measurement-equivalent。Profiler 只顯示 dynamic shared memory 約下降 0.90 KB/block，缺少 stall 與 memory throughput，因此不能做因果宣稱。

Top-k 與 shmembench 也提供 supporting evidence。Top-k Mode A 顯示即使沒有有效修改，也可能因 baseline CV 過高產生 pseudo-speedup，因此 Mode B 前必須 robust baseline remeasurement。Shmembench 則顯示只有 `block_size=256` 是 official validated comparison，其餘 block size 應保留為 diagnostic failure，不可納入 official speedup。

## 6. 討論

結果顯示，LLM agent 對規律、高平行度且記憶體存取清楚的 workload 較可靠。Softmax、top-k、Adam、adjacent、randomAccess 與 sorting kernels 中，agent 能提出 workspace reuse、branch elimination、loop-invariant hoisting、incremental arithmetic 與 shape-aware dispatch 等合理策略。

相反地，irregular graph 與 convergence-based 程式風險較高。外部測試中的 `cc`、`gc`、`mis` 可能破壞 frontier propagation、atomic update 或 convergence logic。Floyd-Warshall 則說明極端加速更可能是 incomplete computation 或 invalid measurement，而非真實算法改善。

Prompt 約束對研究可信度影響極大。弱 prompt 能產生漂亮數字，但常缺 baseline、raw output 與 contradiction check。強 prompt 透過 correctness、CSV、variance 與 result type 限制過度宣稱。人類審查在 partial optimization 中尤其重要：softmax 案例顯示，人類能拒絕錯誤的 universal replacement，同時保留有效的 shape-dependent strategy。

本研究觀察到的失敗與警示類型如下：

| Pattern | Example | Likely cause |
|---|---|---|
| Correctness failure | cc, gc, mis | broken iteration, frontier, or atomic semantics |
| Implausible speedup | floydwarshall | skipped work or invalid measurement |
| Regression | floydwarshall2, split-env | bad shared memory/occupancy or memory-bound tuning |
| Measurement equivalent | p2p, quicksort | improvement below noise threshold |
| Environment repair | allreduce | launcher/UCX path, not kernel algorithm |
| Benchmark-aware shortcut | minmax, scan, reverse | fixed input or validation structure |
| Partial optimization | softmax round 1 | large-slice gain but small-slice/correctness failure |

## 7. 結論

LLM agent 可以優化 CUDA 程式，但必須置於嚴格審核流程中。可信 speedup 主要出現在規律 kernel 與經過審查的 softmax/topk 案例；失敗則集中於 irregular graph、極端可疑加速、測量噪音與 benchmark-aware shortcut。本文的核心結論是：agent optimization 不應被視為單次產生 patch，而應被視為可審核工作流。實測 baseline、correctness gate、repeated trials、result type classification 與 human review 是區分真實 kernel improvement、measurement artifact、environment fix 與 benchmark-specific shortcut 的必要條件。

## Artifact 位置

完整 input prompt、system-like rules、對應 output、agent changes 已整理於：

`/home/a/HeCBench_CUDA_Optimization/docs/agent_io_change_archive/benchmark_view/`

若原始資料夾沒有 `prompt.md`，對應 benchmark 的 `input_prompt.md` 會使用指定的預設硬體與 Slurm prompt。外部基本測試沒有 source diff、raw logs 或完整 agent response，相關欄位標為 N/A。
