# Phase 2 正式研究報告：Prompt 約束層級對 AI 輔助 CUDA 優化之影響

## 摘要

本研究比較 P1、P2、P3 三種 prompt 約束層級在 HeCBench CUDA benchmark 優化任務中的表現。P1 代表弱約束、接近日常對話式要求；P2 代表具備基本工程規格的中約束 prompt；P3 則是包含 baseline、correctness gate、CSV schema、variance/profiler、contradiction check 與 result-type classification 的強約束實驗 protocol。

已完成分析的 benchmark 包含 `allreduce-cuda`、`moe-align-cuda`、`moe-cuda`、`p2p-cuda`、`pingpong-cuda`、`prefetch-cuda`、`shmembench-cuda`、`simpleMultiDevice-cuda`、`softmax-cuda`、`topk-cuda`，共 30 筆 final-level 摘要。結果顯示：P1 有較高機率產生看似亮眼的效能數字，但資料完整性與可審核性不足；P2 能明顯改善 rejected attempt 紀錄與 baseline 對照；P3 不一定追求最高 speedup，但最能產出可重現、可審核、可分類的研究資料。

## 研究問題

- RQ2-1：prompt 約束強度是否影響 AI agent 的正確性、效能與可審核性？
- RQ2-2：強約束 prompt 是否能降低偽加速、錯誤報告與不可重現結果？
- RQ2-3：哪些 prompt 條款最關鍵？
- RQ2-4：prompt.md 是否比一般網頁對話更適合作為工程協作介面？

## 方法

本報告使用 `/home/a/PP/phase2/reports/phase2_level_summary.csv` 作為主要資料來源。效能指標依 benchmark 類型分為 latency/time 與 bandwidth/throughput：latency 類以 `baseline / final` 作為 improvement ratio；bandwidth 類以 `final / baseline` 作為 improvement ratio。對於 baseline invalid 或缺失的案例，不計算 speedup。

可審核性分數由六項構成：correctness 是否記錄、是否有 measured baseline、是否能計算 speedup、是否有 CSV source、是否有 variance/profiler 或 P3 protocol 紀錄、是否沒有 caution/weak auditability 標記。此分數不是效能分數，而是研究資料品質指標。

## 圖表

![Figure 1](formal_figures/figure1_speedup_by_level.svg)

![Figure 2](formal_figures/figure2_auditability_by_level.svg)

![Figure 3](formal_figures/figure3_result_type_distribution.svg)

## 統計摘要

| Level | N | Speedup mean | Speedup median | Numeric speedups | Mean auditability | CSV source count | Missing/invalid baseline | Measurement-equivalent/no-speedup |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| P1 | 10 | 1.552 | 1.128 | 6 | 46.0 | 0 | 4 | 1 |
| P2 | 10 | 1.369 | 1.117 | 10 | 65.0 | 0 | 0 | 4 |
| P3 | 10 | 1.131 | 1.097 | 8 | 93.0 | 10 | 2 | 4 |

## 主要發現

### Finding 1：P1 提升探索性，但也提高審核風險

P1 的平均 speedup 受到 `pingpong-cuda` NCCL 1GiB 約 2x 的結果拉高，但 P1 常缺少 CSV、variance、完整 baseline 或完整 case coverage。例如 `moe-align-cuda` 只有 final latency，無 measured baseline；`p2p-cuda` 僅有 2 GPU / 2 direction 結果，不能代表完整 4-GPU topology matrix。這表示 P1 適合探索可能方向，但不適合直接作為正式研究結論。

### Finding 2：P2 是工程可用性的最低門檻

P2 的報告普遍包含 baseline、accepted/rejected attempts 與 final decision。`moe-cuda` 清楚排除 correctness fail 或 timeout 的嘗試；`pingpong-cuda` 保留 invalid setup attempts 並把最後結果標示為 noise-level/measurement-equivalent。這使 P2 已經能支撐基本工程報告，但仍缺少 P3 所要求的統一 CSV、variance/profiler 與 contradiction check。

### Finding 3：P3 強化研究可審核性，且能抑制過度宣稱

P3 的平均可審核性最高。`allreduce-cuda` 在 P3 中被明確分類為 `ENV_FIX`，不宣稱 kernel speedup；`pingpong-cuda` 因 baseline NCCL executable missing 而不計算 speedup，只回報 measurement recovery；`p2p-cuda` 則將 full directed topology sweep 標記為 `MEASURE_FIX` 與 measurement-equivalent。這些案例顯示 P3 的核心價值是避免把環境修復、測量修復或 topology coverage 誤寫成實際效能優化。

### Finding 4：Prompt 條款中最關鍵的是 correctness gate、baseline、CSV schema 與 contradiction check

結果顯示，單純要求「保持 correctness」不足以產生可審核資料；必須進一步要求 measured baseline、raw output、CSV schema、accepted/rejected 分類、variance/trials 與 contradiction check。尤其在 `pingpong-cuda`、`allreduce-cuda` 這類環境與 launcher 敏感的 benchmark 中，result type classification 是避免錯誤結論的關鍵。

## Benchmark 細部結論

### allreduce-cuda
- P1: speedup `1.1635`，status `valid`，type `KERNEL_OPT/ENV_FIX`。P1 replaced measured allreduce path with NCCL; strong speedup on many sizes, largest-size speedup modest.
- P2: speedup `2.7280`，status `valid`，type `KERNEL_OPT`。Geomean speedup 2.728x; largest buffer is measurement-equivalent/slightly slower (120294 -> 120423 us).
- P3: speedup `n/a`，status `success_no_speedup_claim`，type `ENV_FIX`。P3 correctly treats this as launcher/environment repair with three reproducibility runs.

### moe-align-cuda
- P1: speedup `n/a`，status `valid_no_baseline`，type `PARAM_TUNE`。Summary reports final mean but no measured baseline, so speedup cannot be audited.
- P2: speedup `1.2237`，status `valid`，type `PARAM_TUNE`。Cached cumsum workspace; rejected slower variants documented.
- P3: speedup `1.1504`，status `valid_with_variance_profiler`，type `PARAM_TUNE`。Three accepted trials plus profiler notes; rejected regression excluded.

### moe-cuda
- P1: speedup `1.0927`，status `valid`，type `KERNEL_OPT`。Fused softmax+topk; P1 summary lacks variance/profiler but reports baseline and final.
- P2: speedup `1.0339`，status `valid`，type `KERNEL_OPT`。Hybrid path gives real topk=1 gain but mean speedup only 3.39%.
- P3: speedup `1.0778`，status `valid_with_variance_profiler`，type `KERNEL_OPT`。topk 1/2/4 improve; topk=8 explicitly classified measurement-equivalent.

### p2p-cuda
- P1: speedup `n/a`，status `weak_auditability`，type `MEASURE_FIX`。No summary.md; final file reports only 2 GPUs/2 directions, not the full 4-GPU topology matrix.
- P2: speedup `1.0021`，status `measurement_equivalent`，type `MEASURE_FIX`。Directional sweep improves auditability; speedup below 1%.
- P3: speedup `1.0022`，status `measurement_equivalent`，type `TOPOLOGY_MEASURE/MEASURE_FIX`。Full directed 4-GPU topology coverage; performance change below 1%.

### pingpong-cuda
- P1: speedup `1.9990`，status `valid_with_caution`，type `KERNEL_OPT/MEASURE_FIX`。NCCL grouping doubles reported 1GiB bandwidth; P1 lacks CSV/variance and has earlier invalid attempts.
- P2: speedup `1.0000`，status `measurement_equivalent`，type `PARAM_TUNE/MEASURE_FIX`。Five optimizations tried; final improvement is noise-level but invalid setup attempts are clearly rejected.
- P3: speedup `n/a`，status `success_no_speedup_claim`，type `MEASURE_FIX`。Baseline NCCL executable missing; P3 correctly reports measurement recovery and avoids speedup claim.

### prefetch-cuda
- P1: speedup `n/a`，status `valid_no_baseline`，type `MEASURE_FIX/PARAM_TUNE`。Only one P1 result file was available; final timing can be summarized but speedup cannot be audited against measured baseline.
- P2: speedup `1.6234`，status `valid`，type `PARAM_TUNE/MEASURE_FIX`。Separates prefetch setup from timed kernel execution; primary repeat=100 with_prefetch improves while no-prefetch also improves.
- P3: speedup `1.1163`，status `valid_with_variance_profiler`，type `PARAM_TUNE`。No-prefetch block-size tuning improves demand-paging path; repeat=100 with_prefetch is explicitly measurement-equivalent.

### shmembench-cuda
- P1: speedup `1.0280`，status `valid_with_caution`，type `KERNEL_OPT`。Best valid raw run improved modestly; one faster P1 attempt had checksum failure and is excluded.
- P2: speedup `1.0013`，status `measurement_equivalent`，type `KERNEL_OPT`。Valid final optimization is only about 0.13% faster; checksum-failing faster attempt is rejected.
- P3: speedup `1.0293`，status `valid_with_variance`，type `KERNEL_OPT`。Removing unneeded synchronization gives a modest 2.85% time improvement; checksum-failing block-size sweep is rejected.

### simpleMultiDevice-cuda
- P1: speedup `n/a`，status `success_no_speedup_claim`，type `MEASURE_FIX/KERNEL_OPT`。Raw P1 logs show a dramatic total_us drop, but H2D/D2H timing scope appears changed; no speedup claim is counted.
- P2: speedup `1.0108`，status `measurement_equivalent`，type `KERNEL_OPT`。Block-level reduction improves kernel/D2H components, but total time remains H2D-copy-limited with about 1% speedup.
- P3: speedup `1.0121`，status `measurement_equivalent`，type `KERNEL_OPT`。Final kernel optimization is real but total-time speedup is only about 1.2% because H2D copy dominates.

### softmax-cuda
- P1: speedup `1.0330`，status `valid_with_caution`，type `KERNEL_OPT`。Raw logs show slice=784 impl=1 improved, but P1 has no structured summary, rejected-attempt table, or variance.
- P2: speedup `1.8704`，status `valid`，type `KERNEL_OPT`。Manual warp reductions plus slice=784 specialization produce a clear implementation-1 speedup; slower attempts are documented.
- P3: speedup `1.4575`，status `valid_with_variance_profiler`，type `KERNEL_OPT`。Block-per-slice kernel improves large-slice implementation 1; final result includes 3 trials and contradiction check.

### topk-cuda
- P1: speedup `2.9936`，status `valid_with_caution`，type `KERNEL_OPT`。Raw outputs report PASS and lower mean than first run, but P1 lacks accepted/rejected rationale and variance.
- P2: speedup `1.1991`，status `valid`，type `KERNEL_OPT`。Cached radix workspace removes timed cudaMalloc/cudaFree overhead; rejected block-size variants are recorded.
- P3: speedup `1.1995`，status `valid_with_variance`，type `KERNEL_OPT`。Hybrid workspace/block-size strategy improves mean top-k time with low trial variance; rejected block512 regression excluded.

## 後續優化建議

- `allreduce-cuda`：後續應分離 launcher/environment repair 與 collective algorithm/kernel optimization，並固定 NCCL、MPI、GPU topology 與 buffer-size sweep 條件。若最大 size 未改善，報告應避免只引用 geomean。
- `moe-align-cuda`：目前最穩定的方向是 workspace/cache 與參數調校。下一輪可加入不同 token/expert 分佈，檢查優化是否只對單一 workload 有效。
- `moe-cuda`：P1/P2/P3 都顯示 topk=1/2/4 較有優化空間，topk=8 可能接近 memory 或 algorithm bottleneck。建議分別記錄 per-topk speedup，不只看 arithmetic mean。
- `p2p-cuda`：效能變化接近 measurement-equivalent，研究價值主要在完整 topology coverage。後續應將 GPU pair、方向、NUMA/PCIe/NVLink 資訊納入輸出欄位。
- `pingpong-cuda`：P1 的高 speedup 需要以完整 baseline 與重複試驗重新驗證。下一輪應固定 MPI/NCCL executable、message-size sweep、warmup、iteration count，並把 invalid setup 與 performance result 分開統計。
- `softmax-cuda`：P2/P3 證明大 slice 的 implementation 1 可由 block-per-slice reduction 受益。後續應分 slice 分組報告，避免小 slice measurement-equivalent 掩蓋大 slice speedup。
- `topk-cuda`：workspace reuse 是穩定收益來源；後續可把 allocation time、kernel time 與 workspace size 分欄，確認 speedup 來自 timed-loop allocation removal 還是真正 kernel improvement。
- `prefetch-cuda`：P2/P3 指向不同 primary metric，後續應明確指定 with_prefetch 與 without_prefetch 的主結論是否分開，並把 prefetch API cost 與 steady-state kernel time 分開統計。
- `simpleMultiDevice-cuda`：總時間受 H2D copy 主導，kernel speedup 會被 total_us 稀釋。後續應同時報告 total/h2d/kernel/d2h，並禁止將改變測量範圍的結果列入 speedup。
- `shmembench-cuda`：同步移除可帶來小幅但穩定改善；後續需補 Nsight Compute 的 bank conflict、occupancy 與 instruction mix，並嚴格排除 checksum failed 的較快結果。

## Prompt 改善建議

- 保留 P3 作為正式實驗主 prompt，並把 `result_type` 設為必填欄位，例如 `KERNEL_OPT`、`PARAM_TUNE`、`ENV_FIX`、`MEASURE_FIX`、`TOPOLOGY_MEASURE`、`NO_VALID_SPEEDUP`。
- 要求每個 benchmark 都輸出同一組檔案：raw log、baseline CSV、final CSV、accepted/rejected attempts、summary.md。缺任一檔案時，必須在 summary 中標記 `incomplete_audit_trail`。
- 在 prompt 中明確禁止只回報最佳單點數字；必須同時回報 full sweep、mean/median、重複次數、是否有 outlier，以及 largest-size 或主要 workload 的單獨結果。
- 加入 contradiction check：若 correctness fail、baseline invalid、缺少 baseline、或 metric direction 不一致，agent 必須輸出 `no_speedup_claim`，不能宣稱優化成功。
- 對 P1/P2 對照組可保留較少限制，但仍建議最低限度加入運行指令、correctness command、benchmark command 與 output path，避免資料無法重現。

## 威脅與限制

- 目前摘要表已涵蓋 10 個 benchmark 的 P1/P2/P3 三層分析；後續限制主要來自各 benchmark metric 不同與部分 P1 資料缺少結構化紀錄。
- 不同 benchmark 的 metric 單位不同，跨 benchmark 的平均 speedup 只能作為方向性指標，不應視為統一排行榜。
- P1 缺少結構化資料，部分數值需由 summary 或 raw log 解析，存在較高整理誤差風險。
- 部分 P3 結果刻意不計算 speedup，因其任務本質是環境修復或測量恢復；這會降低表面效能平均，但提高研究可信度。

## 研究結論

Phase 2 結果支持以下結論：prompt 約束層級不只影響 agent 是否能優化程式，也深刻影響結果是否可驗證、可重現與可用於研究。P1 適合探索，P2 適合工程協作，P3 最適合作為正式實驗 protocol。若目標是撰寫研究報告或比較 AI agent 優化能力，建議以 P3 作為主實驗條件，並將 P1/P2 作為對照組，用來量化 prompt 約束不足時產生的偽加速、缺 baseline、缺 raw output 與過度宣稱問題。
