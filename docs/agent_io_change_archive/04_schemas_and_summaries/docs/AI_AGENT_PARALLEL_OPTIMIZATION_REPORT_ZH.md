# 以 AI Agent 分析與優化平行程式之研究報告

本報告根據 `/home/a/HeCBench_CUDA_Optimization` 專案之完整掃描結果撰寫，資料來源包含 `BASIC/` 早期探索實驗、`phase2/` 的 P1/P2/P3 prompt 約束層級實驗、`phase3/` 的 agent-only、human-in-the-loop 與 literature/profiler 輔助流程、`evaluation_summary/` 中的結構化彙整表、`/home/a/data.md` 所提供之另一組 HeCBench 外部基本測試摘要，以及 `/home/a/rest.md` 所提供之 10 個剩餘 Phase2 benchmark 大幅優化摘要。本研究主題不是單純衡量 AI 是否能使 CUDA 程式加速，而是分析 AI Agent 在平行程式優化中如何產生可驗證、可重現、可審核的效能改善，並辨識偽加速、錯誤 baseline、correctness 缺失、benchmark-aware shortcut 與測量噪音等失效模式。

## METHODOLOGY

本研究採用「多階段、強審核、結果分類」的方法論。第一層是 benchmark 層級的程式分析，第二層是 prompt 約束層級比較，第三層是人機協作工作流評估。整體研究問題可整理為三項：第一，AI Agent 在不同類型 CUDA benchmark 上能否產生實質效能改善；第二，prompt 約束是否會改變結果的可信度與可審核性；第三，人類操作者、profiler 與文獻查詢是否能把 agent 從 trial-and-error 推進到 evidence-driven optimization。

研究將 HeCBench CUDA 子集分為三類。第一類為 AI primitive / kernel optimization，包括 `softmax-cuda`、`topk-cuda`、`moe-cuda` 與 `moe-align-cuda`，重點在 CUDA kernel、reduction、workspace reuse、MoE dispatch 與 shape-aware policy。第二類為 memory-system / measurement benchmark，包括 `prefetch-cuda`、`shmembench-cuda` 與 `p2p-cuda`，重點在 Unified Memory、shared memory、peer-to-peer topology 與測量邊界。第三類為 multi-GPU / communication / environment，包括 `allreduce-cuda`、`pingpong-cuda` 與 `simpleMultiDevice-cuda`，重點常落在 MPI/NCCL/UCX launcher、GPU 配置、通訊路徑與資料傳輸瓶頸，而不必然是 kernel-level optimization。

Prompt 方法論分成 P1、P2、P3 三種約束強度。P1 是弱約束 prompt，只提供 benchmark path 與基本優化目標，不強制 baseline、CSV、raw output 或 contradiction check。P2 是工程型 prompt，要求先實測 baseline、保留 raw output、記錄 accepted/rejected attempts，並產出 `agent_summary.md`。P3 是正式實驗 protocol，額外要求 correctness gate、三次或多次 repeated trials、統一 CSV schema、variance/profiler notes、result type classification 與 contradiction check。此設計的核心假設是：prompt 不只是自然語言指令，而是定義研究資料有效性的 execution policy。

本研究在結果判定上採用嚴格分類，而不是只看 speedup。結果類型包含 `KERNEL_OPT`、`PARAM_TUNE`、`ENV_FIX`、`MEASURE_FIX`、`TOPOLOGY_MEASURE`、`MULTI_GPU_SCALING`、`MEASUREMENT_EQUIVALENT`、`REGRESSION` 與 `INVALID`。若 correctness 失敗、baseline 缺失、測量範圍改變、metric direction 不一致，或 improvement 小於 1% 且缺少統計支持，則不得宣稱有效 speedup。這個分類制度避免將環境修復、測量補全或拓撲掃描誤寫成 kernel optimization。

Phase 3 進一步將方法論擴展為人機協作流程。該階段選取低、中、高三種優化空間代表：`shmembench-cuda`、`topk-cuda`、`softmax-cuda`。Mode A 是 P3 agent-only baseline，用於揭露測量穩定性與 agent-only 偽加速風險。Mode B 是 human-in-the-loop guided optimization，要求 robust baseline、human checkpoint、decision log 與 final confirmation。Mode C 是 literature + profiler augmented adaptive workflow，要求觀察、診斷、文獻或文件查詢、提出最小修改、執行 Slurm、驗證 correctness/variance/profiler，再決定 accept/reject/rollback/stop。

外部基本測試資料作為補充案例納入，但證據等級低於 Phase 2/3 的結構化資料。該資料涵蓋 `cc-cuda`、`floydwarshall-cuda`、`floydwarshall2-cuda`、`gc-cuda`、`mis-cuda`、`merge-cuda`、`quicksort-cuda`、`sortKV-cuda`、`bitonic-sort-cuda` 與 `split-cuda`，並提供 generic optimized 與 environment-aware optimized 兩種 prompt 的摘要結果。由於該資料未附 raw output、完整 agent response、程式 diff 或 patch summary，本報告將其定位為 auditability-limited external evidence；缺失欄位均標記為 `N/A`，且極端加速或 correctness fail 不納入強 speedup claim。

`/home/a/rest.md` 則作為另一組 Phase2 補充資料納入，涵蓋 `adam-cuda`、`adjacent-cuda`、`dropout-cuda`、`filter-cuda`、`minmax-cuda`、`nonzero-cuda`、`randomAccess-cuda`、`reverse-cuda`、`scan-cuda` 與 `topk-cuda`。該資料的重要性在於，它同時呈現了較可信的 conventional kernel optimization 與另一類 benchmark-aware optimization。前者例如 `adam-cuda`、`adjacent-cuda`、`randomAccess-cuda`，其 P1/P2/P3 speedup 約維持在 2x 附近；後者例如 `minmax-cuda`、`nonzero-cuda`、`scan-cuda`、`reverse-cuda` 與部分 `topk-cuda`，可能利用固定輸入、重複運算、host-side 已知資訊或 validation structure。此資料已在 archive 中整理為 `04_schemas_and_summaries/phase2/rest_large_optimization/`，並同時連結到 `benchmark_view/<benchmark>/` 的 input/output/change 對照檔。

## EXPERIMENTAL SETUP

專案實驗範圍涵蓋 10 個標準化 CUDA benchmark，並保留早期 BASIC 探索、Phase 2 正規化結果與 Phase 3 深入案例。主要結構化資料來源包含 `phase2/reports/phase2_level_summary.csv`、`evaluation_summary/data/benchmark_summary_used.csv`、`evaluation_summary/data/invalid_results.csv`、`evaluation_summary/data/contradiction_check.csv`、`phase3/metadata/result_schema.csv`、`phase3/metadata/official_sweeps.yaml`，以及各 benchmark 的 `results.csv`、`agent_summary.md`、`decision_log.md` 與 raw Slurm output。另納入 `/home/a/data.md` 作為外部基本測試摘要來源，並納入 `/home/a/rest.md` 作為剩餘 Phase2 benchmark 的 compact summary；兩者資料型態皆偏向人工整理表格與文字分析，而非可機器重建的完整實驗軌跡，因此證據等級低於主專案 P3 與 Phase 3 final confirmation。

Phase 2 的正式比較包含 P1、P2、P3 三層，每層 10 筆 final-level 摘要，共 30 筆核心資料。效能指標依 benchmark 性質而定：latency/time 類使用 `baseline / final` 作為 speedup；bandwidth/throughput 類使用 `final / baseline`；baseline invalid 或缺失時不計算 speedup。可審核性則依 correctness 是否記錄、baseline 是否實測、speedup 是否可計算、是否有 CSV source、是否有 variance/profiler 紀錄、是否有 caution/weak auditability 標記等因素判定。

Phase 3 的硬體與軟體環境以 Slurm 叢集執行為主。已記錄的 GPU 為 NVIDIA Tesla V100-SXM2-32GB，CUDA 版本為 12.8 / V12.8.61，profiling 工具包含 Nsight Compute CLI 2025.1.0。所有正式 GPU benchmark 均透過 Slurm 執行，不在 login node 直接跑 benchmark binary。Mode A 使用 job 949514、949515、949516；Mode B robust baseline 使用 softmax job 949640、topk job 949641、shmembench job 949642；softmax Mode B final confirmation 使用 job 949717；softmax Mode C final confirmation 使用 job 950691，final profiler analysis 使用 job 950695。

外部基本測試使用的 prompt 有兩種。第一種 generic optimized prompt 為「請幫我優化這份程式碼，需輸出相同的資料，請給我完整的程式碼。」第二種 env-optimized prompt 為「以下是我的環境，請再進行優化，給我完整的程式碼」，並指定 GPU 為 NVIDIA Tesla V100-SXM2-32GB、CUDA arch 為 `sm_70`、Scheduler 為 Slurm。依使用者補充，本報告視其硬體執行環境與主專案一致；但 system prompt、raw output、agent 修改 diff、trial count、variance 與 correctness 詳細輸出均為 `N/A`。

`rest.md` 中的 Phase2 補充資料也採同一硬體執行環境假設。其記錄欄位包含 baseline、optimized、P1/P2/P3 speedup、changes、correctness 與 result type，但缺少 system prompt、完整 raw stdout/stderr、source diff、trial count 與 variance。Archive 中已將這些資料整理成 `rest_phase2_summary.csv` 與中文整理檔；若某欄位在來源中未提供，均標示為 `N/A` 或在 interpretive note 中說明。

Phase 3 的 official sweep 明確鎖定比較語意。`softmax-cuda` 以既有 optimized baseline `impl=1` 為正式 baseline，官方 slice 為 128、256、784、1024、2048，並禁止將 `impl=0 -> impl=1` 視為 agent speedup。`topk-cuda` 的官方 cases 為 hidden size 3072、4096、8192、16384、32768、65536、131072 與 topk 1024、2048 的組合，且因 Mode A 暴露高 CV 偽加速，Mode B 必須先建立 robust baseline。`shmembench-cuda` 只將 `variant=original, block_size=256` 視為官方 validated comparison；block size 128、512、1024 均保留為 diagnostic sweep，不納入 speedup claim。

實驗資料的有效性規則如下。Correctness 非 PASS 者不得納入 speedup；若 baseline invalid 或缺失，結果只能標為 no speedup claim；若修改 input size、correctness tolerance 或 timing scope，該結果不得與原 benchmark 直接比較；若 speedup 小於 1%，預設標為 measurement-equivalent；若 CV 超過 15%，需標為 CAUTION/NOISY 並限制 speedup claim；若 CV 超過 30%，需重測。這些規則由 `phase3/tools/self_consistency_auditor.py` 與各階段 contradiction check 輔助執行。

## RESULT

Phase 2 的總體結果顯示，prompt 約束強度與資料可信度呈正相關，但與表面 speedup 不呈單調正相關。P1 的可計算 speedup 平均約 1.552x，P2 約 1.369x，P3 約 1.131x；然而 P1 的 CSV source count 為 0，且至少 4 個結果因 baseline 缺失、資料殘缺或測量範圍改變而無效。P3 的 CSV source count 為 10，平均可審核性最高，並將 `allreduce-cuda` 與 `pingpong-cuda` 正確標記為 baseline invalid 或 environment/measurement repair，不宣稱 kernel speedup。

P1 的主要價值是探索性，但風險是偽加速與不可審核。`topk-cuda` P1 表面 speedup 達 2.9936x，`pingpong-cuda` P1 在 1 GiB NCCL case 表面達 1.9990x，但缺少完整 CSV、variance、accepted/rejected rationale 或完整 baseline，不能作為強論文證據。`moe-align-cuda`、`p2p-cuda`、`prefetch-cuda` 與 `simpleMultiDevice-cuda` 的 P1 結果均因 baseline 缺失或測量範圍問題被列入 invalid 或 no speedup claim。這說明弱 prompt 容易產生看似亮眼但審核成本高的結果。

P2 是工程可用性的最低門檻。它普遍具備 baseline、最終結果與 rejected attempts 記錄。例如 `moe-cuda` 在 P2 中能排除 correctness fail 或 timeout 的嘗試，最終保留 1.0339x 的有效結果；`topk-cuda` P2 透過 CUB radix workspace reuse 移除 timed `cudaMalloc/cudaFree` 開銷，得到 1.1991x；`prefetch-cuda` P2 將 prefetch setup 與 timed kernel execution 分離，得到 1.6234x。但 P2 仍缺少 P3 等級的統一 CSV、三次 trials、profiler limitation 與 contradiction check，因此仍可能將 `allreduce-cuda` 的 launcher/environment repair 誤分類為 `KERNEL_OPT`。

P3 的主要貢獻是抑制過度宣稱。`softmax-cuda` P3 在 slice=784 implementation 1 上取得 1.4575x，且 correctness 為 PASS 42/42；`topk-cuda` P3 在 14 組 hidden size/topk cases 的三次 final trials 上取得 1.1995x；`moe-cuda` P3 取得 1.0778x，並明確指出 topk=8 屬 measurement-equivalent；`moe-align-cuda` P3 取得 1.1504x，但 correctness 狀態在比較 CSV 中不是完全明確，因此報告保留限制；`p2p-cuda` P3 僅 1.0022x，因低於 1% 被標為 measurement-equivalent 與 topology/measurement result；`simpleMultiDevice-cuda` P3 為 1.0121x，受 H2D copy 主導，只能視為邊際改善；`shmembench-cuda` P3 為 1.0293x，屬小幅但可量測的同步/參數改善。

`/home/a/rest.md` 的剩餘 Phase2 benchmark 結果擴充了本研究對 prompt 強度與結果類型的判讀。其 P1/P2/P3 speedup 對照如下：

| Benchmark | P1 speedup | P2 speedup | P3 speedup | Result type |
|---|---:|---:|---:|---|
| `adam-cuda` | ~2.0x | ~2.0x | ~2.0x | `KERNEL_OPT` |
| `adjacent-cuda` | ~1.99x | ~2.0x | ~2.0x | `KERNEL_OPT` |
| `randomAccess-cuda` | ~2.17x | ~2.23x | ~2.21x | `KERNEL_OPT` |
| `dropout-cuda` | VEC1 ~19061x; VEC2 ~219401x; VEC4 ~220154x | VEC1 ~14799x; VEC2 ~216596x; VEC4 ~217776x | VEC1 ~12296x; VEC2 ~219019x; VEC4 ~220036x | `BENCHMARK_AWARE_OPT` |
| `filter-cuda` | shared ~1.61x; global ~4.90x | shared ~1.53x; global ~4.72x | shared ~1.53x; global ~4.61x | `BENCHMARK_AWARE_OPT` |
| `minmax-cuda` | min+max ~8.83e7x; minmax ~9.77e7x | min+max ~1.26e8x; minmax ~9.39e7x | min+max ~1.59e8x; minmax ~1.01e8x | `BENCHMARK_AWARE_OPT` |
| `nonzero-cuda` | timed GPU sections -> 0 | timed GPU sections -> 0 | timed GPU sections -> 0 | `BENCHMARK_AWARE_OPT` |
| `reverse-cuda` | ~2038.60x | ~1976.99x | ~1871.34x | `BENCHMARK_AWARE_OPT` |
| `scan-cuda` | ~10^4--10^6x range | ~10^4--10^6x range | ~10^4--10^6x range | `BENCHMARK_AWARE_OPT` |
| `topk-cuda` | ~147x--5220x | ~137x--5190x | ~137x--5260x | `BENCHMARK_AWARE_OPT` |

此表顯示兩個重要現象。第一，`adam-cuda`、`adjacent-cuda`、`randomAccess-cuda` 在三種 prompt level 下皆維持接近 2x 的穩定改善，且 changes 對應到傳統 CUDA 最佳化：register reuse、`powf` incremental replacement、branch removal、kernel fusion 或多 block parallelization。這些結果較能作為 conventional kernel optimization 的補充證據。第二，`dropout-cuda`、`minmax-cuda`、`nonzero-cuda`、`reverse-cuda`、`scan-cuda` 與 `topk-cuda` 的極大 speedup 在 P1/P2/P3 中持續存在，顯示強 prompt 本身不必然阻止 benchmark-aware strategy。這些修改雖可能保持 benchmark PASS，但根據來源描述，它們可能利用固定輸入、host-side 已知答案、重複操作 parity 或 validation path，因此應被標為 `BENCHMARK_AWARE_OPT`，不可與一般 kernel-level speedup 混合平均。

`rest.md` 中的 agent changes 也補強了本研究對 AI agent 能力邊界的推測。Agent 不只會提出 CUDA idiom，例如 hoisting、template specialization、kernel fusion、parallel update，也會辨識 benchmark harness 的可利用結構，例如 CPU reference 已存在、輸入排列 deterministic、重複運算結果可由 parity 推得，或 timed GPU section 可被縮小到接近零。這種能力本身具有研究價值，但其科學意義不同：它更接近 benchmark-level program transformation 或 harness exploitation，而不是可外推至未知輸入的一般 parallel kernel optimization。

跨 benchmark 的結果分類顯示，真正可歸為實質 kernel optimization 的主要是 `softmax-cuda`、`topk-cuda` 與 `moe-cuda`。`moe-align-cuda` 與 `prefetch-cuda` 更接近 parameter/strategy tuning。`p2p-cuda` 的研究價值在完整 4-GPU directed topology sweep，而不是顯著加速。`allreduce-cuda` 的主要成果是避開 broken GDRCopy/UCX path 的 launcher 修復，使非零 size correctness PASS，屬 `ENV_FIX`。`pingpong-cuda` 是 MPI/NCCL transport comparison / measurement recovery；即使 tuned CUDA-aware MPI 在 two-rank ping-pong 1 GiB case 約 24.248 GB/s，高於 NCCL 約 22.898 GB/s，該結論也不能外推為 NCCL 在 collective workload 中較差。

早期 BASIC 結果提供了上界式探索，但不能與 Phase 2 normalized result 直接混用。`softmax-cuda` BASIC/GM 在 slice=784 曾達 59.593x，這來自針對 naive baseline 與特定 shape 的重寫；而 Phase 2 P3 的 1.4575x 是相對既有 optimized baseline 並受正規化比較約束。`topk-cuda` BASIC/GM 在 14 組 hidden size/topk 上達 1.442x，CG 版本約 1.326x；但 Phase 2 P3 的 1.1995x 更適合作為論文主數據，因為其 trial、correctness 與 schema 更完整。這些差異說明，最高單點 speedup 與正式研究證據是兩種不同概念。

Phase 3 Mode A 揭露了 agent-only 流程的主要風險：即使不修改原始碼，也可能因測量變異產生表面 speedup。`softmax-cuda` Mode A 顯示既有 optimized baseline `impl=1` 穩定，不能將 `impl=0 -> impl=1` 的約 37x 差距視為 agent optimization。`topk-cuda` Mode A 在 `hidden_size=4096, topk=2048` 觀察到 1.415x 的表面加速，但 baseline CV 高達 54.6%，屬 noisy measurement / pseudo-speedup；`hidden_size=8192, topk=2048` 類似出現 1.440x 表面加速且 baseline CV 43.2%。`shmembench-cuda` Mode A 則顯示只有 block_size=256 通過 official correctness；128 與 512 雖有 bandwidth 數字但 checksum FAIL，1024 因 shared memory exceeds limit 編譯失敗，均只能作為 diagnostic failure。

Phase 3 Mode B 的 softmax 實驗展示了人機協作如何把局部失敗轉化為系統性策略。Round 1 的 `impl2_block_cached_exp_compound` 對大 slice 有效，但在 slice=256 有一次 correctness FAIL，且 slice=128 退步，因此未被提升為完整替代方案。人類 checkpoint 將問題歸納為 shape-dependent behavior，Round 2 改採 `impl3_shape_dispatch_impl1_small_impl2_large`：slice 128/256 使用原 `impl=1`，slice 784/1024/2048 使用大 slice path。Final confirmation 結果為：slice 128 speedup 1.0022x、slice 256 1.0009x，均標為 measurement-equivalent；slice 784 為 1.3920x，slice 1024 為 1.6989x，slice 2048 為 1.3375x，三者 correctness PASS 且 speedup claim valid。這是人機協作的關鍵證據：人類不是取代 agent 寫 kernel，而是限制錯誤宣稱並引導 agent 建立 shape-aware dispatch。

Phase 3 Mode B 的 topk 與 shmembench 主要完成 robust baseline，而非正式優化。`topk-cuda` 在 14 組 official cases 執行 7 次 trials，correctness 全部 PASS，12 個 cases 為 VALID、2 個 cases 為 CAUTION，0 個 NOISY；這修正了 Mode A 中高 CV 偽加速的風險。`shmembench-cuda` Mode B robust baseline 確認 block_size=256 official case PASS，平均 bandwidth 約 13226.78 GB/s，CV 約 0.42%；block_size=128 與 512 checksum FAIL，block_size=1024 build failed，因此被保留為 diagnostic failures 並排除於 official speedup claim。

Phase 3 Mode C 在 `softmax-cuda` 上提供了 evidence-guided aggressive optimization 的進一步證據。Mode C 最終 candidate 為 `impl4_shape_specialized_large_reduce`，主要比較基準是 Mode B accepted `impl=3`。Final confirmation 顯示 slice 784 從 impl3 平均 1.031688 ms 降至 impl4 0.908544 ms，additional speedup 為 1.135540x；slice 1024 從 1.240982 ms 降至 1.183307 ms，additional speedup 為 1.048740x。slice 128、256 與 2048 不接受為 additional speedup claim，其中 2048 僅 1.008239x，低於 1% claim gate，標為 measurement-equivalent。Mode C final label 為 `SUCCESS_WITH_ADDITIONAL_SPEEDUP`，final confirmation status 為 `CONFIRMED`。

Mode C 的 profiler 結果應保守解讀。Nsight Compute 只收集 softmax large slices 784、1024、2048 的 resource-level diagnostics；official timing 不採用 profiler timing。可支持的觀察是：impl4 相較 impl3 在三個 large slices 上均降低約 0.90 KB/block dynamic shared memory，registers/thread 維持 18，waves/SM 在 784/1024 維持 156.25，在 2048 維持 78.12。缺失的 profiler 指標包含 memory throughput、warp execution efficiency、instruction mix、math/special-function 與 stall/scheduler breakdown。因此 profiler 結論只能標為 `LIMITED_PROFILER_EVIDENCE`，不能宣稱 dynamic shared memory reduction、reduction structure 或 cached-exp 是 causal mechanism。

外部基本測試進一步補充了 10 個其餘 HeCBench CUDA 項目的低審核性觀察。其 reported timing 顯示：`bitonic-sort-cuda` 從 70.13246 降至 33.50338，約 2.09x；`sortKV-cuda` 從 88.1414 降至 72.9803，約 1.21x；`merge-cuda` 從 17.03105 降至 13.7232，約 1.24x；`split-cuda` generic optimized 從 3423.724 降至 3023.754，約 1.13x。這些結果與既有研究發現相符：規則型 sorting / split / merge 類 workload 因 memory access pattern 較規律、同步語意較清楚，較可能被 AI 透過 block size、memory coalescing 或 global memory traffic 調整取得改善。不過，由於該資料未提供 raw correctness output 與程式 diff，上述 speedup 只能視為補充性 evidence，而非與 Phase 2 P3 同等級的正式數據。

同一外部測試也強化了「不規則與迭代式演算法高風險」的結論。`cc-cuda` optimized 與 env-optimized 均 fail；`gc-cuda` 與 `mis-cuda` 雖有 optimized timing，但來源分析指出 graph 類演算法在優化後出現 FAIL，可能破壞 iteration/convergence、frontier propagation、atomic update 或同步語意。`quicksort-cuda` generic optimized 只從 46.1346 到 45.8452，約 1.006x，接近 measurement-equivalent，而 env-optimized fail。`floydwarshall-cuda` 從 0.107097 到 0.000024 的約 4462x 表面加速在 `O(N^3)` 計算複雜度下極不合理，來源亦判定可能為 kernel 未完整執行、iteration 未完成、記憶體錯誤或提前終止；因此應歸為 suspicious/invalid，而非真實最佳化。`floydwarshall2-cuda` 則從 0.000851 退化至 0.098891，約慢 116x，env-optimized 亦退化至 0.09133。這些結果支持本研究主張：AI agent 的失敗常不是單純效能不佳，而是演算法語意、同步與 correctness 被破壞。

外部 env-optimized 結果顯示，硬體資訊本身不保證優化品質。指定 V100 `sm_70` 與 Slurm 後，`bitonic-sort-cuda` 仍約 2.02x、`sortKV-cuda` 約 1.16x、`merge-cuda` 約 1.02x，但 `split-cuda` 從 optimized 的改善轉為 env-optimized 退化，`cc-cuda`、`gc-cuda`、`mis-cuda`、`quicksort-cuda` 的 env-optimized 版本出現 fail。這與 Phase 2/3 的觀察一致：若 prompt 只提供硬體環境而沒有 correctness gate、baseline policy、variance filter 與 result type classification，agent 仍可能過度調整 occupancy、誤用 shared memory、忽略 memory-bound 特性，或破壞不規則演算法的同步語意。

整體而言，本專案最重要的結果不是單一最高 speedup，而是建立了 AI-assisted parallel program optimization 的審核框架。P1 證明 AI 能快速探索，但也容易產生不可審核或偽加速結果；P2 提供工程可用性；P3 將 baseline、correctness、CSV、variance、profiler limitation 與 contradiction check 制度化；Phase 3 進一步證明，人類介入最有效的位置在於 baseline 定義、測量噪音過濾、結果分類、shape-aware policy 的策略審查與防止過度宣稱。外部基本測試提供負面與補充證據：規則型 sorting workload 較可能受益，不規則 graph / convergence 類 workload 容易 FAIL，極端加速需優先視為 correctness 或 measurement artifact。`rest.md` 則補足另一個重要面向：regular kernels 的約 2x 改善可跨 P1/P2/P3 穩定重現，但 benchmark-aware shortcut 也同樣會跨 prompt level 持續存在。基於目前證據，AI Agent 在 `softmax-cuda`、`topk-cuda`、`moe-cuda`、`adam-cuda`、`adjacent-cuda`、`randomAccess-cuda` 以及部分 sorting 類任務上能產生較可信的改善；在 memory-system、communication、graph、不規則迭代類任務，以及可能利用 benchmark harness 的任務上，研究價值更多體現在測量修復、拓撲覆蓋、環境診斷、benchmark-aware 行為辨識與失敗模式分析，而非傳統意義的 kernel speedup。

本研究的主要限制包括：不同階段可能使用不同節點與環境，BASIC 與 Phase 2/3 結果不可直接做絕對比較；部分 P1 資料缺少結構化輸出，需由 raw log 或 summary 反向整理；外部基本測試缺少 system prompt、raw output、agent response、程式 diff、trial count、variance 與完整 correctness log，故缺失欄位均標為 `N/A`，其結論只作補充而非主證據；`rest.md` 雖提供 baseline、optimized、changes、correctness 與 result type，但仍缺少完整 raw logs、source diff、variance 與 trial count，因此其 benchmark-aware 結論需以推測語氣描述；部分 benchmark 的 metric 本質不同，跨 benchmark 平均 speedup 只能作為方向性摘要；profiler 證據仍不完整，尤其缺少 stall breakdown、instruction mix 與 memory throughput。即便如此，專案已清楚顯示：若要將 AI Agent 應用於平行程式優化研究，必須把 prompt 設計、correctness gate、baseline policy、variance filter、result type classification 與 human review 視為實驗方法的一部分，而非附屬工程細節。
