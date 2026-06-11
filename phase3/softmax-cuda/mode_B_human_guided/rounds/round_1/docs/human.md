以下先給「可引用文獻」與「softmax-cuda 可測的優化假設」。重點：**Mode B/C 的 baseline 必須是 `impl=1` 的 `softMax2`，不是 `impl=0`。不得把 `impl=0 → impl=1` 寫成 Phase 3 speedup。**

***

## 1. 可引用參考文獻與資料

| title                                                                       |                                                          authors |                 year | venue / source         | source type                    | relevance to this research                                                         | usable claim                                                                                                                               | limitation                                                       | where to cite                                                                                                                                                                                                                                                                       |
| --------------------------------------------------------------------------- | ---------------------------------------------------------------: | -------------------: | ---------------------- | ------------------------------ | ---------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Online normalizer calculation for softmax                                   |                                Maxim Milakov, Natalia Gimelshein |                 2018 | arXiv                  | arXiv technical paper          | 直接針對 softmax 的 memory access 減少與 online normalizer                                 | safe softmax 可透過 online normalizer 合併 max 與 normalizer 計算，論文報告 softmax 最多約 1.3x，加上 TopK fusion 可更高，但 fusion 效果不可直接套到本題單獨 softmax           | 非 peer-reviewed；硬體與 workload 不一定等同 HeCBench                      | CUDA softmax algorithm rationale / memory-access reduction [\[arxiv.org\]](https://arxiv.org/abs/1805.02867)                                                                                                                                         |
| FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness | Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, Christopher Ré |                 2022 | NeurIPS / arXiv        | peer-reviewed conference paper | 提供 IO-aware 精確計算的代表性論證，支持「減少 HBM 讀寫比單純減 FLOPs 更重要」的設計思路                            | 透過 tiling 與 online softmax 減少 HBM 與 SRAM 間讀寫，仍維持 exact attention；可引用其 IO-aware 方法論，不可引用其 attention speedup 作為本 benchmark 的 softmax speedup | 研究對象是 attention，不是單獨 row-wise softmax；speedup 不可外推               | Motivation for IO-aware CUDA optimization / methodology [\[arxiv.org\]](https://arxiv.org/abs/2205.14135), [\[proceeding...neurips.cc\]](https://proceedings.neurips.cc/paper_files/paper/2022/hash/67d57c32e20fd0a7a302cb81d36e40d5-Abstract-Conference.html) |
| CUDA C++ Programming Guide: Cooperative Groups                              |                                                           NVIDIA | current official doc | NVIDIA Docs            | official documentation         | 目前原始碼已使用 `cooperative_groups` 與 warp-level reduction；可作為 warp/tile reduction 合法性來源 | Cooperative Groups 提供 thread groups、tiled partition、scan、parallel reduce 等 primitive，可用來表達 warp/block 粒度協作                                 | 官方 API 文件，不提供本 benchmark 效能保證                                    | Explaining current `softMax2` and reduction alternatives [\[docs.nvidia.com\]](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cooperative-groups.html)                                                                              |
| CUB Developer Overview                                                      |                                                           NVIDIA | current official doc | NVIDIA CCCL / CUB Docs | official documentation         | 可用 CUB `WarpReduce` / `BlockReduce` 作為 block-per-row softmax 的 reduction primitive | CUB 提供 Thread/Warp/Block/Device 層級 reduction；`WarpReduce` 與 `BlockReduce` 是 cooperative parallel primitives                                | CUB 可能增加模板與 temporary storage 複雜度；效能需實測                          | Implementation alternatives for block-level reductions [\[nvidia.github.io\]](https://nvidia.github.io/cccl/unstable/cub/developer_overview.html)                                                                                                           |
| Nsight Compute Profiling Guide                                              |                                                           NVIDIA | current official doc | NVIDIA Docs            | official documentation         | Mode C profiler-augmented workflow 的依據                                             | Nsight Compute 可收集 kernel metrics；不同 section sets 會影響 profiling overhead，因此 profiler 結果不可直接當 benchmark timing                              | profiler unavailable 不等於實驗失敗；profiling run 應和 official timing 分開 | Profiling methodology / profiler limitation [\[docs.nvidia.com\]](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)                                                                                                                        |
| CUDA C++ Best Practices Guide                                               |                                                           NVIDIA | current official doc | NVIDIA Docs            | official documentation         | 支持 verification、timing、memory coalescing、occupancy、shared memory 等 CUDA 優化與評估原則    | CUDA Best Practices Guide 包含 verification、numerical accuracy、timing、bandwidth、coalesced global memory、shared memory、occupancy 等主題          | 是通用指南，不保證特定 kernel speedup                                       | Benchmark validity / CUDA optimization methodology [\[docs.nvidia.com\]](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html), [\[docs.nvidia.com\]](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/contents.html)                 |

***

## 2. 對目前原始碼的技術判讀

### 現有 baseline

* `softMax` 是 `impl=0`，一個 thread 處理一個 slice，屬於 naive reference，不應作為 Mode B/C 優化 baseline。
* `softMax2` 是 `impl=1`，每個 warp 處理一個 slice，使用 Cooperative Groups 的 `thread_block_tile<32>` 與 warp-level reduction。這已經是 HeCBench 內建 optimized baseline，Mode B/C 必須以它為 baseline。
* official slices 固定為：
  * 128
  * 256
  * 784
  * 1024
  * 2048

### 目前 `softMax2` 的主要優化空間

`softMax2` 的核心限制是：**每個 slice 只用 32 個 threads**。對 `sliceSize=1024` 或 `2048`，每個 lane 需要處理 32 或 64 個元素，row 內平行度有限。Cooperative Groups 與 CUB 都支援更大粒度的 block-level reduction，因此可測試「一個 CTA 處理一個 slice」的替代版本。 [\[docs.nvidia.com\]](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cooperative-groups.html), [\[nvidia.github.io\]](https://nvidia.github.io/cccl/unstable/cub/developer_overview.html)

***

## 3. 建議優化方法，按優先順序

以下都是**假設**，不是已證明結果。每一項都必須經過 correctness gate、official slices、重複測量與 raw output 保存。

### H1. Large-slice 使用 block-per-row softmax

**方法**

新增 `softMax_block_row`：

* grid：`grid.x = numSlice`
* block：128 或 256 threads
* 每個 block 處理一個 slice
* thread 以 `for (j = threadIdx.x; j < sliceSize; j += blockDim.x)` 掃描 row
* block-level reduce max
* block-level reduce sum
* normalize 並寫回

**適用 slice**

優先測：

* 784
* 1024
* 2048

對 128、256 不一定有利，因為 warp-per-row 的 overhead 較低。

**文獻依據**

Cooperative Groups 與 CUB 都提供 block/warp 層級 reduction primitive，可支援這種 block-per-row 設計。 [\[docs.nvidia.com\]](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cooperative-groups.html), [\[nvidia.github.io\]](https://nvidia.github.io/cccl/unstable/cub/developer_overview.html)

**風險**

* 小 slice 可能 regression。
* block-per-row 會降低同時處理的 rows per block，可能降低 occupancy 或造成 tail effect。
* 必須分 slice 報告，不可只挑快的 slice。

**result\_type**

若修改 kernel 並通過 correctness，可標：

* `KERNEL_OPT`

若只調 block size，可標：

* `PARAM_TUNE`

***

### H2. Shared memory 暫存 exp，避免重複計算 `expf`

目前 `softMax2` 對每個元素大致會計算兩次 `expf`：

1. sum pass：`sum += expf(...)`
2. output pass：`dest[...] = expf(...) / sum`

**方法**

在 block-per-row kernel 中：

1. reduce max
2. 第二 pass 計算 `e = expf(x - max)`，存入 shared memory，同時計算 sum
3. reduce sum
4. 第三 pass 從 shared memory 讀 `e`，寫 `e / sum`

**適用 slice**

official slices 最大 2048，`float` shared buffer 需要：

* 2048 × 4 bytes = 8192 bytes

這在一般 CUDA shared memory 限制內通常可行，但實際 occupancy 需由 Nsight Compute 或 compiler resource report 驗證。CUDA Best Practices Guide 將 shared memory、occupancy、memory optimization 列為 CUDA 優化核心議題。 [\[docs.nvidia.com\]](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html), [\[docs.nvidia.com\]](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/contents.html)

**風險**

* shared memory 使用量增加，可能降低 occupancy。
* 可能改善 compute-heavy 情況，但若瓶頸是 global memory 或 launch overhead，效果有限。
* 不可改用 approximate softmax，也不可放寬 `1e-3` tolerance。

***

### H3. Online softmax normalizer

**方法**

用 online normalizer 在一次 streaming pass 中同時計算 row max 與 normalizer：

```text
m_new = max(m_old, x)
d_new = d_old * exp(m_old - m_new) + exp(x - m_new)
m = m_new
d = d_new
```

最後仍需至少一個 pass 寫出 `exp(x - m) / d`。

**文獻依據**

Milakov 與 Gimelshein 提出 online normalizer，目標是減少 classical softmax 的 memory accesses；論文報告 softmax 最多約 1.3x，但這是其 benchmark 條件下的結果，不可直接當作本研究結果。 [\[arxiv.org\]](https://arxiv.org/abs/1805.02867)

**適用性**

* 對 `sliceSize=1024/2048` 較值得測。
* 對 128/256 可能 overhead 抵消收益。
* 與 H2 shared memory exp cache 可能互斥或需分別 ablation，不應一次混合多個改動。

**風險**

* online normalizer 的 reduction 結合方式更複雜。
* 浮點加總順序改變，必須確認 `fabsf(diff) <= 1e-3`。
* 若只改善單一 slice，論文中必須標為 partial，不可寫成 full success。

***

### H4. Vectorized load/store：`float4`

**方法**

因 official slices 128、256、784、1024、2048 都可被 4 整除，可測試 `float4` vectorized load。輸入來自 `aligned_alloc(1024, ...)`，且 row offset 為 `sliceSize * sizeof(float)`，official slices 也維持 16-byte 對齊條件。

**文獻依據**

CUDA Best Practices Guide 明確涵蓋 global memory coalescing、memory spaces、bandwidth、shared memory 等 memory optimization 主題；vectorized load/store 的實際效益仍需以 benchmark 與 profiler 驗證。 [\[docs.nvidia.com\]](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html), [\[docs.nvidia.com\]](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/contents.html)

**風險**

* 必須保證 alignment，不可對非 official slice 泛化。
* `float4` 會增加暫存器壓力。
* 若只改 memory access pattern，仍需 correctness PASS。

***

### H5. 針對固定 official slices 做 template specialization

**方法**

建立固定 slice kernel：

* `softmax_block_row<128>`
* `softmax_block_row<256>`
* `softmax_block_row<784>`
* `softmax_block_row<1024>`
* `softmax_block_row<2048>`

好處：

* compiler 可做 loop unroll
* shared memory size 可靜態化
* 減少 runtime branch

**文獻依據**

CUDA Best Practices Guide 將 instruction optimization、occupancy、register pressure、shared memory 等列為優化評估項目；template specialization 是否有效要看 compiler output 與實測。 [\[docs.nvidia.com\]](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/contents.html)

**風險**

* 增加程式複雜度。
* 若 agent 擅自只測某一個 slice，結果 invalid。
* 若 specialization 只支援 official slices，必須明確保留或處理其他輸入，不可破壞 benchmark CLI 行為。

***

### H6. Reduction primitive 對照實驗：Cooperative Groups vs CUB vs hand-written shuffle

**方法**

針對同一個 block-per-row algorithm，比較三種 reduction：

1. Cooperative Groups `cg::reduce`
2. CUB `BlockReduce`
3. hand-written warp shuffle + shared memory cross-warp reduction

**文獻依據**

Cooperative Groups 提供 safe/future-proof 的 group-level primitive；CUB 提供 thread/warp/block/device 層級 reduction primitive。 [\[docs.nvidia.com\]](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cooperative-groups.html), [\[nvidia.github.io\]](https://nvidia.github.io/cccl/unstable/cub/developer_overview.html)

**風險**

* 這是 implementation primitive 比較，不是 algorithmic optimization。
* 若三者效能差 <1%，應標 `MEASUREMENT_EQUIVALENT`。

***

## 4. 不建議或禁止的方法

以下方法不應讓 agent 執行，或執行後不得作為有效 speedup：

1. **不得把 `impl=0 → impl=1` 當作 Mode B/C speedup。**
2. **不得放寬 correctness tolerance。**
3. **不得改成 approximate softmax。**
4. **不得只測 2048 或只保留快的 slice。**
5. **不得刪除慢 case 或 failing case。**
6. **不得把 measurement fix 寫成 kernel optimization。**
7. **不得把 profiler run 的 timing 當 official benchmark timing。Nsight Compute 收集 metrics 會引入 profiling overhead，official timing 應獨立執行。** [\[docs.nvidia.com\]](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)
8. **不得在 login node 直接執行 GPU benchmark binary，必須透過 sbatch。**

***

## 5. 建議 Mode B 第一輪實驗設計

### Round B1: block-per-row large-slice kernel

**Hypothesis**

`softMax2` 對每個 row 只使用一個 warp。對 `sliceSize >= 784`，改成一個 CTA 處理一個 row，並使用 block-level reduction，可能提高 row 內平行度。

**Change**

新增 `impl=2`：

* 不刪除 `impl=0`、`impl=1`
* `impl=1` 保持 baseline
* `impl=2` 為 block-per-row candidate
* 不改 correctness tolerance
* 不改 CPU reference

**Official sweep**

必須測：

```text
sliceSize = 128, 256, 784, 1024, 2048
impl = 1 baseline
impl = 2 candidate
repeat = same as baseline
numSlice = same as baseline
```

**Required output**

每個 case 保存：

```text
benchmark
mode
round_id
impl
numSlice
sliceSize
repeat
trial_id
time_ms
correctness_status
raw_stdout_path
raw_stderr_path
speedup_vs_impl1
cv
measurement_validity
speedup_claim_valid
result_type
notes
```

**Classification rule**

* correctness FAIL → `INVALID`
* improvement <1% → `MEASUREMENT_EQUIVALENT`
* only some slices improve → `PARTIAL_SUCCESS` in narrative, CSV still per-case
* profiler unavailable → record limitation, not failure

***

## 6. 可直接貼給 agent 的 prompt 修正

```text
You are optimizing HeCBench softmax-cuda in Phase 3 Mode B.

Baseline rule:
- impl=1 softMax2 is the official baseline.
- impl=0 is naive reference only.
- Do not claim impl=0 -> impl=1 as Phase 3 speedup.

Execution rule:
- All GPU benchmark runs must be executed through sbatch.
- Do not run ./main or any GPU benchmark binary on the login node.

Correctness rule:
- Do not change CPU reference.
- Do not relax tolerance.
- Do not implement approximate softmax.
- correctness FAIL means result invalid.

Official cases:
- sliceSize = 128, 256, 784, 1024, 2048.
- Use identical numSlice and repeat for baseline and candidate.
- Do not skip slow or failing cases.

Round objective:
- Add a new candidate impl=2 using one CUDA block per slice.
- Keep impl=1 unchanged.
- Use block-level reduction for max and sum.
- Optionally use shared memory to cache exp(x - max), but if used, record it as a separate hypothesis or ablation.
- Do not combine multiple unrelated optimizations in one round unless explicitly approved.

Output requirement:
- Save raw stdout and stderr for every run.
- Produce CSV with:
  benchmark, mode, round_id, impl, numSlice, sliceSize, repeat, trial_id,
  time_ms, correctness_status, raw_stdout_path, raw_stderr_path,
  baseline_impl, speedup_vs_impl1, cv, measurement_validity,
  speedup_claim_valid, result_type, notes.

Result interpretation:
- speedup is valid only if correctness PASS and baseline impl=1 is valid.
- improvement <1% must be classified as MEASUREMENT_EQUIVALENT.
- If only some slices improve, report partial success only.
- ENV_FIX, MEASURE_FIX, BUILD_FIX, or profiler availability changes must not be reported as KERNEL_OPT.
```

***

## 7. 可直接放入論文或報告的保守文字

> For `softmax-cuda`, Phase 3 uses `impl=1` as the optimized baseline because `impl=0` is a naive reference implementation. Therefore, transitions from `impl=0` to `impl=1` are excluded from agent optimization claims. The optimization hypotheses focus on row-wise CUDA softmax variants that modify intra-row parallelism and reduction strategy while preserving exact softmax semantics and the original correctness tolerance. Candidate methods include block-per-row reduction, shared-memory caching of exponentials, online normalizer calculation, and vectorized memory access. These hypotheses are motivated by prior work on online softmax and IO-aware GPU algorithms, as well as NVIDIA CUDA documentation on cooperative groups, CUB reductions, profiling, and memory optimization. However, all speedup claims are accepted only after correctness PASS, identical official slice coverage, repeated measurements, and raw-output auditability. [\[arxiv.org\]](https://arxiv.org/abs/1805.02867), [\[arxiv.org\]](https://arxiv.org/abs/2205.14135), [\[docs.nvidia.com\]](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cooperative-groups.html), [\[nvidia.github.io\]](https://nvidia.github.io/cccl/unstable/cub/developer_overview.html), [\[docs.nvidia.com\]](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html), [\[docs.nvidia.com\]](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html)

***

## 8. Blocking fixes before starting Mode B

1. 固定 baseline：`impl=1`。
2. 固定 official sweep：128、256、784、1024、2048。
3. 新增 candidate 時必須保留 `impl=1`。
4. 所有 benchmark 必須 sbatch。
5. 每個 case 必須保存 raw stdout/stderr。
6. correctness FAIL 直接 invalid。
7. 不得將 <1% improvement 宣稱為顯著加速。
8. 不得把 profiler timing 當 official timing。
