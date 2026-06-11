
## 中文摘要：與 `block-per-slice + shared-memory cached exp` 的關係

### 1. 為什麼這個方向合理

穩定 softmax 通常包含三個概念階段：求 row max、計算 `exp(x - max)` 並求和、輸出 `exp(x - max) / sum`；其中 max 與 sum 都是 row-wise reduction，需要 thread 間通訊。AITemplate wiki 明確指出 softmax 的 subtract/divide 是 elementwise operation，但 max/sum 是 reduction operation，因此需要在 GPU threads 間做資料交換。 [\[github.com\]](https://github.com/facebookincubator/AITemplate/wiki/How-to-write-a-fast-Softmax-CUDA-kernel%3F)

HeCBench 目前 `impl=1` 是 warp-per-slice，每列只用一個 warp；對 `sliceSize=784/1024/2048`，每個 warp lane 要處理多個元素。OneFlow 的工程資料也採用「依 num\_cols 分段」策略：短 row 可用 warp-per-row，中等 row 可用 block-per-row 並借助 shared memory 保存中間結果。 [\[segmentfault.com\]](https://segmentfault.com/a/1190000038894591)

因此，對 HeCBench official slices 中的 784、1024、2048，測試 block-per-slice 是合理假設；但對 128、256，warp-per-slice 可能更低 overhead，block-per-slice 可能 regression。這點與 OneFlow「不同 num\_cols 需要不同實作」的結論一致。 [\[segmentfault.com\]](https://segmentfault.com/a/1190000038894591)

### 2. shared-memory cached exp 的真正作用

目前 `impl=1` 在 sum pass 計算一次 `expf(src - max)`，在 output pass 又計算一次同樣的 `expf(src - max)`。cached-exp 的想法是把第二階段算出的 exponential 暫存在 shared memory，第三階段直接讀取 cached value 做除法。NVIDIA 的 shared memory 說明指出 shared memory 可作為 block 內 threads 共享的 user-managed cache，也常用於 parallel reduction，但需要同步以避免 race condition。 [\[developer.nvidia.cn\]](https://developer.nvidia.cn/blog/using-shared-memory-cuda-cc/)

這個方法的潛在收益是減少一次 `expf` 計算；代價是增加 shared memory write/read、`__syncthreads()`、block-level reduction overhead，以及可能降低 occupancy。CUDA Best Practices Guide 明確把 memory usage、parallel execution、instruction-level efficiency、profiling 都列為 CUDA optimization 的核心面向，因此這類 trade-off 必須實測而不能只靠推論。 [\[docs.nvidia.com\]](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html)

### 3. 和 online softmax 的差異

Online normalizer paper 的重點是用 running max 與 running normalizer 減少 softmax memory access；該文摘要報告 classical softmax 最多約 1.3x，Softmax+TopK fused 最多約 5x。 [\[arxiv.org\]](https://arxiv.org/abs/1805.02867)

但 `block-per-slice + shared-memory cached exp` 不是 online normalizer。cached-exp 仍然需要先得到 row max，然後計算 exp/sum，再輸出；它主要是在「避免重算 exp」與「用 shared memory 換取少一次 expensive computation」之間做 trade-off。Online softmax 則是改變 normalizer 的計算方式，用 streaming update 合併 max 與 denominator 計算。 [\[arxiv.org\]](https://arxiv.org/abs/1805.02867)

因此，若 Round 1 採用 cached-exp，不應引用 Milakov & Gimelshein 來聲稱「這就是 online softmax」。正確引用方式是：該文支持「減少 softmax memory access / pass count 可能改善效能」這個一般動機，但本 round 的 candidate 是不同實作。 [\[arxiv.org\]](https://arxiv.org/abs/1805.02867)

### 4. 和 FlashAttention 的關係

FlashAttention 的重點是 attention，不是 standalone softmax；它透過 tiling 減少 HBM 與 SRAM 間讀寫，並維持 exact attention。 [\[arxiv.org\]](https://arxiv.org/abs/2205.14135), [\[proceeding...neurips.cc\]](https://proceedings.neurips.cc/paper_files/paper/2022/hash/67d57c32e20fd0a7a302cb81d36e40d5-Abstract-Conference.html)

對 HeCBench softmax 而言，可引用 FlashAttention 的地方是「IO-aware 設計原則」：GPU kernel 優化不能只看 FLOPs，也要計算 HBM/SRAM/register 間資料移動。不可引用 FlashAttention 的 GPT-2 或 BERT speedup 來支持 standalone `softmax-cuda` 的 speedup。 [\[arxiv.org\]](https://arxiv.org/abs/2205.14135), [\[proceeding...neurips.cc\]](https://proceedings.neurips.cc/paper_files/paper/2022/hash/67d57c32e20fd0a7a302cb81d36e40d5-Abstract-Conference.html)

***

## 對 HeCBench `softmax-cuda` Round 1 的可用優化假設

### H1: block-per-slice 對 large slice 可能提升 row 內平行度

* `impl=1`：one warp per slice。
* candidate：one block per slice，256 threads。
* 預期可能有利於 784、1024、2048。
* 預期可能不利於 128、256。

可引用依據：OneFlow 採用 row length dependent dispatch，並說明 warp-per-row 與 block-per-row 適合不同 `num_cols` 區間。 [\[segmentfault.com\]](https://segmentfault.com/a/1190000038894591)

### H2: shared-memory cached exp 可能減少重複 `expf`

* 現有 `impl=1` 對每個元素有兩次 `expf(src - max)`。
* cached-exp candidate 只算一次，存入 shared memory，再用 cached value 做 normalization。
* 對 `sliceSize=2048`，暫存 2048 個 float 約 8 KiB，容量上通常可行，但是否降低 occupancy 要看 compiler 與 profiler。

可引用依據：NVIDIA shared memory blog 支持 shared memory 作為 block 內 user-managed cache 與 reduction 空間；CUDA Best Practices Guide 支持以 profiling 與 memory optimization 方式評估這種 trade-off。 [\[developer.nvidia.cn\]](https://developer.nvidia.cn/blog/using-shared-memory-cuda-cc/), [\[docs.nvidia.com\]](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html)

### H3: 不應把結果單獨歸因於 cached exp

Round 1 candidate 同時改了：

1. warp-per-slice → block-per-slice
2. warp reduction → block reduction
3. recompute exp → shared-memory cached exp

所以若加速，只能說「compound candidate 加速」，不能說「cached exp 單獨造成加速」。這與 OneFlow、AITemplate 都強調 kernel strategy 與 row length、memory access、reduction design 共同影響效能的描述一致。 [\[segmentfault.com\]](https://segmentfault.com/a/1190000038894591), [\[github.com\]](https://github.com/facebookincubator/AITemplate/wiki/How-to-write-a-fast-Softmax-CUDA-kernel%3F)
