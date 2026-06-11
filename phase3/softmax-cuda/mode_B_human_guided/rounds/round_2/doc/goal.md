建立 shape-aware dispatch:
- slice=128 使用 impl=1
- slice=256 暫時使用 impl=1，除非修正 correctness
- slice=784/1024/2048 使用 impl=2

原因是：**目前 Round 1 的證據支持「impl=2 對 large slices 有效」，但不支持「impl=2 可取代 impl=1 成為全域實作」。**  
因此 shape-aware dispatch 是合理的下一步，但要注意它屬於**新一輪 candidate**，不是 Round 1 的重新解釋。

***

## 為什麼要建立 shape-aware dispatch

### 1. `slice=128` 必須使用 `impl=1`

原因：

* Round 1 中 `slice=128`：
  * `impl=1 mean = 0.135152 ms`
  * `impl=2 mean = 0.554750 ms`
  * speedup 約 `0.244x`
  * 也就是 `impl=2` 約慢 4.1 倍

這不是 measurement-equivalent，而是明確 regression。

所以：

```text
slice=128 -> impl=1
```

是必要規則。否則 shape-aware candidate 會把已知 regression 帶入 official result。

***

### 2. `slice=256` 暫時使用 `impl=1`

原因：

* Round 1 中 `slice=256`：
  * `impl=2` correctness = PASS 2/3, FAIL 1/3
  * correctness 不穩定
  * measurement\_validity = INVALID
  * speedup\_claim\_valid = false

根據固定規則：

```text
correctness FAIL -> result invalid
```

因此即使 `impl=2` 在某些 trial 可能有時間數據，也不能使用。

目前合理規則是：

```text
slice=256 -> impl=1
```

除非下一輪先修正 correctness，並重新通過完整 official validation。

***

### 3. `slice=784/1024/2048` 可使用 `impl=2`

原因：

Round 1 中 large slices 都是：

* correctness PASS 3/3
* measurement\_validity VALID
* per-slice speedup > 1%
* result\_type = KERNEL\_OPT

具體結果：

| slice | impl=1 mean ms | impl=2 mean ms |  speedup | interpretation              |
| ----: | -------------: | -------------: | -------: | --------------------------- |
|   784 |       1.434026 |       1.108087 | 約 1.294x | valid per-slice improvement |
|  1024 |       2.068956 |       1.300902 | 約 1.590x | valid per-slice improvement |
|  2048 |       2.212359 |       1.680560 | 約 1.316x | valid per-slice improvement |

所以：

```text
slice=784/1024/2048 -> impl=2
```

是基於 Round 1 實測結果的 conservative dispatch policy。

***

## 這個策略的研究意義

shape-aware dispatch 的本質不是「單一 kernel 全面優化」，而是：

> 根據 slice size 選擇已驗證有效且正確的 implementation。

這符合 softmax 類 kernel 的常見特性：不同 row length 對最佳 parallelization strategy 的需求不同。小 slice 可能更適合 warp-per-slice，因為 overhead 低；大 slice 可能更適合 block-per-slice，因為 row 內平行度較高，且 shared-memory cached exp 的成本可能被減少 `expf` 重算的收益抵消。

但在你的實驗裡，這個說法必須限縮為：

> Round 1 empirical results suggest that a shape-aware dispatch may preserve the stable baseline for small or invalid slices while using the compound impl=2 candidate for larger slices where correctness and speedup were both valid.

***

## 必須注意的分類

如果下一輪實作 shape-aware dispatch，建議 result\_type 不要直接全部寫成 `KERNEL_OPT`。

更精確分類：

| slice | dispatch choice | expected classification basis           |
| ----: | --------------- | --------------------------------------- |
|   128 | impl=1          | `BASELINE` 或 `NO_EFFECT`，不是 KERNEL\_OPT |
|   256 | impl=1          | `BASELINE`，除非修正 impl=2 correctness      |
|   784 | impl=2          | 可候選 `KERNEL_OPT`                        |
|  1024 | impl=2          | 可候選 `KERNEL_OPT`                        |
|  2048 | impl=2          | 可候選 `KERNEL_OPT`                        |

整體 candidate 可稱為：

```text
PARAM_TUNE / SHAPE_AWARE_DISPATCH + KERNEL_OPT for selected slices
```

或在你的 result\_type schema 若只能單選，建議：

```text
PARAM_TUNE
```

並在 notes 中寫：

```text
Uses impl=1 for 128/256 and impl=2 for 784/1024/2048 based on Round 1 per-slice validity.
```

如果你想強調 source-level dispatch 加上 kernel candidate，也可以在 narrative 說：

> The shape-aware dispatch combines validated per-slice kernel selection with the Round 1 compound large-slice kernel.

但不要寫：

> all slices optimized by impl=2

這是錯的。

