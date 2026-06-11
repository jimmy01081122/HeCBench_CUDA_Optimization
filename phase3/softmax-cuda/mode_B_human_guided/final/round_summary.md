# softmax-cuda Mode B Final Confirmation Summary

## English

Slurm job: 949717 on `gn1228.twcc.ai`.

Accepted candidate: `impl3_shape_dispatch_impl1_small_impl2_large`.

Profiler status: `NOT_RUN`.

| slice | dispatch_selected_impl | candidate mean ms | paired impl=1 mean ms | correctness | validity | result_type | aggregate speedup |
|---:|---:|---:|---:|---|---|---|---:|
| 128 | 1 | 0.134574 | 0.134869 | 3/3 PASS | VALID | MEASUREMENT_EQUIVALENT | 1.002197 |
| 256 | 1 | 0.321505 | 0.321793 | 3/3 PASS | VALID | MEASUREMENT_EQUIVALENT | 1.000895 |
| 784 | 2 | 1.036402 | 1.442716 | 3/3 PASS | VALID | PARAM_TUNE | 1.392043 |
| 1024 | 2 | 1.238443 | 2.104045 | 3/3 PASS | VALID | PARAM_TUNE | 1.698944 |
| 2048 | 2 | 1.672904 | 2.237452 | 3/3 PASS | VALID | PARAM_TUNE | 1.337466 |

Interpretation:
- This is a shape-aware dispatch result, not a universal `impl=2` or universal `KERNEL_OPT` result.
- `slice=128` and `slice=256` dispatch to unchanged `impl=1`; they are classified as `MEASUREMENT_EQUIVALENT` with `speedup_claim_valid=false`.
- `slice=784`, `slice=1024`, and `slice=2048` dispatch to unchanged `impl=2`; speedup claims are valid only because correctness passed and CV was stable.
- Final confirmation label: `SUCCESS`.

## 繁體中文

Slurm 作業：`949717`，節點 `gn1228.twcc.ai`。

接受的候選版本：`impl3_shape_dispatch_impl1_small_impl2_large`。

Profiler 狀態：`NOT_RUN`。

| slice | dispatch_selected_impl | candidate mean ms | paired impl=1 mean ms | correctness | validity | result_type | aggregate speedup |
|---:|---:|---:|---:|---|---|---|---:|
| 128 | 1 | 0.134574 | 0.134869 | 3/3 PASS | VALID | MEASUREMENT_EQUIVALENT | 1.002197 |
| 256 | 1 | 0.321505 | 0.321793 | 3/3 PASS | VALID | MEASUREMENT_EQUIVALENT | 1.000895 |
| 784 | 2 | 1.036402 | 1.442716 | 3/3 PASS | VALID | PARAM_TUNE | 1.392043 |
| 1024 | 2 | 1.238443 | 2.104045 | 3/3 PASS | VALID | PARAM_TUNE | 1.698944 |
| 2048 | 2 | 1.672904 | 2.237452 | 3/3 PASS | VALID | PARAM_TUNE | 1.337466 |

解讀：
- 這是 shape-aware dispatch 結果，不是通用的 `impl=2` 或通用 `KERNEL_OPT` 結果。
- `slice=128` 與 `slice=256` 會派發到未修改的 `impl=1`；分類為 `MEASUREMENT_EQUIVALENT`，且 `speedup_claim_valid=false`。
- `slice=784`、`slice=1024`、`slice=2048` 會派發到未修改的 `impl=2`；只有在 correctness PASS 且 CV 穩定時才允許 speedup claim。
- 最終確認標籤：`SUCCESS`。
