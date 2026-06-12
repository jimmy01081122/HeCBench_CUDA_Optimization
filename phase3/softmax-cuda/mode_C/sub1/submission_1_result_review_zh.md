# Submission 1 結果審查

## 結論

人類 planner 對 Mode C Submission 1 的決策是：

- `ACCEPT_WITH_LIMITATIONS`
- Submission 1 final label: `SUCCESS_WITH_ADDITIONAL_SPEEDUP`

候選版本：

- `impl4_shape_specialized_large_reduce`

主要比較基準：

- `impl=4` 對 Mode B accepted `impl=3`

次要比較可報告：

- `impl=4` 對 `impl=1`

但 Mode C 成功主張只能基於 `speedup_vs_impl3`。

## 接受的額外加速

以下 slice 可接受為 Mode C 對 Mode B `impl=3` 的額外加速：

| slice | speedup_vs_impl3 | 狀態 |
|---:|---:|---|
| 784 | 1.131x | accepted additional speedup |
| 1024 | 1.049x | accepted additional speedup |

## 不接受為額外加速

以下結果不得宣稱為 Mode C 額外加速：

| slice | 說明 |
|---:|---|
| 128 | `impl=4` dispatch 到未修改的 `impl=1`，不得宣稱 Mode C speedup |
| 256 | `impl=4` dispatch 到未修改的 `impl=1`，不得宣稱 Mode C speedup |
| 2048 | `speedup_vs_impl3=1.008x`，低於 1% 門檻，分類為 measurement-equivalent |

## Correctness

所有 official slices correctness 均為 PASS：

- 128: PASS
- 256: PASS
- 784: PASS
- 1024: PASS
- 2048: PASS

## Auditor

Auditor status: PASS。

已通過的重點：

- official cases present
- correctness PASS before speedup claim
- `speedup_vs_impl3` present for `impl=4`
- small slices not claimed as kernel improvements
- no profiler-supported claim without profiler data
- no cached-exp causality claim without ablation
- no aggregate-only success hiding per-slice regression

## 限制

- Profiler: `NOT_RUN`
- Ablation: `NOT_RUN`
- 目前不能做 causal attribution。
- 不能說明 `impl=4` 的 source patch 在因果上獨立解釋了效能提升。
- 只能說 Submission 1 timing evidence 顯示 `impl=4` 在 784 與 1024 對 `impl=3` 有額外加速。

## Do-Not-Claim

不得宣稱：

- `impl=4` 是 universal kernel optimization。
- `slice=128` 有 Mode C speedup。
- `slice=256` 有 Mode C speedup。
- `slice=2048` 有 Mode C speedup。
- profiler 證明 reduction 或 synchronization 是 bottleneck。
- cached exponentials 是改善原因。
- warp reduction 是改善原因。
- `speedup_vs_impl1` 是 Mode C 的主要成功指標。
- aggregate speedup 可取代 per-slice 判讀。

## 建議

Submission 1 已經有可接受的額外加速，但解釋力不足。若要讓 final report 更可信，下一步應優先取得 profiler 或 ablation evidence，而不是盲目 tuning。
