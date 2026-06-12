## Result Audit Summary

* **verdict:** ACCEPT\_WITH\_LIMITATIONS

* **final label:** SUCCESS\_WITH\_ADDITIONAL\_SPEEDUP

* **accepted claims:**
  1. Submission 1 `impl=4_shape_specialized_large_reduce` 完整覆蓋 official slices。
  2. correctness 全部 PASS。
  3. auditor 與 contradiction checks 全部 PASS。
  4. `impl=4` 相對 Mode B `impl=3` 在：
     * `slice=784` 有有效 additional speedup：`1.131x`
     * `slice=1024` 有有效 additional speedup：`1.049x`
  5. `slice=2048` 相對 `impl=3` 為 `1.008x`，低於 1%，正確分類為 `MEASUREMENT_EQUIVALENT`。
  6. `slice=128/256` 不宣稱 Mode C speedup，正確。
  7. profiler 未執行，未做 profiler-supported bottleneck claim，正確。
  8. 未宣稱 cached-exp causality，正確。

* **rejected claims:**
  1. 不可宣稱 `impl=4` 是 universal kernel optimization。
  2. 不可宣稱 `slice=128` 有 Mode C speedup，即使 `speedup_vs_impl3=1.144855`，因為該 comparison 的 `impl=3` baseline 在 128 是 `NOISY`，且 small slice fallback 到 `impl=1`。
  3. 不可宣稱 `slice=2048` 有有效 Mode C speedup，因為 `speedup_vs_impl3=1.007945 < 1.01`。
  4. 不可宣稱 profiler 證明 reduction/synchronization 是 bottleneck。
  5. 不可宣稱 cached-exp 或 warp reduction 單獨造成改善，因為沒有 ablation。

* **missing evidence:**
  1. 沒有 profiler evidence。
  2. 沒有 ablation evidence。
  3. 沒有提供完整 raw stdout/stderr 內容；目前只能依據 CSV path、auditor PASS 與 summary 判定。
  4. `impl=4` 的 source patch 細節沒有貼出完整程式碼，因此無法獨立審查 kernel 是否完全只改 reduction path。

***

## Per-Case Audit Table

| slice | implementation / variant                                      | correctness | measurement validity                              |                            speedup\_vs\_impl1 |           speedup\_vs\_impl3 | speedup claim validity | result type             | notes                                                                                         |
| ----: | ------------------------------------------------------------- | ----------- | ------------------------------------------------- | --------------------------------------------: | ---------------------------: | ---------------------- | ----------------------- | --------------------------------------------------------------------------------------------- |
|   128 | `impl=4_shape_specialized_large_reduce`, dispatch to `impl=1` | PASS        | VALID for impl4, but paired `impl=3` row is NOISY | 0.997x vs impl1 mean comparison approximately | 1.145x from noisy impl3 mean | false                  | MEASUREMENT\_EQUIVALENT | No Mode C claim allowed. `impl=3` at 128 is noisy; `impl=4` falls back to unchanged `impl=1`. |
|   256 | `impl=4_shape_specialized_large_reduce`, dispatch to `impl=1` | PASS        | VALID                                             | 0.999x vs impl1 mean comparison approximately |                       1.001x | false                  | MEASUREMENT\_EQUIVALENT | Correct classification. Same-path small slice.                                                |
|   784 | `impl=4_shape_specialized_large_reduce`                       | PASS        | VALID                                             |                                        1.593x |                       1.131x | true                   | MODE\_C\_CANDIDATE      | Valid additional Mode C speedup over `impl=3`.                                                |
|  1024 | `impl=4_shape_specialized_large_reduce`                       | PASS        | VALID                                             |                                        1.782x |                       1.049x | true                   | MODE\_C\_CANDIDATE      | Valid additional Mode C speedup over `impl=3`, but modest.                                    |
|  2048 | `impl=4_shape_specialized_large_reduce`                       | PASS        | VALID                                             |                                        1.347x |                       1.008x | false                  | MEASUREMENT\_EQUIVALENT | Correctly not claimed as Mode C speedup because <1%.                                          |

Mean-based values from provided summary:

```text
128:  impl3 mean 0.157587, impl4 mean 0.137648, speedup_vs_impl3 = 1.144855
256:  impl3 mean 0.307093, impl4 mean 0.306706, speedup_vs_impl3 = 1.001262
784:  impl3 mean 1.028795, impl4 mean 0.909558, speedup_vs_impl3 = 1.131093
1024: impl3 mean 1.241707, impl4 mean 1.183763, speedup_vs_impl3 = 1.048949
2048: impl3 mean 1.675165, impl4 mean 1.661960, speedup_vs_impl3 = 1.007945
```

***

## Contradictions

### 1. `slice=128` 的 `speedup_vs_impl3=1.144855` 不可作為有效改善

這不是資料矛盾，但容易被誤讀。

原因：

* `slice=128` 的 `impl=3` baseline row 被標為 `NOISY`，CV = `0.224170`，也就是 22.4%。
* `impl=4` 在 `slice=128` dispatch 到 unchanged `impl=1`。
* small slices 被 proposal 明確規定不得宣稱 Mode C optimization speedup。

因此 summary 正確將其分類為：

```text
MEASUREMENT_EQUIVALENT
speedup_claim_valid=false
```

但報告中必須避免寫：

```text
slice=128 improved by 1.145x over impl=3
```

應寫：

```text
slice=128 is a same-path fallback case and is not considered a Mode C speedup claim.
```

***

### 2. `mode_c_final_label=SUCCESS_WITH_ADDITIONAL_SPEEDUP` 出現在 baseline rows

CSV 中 baseline rows 也帶有：

```text
mode_c_final_label=SUCCESS_WITH_ADDITIONAL_SPEEDUP
```

這不是致命錯誤，但欄位語意容易混淆。`mode_c_final_label` 是 submission-level label，不是每一列 baseline 的 row-level result。

建議後續報告中說明：

```text
mode_c_final_label is a submission-level label repeated across rows for convenience.
Row-level result_type remains authoritative for each implementation/case.
```

***

### 3. `slice=2048` 不能算 Mode C speedup

`speedup_vs_impl3=1.007945`，低於 1% 門檻。summary 已正確標為：

```text
MEASUREMENT_EQUIVALENT
speedup_claim_valid=false
```

不可把 2048 放進「Mode C additional speedup」清單。

***

### 4. 無 profiler / 無 ablation

summary 已正確承認：

```text
Profiler status: NOT_RUN
No cached-exp causality claim
```

因此沒有矛盾。  
但任何後續報告若寫成「profiling shows」或「cached-exp caused」都是不支援。

***

## Paper-Safe Interpretation

可直接放入論文或報告：

```text
Mode C Submission 1 evaluated an additive candidate, impl4_shape_specialized_large_reduce, against the accepted Mode B impl=3 shape-aware dispatch policy. The candidate preserved impl=0/1/2/3 and introduced a new large-slice kernel intended to reduce shared-memory reduction and synchronization overhead through warp-level reductions and compact cross-warp reduction. All official slices passed correctness, and the auditor and contradiction checks passed.

The primary Mode C comparison is against impl=3. The small slices 128 and 256 dispatch to the unchanged impl=1 path and are not considered optimization speedup claims. For large slices, impl=4 achieved valid additional speedups over impl=3 on slice=784 and slice=1024, with speedup_vs_impl3 of 1.131x and 1.049x respectively. Slice=2048 showed only 1.008x over impl=3 and is therefore classified as measurement-equivalent rather than a valid additional speedup. Overall, Submission 1 is classified as SUCCESS_WITH_ADDITIONAL_SPEEDUP, but the accepted claim is limited to per-slice additional improvements on 784 and 1024. Since profiler and ablation were not run, no profiler-supported bottleneck conclusion or cached-exp causality claim is made.
```

***

## Do-Not-Claim

以下不能寫入論文或 summary：

1. `impl=4` 是 universal kernel optimization。
2. `impl=4` 改善所有 official slices。
3. `slice=128` 有有效 Mode C speedup。
4. `slice=256` 有有效 Mode C speedup。
5. `slice=2048` 有有效 Mode C speedup。
6. Mode C 的成功可以只用 aggregate speedup 表示。
7. `impl=4` 的改善證明 bottleneck 是 shared-memory reduction。
8. profiler 支持 reduction bottleneck。
9. cached exponentials alone caused speedup。
10. warp-level reduction alone caused speedup。
11. `speedup_vs_impl1` 可以作為 Mode C additional speedup 的主要依據。
12. `impl=3` 被完全取代。更精確說法是：`impl=4` 是目前 Mode C candidate，在 784/1024 有 additional speedup，但 2048 與 small slices 不支持全面替代宣稱。

***

## Final Label Decision

根據你給的 criteria：

```text
SUCCESS_WITH_ADDITIONAL_SPEEDUP:
- all official slices correctness PASS
- all official cases present
- auditor PASS
- no hidden regression
- at least one large slice has valid speedup_vs_impl3 >= 1.01
- no large-slice regression vs impl=3
- small slices are measurement-equivalent or valid
```

Submission 1 符合。

正式判定：

```text
Mode C Submission 1 verdict: ACCEPT_WITH_LIMITATIONS
Mode C final label for Submission 1: SUCCESS_WITH_ADDITIONAL_SPEEDUP
Accepted additional-speedup claims:
- slice=784: 1.131x vs impl=3
- slice=1024: 1.049x vs impl=3

Not accepted as additional speedup:
- slice=128
- slice=256
- slice=2048
```

***

## 建議給 CLI 的回覆

```text
Submission 1 result audit: ACCEPT_WITH_LIMITATIONS.

Accepted:
- correctness PASS for all official slices
- auditor PASS
- contradiction checks PASS
- valid additional Mode C speedup vs impl=3 for:
  - slice=784: 1.131x
  - slice=1024: 1.049x

Not accepted as speedup claims:
- slice=128: small-slice fallback to impl=1; impl=3 baseline is noisy; no Mode C speedup claim
- slice=256: small-slice fallback to impl=1; measurement-equivalent
- slice=2048: speedup_vs_impl3=1.008x < 1%; measurement-equivalent

Final label:
- SUCCESS_WITH_ADDITIONAL_SPEEDUP

Required reporting constraints:
- Report per-slice results first.
- Do not report only aggregate speedup.
- Do not claim impl=4 is universal.
- Do not claim profiler-supported bottleneck because profiler_status=NOT_RUN.
- Do not claim cached-exp or warp-reduction causality without ablation.
- Do not use speedup_vs_impl1 as the main Mode C success metric.
- State that accepted additional Mode C speedup is limited to slices 784 and 1024.
```

***

## 是否建議進 Submission 2

目前不必立刻進 Submission 2。  
建議先做兩件事之一：

### 選項 A：凍結 Submission 1，寫入 Mode C 初步成果

適合目前想收斂報告。  
可寫：

```text
Mode C Submission 1 produced additional per-slice speedup beyond Mode B on 784 and 1024, while 2048 was measurement-equivalent and small slices were fallback cases.
```

### 選項 B：Submission 2 做 ablation 或 profiler

如果研究目標需要因果解釋，下一步不應直接再調參，而應做：

1. **profiler run**
   * Compare `impl=3` vs `impl=4` on 784/1024/2048
   * official timing 與 profiler timing 分離

2. **ablation**
   * `impl=2`: block-per-slice + cached exp + shared reduction
   * `impl=4`: block-per-slice + cached exp + warp/cross-warp reduction
   * optional no-cache version if approved

這樣才有資格討論 reduction-overhead 或 cached-exp contribution。
