# softmax-cuda Mode B Agent Summary

## English

Robust baseline used `impl=1` as the official baseline. Round 1 tested `impl2_block_cached_exp_compound`; it improved large slices but failed correctness on one `slice=256` trial and regressed `slice=128`, so it was not promoted as a full replacement.

Round 2 introduced the accepted shape-aware dispatch candidate `impl3_shape_dispatch_impl1_small_impl2_large`: `slice=128/256 -> impl=1`, `slice=784/1024/2048 -> impl=2`.

Final confirmation ran as Slurm job 949717. All official candidate trials passed correctness. Final artifacts are in `final/`.

Final conclusion label: `SUCCESS`.

Mode C has not been started.

## 繁體中文

Robust baseline 使用 `impl=1` 作為官方基準。Round 1 測試 `impl2_block_cached_exp_compound`；大尺寸 slice 有改善，但 `slice=256` 有一個 trial correctness 失敗，且 `slice=128` 退步，因此沒有被提升為完整替代方案。

Round 2 導入已接受的 shape-aware dispatch 候選版本 `impl3_shape_dispatch_impl1_small_impl2_large`：`slice=128/256 -> impl=1`，`slice=784/1024/2048 -> impl=2`。

Final confirmation 透過 Slurm job 949717 執行。所有官方 candidate trials correctness 皆 PASS。最終 artifacts 位於 `final/`。

最終結論標籤：`SUCCESS`。

尚未進入 Mode C。
