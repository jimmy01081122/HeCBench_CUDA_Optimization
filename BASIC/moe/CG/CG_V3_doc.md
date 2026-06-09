## CG_V3 optimization

Built on the validated CG_V2 hybrid implementation.

Changes from CG_V2:
- Added a dedicated `moeSoftmaxTop1<TPB>` kernel for the fixed top-k 1 case.
- The top-k 1 path now performs only:
  - one block reduction to find the maximum expert with the same smaller-index
    tie behavior as the reference,
  - one block reduction for the softmax denominator,
  - one output write for the selected expert.
- This avoids the CG_V2 top-k 1 fused path's full 384-entry shared softmax
  cache and extra top-k reduction.

Unchanged from CG_V2:
- top-k 2 and 4 continue to use the fused softmax + top-k path.
- top-k 8 continues to use the original two-kernel path, which measured faster
  than the fused top-k 8 attempt.

Correctness:
- top-k 1 selection is monotonic with softmax, and equal logits produce equal
  softmax probabilities, so selecting the smaller expert index on equal logits
  preserves the reference tie behavior.
- The output weight is `1 / sum(exp(input - row_max))`, matching the softmax
  probability of the selected maximum expert.



最後一次 V3 已完成，job id `944997`，結果檔已產生：

- 說明檔：[CG_V3_doc.md](/home/r14525078/HeCBench/src/moe-cuda/results/CG_V3_doc.md)
- 結果檔：[moe_cuda_result_944997.txt](/home/r14525078/HeCBench/src/moe-cuda/results/moe_cuda_result_944997.txt)
- PASS: `40`
- FAIL: `0`

V3 主要新增 topk=1 專用 kernel：[kernels.h](/home/r14525078/HeCBench/src/moe-cuda/kernels.h:165)，並在 [main.cu](/home/r14525078/HeCBench/src/moe-cuda/main.cu:83) dispatch。topk=1 現在只做 max expert reduction + denominator reduction，直接輸出選中 expert 的 softmax weight，不再建立 384-entry softmax cache。

V3 平均時間：

```text
topk=1 avg=168.573779 us
topk=2 avg=344.903119 us
topk=4 avg=561.396344 us
topk=8 avg=949.431598 us
```

相對 V2，主要改善是 topk=1：`239.096091 -> 168.573779 us`。其餘 topk=2/4/8 維持 V2 路線並略有波動。