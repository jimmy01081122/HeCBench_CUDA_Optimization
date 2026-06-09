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
