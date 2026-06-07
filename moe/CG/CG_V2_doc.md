## CG_V2 optimization

Built on CG_V1 after measuring all judged top-k cases.

Changes from CG_V1:
- Keep the fused `moeSoftmaxTopK` dispatch for top-k 1, 2, and 4, where it
  reduced runtime by avoiding the full softmax workspace traffic and one kernel
  launch per repeat.
- Route top-k 8 back to the original two-kernel implementation. CG_V1 showed
  that the fused top-k 8 specialization performed more per-block reductions and
  was slower than the original path for this case.

Correctness:
- The fused paths still select over computed softmax probabilities rather than
  raw logits, preserving the reference behavior for equal probabilities and
  underflow-to-zero ties.
- The original path remains unchanged for top-k 8.
