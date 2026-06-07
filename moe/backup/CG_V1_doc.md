## CG_V1 optimization

Implemented a fused softmax + top-k CUDA kernel for the fixed benchmark shape
with 384 experts and top-k values 1, 2, 4, and 8.

Changes:
- Added `moeSoftmaxTopK<TPB, TOPK>` specializations in `kernels.h`.
- The fused kernel computes the row maximum, softmax denominator, and cached
  per-expert softmax probabilities inside one block per token.
- Top-k selection is performed over the cached softmax probabilities, preserving
  the reference tie behavior by selecting the smaller expert index when values
  are equal.
- `main.cu` dispatches to the fused specializations for the judged 384-expert
  configurations and keeps the original two-kernel path as a fallback.

Expected benefit:
- Removes the full softmax workspace global write/read traffic from the judged
  path.
- Reduces the timed loop from two kernel launches per repeat to one kernel
  launch per repeat.
- Specializes away the runtime top-k loop bound for top-k 1/2/4/8.
