# moe-cuda Agent Summary

## Baseline

- Job: 948372
- Node: gn1230
- Status: PASS for topk 1/2/4/8
- Metric, average execution time of kernels:
  - topk=1: 304.829926 us
  - topk=2: 347.498291 us
  - topk=4: 518.938232 us
  - topk=8: 949.734314 us
  - arithmetic mean: 530.250191 us

## Optimization Submissions

| Submission | Job | Node | Correctness | topk=1 us | topk=2 us | topk=4 us | topk=8 us | Mean us | Speedup vs baseline mean | Decision |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| opt1 | 948373 | gn1230 | FAIL for topk=2/4, canceled before topk=8 | 8242.471680 | 11859.618164 | 23565.798828 | n/a | n/a | n/a | Rejected |
| opt2 | 948374 | gn1230 | PASS for completed topk=1/2/4, canceled before topk=8 | 8216.875977 | 11786.789062 | 23447.560547 | n/a | n/a | n/a | Rejected |
| opt3 | 948376 | gn1230 | PASS for topk 1/2/4/8 | 231.290329 | 351.397949 | 519.070435 | 949.677307 | 512.859005 | 1.0339x | Accepted |

## Final Result

- Accepted job: 948376
- Node: gn1230
- Final correctness: PASS for topk 1/2/4/8
- Final metric, average execution time of kernels:
  - topk=1: 231.290329 us, speedup 1.3181x
  - topk=2: 351.397949 us, speedup 0.9889x
  - topk=4: 519.070435 us, speedup 0.9997x
  - topk=8: 949.677307 us, speedup 1.0001x
  - arithmetic mean: 512.859005 us, speedup 1.0339x

## Best Strategy

The accepted version uses a hybrid path. For topk=1 it fuses softmax normalization and top-1 selection into one block-level CUB reduction kernel, avoiding the full softmax workspace write and the separate TopK kernel launch. For topk=2/4/8 it keeps the original two-kernel path, because full fusion with serial per-block merging was correct only after ranking post-softmax probabilities and was much slower on V100.


TOKENS : 40,703 used