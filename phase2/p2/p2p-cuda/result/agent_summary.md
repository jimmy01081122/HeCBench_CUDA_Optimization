# p2p-cuda Agent Summary

## Baseline

- Job id: 948446
- Node: gn1115
- Status: PASS
- Raw files:
  - result/baseline-948446.txt
  - result/p2p-cuda-948446.out
  - result/p2p-cuda-948446.err
- Metric: 36.1700 GB/s average over 6 reported unordered GPU pairs
- Range: 24.1700 to 48.1800 GB/s

## Optimization Submissions

| Submission | Job id | Node | Correctness | Metric | Speedup vs baseline | Result |
| --- | ---: | --- | --- | ---: | ---: | --- |
| opt1_peer_async_directional | 948447 | gn1115 | PASS, 12/12 directional pairs | 36.2450 GB/s avg | 1.0021x (+0.21%) | Measurement-equivalent, final valid result |
| opt2_multistream_directional | 948454 | gn1115 | PASS, 12/12 directional pairs | 35.8908 GB/s avg | 0.9923x (-0.77%) | Rejected, slower |

## Final Result

- Final source strategy: explicit directional peer copies with `cudaMemcpyPeerAsync`, CUDA event timing, and a full `i != j` directional GPU-pair sweep.
- Final correctness: PASS for all 12 directional GPU pairs.
- Final metric: 36.2450 GB/s average over all 12 directional GPU pairs.
- Final speedup: 1.0021x (+0.21%) versus baseline average.
- Acceptance note: The final result is correctness-valid and preserves the full topology/directional sweep, but the performance delta is below 1%, so it is marked measurement-equivalent rather than a meaningful speedup.

## Best Strategy Explanation

The best valid strategy removed host-side timing and implicit copy-direction selection from the measured path by using explicit `cudaMemcpyPeerAsync` calls timed with CUDA events. It also reports every directional GPU pair instead of only unordered pairs. The V100 NVLink topology was already near saturation in the baseline, so the measured improvement is small and within run-to-run noise. Multi-stream chunking added scheduling overhead and reduced average bandwidth.

TOKENS : 51,430 used