# p2p-cuda Agent Summary

## Environment

- Date: 2026-06-09
- Node: `gn1115.twcc.ai`
- GPUs: 4 visible, `CUDA_VISIBLE_DEVICES=0,1,2,3`
- GPU model: NVIDIA Tesla V100-SXM2-32GB
- CUDA module: `cuda/12.8`
- NVCC: CUDA compilation tools release 12.8, V12.8.61
- Build command: `make clean || true && make ARCH=sm_70`
- Benchmark command: `./main 2000`

## Prompt / Submission Limit

- Prompt level: P3
- Submission limit: 5 optimization submissions
- Baseline counts as submission: no
- Optimization submissions used: 1

## Topology

`nvidia-smi topo -m` was recorded in:

- `result/p2p-cuda_result_948464.txt`
- `result/p2p-cuda_result_948466.txt`

Observed GPU links:

| Pair | Link |
| --- | --- |
| GPU0-GPU1 | NV2 |
| GPU0-GPU2 | NV1 |
| GPU0-GPU3 | NV2 |
| GPU1-GPU2 | NV2 |
| GPU1-GPU3 | NV1 |
| GPU2-GPU3 | NV1 |

## Baseline Result

- Job ID: 948464
- Status: valid baseline
- Result type: TOPOLOGY_MEASURE
- Raw stdout: `result/p2p-cuda_948464.out`
- Raw stderr: `result/p2p-cuda_948464.err`
- Raw benchmark output: `result/p2p-cuda_result_948464.txt`
- Correctness: 6 PASS, 0 FAIL
- Limitation: baseline measured 6 unordered peer pairs using ping-pong copies, while the P3 prompt expects directional GPU-pair coverage.

| Case | Bandwidth GB/s | Correctness |
| --- | ---: | --- |
| GPU0-GPU1 | 48.14 | PASS |
| GPU0-GPU2 | 24.17 | PASS |
| GPU0-GPU3 | 48.16 | PASS |
| GPU1-GPU2 | 48.16 | PASS |
| GPU1-GPU3 | 24.18 | PASS |
| GPU2-GPU3 | 24.18 | PASS |

## Submission History

| Submission | Job ID | Variant | Hypothesis | Result | Accepted |
| --- | --- | --- | --- | --- | --- |
| 1 | 948466 | directed-p2p-copy | Measuring `GPUi -> GPUj` with `cudaMemcpyPeer` covers all directed topology cases while preserving per-case remote-kernel correctness validation. | 36 PASS, 0 FAIL; 12 directed cases x 3 trials. | yes |

No rejected optimization submissions.

## Correctness Table

| Run | PASS cases | FAIL cases | Notes |
| --- | ---: | ---: | --- |
| Baseline 948464 | 6 | 0 | Unordered measured pairs |
| Submission 1 / 948466 | 36 | 0 | 12 directed pairs across 3 trials |

## Performance Table

Final accepted candidate, mean over 3 trials:

| Directed case | Mean GB/s | Min | Max | Stddev | CV |
| --- | ---: | ---: | ---: | ---: | ---: |
| GPU0->GPU1 | 48.28 | 48.28 | 48.28 | 0.00 | 0.00% |
| GPU0->GPU2 | 24.21 | 24.21 | 24.21 | 0.00 | 0.00% |
| GPU0->GPU3 | 48.28 | 48.28 | 48.28 | 0.00 | 0.00% |
| GPU1->GPU0 | 48.28 | 48.28 | 48.28 | 0.00 | 0.00% |
| GPU1->GPU2 | 48.28 | 48.28 | 48.28 | 0.00 | 0.00% |
| GPU1->GPU3 | 24.21 | 24.21 | 24.21 | 0.00 | 0.00% |
| GPU2->GPU0 | 24.21 | 24.21 | 24.21 | 0.00 | 0.00% |
| GPU2->GPU1 | 48.28 | 48.28 | 48.28 | 0.00 | 0.00% |
| GPU2->GPU3 | 24.21 | 24.21 | 24.21 | 0.00 | 0.00% |
| GPU3->GPU0 | 48.28 | 48.28 | 48.28 | 0.00 | 0.00% |
| GPU3->GPU1 | 24.21 | 24.21 | 24.21 | 0.00 | 0.00% |
| GPU3->GPU2 | 24.21 | 24.21 | 24.21 | 0.00 | 0.00% |

The final accepted candidate has two topology bands: NV2 paths at about 48.28 GB/s and NV1 paths at about 24.21 GB/s.

For overlapping unordered baseline pairs, the directional measurement is within 1% of baseline and is classified as measurement-equivalent rather than a real speedup:

| Case | Baseline GB/s | Candidate comparable GB/s | Change | Classification |
| --- | ---: | ---: | ---: | --- |
| GPU0-GPU1 | 48.14 | 48.28 | +0.29% | MEASUREMENT_EQUIVALENT |
| GPU0-GPU2 | 24.17 | 24.21 | +0.17% | MEASUREMENT_EQUIVALENT |
| GPU0-GPU3 | 48.16 | 48.28 | +0.25% | MEASUREMENT_EQUIVALENT |
| GPU1-GPU2 | 48.16 | 48.28 | +0.25% | MEASUREMENT_EQUIVALENT |
| GPU1-GPU3 | 24.18 | 24.21 | +0.12% | MEASUREMENT_EQUIVALENT |
| GPU2-GPU3 | 24.18 | 24.21 | +0.12% | MEASUREMENT_EQUIVALENT |

## CSV

CSV results are saved at `result/p2p-cuda_results.csv`.

## Profiler / Measurement Notes

`nvidia-smi topo -m` was recorded as required. No kernel profiler was collected because the accepted change is a measurement/topology coverage fix, not a kernel optimization.

## Classification

- Primary result type: TOPOLOGY_MEASURE
- Accepted attempt type: MEASURE_FIX
- Speedup label: MEASUREMENT_EQUIVALENT for overlapping baseline cases

## Contradiction Check

- Raw baseline output contains 6 measured bandwidth lines, 6 PASS, and 0 FAIL.
- Raw attempt output contains 36 measured bandwidth lines, 36 PASS, and 0 FAIL.
- The final result uses measured baseline job 948464 and measured candidate job 948466.
- No rejected attempt metrics are used.
- It is correct to state that all required directed cases in the accepted candidate PASS.

## Final Conclusion

SUCCESS

The benchmark now reports the full directed 4-GPU P2P topology sweep with per-case correctness validation and 3-trial variance data. The measured bandwidths match the NVLink topology bands and do not demonstrate a real speedup over overlapping baseline cases; the accepted outcome is a topology measurement fix.


TOKENS : 52,797 used