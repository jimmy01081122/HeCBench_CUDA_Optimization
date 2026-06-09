# pingpong-cuda Agent Summary

## Baseline

- Valid baseline job: 948506
- Node: gn1224
- Correctness: PASS, no `ERROR` lines and full 12-size MPI plus 12-size NCCL sweep completed.
- Environment: `UCX_TLS=self,shm,cuda_copy,cuda_ipc`, `NCCL_DEBUG=unset`, 2 Tesla V100-SXM2-32GB GPUs.
- Baseline 1 GiB metric:
  - MPI: 0.044279685 s, 24.249084283 GB/s
  - NCCL: 0.046889752 s, 22.899285455 GB/s

Invalid setup attempts before the valid baseline:

| Job | Node | Status | Reason |
| --- | --- | --- | --- |
| 948499 | unknown | Rejected | Slurm output directory did not exist before submission. |
| 948500 | gn1224 | Rejected | Metadata command used invalid `nvidia-smi` query field. |
| 948505 | gn1224 | Rejected | Built only `main-mpi`; `main-nccl` executable was missing. |

## Optimization Submissions

| Submission | Job | Node | Change | Correctness | 1 GiB MPI GB/s | 1 GiB NCCL GB/s | Decision |
| --- | --- | --- | --- | --- | ---: | ---: | --- |
| 1 | 948510 | gn1224 | Prioritized CUDA IPC in `UCX_TLS`; set `NCCL_P2P_LEVEL=NVL`. | PASS | 24.254632410 | 22.899288019 | Rejected, NCCL small sizes regressed and large-size gain was noise-level. |
| 2 | 948513 | gn1224 | Used `cudaStreamCreateWithFlags(..., cudaStreamNonBlocking)` for NCCL. | PASS | 24.251932705 | 22.899447218 | Rejected as standalone, MPI regressed slightly; NCCL large-size gain was tiny. |
| 3 | 948517 | gn1224 | Added `UCX_RNDV_SCHEME=put_zcopy` on top of submission 2. | PASS | 24.252626993 | 22.899530319 | Rejected, small NCCL sizes regressed and MPI did not improve. |
| 4 | 948522 | gn1224 | Replaced MPI blocking calls with persistent requests; kept NCCL nonblocking stream. | PASS | 24.251578905 | 22.899456238 | Rejected, MPI was slower than baseline across most sizes. |
| 5 | 948523 | gn1224 | Pre-posted rank 0 MPI return receive with `MPI_Irecv`; kept NCCL nonblocking stream. | PASS | 24.254192499 | 22.899517934 | Accepted, best final valid source variant with small but positive 1 GiB speedup. |

## Final Result

- Final accepted job: 948523
- Node: gn1224
- Correctness: PASS, no `ERROR` lines and full size sweep completed.
- Final 1 GiB metric:
  - MPI: 0.044270360 s, 24.254192499 GB/s, speedup 1.00021x versus baseline time.
  - NCCL: 0.046889276 s, 22.899517934 GB/s, speedup 1.00001x versus baseline time.

## Best Strategy

The best valid strategy was conservative: keep the required pingpong ordering, but pre-post rank 0's return receive before the outbound send so MPI can prepare the matching receive earlier. The NCCL nonblocking stream change produced only noise-level large-message benefit and did not materially change the benchmark. Overall gains are very small because the baseline was already near the V100 peer-transfer plateau for the largest sizes.


TOKENS : 53,588 used