# softmax-cuda Mode B Round 2 Summary

Slurm job: 949703 on `gn1221.twcc.ai`

Candidate: `impl3_shape_dispatch_impl1_small_impl2_large`

Dispatch map:

| slice | dispatch_selected_impl | candidate mean ms | paired impl=1 mean ms | correctness | validity | result_type | aggregate speedup |
|---:|---:|---:|---:|---|---|---|---:|
| 128 | 1 | 0.135674 | 0.144732 | 3/3 PASS | VALID | MEASUREMENT_EQUIVALENT | 1.066763 |
| 256 | 1 | 0.306408 | 0.305251 | 3/3 PASS | VALID | MEASUREMENT_EQUIVALENT | 0.996224 |
| 784 | 2 | 1.107988 | 1.437362 | 3/3 PASS | VALID | PARAM_TUNE | 1.297273 |
| 1024 | 2 | 1.300765 | 2.082344 | 3/3 PASS | VALID | PARAM_TUNE | 1.600861 |
| 2048 | 2 | 1.670514 | 2.213330 | 3/3 PASS | VALID | PARAM_TUNE | 1.324940 |

Profiler status: `NOT_RUN`.

Interpretation:
- This is a shape-aware dispatch result, not a universal `impl=2` or universal `KERNEL_OPT` result.
- `slice=128` and `slice=256` intentionally dispatch to unchanged `impl=1`; measurement-equivalent behavior is expected and no kernel optimization or speedup claim is made for those slices.
- Large-slice improvements are dispatch-policy outcomes that reuse the unchanged `impl=2` candidate from Round 1.
- All official slices passed correctness; promotion depends on human review of per-slice validity and dispatch-policy interpretation.
