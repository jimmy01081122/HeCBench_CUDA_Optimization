# pingpong-cuda Agent Summary

## 1. Environment

- GPU model: NVIDIA Tesla V100-SXM2-32GB
- CUDA_VISIBLE_DEVICES: `0,1`
- Number of GPUs visible: 2
- Final node: `gn1230`
- Final GPU UUIDs:
  - GPU 0: `GPU-f19ea2fc-4fa5-b54a-8007-bc60b89e9c8b`
  - GPU 1: `GPU-034eec08-f857-349b-a782-5f038bdfe219`
- nvcc: CUDA 12.6, V12.6.77
- mpirun: Open MPI 4.1.7a1
- NCCL link: `libnccl.so.2 => /work/HPC_SYS/twnia2/pkg-rocky8/nvidia/hpc_sdk/Linux_x86_64/24.11/comm_libs/12.6/nccl/lib/libnccl.so.2`
- MPI_ROOT: `/work/HPC_SYS/twnia2/pkg-rocky8/nvidia/hpc_sdk/Linux_x86_64/24.11/comm_libs/12.6/hpcx/hpcx-2.20/ompi`
- ARCH: `sm_70`
- Launcher: `mpirun -x UCX_TLS --mca coll ^hcoll,ucc --mca pml ucx -n 2`
- UCX_TLS: `self,shm,cuda_copy,cuda_ipc`

## 2. Baseline

- Baseline job id: `946551`
- Build: PASS
- MPI run: PASS
- NCCL run: PASS
- First failed size: none
- Error message: none
- Data valid: yes
- Notes: baseline used the tuned UCX launcher, but `UCX_TLS` was not exported in the first script and OpenMPI warned that it could not forward the variable. Correctness still passed for both MPI and NCCL with complete size sweeps.

## 3. Submission History

| Submission | Job id | Modification | Hypothesis | Result | Correctness | Performance |
|---:|---:|---|---|---|---|---|
| 1 | 946555 | Exported `UCX_TLS`, split MPI/NCCL raw logs, added parsed `RESULT` rows. | Baseline was correct but launcher env and output parsing needed to be reproducible. | PASS, but NCCL parsed rows duplicated due nested `tee`. | MPI PASS, NCCL PASS | Valid raw data; parser duplication fixed later. |
| 2 | 946567 | Added barriers/synchronization around timing and direct `RESULT` output with `loop_count=50`. | Timing needed less rank skew and explicit completion. | PASS with exactly 24 `RESULT` rows. | MPI PASS, NCCL PASS | MPI/NCCL measured across full sweep. |
| 3 | 946573 | Added 3 trials per size, `SUMMARY` avg/min/max, parser, CSV, abort on correctness failure. | Single-trial timing was noisy; CSV should mark valid data only after correctness pass. | PASS; CSV `result/pingpong_results_946573.csv` has 72 data rows. | MPI PASS, NCCL PASS | Valid multi-trial data. |
| 4 | 946584 | Changed NCCL stream to `cudaStreamNonBlocking`. | Non-blocking stream may avoid incidental default-stream serialization. | PASS; CSV `result/pingpong_results_946584.csv` has 72 data rows. | MPI PASS, NCCL PASS | Large-size throughput reached ~24 GB/s MPI and ~22.9 GB/s NCCL. |
| 5 | 946595 | Final confirmation using best valid settings from submission 4. | Confirm current best configuration remains correct and stable. | PASS; CSV `result/pingpong_results_946595.csv` has 72 data rows. | MPI PASS, NCCL PASS | Final valid data used below. |

## 4. Correctness Summary

- MPI pingpong correctness: PASS
- NCCL pingpong correctness: PASS
- Failed size: none
- Ranks/GPUs: 2 MPI ranks and 2 GPUs used
- Waivers: none
- Final stderr: non-fatal module load banner only

## 5. Performance Table

Final confirmation job: `946595`. Values are per one-way transfer average over 3 trials, with 50 ping-pong iterations per trial.

| size_bytes | MPI avg_time_s | MPI GB/s | NCCL avg_time_s | NCCL GB/s | faster | speedup |
|---:|---:|---:|---:|---:|---|---:|
| 524288 | 0.000034077 | 15.385 | 0.000051315 | 10.217 | MPI | 1.51x |
| 1048576 | 0.000053718 | 19.520 | 0.000073982 | 14.173 | MPI | 1.38x |
| 2097152 | 0.000097241 | 21.566 | 0.000120382 | 17.421 | MPI | 1.24x |
| 4194304 | 0.000184562 | 22.726 | 0.000216224 | 19.398 | MPI | 1.17x |
| 8388608 | 0.000359130 | 23.358 | 0.000416881 | 20.122 | MPI | 1.16x |
| 16777216 | 0.000706188 | 23.757 | 0.000801410 | 20.935 | MPI | 1.13x |
| 33554432 | 0.001398558 | 23.992 | 0.001495061 | 22.444 | MPI | 1.07x |
| 67108864 | 0.002782567 | 24.118 | 0.002959643 | 22.675 | MPI | 1.06x |
| 134217728 | 0.005549515 | 24.185 | 0.005888496 | 22.793 | MPI | 1.06x |
| 268435456 | 0.011083586 | 24.219 | 0.011746436 | 22.853 | MPI | 1.06x |
| 536870912 | 0.022148203 | 24.240 | 0.023461629 | 22.883 | MPI | 1.06x |
| 1073741824 | 0.044281424 | 24.248 | 0.046892388 | 22.898 | MPI | 1.06x |

## 6. Interpretation

- Small sizes: tuned CUDA-aware MPI is faster in the final run, especially at 512 KiB to 2 MiB.
- Large sizes: both backends plateau near the same transport limit; MPI is still about 6% faster in the final run.
- Topology: final large-size bandwidth around 23-24 GB/s is consistent with a fast intra-node GPU-GPU path, likely NVLink-class rather than PCIe-only.
- NCCL vs MPI: NCCL does not outperform MPI for this two-rank ping-pong pattern in the final confirmation.
- Tuned UCX: the `self,shm,cuda_copy,cuda_ipc` UCX launcher makes MPI very competitive and ultimately fastest in this test.

## 7. Limitations

- Only 2 ranks / 2 GPUs tested.
- Single node only.
- No multi-node test.
- No full GPU pair matrix beyond the Slurm-assigned pair.
- Submit limit respected: baseline + 5 optimization jobs.
- Results depend on the GPU pair assigned by Slurm; early jobs showed lower bandwidth on a different visible GPU UUID pair.

## 8. Final Conclusion

SUCCESS: MPI and NCCL correctness PASS, performance measured, full size sweep preserved, raw output/CSV/summary saved.


## CG 5.5 web
你的結論大致正確，但需要加上兩個邊界條件：

1. **這是 2 ranks / 2 GPUs / single-node / Slurm 指派 GPU pair 下的結果。**
2. **這是 ping-pong send/recv pattern，不代表 NCCL 在 allreduce 或多 GPU collective 上一定比較慢。**

***

## 結果判定

Codex 這次結果可以標記為：

```text
SUCCESS
MPI correctness: PASS
NCCL correctness: PASS
full size sweep: PASS
CSV / raw output / summary: PASS
submission limit: baseline + 5 jobs, 符合限制
```

Final job `946595` 使用 2 張 V100，`CUDA_VISIBLE_DEVICES=0,1`，MPI ranks 為 2，MPI 與 NCCL 都完成完整 size sweep，且 correctness 都是 `PASS`。 [\[ntums365-m...epoint.com\]](https://ntums365-my.sharepoint.com/personal/r14525078_m365_ntu_edu_tw/Documents/Microsoft%20Copilot%20Chat%20%E6%AA%94%E6%A1%88/agent_summary.md), [\[pingpong_c...ult_946595 \| Txt\]](https://ntums365-my.sharepoint.com/personal/r14525078_m365_ntu_edu_tw/Documents/Microsoft%20Copilot%20Chat%20%E6%AA%94%E6%A1%88/pingpong_cuda_result_946595.txt), [\[pingpong_r...lts_946595 \| Excel\]](https://ntums365-my.sharepoint.com/personal/r14525078_m365_ntu_edu_tw/_layouts/15/Doc.aspx?sourcedoc=%7BC6F1554A-E982-4EBF-8884-66490DADC379%7D&file=pingpong_results_946595.csv&action=default&mobileredirect=true)

***

## 最終效能結論

你的描述：

```text
tuned CUDA-aware MPI was fastest across the final sweep, reaching about 24.25 GB/s at 1 GiB;
NCCL reached about 22.90 GB/s. Both are valid because correctness passed for both backends.
```

這是正確的。

在 `1073741824 bytes` 也就是 1 GiB size 下：

```text
MPI avg GB/s  = 24.248
NCCL avg GB/s = 22.898
```

MPI 約比 NCCL 快：

```text
24.248 / 22.898 ≈ 1.06x
```

也就是約 6% faster。 [\[ntums365-m...epoint.com\]](https://ntums365-my.sharepoint.com/personal/r14525078_m365_ntu_edu_tw/Documents/Microsoft%20Copilot%20Chat%20%E6%AA%94%E6%A1%88/agent_summary.md), [\[pingpong_r...lts_946595 \| Excel\]](https://ntums365-my.sharepoint.com/personal/r14525078_m365_ntu_edu_tw/_layouts/15/Doc.aspx?sourcedoc=%7BC6F1554A-E982-4EBF-8884-66490DADC379%7D&file=pingpong_results_946595.csv&action=default&mobileredirect=true)

完整趨勢也是 MPI 全尺寸都比 NCCL 快：

```text
512 KiB: MPI 15.385 GB/s, NCCL 10.217 GB/s
1 MiB:   MPI 19.520 GB/s, NCCL 14.173 GB/s
32 MiB:  MPI 23.992 GB/s, NCCL 22.444 GB/s
1 GiB:   MPI 24.248 GB/s, NCCL 22.898 GB/s
```

 [\[ntums365-m...epoint.com\]](https://ntums365-my.sharepoint.com/personal/r14525078_m365_ntu_edu_tw/Documents/Microsoft%20Copilot%20Chat%20%E6%AA%94%E6%A1%88/agent_summary.md), [\[pingpong_r...lts_946595 \| Excel\]](https://ntums365-my.sharepoint.com/personal/r14525078_m365_ntu_edu_tw/_layouts/15/Doc.aspx?sourcedoc=%7BC6F1554A-E982-4EBF-8884-66490DADC379%7D&file=pingpong_results_946595.csv&action=default&mobileredirect=true)

***

## 為什麼 MPI 會比較快

這次 MPI 使用的是 tuned CUDA-aware UCX path：

```bash
UCX_TLS=self,shm,cuda_copy,cuda_ipc
mpirun -x UCX_TLS --mca coll ^hcoll,ucc --mca pml ucx -n 2
```

這個設定避開了先前壞掉的 GDRCopy path，並使用 `cuda_copy` / `cuda_ipc` 這類 intra-node GPU buffer transport。Codex summary 也明確指出 tuned UCX launcher 讓 MPI 在此測試中最快。 [\[ntums365-m...epoint.com\]](https://ntums365-my.sharepoint.com/personal/r14525078_m365_ntu_edu_tw/Documents/Microsoft%20Copilot%20Chat%20%E6%AA%94%E6%A1%88/agent_summary.md), [\[pingpong_c...ult_946595 \| Txt\]](https://ntums365-my.sharepoint.com/personal/r14525078_m365_ntu_edu_tw/Documents/Microsoft%20Copilot%20Chat%20%E6%AA%94%E6%A1%88/pingpong_cuda_result_946595.txt)

NCCL 版本雖然也 correctness PASS，但在這個 **two-rank ping-pong send/recv** pattern 下沒有超過 MPI。這不代表 NCCL 一般比較差，只代表在此程式、此 pair、此 message sweep 下，MPI path 較快。 [\[ntums365-m...epoint.com\]](https://ntums365-my.sharepoint.com/personal/r14525078_m365_ntu_edu_tw/Documents/Microsoft%20Copilot%20Chat%20%E6%AA%94%E6%A1%88/agent_summary.md), [\[pingpong_r...lts_946595 \| Excel\]](https://ntums365-my.sharepoint.com/personal/r14525078_m365_ntu_edu_tw/_layouts/15/Doc.aspx?sourcedoc=%7BC6F1554A-E982-4EBF-8884-66490DADC379%7D&file=pingpong_results_946595.csv&action=default&mobileredirect=true)

***

## 是否達到極限

目前可說：

```text
pingpong-cuda 在此 2-GPU pair 上已接近單 pair 大訊息傳輸穩定區間。
```

理由是：

* MPI 在大 size 逐漸 plateau 到約 `24.2 GB/s`。
* NCCL 在大 size plateau 到約 `22.9 GB/s`。
* 這與你前面 `p2p-cuda` 中 NV1 route 約 `24.25 GB/s` 的量級一致。  
  也就是這次 Slurm 指派到的 GPU pair 很可能不是 NV2 48 GB/s pair，而是約 NV1 等級的路徑。 [\[ntums365-m...epoint.com\]](https://ntums365-my.sharepoint.com/personal/r14525078_m365_ntu_edu_tw/Documents/Microsoft%20Copilot%20Chat%20%E6%AA%94%E6%A1%88/agent_summary.md), [\[pingpong_r...lts_946595 \| Excel\]](https://ntums365-my.sharepoint.com/personal/r14525078_m365_ntu_edu_tw/_layouts/15/Doc.aspx?sourcedoc=%7BC6F1554A-E982-4EBF-8884-66490DADC379%7D&file=pingpong_results_946595.csv&action=default&mobileredirect=true)

但不能說：

```text
已達整台 node 的 pingpong 極限
```

因為沒有做：

```text
1. 全 GPU pair matrix
2. NV1 / NV2 / SYS pair 對照
3. 多節點測試
4. NCCL tuning matrix
5. 不同 UCX_TLS 組合比較
```

Codex summary 也有列出限制：只測 2 ranks / 2 GPUs、single node、沒有完整 GPU pair matrix，結果依賴 Slurm 指派的 GPU pair。 [\[ntums365-m...epoint.com\]](https://ntums365-my.sharepoint.com/personal/r14525078_m365_ntu_edu_tw/Documents/Microsoft%20Copilot%20Chat%20%E6%AA%94%E6%A1%88/agent_summary.md)

***

## 建議寫入報告的版本

可以這樣寫：

```text
For pingpong-cuda, both MPI and NCCL backends passed correctness on 2 MPI ranks and 2 V100 GPUs. The final configuration used a tuned CUDA-aware MPI launcher with UCX_TLS=self,shm,cuda_copy,cuda_ipc and disabled hcoll/ucc. In the final sweep, tuned MPI was faster across all message sizes. At 1 GiB, MPI reached 24.248 GB/s, while NCCL reached 22.898 GB/s, corresponding to about 1.06x higher bandwidth for MPI.

This result should be interpreted as specific to the two-rank ping-pong pattern and the Slurm-assigned GPU pair. It does not imply that MPI is generally faster than NCCL for all collectives or all topologies. The result establishes a valid ping-pong baseline and shows that the tuned CUDA-aware MPI path is the best tested configuration for this benchmark.
```

***

## 最終判定

```text
任務結果：成功
最佳 backend：tuned CUDA-aware MPI
最佳大訊息結果：約 24.25 GB/s at 1 GiB
NCCL 大訊息結果：約 22.90 GB/s at 1 GiB
correctness：MPI / NCCL 都 PASS
是否顯著：MPI 對 NCCL 約 6% 優勢，對此 final sweep 有意義
下一步：若要找極限，應做 GPU pair matrix，而不是只重複同一 pair
```

## TOKEN
55% left (115,801 used / 258K)