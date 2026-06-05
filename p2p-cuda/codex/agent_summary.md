# p2p-cuda P2P Bandwidth Search Summary
TOKENS : 11K
## Experiment Environment

- Benchmark: `/home/r14525078/HeCBench/src/p2p-cuda`
- Result directory: `/home/r14525078/HeCBench/src/p2p-cuda/result`
- GPU model: Tesla V100-SXM2-32GB
- CUDA: 12.8, nvcc V12.8.61
- CUDA arch: `sm_70`
- Final successful job: `945206`
- Final successful node: `gn1227.twcc.ai`
- CUDA_VISIBLE_DEVICES: `0,1,2,3`
- Visible GPUs: 4
- Methods tested: `cudaMemcpyPeerAsync`, `cudaMemcpyAsync(Default)`
- Buffer sizes: 1 MiB, 8 MiB, 16 MiB, 64 MiB, 128 MiB, 256 MiB
- Correctness: original cross-GPU kernel verification preserved; all 12 measured directional pairs PASS in final run.

Topology summary from `nvidia-smi topo -m`:

| Pair | Link |
|---|---|
| 0-1 | NV2 |
| 0-2 | NV1 |
| 0-3 | SYS |
| 1-2 | NV2 |
| 1-3 | SYS |
| 2-3 | NV1 |

## Submission History

| Job ID | Change / Purpose | Status | Main Result |
|---|---|---|---|
| 945122 | Stage 0 confirmation; added topology output only, kept previous optimized `main.cu`. | Success | 2 GPUs, NV2, PASS, 48.25 GB/s. |
| 945134 | First 4-GPU all-pair sweep, correctness enlarged to 256 MiB. | Failed | 4 GPUs allocated, but correctness FAIL; results invalid. |
| 945137 | Correctness restored to 64 MiB; full 4-GPU sweep with original large repeats. | Failed | Correctness PASS for completed pairs, but job hit time limit during slow SYS routes. |
| 945206 | Reduced-repeat bounded sweep, 4 GPUs, all pairs/sizes/methods, top-3 variance. | Success | 144 PASS sweep points, 12/12 correctness PASS, best stable avg 48.4455 GB/s. |

Stderr note: final job stderr contains only the non-fatal CUDA 12.8 warning that offline compilation support for pre-sm_75 targets will be removed in a future release.

## Best Stable Results

Top 3 configurations were repeated 5 times:

| Pair | Direction | Method | Buffer | Repeat | Warmup | Avg GB/s | Min GB/s | Max GB/s | Stddev | CV | Correctness |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 2->1 | 2 to 1 | cudaMemcpyAsync(Default) | 268435456 | 64 | 8 | 48.4455 | 48.4441 | 48.4459 | 0.0007 | 0.000014 | PASS |
| 0->1 | 0 to 1 | cudaMemcpyAsync(Default) | 268435456 | 64 | 8 | 48.4451 | 48.4448 | 48.4455 | 0.0003 | 0.000005 | PASS |
| 1->2 | 1 to 2 | cudaMemcpyPeerAsync | 268435456 | 64 | 8 | 48.4449 | 48.4441 | 48.4457 | 0.0007 | 0.000014 | PASS |

Best observed single sweep value:

| Pair | Method | Buffer | Repeat | Warmup | GB/s | Correctness |
|---|---|---:|---:|---:|---:|---|
| 0->1 | cudaMemcpyAsync(Default) | 268435456 | 64 | 8 | 48.4459 | PASS |
| 1->2 | cudaMemcpyPeerAsync | 268435456 | 64 | 8 | 48.4459 | PASS |
| 2->1 | cudaMemcpyAsync(Default) | 268435456 | 64 | 8 | 48.4459 | PASS |

## Best Per Direction

| Pair | Link Class | Best Method | Best Buffer | Repeat | Best GB/s |
|---|---|---|---:|---:|---:|
| 0->1 | NV2 | cudaMemcpyAsync(Default) | 268435456 | 64 | 48.4459 |
| 1->0 | NV2 | cudaMemcpyAsync(Default) | 268435456 | 64 | 48.4447 |
| 1->2 | NV2 | cudaMemcpyPeerAsync | 268435456 | 64 | 48.4459 |
| 2->1 | NV2 | cudaMemcpyAsync(Default) | 268435456 | 64 | 48.4459 |
| 0->2 | NV1 | cudaMemcpyPeerAsync | 268435456 | 64 | 24.2450 |
| 2->0 | NV1 | cudaMemcpyPeerAsync | 268435456 | 64 | 24.2450 |
| 2->3 | NV1 | cudaMemcpyPeerAsync | 268435456 | 64 | 24.2453 |
| 3->2 | NV1 | cudaMemcpyAsync(Default) | 268435456 | 64 | 24.2425 |
| 0->3 | SYS | cudaMemcpyPeerAsync | 268435456 | 64 | 8.8140 |
| 1->3 | SYS | cudaMemcpyAsync(Default) | 268435456 | 64 | 8.5559 |
| 3->0 | SYS | cudaMemcpyAsync(Default) | 1048576 | 8000 | 3.2940 |
| 3->1 | SYS | cudaMemcpyPeerAsync | 1048576 | 8000 | 3.3135 |

## Baseline Comparison

| Reference | GB/s | Delta vs New Stable Avg |
|---|---:|---:|
| Previous baseline | 48.17 | +0.575% |
| Previous optimized final | 48.24 | +0.426% |
| New best stable average | 48.4455 | 0% |
| New best observed | 48.4459 | +0.001% vs stable avg |

The new result is higher than 48.24 GB/s, but the gain is below 1%, so it should be treated as measurement-equivalent rather than a significant acceleration.

## Interpretation

- NV2 pairs saturate around 48.45 GB/s and are the practical single-pair limit observed on this node.
- NV1 pairs saturate around 24.25 GB/s, about half of NV2, matching the topology.
- SYS routes are much slower and asymmetric: GPU0/1 -> GPU3 is about 8.5-8.8 GB/s, while GPU3 -> GPU0/1 is about 3.0-3.3 GB/s.
- Larger buffers improve measured bandwidth. 1 MiB underutilizes the copy path; 64 MiB and above approach steady state; 256 MiB was consistently best or tied for best on NVLink routes.
- `cudaMemcpyPeerAsync` and `cudaMemcpyAsync(Default)` are effectively tied on NV2/NV1 routes. Differences are far below 1% and not significant.
- Results appear topology and hardware limited. No CUDA API change tested here moved NV2 beyond about 48.45 GB/s.

## Limitations

- Used 4 of 8 allowed sbatch submissions in this run.
- Final successful coverage used 4 GPUs on one node.
- No multi-node P2P, NCCL, MPI, or concurrent multi-pair bandwidth testing was performed.
- Slow SYS routes required reduced repeats to finish within scheduler time while still keeping measurement windows in the hundreds of ms to seconds range.
- The top-3 variance trials were done only for the highest NV2 configurations, not every pair/size/method.

## Final Conclusion

Best observed GB/s: 48.4459 GB/s.

Best stable average GB/s: 48.4455 GB/s.

Best stable configuration: GPU2 -> GPU1, `cudaMemcpyAsync(Default)`, 256 MiB buffer, repeat 64, warmup 8, correctness PASS.

This is likely very close to the practical single-pair NV2 P2P copy limit for this V100-SXM2 node. Further meaningful improvement is unlikely from `cudaMemcpyPeerAsync` vs `cudaMemcpyAsync(Default)` selection alone; the next worthwhile directions would be concurrent multi-pair aggregate bandwidth, NCCL/MPI transport comparison, or node/topology selection rather than single-pair API tuning.



根據 `agent_summary.md`，這次 Codex 的結果比上一輪完整很多，結論也比較可信。

## 總評

可以接受，且基本達到「可控範圍內尋找單 pair P2P bandwidth 極限」的目標。

但結論要精準寫成：

```text
已接近此 V100-SXM2 節點上 NV2 link 的單向 single-pair P2P copy 實測極限。
```

不能寫成：

```text
已達整台節點總 P2P 極限
```

因為它沒有測 concurrent multi-pair aggregate bandwidth，也沒有測 NCCL / MPI / 多節點。 [\[ntums365-m...epoint.com\]](https://ntums365-my.sharepoint.com/personal/r14525078_m365_ntu_edu_tw/Documents/Microsoft%20Copilot%20Chat%20%E6%AA%94%E6%A1%88/agent_summary.md)

***

## 1. 實驗覆蓋度

這次比前一版明顯完整：

* 使用 4 張 GPU：`CUDA_VISIBLE_DEVICES=0,1,2,3`。 [\[ntums365-m...epoint.com\]](https://ntums365-my.sharepoint.com/personal/r14525078_m365_ntu_edu_tw/Documents/Microsoft%20Copilot%20Chat%20%E6%AA%94%E6%A1%88/agent_summary.md)
* 測試 12 個 directional pairs，final run 中 12/12 correctness PASS。 [\[ntums365-m...epoint.com\]](https://ntums365-my.sharepoint.com/personal/r14525078_m365_ntu_edu_tw/Documents/Microsoft%20Copilot%20Chat%20%E6%AA%94%E6%A1%88/agent_summary.md)
* 測試兩種 copy method：`cudaMemcpyPeerAsync` 與 `cudaMemcpyAsync(Default)`。 [\[ntums365-m...epoint.com\]](https://ntums365-my.sharepoint.com/personal/r14525078_m365_ntu_edu_tw/Documents/Microsoft%20Copilot%20Chat%20%E6%AA%94%E6%A1%88/agent_summary.md)
* 測試 buffer sizes：1 MiB、8 MiB、16 MiB、64 MiB、128 MiB、256 MiB。 [\[ntums365-m...epoint.com\]](https://ntums365-my.sharepoint.com/personal/r14525078_m365_ntu_edu_tw/Documents/Microsoft%20Copilot%20Chat%20%E6%AA%94%E6%A1%88/agent_summary.md)
* 對 top 3 configurations 重複 5 次並計算平均、min、max、stddev、CV。 [\[ntums365-m...epoint.com\]](https://ntums365-my.sharepoint.com/personal/r14525078_m365_ntu_edu_tw/Documents/Microsoft%20Copilot%20Chat%20%E6%AA%94%E6%A1%88/agent_summary.md)

這已經不是單點測試，而是合理的 topology-aware sweep。

***

## 2. 最佳結果是否有意義

最佳穩定平均值：

```text
48.4455 GB/s
```

最佳觀測值：

```text
48.4459 GB/s
```

最佳穩定設定：

```text
GPU2 -> GPU1
cudaMemcpyAsync(Default)
256 MiB buffer
repeat=64
warmup=8
correctness PASS
```

 [\[ntums365-m...epoint.com\]](https://ntums365-my.sharepoint.com/personal/r14525078_m365_ntu_edu_tw/Documents/Microsoft%20Copilot%20Chat%20%E6%AA%94%E6%A1%88/agent_summary.md)

和前一版比較：

```text
previous baseline: 48.17 GB/s
previous optimized: 48.24 GB/s
new stable avg: 48.4455 GB/s
```

提升：

```text
vs 48.17: +0.575%
vs 48.24: +0.426%
```

 [\[ntums365-m...epoint.com\]](https://ntums365-my.sharepoint.com/personal/r14525078_m365_ntu_edu_tw/Documents/Microsoft%20Copilot%20Chat%20%E6%AA%94%E6%A1%88/agent_summary.md)

這個提升低於 1%，所以不能稱為顯著加速。Codex 自己也正確判定為：

```text
measurement-equivalent rather than significant acceleration
```

這個判斷是對的。 [\[ntums365-m...epoint.com\]](https://ntums365-my.sharepoint.com/personal/r14525078_m365_ntu_edu_tw/Documents/Microsoft%20Copilot%20Chat%20%E6%AA%94%E6%A1%88/agent_summary.md)

***

## 3. 是否接近極限

就目前資料來看，**NV2 pair 幾乎已經達到單向 single-pair copy 的穩定上限**。

原因是不同 NV2 pair 都集中在：

```text
約 48.44 GB/s
```

例如：

```text
0 -> 1: 48.4459 GB/s
1 -> 0: 48.4447 GB/s
1 -> 2: 48.4459 GB/s
2 -> 1: 48.4459 GB/s
```

 [\[ntums365-m...epoint.com\]](https://ntums365-my.sharepoint.com/personal/r14525078_m365_ntu_edu_tw/Documents/Microsoft%20Copilot%20Chat%20%E6%AA%94%E6%A1%88/agent_summary.md)

而 NV1 pair 則大約是：

```text
約 24.24 GB/s
```

例如：

```text
0 -> 2: 24.2450 GB/s
2 -> 0: 24.2450 GB/s
2 -> 3: 24.2453 GB/s
3 -> 2: 24.2425 GB/s
```

 [\[ntums365-m...epoint.com\]](https://ntums365-my.sharepoint.com/personal/r14525078_m365_ntu_edu_tw/Documents/Microsoft%20Copilot%20Chat%20%E6%AA%94%E6%A1%88/agent_summary.md)

這個比例很合理：

```text
NV2 約為 NV1 的 2 倍
```

說明結果主要受 NVLink topology 限制，而不是 copy API 選擇限制。 [\[ntums365-m...epoint.com\]](https://ntums365-my.sharepoint.com/personal/r14525078_m365_ntu_edu_tw/Documents/Microsoft%20Copilot%20Chat%20%E6%AA%94%E6%A1%88/agent_summary.md)

***

## 4. Codex 有沒有真正「優化」

嚴格說，Codex 的主要貢獻不是讓單一 copy API 變快，而是：

```text
把 benchmark 從單點測試改成 topology-aware bandwidth characterization。
```

它做出的有效改進包括：

1. 測了所有方向性 GPU pair。 [\[ntums365-m...epoint.com\]](https://ntums365-my.sharepoint.com/personal/r14525078_m365_ntu_edu_tw/Documents/Microsoft%20Copilot%20Chat%20%E6%AA%94%E6%A1%88/agent_summary.md)
2. 做了 buffer size sweep。 [\[ntums365-m...epoint.com\]](https://ntums365-my.sharepoint.com/personal/r14525078_m365_ntu_edu_tw/Documents/Microsoft%20Copilot%20Chat%20%E6%AA%94%E6%A1%88/agent_summary.md)
3. 比較 `cudaMemcpyPeerAsync` 與 `cudaMemcpyAsync(Default)`。 [\[ntums365-m...epoint.com\]](https://ntums365-my.sharepoint.com/personal/r14525078_m365_ntu_edu_tw/Documents/Microsoft%20Copilot%20Chat%20%E6%AA%94%E6%A1%88/agent_summary.md)
4. 對 top 3 做 variance measurement。 [\[ntums365-m...epoint.com\]](https://ntums365-my.sharepoint.com/personal/r14525078_m365_ntu_edu_tw/Documents/Microsoft%20Copilot%20Chat%20%E6%AA%94%E6%A1%88/agent_summary.md)
5. 保留 correctness，final run 中 144 個 sweep points PASS，12/12 directional pair correctness PASS。 [\[ntums365-m...epoint.com\]](https://ntums365-my.sharepoint.com/personal/r14525078_m365_ntu_edu_tw/Documents/Microsoft%20Copilot%20Chat%20%E6%AA%94%E6%A1%88/agent_summary.md)

所以這題應該分類為：

```text
AI 優化成功，但效能提升不顯著；主要成果是測量方法與拓撲探索完整化。
```

不應分類為：

```text
AI 大幅加速成功。
```

***

## 5. 需要注意的問題

### 問題 1：CV 小到不尋常

Top 3 的 stddev 例如：

```text
0.0007 GB/s
CV = 0.000014
```

 [\[ntums365-m...epoint.com\]](https://ntums365-my.sharepoint.com/personal/r14525078_m365_ntu_edu_tw/Documents/Microsoft%20Copilot%20Chat%20%E6%AA%94%E6%A1%88/agent_summary.md)

這非常穩，可能是因為 measurement window 足夠大，也可能是因為 CUDA event timing 對固定 copy path 很穩。不是錯，但報告中建議保守描述為：

```text
在該測量方法下高度穩定
```

不要過度推論到所有 workload。

### 問題 2：SYS route 沒有 top-3 variance

SYS route 很慢且不對稱：

```text
0/1 -> 3: 約 8.5-8.8 GB/s
3 -> 0/1: 約 3.0-3.3 GB/s
```

 [\[ntums365-m...epoint.com\]](https://ntums365-my.sharepoint.com/personal/r14525078_m365_ntu_edu_tw/Documents/Microsoft%20Copilot%20Chat%20%E6%AA%94%E6%A1%88/agent_summary.md)

但 top-3 variance 只針對最高 NV2 configurations，沒有對 SYS route 做完整 variance。這不是致命問題，因為目標是找 peak P2P bandwidth；但若要分析拓撲不對稱，需要補測。

### 問題 3：沒有 concurrent aggregate bandwidth

目前只測 single-pair：

```text
GPUi -> GPUj
```

沒有測：

```text
GPU0 -> GPU1 同時 GPU2 -> GPU3
```

所以不能推論整個節點 aggregate NVLink bandwidth。 [\[ntums365-m...epoint.com\]](https://ntums365-my.sharepoint.com/personal/r14525078_m365_ntu_edu_tw/Documents/Microsoft%20Copilot%20Chat%20%E6%AA%94%E6%A1%88/agent_summary.md)

***

## 6. 建議寫入報告的結論

可以這樣寫：

```text
在 4-GPU V100-SXM2 allocation 上，p2p-cuda 經 AI agent 擴充為 topology-aware P2P bandwidth sweep。最終成功測試 12 個 directional GPU pairs、6 種 buffer sizes 與 2 種 CUDA copy methods，所有 final measured pairs correctness 均通過。

最佳穩定平均值為 48.4455 GB/s，出現在 GPU2 -> GPU1、cudaMemcpyAsync(Default)、256 MiB buffer 設定。此結果相較先前最佳 48.24 GB/s 僅提升約 0.426%，低於 1%，因此不視為顯著加速，而應視為 measurement-equivalent。

實驗顯示 bandwidth 主要由 NVLink topology 決定：NV2 routes 約 48.45 GB/s，NV1 routes 約 24.25 GB/s，SYS routes 明顯較低且具方向不對稱。cudaMemcpyPeerAsync 與 cudaMemcpyAsync(Default) 在 NVLink routes 上差異不顯著。故此 benchmark 的單 pair P2P copy 已接近該節點 NV2 route 的實測上限，後續若要提升應轉向 concurrent multi-pair aggregate bandwidth、NCCL/MPI transport 或不同 node topology，而非單純替換 CUDA copy API。
```

***

## 7. 最終判定

```text
是否達到單 pair 極限：
是，對 NV2 single-pair P2P copy 而言已非常接近。

是否有顯著加速：
否，提升低於 1%，屬於 measurement-equivalent。

是否值得繼續優化同一方向：
不值得。cudaMemcpyPeerAsync vs cudaMemcpyAsync(Default) 已無明顯差異。

是否值得繼續研究：
值得，但方向要換成：
1. concurrent multi-pair aggregate bandwidth
2. NCCL P2P / NCCL allreduce
3. MPI CUDA-aware transport
4. 不同 GPU topology / node selection
5. SYS route asymmetry 原因分析
```
