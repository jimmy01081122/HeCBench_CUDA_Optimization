建議把下一階段收斂成 **Phase 3：人機協作式 AI 程式優化實驗**。  
重點不要再擴大 benchmark 數量，而是從 Phase 2 中選出「純 AI 程式優化能力」低、中、高三類代表案例，做更深的協作流程、prompt、文獻查詢與可驗證工作流研究。

***

# 1. 我建議的三個代表 benchmark

## 1.1 低優化空間代表：`shmembench-cuda`

選它作為低優化空間代表。

理由：

```text
類型：shared memory microbenchmark
Phase 2 P3 speedup：約 1.0293x
主要限制：接近硬體或微基準測量上限，需要 profiler 才能確認是否真有有效改善
```

這類題目適合研究：

```text
當 benchmark 已接近硬體限制時，AI agent 是否會過度宣稱微小提升？
人類如何透過 profiler、variance、measurement-equivalent 規則限制 agent？
```

Phase 2 資料顯示 `shmembench-cuda` 屬小幅改善案例，且後續建議中明確提到應補 shared bank conflict / profiler 指標。 [\[arxiv.org\]](https://arxiv.org/abs/2510.00555), [\[arxiv.org\]](https://arxiv.org/abs/2403.17134)

***

## 1.2 中優化空間代表：`topk-cuda`

選它作為中等優化空間代表。

理由：

```text
類型：Top-K radix selection / workspace-heavy AI primitive
Phase 2 P3 speedup：約 1.1995x
主要策略：CUB workspace reuse、block size tuning、減少 repeated allocation / synchronization
```

這類題目適合研究：

```text
AI 是否能從程式結構中辨識 repeated allocation / workspace overhead？
人類是否能透過文獻與 profiler 指標，引導 agent 從局部 tuning 走向更有根據的演算法設計？
```

Phase 2 資料顯示 `topk-cuda` 的有效策略是 workspace reuse 與 block-size tuning，GM 版本在 14 組 hidden\_size/topk 組合上達到 1.442x，P3 normalized 結果約 1.1995x。 [\[arxiv.org\]](https://arxiv.org/abs/2510.00555)

***

## 1.3 高優化空間代表：`softmax-cuda`

選它作為高優化空間代表。

理由：

```text
類型：row-wise softmax / reduction + exp + normalization
Phase 2 P3 speedup：約 1.4575x
BASIC/GM 探索性結果：slice=784 曾達 59.593x
主要策略：warp-level / block-level reduction、cached expf、shape-specific dispatch
```

這類題目適合研究：

```text
AI 在有明確演算法重構空間時，是否能產生實質 kernel optimization？
人類如何協助建立 shape-aware dispatch policy？
如何避免只針對單一 shape 過度特化？
```

Phase 2 與 BASIC 資料均顯示 `softmax-cuda` 是最明確的 kernel-level optimization 案例；P3 normalized speedup 約 1.4575x，而 BASIC/GM 探索性實驗在特定 slice=784 下曾達到 59.593x。 [\[arxiv.org\]](https://arxiv.org/abs/2510.00555), [\[arxiv.org\]](https://arxiv.org/abs/2403.17134)

***

# 2. 為什麼不選 allreduce / p2p / pingpong

下一階段你說要研究「純 AI 優化能力」，所以不建議選：

```text
allreduce-cuda
p2p-cuda
pingpong-cuda
simpleMultiDevice-cuda
```

理由：

```text
allreduce-cuda：主要是 ENV_FIX，不是 kernel optimization
p2p-cuda：主要是 TOPOLOGY_MEASURE，提升 <1%，屬 measurement-equivalent
pingpong-cuda：主要是 MPI/NCCL transport comparison
simpleMultiDevice-cuda：主要是 multi-GPU scaling / H2D copy bottleneck
```

這些仍有研究價值，但更適合放在「AI 處理系統環境與通訊設定」章節，不適合代表純程式優化能力。Phase 2 已將 allreduce、p2p、pingpong 類案例明確區分為 environment / measurement / topology 類結果。 [\[arxiv.org\]](https://arxiv.org/abs/2510.00555), [\[arxiv.org\]](https://arxiv.org/abs/2403.17134)

***

# 3. Phase 3 核心研究問題

建議將下一階段收斂為以下 4 個 RQ：

```text
RQ3-1:
在人機協作條件下，AI agent 能否超越 Phase 2 P3 的自動化優化結果？

RQ3-2:
人類介入最有效的位置是：
A. 提供文獻與架構知識
B. 設計驗證規則
C. 限制 agent 搜索空間
D. 判定何時停止優化
哪一項？

RQ3-3:
加入文獻查詢與 profiler 指標後，AI 是否能從 trial-and-error 轉向 hypothesis-driven optimization？

RQ3-4:
低、中、高優化空間 benchmark 中，人機協作的邊際收益是否不同？
```

這組問題比「AI 能不能加速」更有研究價值，因為它聚焦在人機協作與 prompt workflow 的設計。

***

# 4. Phase 3 實驗組設計

每個 benchmark 跑三種協作模式。

## 4.1 Mode A：P3 Agent-only baseline

```text
角色：agent 自主依照 P3 prompt 優化
人類只提供 prompt，不中途介入
```

用途：

```text
作為 Phase 2 的延伸 baseline
```

***

## 4.2 Mode B：Human-in-the-loop guided optimization

```text
角色：
- 人類負責判斷瓶頸、批准假設、要求補驗證
- agent 負責改程式、執行、整理結果
```

這是你的主研究模式。

***

## 4.3 Mode C：Literature-augmented adaptive workflow

```text
角色：
- agent 必須先查文獻 / 文件 / CUDA best practices
- 形成假設
- 再執行優化
- 每次修改都要能回扣資料來源或 profiler observation
```

這一組最接近論文主張：**AI + 人類 + 文獻 + 可驗證工作流**。

互動式 prompt assistant 的研究顯示，使用者在有結構化 prompt 輔助時，任務表現、效率與自主感會提高；這支持你將 prompt 從一次性文字提升成 workflow scaffold 的方向。 [\[arxiv.org\]](https://arxiv.org/abs/2510.00555), [\[aisel.aisnet.org\]](https://aisel.aisnet.org/icis2025/hti/hti/8/)

***

# 5. 每個 benchmark 的 Phase 3 實驗任務

## 5.1 `shmembench-cuda`：低優化空間

### 研究目標

```text
確認 AI 是否能辨識硬體上限，避免對 <1% 或微小結果過度宣稱。
```

### 必做項目

```text
1. 建立 P3 baseline
2. 加 Nsight Compute 或等價 profiler 指標
3. 檢查 shared memory throughput
4. 檢查 bank conflict
5. 測 block size / layout / padding
6. 若提升 <1%，標 measurement-equivalent
```

### 人類介入點

```text
1. 判斷 profiler 指標是否支持 agent 的假設
2. 阻止 agent 將 noise 宣稱為 speedup
3. 決定停止條件
```

### 成功標準

```text
不一定要更快。
成功可以是：
- 證明已接近硬體限制
- 解釋為何 AI 無法進一步提升
- 提供可驗證 profiler-based conclusion
```

這能避免只以 speedup 衡量 AI 能力。

***

## 5.2 `topk-cuda`：中優化空間

### 研究目標

```text
從 workspace reuse 進一步探索 radix selection / block size / dispatch policy。
```

### 必做項目

```text
1. 固定 Phase 2 P3 最佳版本作 baseline
2. 查詢 CUDA Top-K / radix selection / CUB workspace 相關資料
3. 分析 workspace allocation 是否仍在 timed path
4. 測 block size 256 / 512 / 1024
5. 測 hidden_size/topk shape sensitivity
6. 嘗試 shape-aware dispatch
7. 每組 correctness 全通過才納入
```

### 人類介入點

```text
1. 判斷 agent 提出的 radix 改法是否改變 exact top-k 語意
2. 阻止只優化單一 hidden_size
3. 要求輸出完整 shape table
```

### 成功標準

```text
1. final speedup > Phase 2 P3
2. 或產出更穩定的 shape-aware policy
3. correctness 全 shape PASS
```

***

## 5.3 `softmax-cuda`：高優化空間

### 研究目標

```text
從單一 optimized kernel 走向 shape-aware softmax dispatch policy。
```

### 必做項目

```text
1. 使用 Phase 2 P3 normalized result 作正式 baseline
2. 保留 BASIC/GM 59.593x 作 exploratory reference，不直接混入統計
3. 查詢 softmax CUDA optimization / warp-level reduction / block-level reduction
4. 測 slice=128,256,784,1024,2048
5. 對不同 slice 建立 dispatch：
   - small slice: warp-level
   - medium slice: block-level
   - large slice: multi-warp/block strategy
6. 檢查 correctness tolerance 是否合理
7. 用 profiler 檢查 expf、global load/store、occupancy
```

### 人類介入點

```text
1. 決定是否接受 --use_fast_math
2. 判斷 tolerance 是否過寬
3. 防止 agent 只針對 slice=784 過度特化
4. 要求 dispatch policy 必須可解釋
```

### 成功標準

```text
1. P3 baseline 之上有穩定提升
2. 各 slice correctness PASS
3. dispatch policy 可解釋
4. profiler 指標支持優化原因
```

LLM-driven CUDA kernel optimization 的研究已經開始強調 verification、profiling、benchmark diversity 與防止 exploitable loopholes；這與你將 softmax 作為高優化空間、且要求 correctness/profiler/多 shape 驗證的設計一致。 [\[arxiv.org\]](https://arxiv.org/abs/2509.14279), [\[developer.nvidia.com\]](https://developer.nvidia.com/blog/benchmarking-llms-on-ai-generated-cuda-code-with-computeeval-2025-2/)

***

# 6. 自適應工作流程設計

建議 Phase 3 不再使用固定 5 次提交流程，而是改成 **bounded adaptive workflow**。

## 6.1 每輪固定流程

每一輪都必須：

```text
1. Observe
   讀 baseline、profiler、raw log

2. Diagnose
   形成瓶頸假設

3. Retrieve
   查詢論文 / CUDA 文件 / 既有結果

4. Plan
   提出單一可驗證修改

5. Execute
   sbatch 執行

6. Validate
   correctness + metric + profiler

7. Decide
   accept / reject / rollback / stop
```

這和 RepairAgent 類 agentic program repair 工作流相似：agent 不是只一次性生成補丁，而是迭代地收集資訊、形成假設、執行工具並用測試驗證結果；該研究將 LLM 視為能自主規劃與調用工具的 agent，而不是固定 prompt 的被動生成器。 [\[arxiv.org\]](https://arxiv.org/abs/2403.17134), [\[software-lab.org\]](https://software-lab.org/publications/icse2025_RepairAgent.pdf)

***

## 6.2 自適應停止條件

每個 benchmark 最多：

```text
baseline + 6 optimization jobs + 1 final confirmation
```

停止條件：

```text
1. 連續 2 次有效修改 speedup < 1%
2. correctness fail 且無合理修復方向
3. profiler 顯示已接近瓶頸
4. human reviewer 判定再優化會改變 benchmark 語意
```

***

# 7. Phase 3 Prompt 架構

建議每個 benchmark prompt 分成 7 個 section。

```text
1. Phase 2 evidence
2. Literature retrieval task
3. Bottleneck hypothesis
4. Adaptive workflow
5. Human approval checkpoints
6. Validation protocol
7. Final report schema
```

## 7.1 通用 Prompt 骨架

```markdown
# Phase 3 Human-AI Collaborative Optimization Prompt

You are a CUDA performance engineer collaborating with a human researcher.

## Benchmark

- benchmark:
- path:
- category:
- Phase 2 result:
- Phase 2 result type:
- known limitations:

## Goal

Improve beyond the Phase 2 P3 result if possible.
If improvement is not possible, produce a profiler-backed explanation.

## Required Workflow

For each optimization round:

1. Observe:
   - read baseline result
   - read profiler result if available
   - read source code

2. Diagnose:
   - state bottleneck hypothesis

3. Retrieve:
   - search or cite relevant CUDA / paper / documentation
   - summarize how it applies

4. Plan:
   - propose exactly one modification
   - predict expected effect

5. Human checkpoint:
   - wait for approval before modifying benchmark semantics

6. Execute:
   - build and run
   - save raw output

7. Validate:
   - correctness
   - performance
   - profiler
   - contradiction check

8. Decide:
   - accept
   - reject
   - rollback
   - stop

## Hard Rules

- Do not remove correctness.
- Do not reduce official input to fake speedup.
- If correctness FAIL, result invalid.
- If speedup < 1%, mark measurement-equivalent.
- If profiler contradicts hypothesis, reject modification.
- All results must be reproducible.

## Final Report

Write:
- agent_summary.md
- decision_log.md
- profiler_summary.md
- human_intervention_log.md
```

***

# 8. 人類操作者要做什麼

這階段不要讓人類只當「執行者」。人類應負責以下四件事。

## 8.1 設定研究邊界

```text
哪些修改算有效？
哪些算改變 benchmark 語意？
哪些結果能放論文？
```

## 8.2 審查假設

每輪 agent 必須提出：

```text
bottleneck hypothesis
expected metric change
risk
```

人類只批准合理假設。

## 8.3 決定 rollback

如果 agent 做了：

```text
1. 過度特化
2. correctness tolerance 放太寬
3. 改變測量範圍
4. 跳過慢 case
```

人類必須要求 rollback。

## 8.4 最終分類

由人類決定：

```text
KERNEL_OPT
PARAM_TUNE
MEASURE_FIX
NO_EFFECT
MEASUREMENT_EQUIVALENT
```

***

# 9. 必須產出的資料

每個 benchmark 最終產出：

```text
phase3/<benchmark>/
├── prompt.md
├── baseline/
│   ├── baseline.out
│   ├── baseline.err
│   └── baseline.csv
├── rounds/
│   ├── round_1/
│   ├── round_2/
│   └── ...
├── final/
│   ├── final.out
│   ├── final.err
│   ├── final.csv
│   └── profiler_summary.md
├── decision_log.md
├── human_intervention_log.md
└── agent_summary.md
```

***

# 10. 評估指標

## 10.1 效能指標

```text
speedup over Phase 2 P3
speedup over BASIC if comparable
mean / stddev / CV
```

## 10.2 協作效率

```text
number of human interventions
accepted modifications / total modifications
jobs to first valid improvement
time to final result
```

## 10.3 嚴謹性

```text
correctness pass rate
contradiction count
invalid result count
rollback count
profiler-supported claims
```

## 10.4 人機協作收益

定義：

```text
collaboration_gain =
  final_human_ai_speedup / phase2_p3_agent_only_speedup
```

但只對同一 metric、同一 input、同一 correctness 條件下計算。

***

# 11. 我建議的實驗順序

不要三題一起做。順序應該是：

```text
1. softmax-cuda
2. topk-cuda
3. shmembench-cuda
```

理由：

```text
softmax:
  最容易產生有意義的人機協作加速與論文亮點。

topk:
  適合測 workspace / dispatch / shape-aware reasoning。

shmembench:
  最後做，用來證明人機協作不只是追求 speedup，也能判定硬體限制與停止條件。
```

***

# 12. 最重要的建議

你的研究應收斂成這句主張：

```text
人機協作的價值不在於讓 AI 每次都產生更大 speedup，而在於讓 AI 的優化過程從 trial-and-error 轉為 evidence-driven、correctness-gated、profiler-supported 的可審核工作流。
```

這會比單純展示「AI 加速多少倍」更有論文價值。

***

# 13. 下一步

我建議下一步先做：

```text
Phase 3 softmax-cuda detailed protocol
```

因為它是高優化空間代表，最適合建立人機協作流程模板。

接著依序套用到：

```text
topk-cuda
shmembench-cuda
```

如果你要，我可以下一則直接幫你產出：

```text
Phase 3 softmax-cuda 人機協作 prompt
+ 實驗流程
+ human checkpoint 表
+ decision log 模板
+ profiler 指標要求
```

並且保持後續三題格式一致。