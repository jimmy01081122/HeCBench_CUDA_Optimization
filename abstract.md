# 中文研究摘要：AI 輔助程式優化中的 Prompt 約束、人機協作與可驗證工作流

## 一、研究主題與核心問題

本研究探討在程式優化領域中，如何透過 **prompt.md、AI agent、操作者介入、一般網頁對話、CLI agent、自動化實驗流程** 等不同人機協作形式，最大化 AI 與人類協作的效率、效能與結果可信度。

研究的核心不是單純回答「AI 能否讓程式變快」，而是進一步追問：

```text
1. 什麼樣的 prompt 設計能讓 AI 產生可驗證的優化結果？
2. AI agent 在程式優化中容易犯哪些錯？
3. 人類操作者應該在哪些節點介入？
4. 如何避免偽加速、錯誤 baseline、correctness 缺失與不可重現結果？
5. 如何把 AI 輔助優化從一次性嘗試，轉化為論文級可審核實驗流程？
```

目前研究已累積 HeCBench CUDA benchmark 的多組 AI 輔助優化結果，並整理出跨 benchmark 的結果摘要、prompt inventory 與中文總報告。既有資料顯示，多數完整 prompt 已包含 baseline、correctness、sbatch、submission limit、raw output 等約束，但仍普遍缺少 profiler 指標與統計變異要求，這構成後續研究改進重點。 [\[arxiv.org\]](https://arxiv.org/html/2603.07169v1)

***

## 二、研究資料集與 Benchmark 範圍

本研究目前聚焦於 10 個 HeCBench CUDA benchmark，涵蓋 AI primitive、memory system、multi-GPU communication 與 environment repair 等類型：

```text
1. softmax-cuda
2. topk-cuda
3. moe-cuda
4. moe-align-cuda
5. prefetch-cuda
6. shmembench-cuda
7. p2p-cuda
8. allreduce-cuda
9. pingpong-cuda
10. simpleMultiDevice-cuda
```

這些 benchmark 被分為三大類：

### 1. AI Primitive / Kernel Optimization

包含：

```text
softmax-cuda
topk-cuda
moe-cuda
moe-align-cuda
```

這一類代表純程式碼或 kernel 層級的優化能力。例如 `softmax-cuda` 在 Phase 2 P3 中取得 1.4575x 加速，而早期 BASIC/GM 探索性結果中，特定 slice=784 曾達到 59.593x，但兩者基準不同，不可直接混用。 [\[arxiv.org\]](https://arxiv.org/abs/2510.00555), [\[arxiv.org\]](https://arxiv.org/abs/2403.17134)

### 2. Memory-System / Measurement Benchmark

包含：

```text
prefetch-cuda
shmembench-cuda
p2p-cuda
```

這一類用於分析 AI 在記憶體限制、Unified Memory、shared memory、P2P topology 等情境下的優化邊界。例如 `p2p-cuda` 的提升約 1.004x，屬於 topology-aware measurement 而非顯著 kernel 加速。 [\[arxiv.org\]](https://arxiv.org/abs/2510.00555)

### 3. Multi-GPU / Communication / Environment

包含：

```text
allreduce-cuda
pingpong-cuda
simpleMultiDevice-cuda
```

這一類重點不一定是 kernel 改寫，而是 MPI、NCCL、UCX、Slurm、GPU allocation 與多 GPU scaling。`allreduce-cuda` 的主要成果是透過 tuned UCX launcher 避開 broken GDRCopy path，使原本 baseline 在 size 0 後失敗的 benchmark 得以通過非零 size correctness。 [\[arxiv.org\]](https://arxiv.org/abs/2510.00555), [\[arxiv.org\]](https://arxiv.org/abs/2403.17134)

***

## 三、Prompt 分層：P1、P2、P3

本研究將 prompt 約束強度分成三層。

### P1：弱約束 Prompt

P1 只提供 benchmark path、基本任務目標與少量環境提示，不強制 baseline、raw output、CSV、contradiction check 或 repeated trials。

P1 的問題是：即使 agent 可能產生有效加速，也常缺乏審核軌跡。在 10 個 P1 結果中，至少 4 個存在 baseline 缺失、資料殘缺或測量範圍改變等問題，另有 1 個出現邏輯矛盾。這使 P1 結果不適合作為核心論文證據，除非重新通過 P3 等級驗證。 [\[arxiv.org\]](https://arxiv.org/abs/2403.17134)

### P2：中約束 Prompt

P2 增加基本工程約束，包括：

```text
1. 必須先跑 baseline
2. 必須保留 raw output
3. 設定 submission limit
4. 不得刪除 correctness
5. 需要產出 agent_summary.md
```

P2 已可避免部分明顯錯誤，但仍缺少 P3 等級的 CSV schema、contradiction check、repeated trials 與 profiler limitation 記錄。

### P3：強約束 Prompt

P3 是目前研究中最具可審核性的 prompt 形式。它要求：

```text
1. baseline 必須實測，不得估算
2. baseline 不計入優化提交次數
3. 每次提交前需說明修改內容、假設、預期改善與驗證目標
4. 每次提交後必須讀取 .out / .err / result.txt
5. correctness FAIL 則結果 invalid
6. 必須產出 CSV 或 RESULT row
7. 必須區分 result type
8. 必須執行 contradiction check
9. 提升小於 1% 必須標記 measurement-equivalent
10. 若沒有 profiler，必須列入 limitation
```

P3 的核心價值不是保證最高 speedup，而是將 AI 優化過程轉化成可重現、可審核、可排除偽加速的實驗流程。

***

## 四、目前主要實驗結果

根據已整理的 benchmark summary，幾個代表性結果如下。

### 1. `softmax-cuda`

`softmax-cuda` 是最明確的 kernel-level optimization 案例。Phase 2 P3 中取得 1.4575x 加速；早期 BASIC/GM 探索性實驗在 slice=784 下曾達 59.593x，但該結果是特定 shape 與特定優化策略下的結果，不可直接與 Phase 2 normalized result 混用。 [\[arxiv.org\]](https://arxiv.org/abs/2510.00555), [\[arxiv.org\]](https://arxiv.org/abs/2403.17134)

### 2. `topk-cuda`

`topk-cuda` 代表中等優化空間。有效策略主要是 CUB workspace reuse、block size tuning、移除 repeated allocation / synchronization。Phase 2 中 P3 speedup 約 1.1995x，早期 GM 版本在 14 組 hidden\_size/topk 組合上達到 1.442x。 [\[arxiv.org\]](https://arxiv.org/abs/2510.00555)

### 3. `allreduce-cuda`

`allreduce-cuda` 不是 kernel optimization，而是 environment / launcher repair。baseline 在 size 0 後失敗，主因是 UCX/GDRCopy path 錯誤；後續透過 `UCX_TLS=self,shm,cuda_copy,cuda_ipc` 等設定避開 broken GDRCopy path，使 RingAllreduce 全尺寸 correctness PASS。 [\[arxiv.org\]](https://arxiv.org/abs/2510.00555), [\[arxiv.org\]](https://arxiv.org/abs/2403.17134)

### 4. `p2p-cuda`

`p2p-cuda` 是 topology-aware measurement 案例。Codex 擴充為 4-GPU all-pair sweep 後，最佳穩定平均約 48.4455 GB/s，但相較先前約 48.24 GB/s 的提升低於 1%，因此應標記為 measurement-equivalent，而非顯著加速。 [\[arxiv.org\]](https://arxiv.org/abs/2403.17134), [\[arxiv.org\]](https://arxiv.org/abs/2510.00555)

### 5. `pingpong-cuda`

`pingpong-cuda` 是 MPI/NCCL point-to-point communication 比較。Final sweep 顯示 tuned CUDA-aware MPI 在 1 GiB 約 24.248 GB/s，NCCL 約 22.898 GB/s，MPI 約 1.059x faster。但這只適用於 two-rank ping-pong pattern，不代表 NCCL 在 collective 類 workload 中較差。 [\[arxiv.org\]](https://arxiv.org/abs/2403.17134), [\[arxiv.org\]](https://arxiv.org/html/2603.07169v1)

***

## 五、研究發現：Prompt 約束的作用

目前結果支持以下觀察。

### 1. Prompt 不是普通指令，而是實驗政策

高品質 prompt 的功能不是「叫 AI 寫快一點」，而是定義：

```text
1. 什麼是有效 baseline
2. 什麼是 correctness PASS
3. 什麼結果可以納入比較
4. 什麼結果應標記 invalid
5. 什麼情況不能宣稱 speedup
```

因此 prompt.md 在本研究中扮演的是 **execution policy**，而非單純自然語言提示。

### 2. 強約束 prompt 能降低偽加速

P3 不一定帶來最高 speedup，但能避免以下問題：

```text
1. baseline 缺失仍計算 speedup
2. correctness FAIL 仍宣稱成功
3. 只通過 partial case 卻寫 all PASS
4. environment fix 被誤寫成 kernel optimization
5. <1% noise 被誤寫成顯著加速
```

既有報告也指出，若沒有 baseline、矛盾檢查與 raw output，agent 可能出現失敗與全通過同時宣稱，或使用 estimated baseline 過度推論。 [\[arxiv.org\]](https://arxiv.org/abs/2403.17134)

### 3. AI agent 的能力受 benchmark 類型限制

不同 benchmark 對 AI 的優化空間不同：

```text
softmax-cuda：高優化空間
topk-cuda：中優化空間
shmembench-cuda / p2p-cuda：低優化空間或硬體受限
allreduce-cuda：主要是環境修復，不是 kernel optimization
```

因此研究不能只看 speedup，還必須看 result type。

***

## 六、下一階段研究方向：人機協作優化

下一階段建議收斂為 **Phase 3：人機協作式 AI 程式優化實驗**。不要再擴大 benchmark 數量，而是從 Phase 2 結果中挑選低、中、高三個代表案例做深入研究。

建議選：

```text
低優化空間：shmembench-cuda
中優化空間：topk-cuda
高優化空間：softmax-cuda
```

### 1. `shmembench-cuda`：低優化空間

研究重點：

```text
AI 是否能判斷已接近硬體限制？
是否會將微小提升過度宣稱？
人類如何用 profiler 與 measurement-equivalent 規則限制 agent？
```

### 2. `topk-cuda`：中優化空間

研究重點：

```text
AI 是否能辨識 workspace allocation / synchronization overhead？
人類如何引導 agent 從 trial-and-error 走向 shape-aware dispatch 或 radix strategy？
```

### 3. `softmax-cuda`：高優化空間

研究重點：

```text
AI 是否能進一步設計 shape-aware softmax dispatch？
人類如何防止過度特化單一 slice？
如何使用 profiler 驗證 expf、reduction、memory traffic 的瓶頸？
```

***

## 七、Phase 3 的人機協作工作流

Phase 3 不應再只是「給 prompt → agent 跑完」。應採用自適應工作流：

```text
1. Observe
   讀取 baseline、raw log、profiler 結果

2. Diagnose
   形成 bottleneck hypothesis

3. Retrieve
   查詢論文、CUDA 文件、既有優化案例

4. Plan
   提出單一可驗證修改

5. Human checkpoint
   人類審查是否改變 benchmark 語意

6. Execute
   sbatch 執行，保存 raw output

7. Validate
   correctness、performance、variance、profiler 一起驗證

8. Decide
   accept / reject / rollback / stop
```

這個流程的重點是讓 AI 從 trial-and-error 轉為 evidence-driven optimization。

***

## 八、人類操作者的角色

本研究主張，人類在以下位置不可取代：

```text
1. 設計研究邊界與 prompt 約束
2. 定義 valid / invalid result
3. 判斷 ENV_FIX、MEASURE_FIX、KERNEL_OPT 的差異
4. 審查 agent 提出的 bottleneck hypothesis
5. 防止過度特化、放寬 correctness 或改變測量範圍
6. 決定何時停止優化
```

更精確地說，AI 負責探索與執行；人類負責定義什麼結果能被相信。

***

## 九、論文主張收斂

本研究後續應收斂成以下主張：

```text
人機協作的價值不在於讓 AI 每次都產生更大 speedup，而在於讓 AI 的優化過程從 trial-and-error 轉為 evidence-driven、correctness-gated、profiler-supported 的可審核工作流。
```

這比單純展示「AI 加速多少倍」更具研究價值。

***

## 十、給協作者的理解重點

協作者只需要先掌握以下邏輯：

```text
1. 我們不是只研究 AI 能不能優化程式。
2. 我們研究的是：什麼樣的人機協作流程能產生可信的優化結果。
3. Prompt.md 是實驗政策，不是普通提示詞。
4. P1/P2/P3 代表不同約束強度。
5. P3 的價值是讓結果可審核，而不是保證最高加速。
6. 不同 benchmark 代表不同 AI 優化難度。
7. 下一階段只選三題：shmembench、topk、softmax。
8. 研究將重點放在可驗證自適應工作流，而不是擴大 benchmark 數量。
```

這份摘要可作為協作者 onboarding 文件的基礎版本。
