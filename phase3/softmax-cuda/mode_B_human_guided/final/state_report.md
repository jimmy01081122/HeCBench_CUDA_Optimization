# 1. 證明人類審查能把「局部有效但不完整的 AI 優化」轉成可用策略

Round 1 中，agent 提出的 `impl=2` 並不是完整成功：

```text
slice=128: regression
slice=256: correctness failure
slice=784/1024/2048: 有效提升
```

如果沒有 human review，agent 很可能只拿大 slice 的改善宣稱成功。  
但人類審查拒絕了「全面替代」的說法，改要求建立 shape-aware dispatch。

最後 Round 2 / final confirmation 證明：

```text
slice=128,256 → 保留 impl=1
slice=784,1024,2048 → 使用 impl=2
```

這使候選策略從「不完整 kernel replacement」變成「可驗證 dispatch policy」。

這就是 Mode B 的核心價值：

```text
AI 產生候選方向；
人類判斷其有效邊界；
再把 partial improvement 轉成可用策略。
```

***

# 2. 證明「最大 speedup」不是唯一目標，正確分類更重要

這次 softmax 最終結果不是應該寫成：

```text
AI 產生 universal KERNEL_OPT
```

而應寫成：

```text
PARAM_TUNE / SHAPE_AWARE_DISPATCH
```

原因是：

```text
impl=3 本身不是新的 universal kernel。
它是根據 slice size 選擇既有 impl=1 或 impl=2。
```

這和 Phase 2 的整體觀察一致：不同 benchmark 的有效成果性質不同，不能把 environment fix、measurement fix、parameter tuning、kernel optimization 混為一談。Phase 2 的彙整也顯示，例如 `allreduce-cuda` 是 launcher / UCX 環境修復，`p2p-cuda` 是 topology-aware measurement，而 `softmax-cuda` 才是明確 kernel-level optimization 類案例。 [\[arxiv.org\]](https://arxiv.org/abs/2510.00555), [\[arxiv.org\]](https://arxiv.org/abs/2403.17134)

***

# 3. 證明 P3 / Mode B 的約束能防止偽加速

這次流程中，幾個規則真的發揮作用：

```text
1. impl=0 → impl=1 不得算 Phase 3 speedup
2. correctness FAIL 不得納入有效結果
3. slice=128/256 走同一路徑時，只能標 measurement-equivalent
4. partial improvement 不得寫成 full optimization
5. 所有 official slice 都必須保留
```

這些規則正是 Phase 2 中強約束 prompt 的價值所在。既有 prompt inventory 顯示，較完整的 prompt 通常包含 baseline、correctness、sbatch、submission limit、raw output 等防偽加速條款，而報告也指出缺少這些規則時 agent 容易產生不可審核或矛盾結論。 [\[arxiv.org\]](https://arxiv.org/html/2603.07169v1), [\[arxiv.org\]](https://arxiv.org/abs/2403.17134)

這次 softmax Mode B 實際驗證了這件事：  
如果沒有這些約束，Round 1 大 slice 的改善可能會被錯誤包裝成完整成功。

***

# 4. 證明 robust baseline 與 paired baseline 是必要的

Mode A / robust baseline 的作用很清楚：

```text
先確定 impl=1 baseline 穩定；
再在 Round 1 / Round 2 中做 paired impl=1 vs candidate 比較。
```

這避免了 topk Mode A 中曾出現的問題：沒有程式修改也可能因 baseline CV 過高產生表面 speedup。

因此這次 softmax 實驗證明：

```text
沒有 paired baseline，就不能放心宣稱 speedup。
沒有 variance / CV，就無法區分真改善與測量漂移。
```

這會成為後續 topk-cuda 的重要原則。

***

# 5. 證明 shape-aware dispatch 是 softmax-cuda 的合理優化方向

final confirmation 顯示：

```text
slice=128: 1.002x，measurement-equivalent
slice=256: 1.001x，measurement-equivalent
slice=784: 1.392x，有效
slice=1024: 1.699x，有效
slice=2048: 1.337x，有效
```

這說明：

```text
小 slice 適合保留 warp-level impl=1
大 slice 適合使用 block-level compound impl=2
```

所以 softmax 的最佳策略不是單一 kernel 通吃，而是：

```text
依 shape 選擇不同 implementation
```

這是一個清楚、可解釋、可驗證的工程結果。

***

# 6. 證明人機協作比純 agent 更適合做「有效結果收斂」

Mode A 的主要價值是揭露風險；Mode B 則真正開始產生可用策略。

這次 softmax 流程顯示：

```text
Mode A:
  建立穩定 baseline

Round 1:
  Agent 提出 compound candidate
  結果 partial success

Human review:
  拒絕 full replacement
  要求 shape-aware dispatch

Round 2:
  建立 impl=3 dispatcher

Final:
  全 official cases correctness PASS
  大 slice 有效提升
```

這證明人類不是只負責「看結果」，而是負責：

```text
1. 判斷 partial result 的有效邊界
2. 防止錯誤歸因
3. 防止偽加速
4. 把失敗與 regression 轉化為下一輪策略
```

***

# 7. 這次沒有證明什麼

這也要明確寫清楚。

## 沒有證明 `impl=2` 是 universal best kernel

因為：

```text
impl=2 在 slice=128 regression
impl=2 在 slice=256 有 correctness failure
```

所以不能寫：

```text
impl=2 universally improves softmax
```

***

## 沒有證明 cached exp 單獨造成提升

Round 1 的 `impl=2` 同時改了：

```text
warp-per-slice → block-per-slice
recompute exp → shared-memory cached exp
```

所以目前只能說：

```text
compound block-level cached-exp path 對 large slices 有效
```

若要證明 cached exp 本身的貢獻，需要 Mode C 或 Round 3 ablation。

***

## 沒有證明所有 benchmark 都能透過人機協作有效加速

目前只完成 softmax-cuda Mode B。  
topk-cuda 和 shmembench-cuda 仍未完成 Mode B optimization。

因此目前只能說：

```text
在 softmax-cuda 這個高優化空間 benchmark 上，Mode B 人機協作有效。
```

不能外推成：

```text
Mode B 對所有 benchmark 都有效。
```

***

# 8. 可寫入論文的核心結論

可以這樣寫：

```text
softmax-cuda 的 Mode B 實驗證明，人機協作流程能將 AI agent 產生的 partial optimization 轉化為可驗證的 shape-aware dispatch policy。Agent 在 Round 1 中提出的 compound block-level cached-exp candidate 對 large slices 有效，但在 small slices 上 regression 或 correctness failure。人類審查拒絕其作為 universal replacement，並引導 agent 在 Round 2 中建立 shape-aware dispatcher。Final confirmation 顯示，該 dispatcher 保留 impl=1 給 slice=128/256，並選擇 impl=2 給 slice=784/1024/2048，所有 official cases correctness PASS，large slices 分別取得 1.392x、1.699x 與 1.337x 有效改善。此結果證明，Mode B 的主要價值在於將 trial-and-error 的 AI 候選優化轉化為 correctness-gated、shape-aware、可審核的工程策略。
```

***

# 9. 總結一句話

這次實驗證明了：

```text
人機協作的價值不是讓 AI 一次產生完美 kernel，而是讓 AI 產生的 partial improvement 經由人類審查、baseline 對照、correctness gate 與 per-case 分析，收斂成可驗證、可解釋、可放進論文的優化策略。
```

目前可正式標記：

```text
softmax-cuda Mode B: SUCCESS
結果類型: PARAM_TUNE / SHAPE_AWARE_DISPATCH
下一步: topk-cuda Mode B Round 1 proposal
```
