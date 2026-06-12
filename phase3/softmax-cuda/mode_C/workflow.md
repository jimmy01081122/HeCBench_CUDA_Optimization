以下是 **Mode C 的完整中文 workflow**。  
我依照你目前決策整理成兩層：

```text
主流程：Mode C Main
  Evidence-guided aggressive optimization
  目標是盡可能超越 Mode B 的 impl=3。

附加流程：Mode C-Extra
  Full-Slice Optimization Opportunity Experiment
  目標是檢查所有 slice，尤其是 128/256，是否仍有優化空間。
```

***

# Mode C 中文工作流程總覽

## Mode C 定位

Mode C 不是單純解釋，也不是盲目追求 speedup。它的定位是：

```text
證據導向的積極優化流程
Evidence-guided aggressive optimization
```

也就是同時做兩件事：

```text
1. 積極嘗試在 softmax-cuda 上超越 Mode B 已接受的 impl=3。
2. 用文獻、profiler、ablation、artifact audit 解釋為什麼成功、失敗或無法再提升。
```

主要比較基準：

```text
Primary comparison:
  Mode C candidate vs Mode B impl=3

Secondary comparison:
  Mode C candidate vs impl=1 baseline
```

***

# 一、Mode C Main Workflow

## Stage 0：Inspection and Planning

### 目的

先只讀取現有資料，不改 source，不跑 benchmark，不提交 sbatch。

目標是確認：

```text
1. 目前 source 是否包含 impl=0/1/2/3。
2. impl=3 dispatch 是否仍符合 Mode B final。
3. Mode B final 的限制是什麼。
4. Mode C 還有哪些可能優化空間。
5. 第一個 aggressive candidate 應該嘗試什麼。
```

### 必讀資料

```text
phase3/softmax-cuda/mode_B_human_guided/final/results.csv
phase3/softmax-cuda/mode_B_human_guided/final/round_summary.md
phase3/softmax-cuda/mode_B_human_guided/final/main.cu
phase3/softmax-cuda/mode_B_human_guided/rounds/round_1/round_summary.md
phase3/softmax-cuda/mode_B_human_guided/rounds/round_2/round_summary.md
phase3/softmax-cuda/mode_B_human_guided/rounds/round_1/patch_summary.md
phase3/softmax-cuda/mode_B_human_guided/rounds/round_2/patch_summary.md
/home/r14525078/HeCBench/src/softmax-cuda/main.cu
```

### 必須產出

```text
mode_C_literature_profiler/source_analysis.md
mode_C_literature_profiler/plan.md
mode_C_literature_profiler/literature_notes.md
```

### Stage 0 必須回答

```text
1. runtime source 是否已有 impl=0/1/2/3？
2. impl=3 dispatch 是否符合：
   128/256 -> impl=1
   784/1024/2048 -> impl=2

3. Mode B 的剩餘瓶頸或限制是什麼？
4. 哪些地方仍可能超越 impl=3？
5. Submission 1 要嘗試哪個 candidate？
6. 為什麼它可能比 impl=3 更快？
7. 是否有文獻、CUDA 文件或 source-level reasoning 支持？
8. 風險是什麼？
9. 預計會改哪些 source file？
10. 確認目前尚未修改 source。
```

### Stage 0 停止點

Stage 0 完成後必須停止，等待 human approval。

```text
不得直接修改 source。
不得直接 sbatch。
```

***

# 二、Mode C Main Submission 1

## Submission 1：Aggressive Candidate 1

### 目的

提出第一個有機會超越 Mode B `impl=3` 的候選實作。

### 候選方向

可選方向包括：

```text
1. 改良 impl=2 large-slice path。
2. 對 784 / 1024 / 2048 建立更細的 shape-specific kernel。
3. 調整 block size。
4. 減少 synchronization。
5. 降低 shared memory footprint。
6. 降低 register pressure。
7. 新增 impl=4 或 impl=5。
```

### 優先原則

建議：

```text
新增 impl=4 或 impl=5
```

不建議直接覆蓋：

```text
impl=1
impl=2
impl=3
```

### Submission 1 必須包含

```text
1. hypothesis
2. exact source change
3. 為什麼可能超越 impl=3
4. 預期每個 slice 的影響
5. 風險
6. validation plan
7. rollback plan
```

***

## Submission 1 驗證方式

每個 official slice 都必須測：

```text
impl=1 baseline
impl=3 Mode B accepted candidate
impl=4 或目前 Mode C candidate
```

official cases 固定：

```text
numSlice=100000, sliceSize=128, repeat=100
numSlice=100000, sliceSize=256, repeat=100
numSlice=100000, sliceSize=784, repeat=100
numSlice=100000, sliceSize=1024, repeat=100
numSlice=50000,  sliceSize=2048, repeat=100
```

每組至少：

```text
3 trials
```

建議執行順序：

```text
impl=1
impl=3
impl=4
```

原因：

```text
可以同時比較：
1. speedup_vs_impl1
2. speedup_vs_impl3
```

***

## Submission 1 判定

### 可接受

```text
1. correctness 全 PASS。
2. 至少一個 official slice 相對 impl=3 提升 >=1%。
3. 沒有 official slice regression >=1%。
4. CV 穩定。
5. auditor PASS。
```

### Partial

```text
1. 部分 slice 超越 impl=3。
2. 部分 slice regression 或 measurement-equivalent。
3. correctness 全 PASS。
```

### Reject / Invalid

```text
1. 任一 official slice correctness FAIL。
2. official case 缺失。
3. baseline invalid。
4. speedup_vs_impl3 無效。
5. raw output 缺失。
6. auditor FAIL。
```

***

# 三、Mode C Main Submission 2

## Submission 2：Correction / Ablation / Second Candidate

### 目的

根據 Submission 1 的結果做修正、ablation 或第二候選。

只有在需要時才做。

### 允許啟動的原因

```text
1. Submission 1 large slice 改善，但 small slice regression。
2. Submission 1 改善某一 large slice，但另一個 large slice regression。
3. Submission 1 correctness FAIL，需要小修。
4. Submission 1 顯示需要 ablation 才能解釋原因。
5. Submission 1 沒有提升，但仍有明確第二候選。
```

### 限制

```text
1. Submission 2 必須比 Submission 1 更窄。
2. 不得一次加入多個無關改動。
3. 不得隱藏 Submission 1 的失敗。
4. 仍需測所有 official cases。
5. 仍需比較 impl=1、impl=3、Mode C candidate。
```

***

# 四、Mode C Main Submission 3

## Submission 3：Final Confirmation

### 目的

確認最佳 Mode C candidate。

### 如果 Mode C candidate 超越 impl=3

Final confirmation 必須測：

```text
impl=1
impl=3
best Mode C candidate
```

### 如果沒有 candidate 超越 impl=3

Final confirmation 應測：

```text
impl=1
impl=3
```

並明確寫：

```text
Mode C produced no additional speedup beyond Mode B.
```

### Final confirmation 要求

```text
1. all official cases
2. at least 3 trials
3. correctness PASS
4. measurement validity
5. speedup_vs_impl1
6. speedup_vs_impl3
7. auditor PASS
8. raw stdout/stderr preserved
```

***

# 五、Profiler Policy

## Profiler 定位

Profiler 是輔助解釋，不是 speedup 計算依據。

```text
Official speedup 必須來自 normal timing runs。
不得用 profiler run timing 作 speedup。
```

## 嘗試方式

最多嘗試一次 Nsight Compute。

建議代表 case：

```text
slice=784 impl=3 and best Mode C candidate
slice=1024 impl=3 and best Mode C candidate
slice=2048 impl=3 and best Mode C candidate
```

## 若 profiler 不可用

如果出現：

```text
ncu unavailable
permission denied
hardware counter unavailable
too slow / timeout
```

則：

```text
profiler_status=UNAVAILABLE
記錄原因
不反覆重試
不得做 profiler-supported conclusion
```

## 若 profiler 可用

可收集：

```text
1. achieved occupancy
2. register count
3. shared memory usage
4. memory throughput
5. warp execution efficiency
6. instruction mix
7. math / special-function 相關指標
```

***

# 六、Mode C-Extra Workflow

## Mode C-Extra 定位

Mode C-Extra 是額外實驗，不取代主 Mode C。

名稱：

```text
Mode C-Extra: Full-Slice Optimization Opportunity Experiment
```

目的：

```text
檢查所有 official slices 是否仍有優化空間。
```

尤其要回答：

```text
1. slice=128 / 256 是否真的應該保持 impl=1？
2. large slices 是否仍有進一步優化空間？
3. full-slice candidate 是否能超越 impl=3？
```

***

## Extra Stage 0：實驗原因文件

### 必須產出

```text
mode_C_literature_profiler/full_slice_experiment_reason.md
```

### 文件必須說明

```text
1. 為什麼需要這個額外實驗。
2. 它和主 Mode C 有何不同。
3. 為什麼主 Mode C 可能自然聚焦 large slices。
4. 為什麼 small slices 128/256 仍值得明確評估。
5. 優化 small slices 有哪些風險。
6. 什麼 evidence 可以支持保留 small slices 不變。
7. 什麼 evidence 可以支持新增 small-slice candidate。
8. 結果如何分類。
```

***

## Extra Stage 1：Full-slice opportunity table

### 必須產出

```text
mode_C_literature_profiler/full_slice_opportunity_table.md
```

### 每個 slice 必須列出

```text
slice=128
slice=256
slice=784
slice=1024
slice=2048
```

對每個 slice 回答：

```text
1. Mode B selected implementation
2. current Mode B role
3. remaining optimization opportunity
4. proposed action
5. risk
6. evidence
```

### Proposed action 可以是

```text
keep impl=1
tune impl=1
add small-slice candidate
keep impl=2
tune impl=2
add large-slice candidate
add full-slice dispatch candidate
```

如果不優化 `128/256`，必須明確寫原因，不能沉默忽略。

***

## Extra Stage 2：Full-slice candidate

若要實作 full-slice candidate，建議：

```text
新增 impl=5 或更高
```

不得覆蓋：

```text
impl=0
impl=1
impl=2
impl=3
主 Mode C candidate
```

候選名稱建議：

```text
variant=modeC_extra_full_slice_candidate
```

***

## Extra 驗證方式

必須比較：

```text
impl=1 baseline
impl=3 Mode B accepted candidate
full-slice candidate
```

所有 official cases：

```text
128
256
784
1024
2048
```

每組至少：

```text
3 trials
```

核心比較：

```text
speedup_vs_impl3
```

***

## Extra 結果標籤

```text
FULL_SLICE_SUCCESS
FULL_SLICE_PARTIAL_SUCCESS
FULL_SLICE_NO_ADDITIONAL_GAIN
FULL_SLICE_INCONCLUSIVE
FULL_SLICE_INVALID
```

### FULL\_SLICE\_SUCCESS

```text
1. 所有 official slices correctness PASS。
2. 至少一個 slice 超越 impl=3。
3. 沒有 official slice regression >=1%。
4. auditor PASS。
```

### FULL\_SLICE\_PARTIAL\_SUCCESS

```text
部分 slice 超越 impl=3，
但另一些 slice measurement-equivalent 或 mild regression。
```

### FULL\_SLICE\_NO\_ADDITIONAL\_GAIN

```text
沒有超越 impl=3，
但確認 impl=3 已是合理 dispatch。
```

### FULL\_SLICE\_INVALID

```text
任一 official slice correctness FAIL，
或 missing official case，
或 invalid baseline。
```

***

# 七、Mode C 最終報告 Workflow

## Final report 必須產出

```text
mode_C_literature_profiler/final/mode_C_summary.md
```

若有 Extra：

```text
mode_C_literature_profiler/final/full_slice_experiment_summary.md
```

***

## mode\_C\_summary.md 必須包含

```text
1. Objective
2. Baseline and Mode B reference
3. Submission history
4. Best candidate
5. Final official results
6. speedup_vs_impl1
7. speedup_vs_impl3
8. correctness table
9. measurement validity table
10. profiler status
11. ablation or attribution conclusion
12. limitations
13. do-not-claim list
14. final label
```

***

## full\_slice\_experiment\_summary.md 必須包含

```text
1. Experiment motivation
2. Why main Mode C may focus on large slices
3. Why full-slice evaluation was added
4. Per-slice opportunity table
5. Candidate description
6. Results vs impl=1
7. Results vs impl=3
8. Per-slice correctness
9. Per-slice speedup_claim_valid
10. Small-slice conclusion
11. Large-slice conclusion
12. Whether full-slice candidate should replace impl=3
13. Limitations
14. Final label
15. Do-not-claim list
```

***

# 八、Mode C final labels

主 Mode C final label 必須是以下之一：

```text
SUCCESS_WITH_ADDITIONAL_SPEEDUP
SUCCESS_EXPLANATION_ONLY
PARTIAL_SUCCESS
INCONCLUSIVE
BLOCKED
```

## SUCCESS\_WITH\_ADDITIONAL\_SPEEDUP

```text
Mode C candidate 相對 impl=3 有有效提升，
且 correctness / measurement / auditor 全部通過。
```

## SUCCESS\_EXPLANATION\_ONLY

```text
沒有超越 impl=3，
但提供 profiler / ablation / literature 支持的解釋。
```

## PARTIAL\_SUCCESS

```text
部分 slice 超越 impl=3，
但其他 slice regression、measurement-equivalent 或不穩定。
```

## INCONCLUSIVE

```text
證據不足，無法下結論。
```

## BLOCKED

```text
環境、build、runtime、profiler 或資料缺失導致無法完成。
```

***

# 九、整體決策邏輯

Mode C 完成後，主規劃器要回答：

```text
1. Mode C 是否超越 Mode B impl=3？
2. 若有，是哪些 slice？
3. 是否所有 official cases correctness PASS？
4. 是否有 regression？
5. 是否有 profiler 或 ablation 支持原因？
6. 若沒有超越，是否能解釋為何 Mode B 已足夠？
7. Mode C-Extra 是否支持或推翻「small slices 保留 impl=1」的策略？
8. 最終 accepted candidate 是：
   - Mode B impl=3？
   - main Mode C candidate？
   - Mode C-Extra full-slice candidate？
   - 還是沒有可接受替代？
```

***

# 十、不可宣稱事項

Mode C 報告不得宣稱：

```text
1. Mode C 成功，除非 artifacts 支持。
2. Mode C 有 additional speedup，除非 speedup_vs_impl3 有效。
3. impl=3 是 universal KERNEL_OPT。
4. impl=2 是 universal replacement。
5. slice=128/256 有有效 speedup，除非真的超越 impl=3 且 correctness PASS。
6. profiler-supported conclusion，除非 profiler 實際存在。
7. cached exp 單獨造成提升，除非 ablation 支持。
8. topk/shmembench Mode B 已完成 optimization。
```

***

# 最後總結

Mode C 的完整 workflow 是：

```text
Stage 0:
  只讀檢查與計畫

Submission 1:
  積極 candidate，嘗試超越 impl=3

Submission 2:
  修正、ablation 或第二候選

Submission 3:
  final confirmation

Mode C-Extra:
  獨立檢查所有 slice 是否仍有優化機會

Final:
  決定是否有 additional speedup、是否只是 explanation-only、或是否維持 Mode B impl=3 為最佳策略
```

這樣設計能同時滿足：

```text
1. 盡可能加速
2. 不破壞 correctness
3. 不偽造 profiler 結論
4. 不忽略 small slices
5. 保留 full-slice opportunity analysis
6. 能支撐論文中的成功、失敗與人機協作分析
```
