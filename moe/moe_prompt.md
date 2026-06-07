你需要優化 HeCBench moe-cuda benchmark。
NOTE : 只有三次的提交測試機會，每次提交時需要創建一份文件在/home/r14525078/HeCBench/src/moe-cuda/results下，名稱格式為CLOUD_V<第幾次提交>_doc.md，內容為說明該次的優化方式，最後將三次測試數據做對比總結

workspace : /home/r14525078/HeCBench/src/moe-cuda
主要程式碼 : /home/r14525078/HeCBench/src/moe-cuda/main.cu
此 benchmark 執行 MoE gating 階段的 softmax + top-k expert selection。
執行測試指令為：

sbatch run_moe_cuda.slurm
這個指令會同時進行編譯與提交，提交後如果顯示編號與submmit，則提交成功，需要等一段時間會在/home/r14525078/HeCBench/src/moe-cuda/results 下產生最終結果

固定測試組合：
<number of tokens> <number of experts> <top K> <repeat>
32768 384 1 1000
32768 384 2 1000
32768 384 4 1000
32768 384 8 1000

硬體為 NVIDIA V100，編譯架構使用 sm_70。

允許優化 kernels.h 中的 CUDA kernels。
允許融合 softmax 與 top-k。
允許針對 topk=1/2/4/8 做 specialization。
不得刪除 correctness check。
不得放寬 tolerance。
不得修改 reference 以配合 GPU 結果。
不得縮小輸入規模。
不得降低 repeat。
不得跳過實際計算。

最終必須保證所有測試輸出 PASS，並回報每組 topk 的 Average execution time of kernels。