Proceed to Mode B Round 1 for softmax-cuda ONLY.

Do not submit any sbatch job yet.

First produce a Round 1 proposal containing:

1. Robust baseline summary
   - official cases
   - impl=1 baseline metrics
   - correctness status
   - CV / measurement_validity

2. Bottleneck hypothesis
   - identify exactly one suspected bottleneck
   - explain why this bottleneck is plausible from the baseline/source

3. Proposed change
   - propose exactly one minimal source-level modification
   - do not modify correctness tolerance
   - do not remove any official case
   - do not compare impl=0 -> impl=1 as speedup
   - candidate must be compared against impl=1 robust baseline

4. Expected improvement
   - specify which slice sizes may improve
   - specify expected risk/regression

5. Validation plan
   - run all official softmax cases:
     slice=128,256,784,1024,2048
   - correctness must PASS for all cases
   - at least 3 trials
   - report mean/min/max/stddev/CV
   - run self_consistency_auditor.py
   - set human_decision=Approved only after I approve

Note : Make sure you consider the hardware condition and enviroment here. 
Stop after producing the proposal and wait for human approval.