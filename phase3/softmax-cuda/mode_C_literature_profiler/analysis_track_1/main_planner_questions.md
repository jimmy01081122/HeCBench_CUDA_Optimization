# Main Planner Questions

These choices require human planner decision before any further execution.

## 1. Approve Profiler-Only sbatch?

Should Codex run a profiler-only sbatch job before Submission 2?

Reason:

- `ncu` appears available after `module load cuda/12.8`.
- Prior Mode A logs indicate `ncu` completed once.
- Profiler evidence could explain why 784 and 1024 improved and why 2048 did
  not reach the 1% threshold.

Constraint:

- profiler timing will not be used for official speedup.

## 2. If Profiler Is Not Approved, Use Submission 2 For Ablation?

Recommended fallback:

- Option A: reduction-structure ablation.

Reason:

- It is the clearest way to test whether the `impl=4` improvement is associated
  with the reduction-structure change.

## 3. Should 2048 Be Prioritized?

Submission 1 did not produce accepted additional speedup for 2048.

Question:

- Should Submission 2 prioritize a 2048-specific optimization, or should it
  prioritize explanatory evidence for the accepted 784/1024 gains?

Recommended answer:

- prioritize explanation first unless profiler data clearly points to a
  low-risk 2048-specific change.

## 4. Stop After Submission 1?

Submission 1 already has accepted limited additional speedup.

Question:

- Is the current evidence sufficient to proceed directly to final confirmation?

Tradeoff:

- This minimizes risk and preserves the accepted candidate.
- It leaves causal explanation weaker because profiler and ablation are still
  not run.

## 5. Preferred Final Claim Style

If no further optimization is attempted, final claim should be limited to:

- `impl=4` has accepted additional speedup on 784 and 1024 vs `impl=3`.
- 128, 256, and 2048 are not additional Mode C speedup claims.
- no profiler-supported or ablation-supported causal attribution is available.
