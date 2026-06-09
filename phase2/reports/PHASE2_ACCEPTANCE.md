# PHASE2_ACCEPTANCE

A server-side experiment is acceptable only if:

1. The prompt file used is recorded.
2. Baseline is attempted and classified.
3. Raw `.out`, `.err`, and result `.txt` are preserved.
4. Correctness is explicitly reported.
5. FAIL/skipped/waived cases are not counted as success.
6. Speedup uses measured baseline only.
7. Agent summary includes accepted and rejected attempts.
8. P3 runs include contradiction check, variance notes, and profiler/measurement notes.

When data is returned, include the whole result tree rather than only final tables.
