# Prompt Schema

Each P3 prompt must contain:

- Prompt Metadata
- Environment
- Paths
- Benchmark-Specific Requirements
- Hard Rules
- Baseline Requirements
- Optimization Submission Rules
- Correctness Gate
- Required Result Types
- CSV Result Schema
- Variance / Repeated Trials
- Profiler / Measurement Notes
- Contradiction Check
- Final Output

Required CSV schema:

```csv
benchmark,job_id,node,prompt_level,submission_index,variant,case,metric_name,metric_value,metric_unit,correctness,status,result_type,accepted,reject_reason,notes
```
