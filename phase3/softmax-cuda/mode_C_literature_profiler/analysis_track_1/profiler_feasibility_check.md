# Profiler Feasibility Check

## Scope

This was an inspect-only feasibility check.

No benchmark binary was run on the login node. No source file was modified. No
sbatch job was submitted.

## Commands Checked

Default login environment:

```text
which ncu
```

Result:

```text
which: no ncu in default PATH
```

Default login environment:

```text
ncu --version
```

Result:

```text
ncu: command not found
```

Module availability:

```text
module avail cuda
```

Result:

```text
cuda/10.2 cuda/11.4 cuda/11.7 cuda/12.3 cuda/12.8
```

After loading CUDA 12.8:

```text
module load cuda/12.8
which ncu
ncu --version
```

Result:

```text
/work/HPC_SYS/twnia2/pkg-rocky8/nvidia/cuda/cuda-12.8/bin/ncu
NVIDIA (R) Nsight Compute Command Line Profiler
Version 2025.1.0.0 (build 35237751) (public-release)
```

The module command also printed:

```text
logger: socket /dev/log: Operation not permitted
```

This appears to be a module-load logging limitation in the current shell, not an
Nsight Compute availability failure.

## Prior Logs

Prior softmax Mode A artifacts report:

```text
profiler_available=True
ncu completed; see final/ncu.txt
```

No prior softmax log inspected here showed an Nsight Compute permission error
such as `ERR_NVGPUCTRPERM`.

## Status

profiler_status_candidate: `AVAILABLE`

Reason:

- `ncu` is unavailable in the default login PATH.
- `ncu` is available after `module load cuda/12.8`.
- Prior Mode A logs indicate an earlier `ncu` run completed.

## Rules

- Profiler execution must use sbatch.
- Do not run `./main` or any GPU benchmark binary on the login node.
- Profiler timing must not be used for official speedup.
- Official speedup remains based on normal timing rows in Submission 1 and any
  later final confirmation.

## Recommendation

Profiler should be attempted before Submission 2 if the human planner approves a
profiler-only sbatch run. The run would be explanation-only and should not count
as an optimization submission.
