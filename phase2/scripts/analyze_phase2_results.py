#!/usr/bin/env python3
import csv
import math
import re
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
OUT_CSV = REPORTS / "phase2_level_summary.csv"
OUT_MD = REPORTS / "PHASE2_RESULTS_ANALYSIS.md"


def read(path):
    return path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""


def rows_csv(path):
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8", errors="replace") as f:
        return list(csv.DictReader(f))


def fnum(x):
    try:
        return float(x)
    except Exception:
        return float("nan")


def fmt(x, digits=3):
    if x is None:
        return "n/a"
    try:
        x = float(x)
    except Exception:
        return str(x)
    if math.isnan(x):
        return "n/a"
    return f"{x:.{digits}f}"


def mean(vals):
    vals = [float(v) for v in vals if not math.isnan(float(v))]
    return sum(vals) / len(vals) if vals else float("nan")


def last_match_float(pattern, text):
    vals = re.findall(pattern, text, re.S)
    return float(vals[-1]) if vals else float("nan")


def speedup(old, new):
    old, new = fnum(old), fnum(new)
    if math.isnan(old) or math.isnan(new) or new == 0:
        return float("nan")
    return old / new


def improvement(old, new, unit):
    old, new = fnum(old), fnum(new)
    if math.isnan(old) or math.isnan(new) or old == 0 or new == 0:
        return float("nan")
    # Latency/time metrics are better when lower; throughput metrics are better when higher.
    if "GB/s" in unit or "bandwidth" in unit.lower():
        return new / old
    return old / new


def add(level, benchmark, baseline, final, unit, correctness, result_type, status, notes, source):
    sp = improvement(baseline, final, unit) if baseline not in ("n/a", "", None) and final not in ("n/a", "", None) else float("nan")
    return {
        "level": level,
        "benchmark": benchmark,
        "baseline_metric": fmt(baseline, 6) if isinstance(baseline, (float, int)) else str(baseline),
        "final_metric": fmt(final, 6) if isinstance(final, (float, int)) else str(final),
        "metric_unit": unit,
        "speedup": fmt(sp, 4),
        "correctness": correctness,
        "result_type": result_type,
        "status": status,
        "notes": notes,
        "source": source,
    }


def parse_p1():
    rows = []

    text = read(ROOT / "p1/allreduce-cuda/result/summary.md")
    m = re.search(r"536870912:\s*([0-9.]+)\s*->\s*([0-9.]+)", text)
    if m:
        rows.append(add("P1", "allreduce-cuda", float(m.group(1)), float(m.group(2)), "us/iter at largest size", "PASS all sizes", "KERNEL_OPT/ENV_FIX", "valid", "P1 replaced measured allreduce path with NCCL; strong speedup on many sizes, largest-size speedup modest.", "p1/allreduce-cuda/result/summary.md"))

    text = read(ROOT / "p1/moe-align-cuda/result/summary.md")
    m = re.search(r"30/30.*?Mean latency.*?`([0-9.]+) us`", text, re.S)
    if m:
        rows.append(add("P1", "moe-align-cuda", "n/a", float(m.group(1)), "us mean latency", "PASS 30/30", "PARAM_TUNE", "valid_no_baseline", "Summary reports final mean but no measured baseline, so speedup cannot be audited.", "p1/moe-align-cuda/result/summary.md"))

    text = read(ROOT / "p1/moe-cuda/result/summary.md")
    final_vals = [float(x) for x in re.findall(r"topk=\d+\s+PASS\s+([0-9.]+) us", text)]
    base_match = re.search(r"Earlier baseline run.*?```text\n(.*?)```", text, re.S)
    base_vals = [float(x) for x in re.findall(r"topk=\d+\s+([0-9.]+) us", base_match.group(1))] if base_match else []
    if final_vals and base_vals:
        rows.append(add("P1", "moe-cuda", mean(base_vals), mean(final_vals), "us arithmetic mean over topk 1/2/4/8", "PASS 4/4", "KERNEL_OPT", "valid", "Fused softmax+topk; P1 summary lacks variance/profiler but reports baseline and final.", "p1/moe-cuda/result/summary.md"))

    p2p_files = sorted((ROOT / "p1/p2p-cuda/result").glob("p2p_cuda_result_*.txt"))
    if p2p_files:
        latest = p2p_files[-1]
        vals = [float(x) for x in re.findall(r":\s*([0-9.]+) GB/s", read(latest))]
        final = mean(vals) if vals else float("nan")
        rows.append(add("P1", "p2p-cuda", "n/a", final, "GB/s average over reported directed pairs", "PASS reported", "MEASURE_FIX", "weak_auditability", "No summary.md; final file reports only 2 GPUs/2 directions, not the full 4-GPU topology matrix.", latest.relative_to(ROOT).as_posix()))

    text = read(ROOT / "p1/pingpong-cuda/result/summary.md")
    m = re.search(r"NCCL improved from `([0-9.]+) GB/s` to `([0-9.]+) GB/s`", text)
    if m:
        rows.append(add("P1", "pingpong-cuda", float(m.group(1)), float(m.group(2)), "GB/s NCCL at 1GiB", "no correctness errors printed", "KERNEL_OPT/MEASURE_FIX", "valid_with_caution", "NCCL grouping doubles reported 1GiB bandwidth; P1 lacks CSV/variance and has earlier invalid attempts.", "p1/pingpong-cuda/result/summary.md"))

    # Additional Phase 2 benchmarks. P1 often lacks structured summaries, so these
    # rows use raw outputs and intentionally mark weak auditability where baseline
    # or measurement scope cannot be proven equivalent.
    soft_base = read(ROOT / "p1/softmax-cuda/result/softmax_cuda_result_948543.txt")
    soft_final = read(ROOT / "p1/softmax-cuda/result/softmax_cuda_result_948546.txt")
    mb = re.search(r"^784,1,([0-9.]+),PASS$", soft_base, re.M)
    mf = re.search(r"^784,1,([0-9.]+),PASS$", soft_final, re.M)
    if mb and mf:
        rows.append(add("P1", "softmax-cuda", float(mb.group(1)), float(mf.group(1)), "ms avg latency for slice=784 impl=1", "PASS reported", "KERNEL_OPT", "valid_with_caution", "Raw logs show slice=784 impl=1 improved, but P1 has no structured summary, rejected-attempt table, or variance.", "p1/softmax-cuda/result/softmax_cuda_result_948546.txt"))

    def topk_mean(path):
        vals = [float(x) for x in re.findall(r"Average execution time of topk\s*:\s*([0-9.]+)", read(path))]
        return mean(vals)

    p1_topk_base = topk_mean(ROOT / "p1/topk-cuda/result/topk_cuda_result_948615.txt")
    p1_topk_final = topk_mean(ROOT / "p1/topk-cuda/result/topk_cuda_result_948618.txt")
    if not math.isnan(p1_topk_base) and not math.isnan(p1_topk_final):
        rows.append(add("P1", "topk-cuda", p1_topk_base, p1_topk_final, "us mean over reported hidden_size/topk cases", "PASS reported", "KERNEL_OPT", "valid_with_caution", "Raw outputs report PASS and lower mean than first run, but P1 lacks accepted/rejected rationale and variance.", "p1/topk-cuda/result/topk_cuda_result_948618.txt"))

    prefetch_text = read(ROOT / "p1/prefetch-cuda/result/prefetch_cuda_result_948600.txt")
    prefetch_vals = [float(x) for x in re.findall(r"Average execution time:\s*([0-9.]+)\s*\(ms\)", prefetch_text)]
    if prefetch_vals:
        rows.append(add("P1", "prefetch-cuda", "n/a", mean(prefetch_vals[:10]), "ms mean with_prefetch raw samples", "PASS reported", "MEASURE_FIX/PARAM_TUNE", "valid_no_baseline", "Only one P1 result file was available; final timing can be summarized but speedup cannot be audited against measured baseline.", "p1/prefetch-cuda/result/prefetch_cuda_result_948600.txt"))

    smd_base = read(ROOT / "p1/simpleMultiDevice-cuda/result/simpleMultiDevice_cuda_result_948642.txt")
    smd_final = read(ROOT / "p1/simpleMultiDevice-cuda/result/simpleMultiDevice_cuda_result_948646.txt")
    smd_base_total = last_match_float(r"total_us=([0-9.]+)", smd_base)
    smd_final_total = last_match_float(r"total_us=([0-9.]+)", smd_final)
    if not math.isnan(smd_base_total) and not math.isnan(smd_final_total):
        rows.append(add("P1", "simpleMultiDevice-cuda", "measurement scope changed", smd_final_total, "us total_us raw final", "PASS reported", "MEASURE_FIX/KERNEL_OPT", "success_no_speedup_claim", "Raw P1 logs show a dramatic total_us drop, but H2D/D2H timing scope appears changed; no speedup claim is counted.", "p1/simpleMultiDevice-cuda/result/simpleMultiDevice_cuda_result_948646.txt"))

    shmem_base = read(ROOT / "p1/shmembench-cuda/result/shmembench_cuda_result_948665.txt")
    shmem_final = read(ROOT / "p1/shmembench-cuda/result/shmembench_cuda_result_948670.txt")
    shmem_base_ms = last_match_float(r"Average kernel execution time\s*:\s*([0-9.]+)", shmem_base)
    shmem_final_ms = last_match_float(r"Average kernel execution time\s*:\s*([0-9.]+)", shmem_final)
    if not math.isnan(shmem_base_ms) and not math.isnan(shmem_final_ms):
        rows.append(add("P1", "shmembench-cuda", shmem_base_ms, shmem_final_ms, "ms avg kernel time", "PASS/no checksum failed in final", "KERNEL_OPT", "valid_with_caution", "Best valid raw run improved modestly; one faster P1 attempt had checksum failure and is excluded.", "p1/shmembench-cuda/result/shmembench_cuda_result_948670.txt"))

    return rows


def parse_p2():
    rows = []
    text = read(ROOT / "p2/allreduce-cuda/result/agent_summary.md")
    m = re.search(r"Geomean speedup across all buffer sizes:\s*([0-9.]+)x", text)
    largest = re.search(r"Final largest-buffer metric:\s*([0-9.]+).*?Baseline largest-buffer metric:\s*([0-9.]+)", text, re.S)
    if m and largest:
        rows.append(add("P2", "allreduce-cuda", 1.0, 1.0 / float(m.group(1)), "geomean relative latency", "PASS 12/12", "KERNEL_OPT", "valid", f"Geomean speedup {m.group(1)}x; largest buffer is measurement-equivalent/slightly slower ({largest.group(2)} -> {largest.group(1)} us).", "p2/allreduce-cuda/result/agent_summary.md"))

    text = read(ROOT / "p2/moe-align-cuda/result/agent_summary.md")
    m = re.search(r"Baseline.*?Metric:\s*([0-9.]+).*?Final metric:\s*([0-9.]+).*?Final speedup:\s*([0-9.]+)x", text, re.S)
    if m:
        rows.append(add("P2", "moe-align-cuda", float(m.group(1)), float(m.group(2)), "us mean latency", "PASS 30/30", "PARAM_TUNE", "valid", "Cached cumsum workspace; rejected slower variants documented.", "p2/moe-align-cuda/result/agent_summary.md"))

    text = read(ROOT / "p2/moe-cuda/result/agent_summary.md")
    m = re.search(r"arithmetic mean:\s*([0-9.]+) us.*?arithmetic mean:\s*([0-9.]+) us, speedup ([0-9.]+)x", text, re.S)
    if m:
        rows.append(add("P2", "moe-cuda", float(m.group(1)), float(m.group(2)), "us arithmetic mean over topk 1/2/4/8", "PASS 4/4 final; invalid attempts rejected", "KERNEL_OPT", "valid", "Hybrid path gives real topk=1 gain but mean speedup only 3.39%.", "p2/moe-cuda/result/agent_summary.md"))

    text = read(ROOT / "p2/p2p-cuda/result/agent_summary.md")
    m = re.search(r"Metric:\s*([0-9.]+) GB/s.*?Final metric:\s*([0-9.]+) GB/s", text, re.S)
    if m:
        rows.append(add("P2", "p2p-cuda", float(m.group(1)), float(m.group(2)), "GB/s average", "PASS 12/12 final", "MEASURE_FIX", "measurement_equivalent", "Directional sweep improves auditability; speedup below 1%.", "p2/p2p-cuda/result/agent_summary.md"))

    text = read(ROOT / "p2/pingpong-cuda/result/agent_summary.md")
    m = re.search(r"Baseline 1 GiB metric:.*?NCCL:.*?([0-9.]+) GB/s.*?Final 1 GiB metric:.*?NCCL:.*?([0-9.]+) GB/s", text, re.S)
    if m:
        rows.append(add("P2", "pingpong-cuda", float(m.group(1)), float(m.group(2)), "GB/s NCCL at 1GiB", "PASS full sweep", "PARAM_TUNE/MEASURE_FIX", "measurement_equivalent", "Five optimizations tried; final improvement is noise-level but invalid setup attempts are clearly rejected.", "p2/pingpong-cuda/result/agent_summary.md"))

    text = read(ROOT / "p2/softmax-cuda/result/agent_summary.md")
    m = re.search(r"Baseline metric, implementation 1:\s*([0-9.]+) ms.*?Final metric, implementation 1:\s*([0-9.]+) ms", text, re.S)
    if m:
        rows.append(add("P2", "softmax-cuda", float(m.group(1)), float(m.group(2)), "ms slice=784 implementation 1", "PASS impl 0/1", "KERNEL_OPT", "valid", "Manual warp reductions plus slice=784 specialization produce a clear implementation-1 speedup; slower attempts are documented.", "p2/softmax-cuda/result/agent_summary.md"))

    text = read(ROOT / "p2/topk-cuda/result/agent_summary.md")
    m = re.search(r"Metric:\s*([0-9.]+) us average.*?Final metric:\s*([0-9.]+) us", text, re.S)
    if m:
        rows.append(add("P2", "topk-cuda", float(m.group(1)), float(m.group(2)), "us mean over 14 hidden_size/topk cases", "PASS 14/14", "KERNEL_OPT", "valid", "Cached radix workspace removes timed cudaMalloc/cudaFree overhead; rejected block-size variants are recorded.", "p2/topk-cuda/result/agent_summary.md"))

    text = read(ROOT / "p2/prefetch-cuda/result/agent_summary.md")
    m = re.search(r"repeat=100 with_prefetch:\s*([0-9.]+) ms.*?with_prefetch:\s*([0-9.]+) ms,\s*([0-9.]+)x speedup", text, re.S)
    if m:
        rows.append(add("P2", "prefetch-cuda", float(m.group(1)), float(m.group(2)), "ms repeat=100 with_prefetch", "PASS all trials", "PARAM_TUNE/MEASURE_FIX", "valid", "Separates prefetch setup from timed kernel execution; primary repeat=100 with_prefetch improves while no-prefetch also improves.", "p2/prefetch-cuda/result/agent_summary.md"))

    text = read(ROOT / "p2/simpleMultiDevice-cuda/result/agent_summary.md")
    m = re.search(r"Average total_us:\s*([0-9.]+).*?Final total_us:\s*([0-9.]+)", text, re.S)
    if m:
        rows.append(add("P2", "simpleMultiDevice-cuda", float(m.group(1)), float(m.group(2)), "us total time over 4 GPUs", "PASS", "KERNEL_OPT", "measurement_equivalent", "Block-level reduction improves kernel/D2H components, but total time remains H2D-copy-limited with about 1% speedup.", "p2/simpleMultiDevice-cuda/result/agent_summary.md"))

    text = read(ROOT / "p2/shmembench-cuda/result/agent_summary.md")
    m = re.search(r"Average kernel execution time:\s*([0-9.]+) ms.*?Final average kernel execution time:\s*([0-9.]+) ms", text, re.S)
    if m:
        rows.append(add("P2", "shmembench-cuda", float(m.group(1)), float(m.group(2)), "ms avg kernel time", "PASS final", "KERNEL_OPT", "measurement_equivalent", "Valid final optimization is only about 0.13% faster; checksum-failing faster attempt is rejected.", "p2/shmembench-cuda/result/agent_summary.md"))

    return rows


def parse_p3_csv_level():
    rows = []

    # allreduce: no optimization speedup, summarize repeated measurement at largest size.
    data = rows_csv(ROOT / "p3/allreduce-cuda/result/allreduce-cuda_results.csv")
    if data:
        largest = [r for r in data if r["case"] == "size_bytes=536870912"]
        vals = [fnum(r["metric_value"]) for r in largest]
        rows.append(add("P3", "allreduce-cuda", "n/a", mean(vals), "us/iter repeated mean at largest size", "PASS 36/36 total", "ENV_FIX", "success_no_speedup_claim", "P3 correctly treats this as launcher/environment repair with three reproducibility runs.", "p3/allreduce-cuda/result/allreduce-cuda_results.csv"))

    data = rows_csv(ROOT / "p3/moe-align-cuda/result/moe-align_results.csv")
    if data:
        base = [fnum(r["metric_value"]) for r in data if r["variant"] == "baseline"]
        accepted = [fnum(r["metric_value"]) for r in data if r["variant"] == "cached_cumsum" and r["accepted"].lower() == "true"]
        rows.append(add("P3", "moe-align-cuda", mean(base), mean(accepted), "us mean latency across accepted rows", "PASS all scored runs", "PARAM_TUNE", "valid_with_variance_profiler", "Three accepted trials plus profiler notes; rejected regression excluded.", "p3/moe-align-cuda/result/moe-align_results.csv"))

    data = rows_csv(ROOT / "p3/moe-cuda/result/moe-cuda_results.csv")
    if data:
        base = [fnum(r["metric_value"]) for r in data if r["variant"] == "baseline"]
        accepted = [fnum(r["metric_value"]) for r in data if r["variant"] == "fused_smallk"]
        rows.append(add("P3", "moe-cuda", mean(base), mean(accepted), "us arithmetic mean over all topk/trials", "PASS all official cases", "KERNEL_OPT", "valid_with_variance_profiler", "topk 1/2/4 improve; topk=8 explicitly classified measurement-equivalent.", "p3/moe-cuda/result/moe-cuda_results.csv"))

    data = rows_csv(ROOT / "p3/p2p-cuda/result/p2p-cuda_results.csv")
    if data:
        base = [fnum(r["metric_value"]) for r in data if r["variant"] == "baseline"]
        accepted = [fnum(r["metric_value"]) for r in data if r["variant"] == "directed-p2p-copy"]
        rows.append(add("P3", "p2p-cuda", mean(base), mean(accepted), "GB/s average", "PASS 36/36 directed final", "TOPOLOGY_MEASURE/MEASURE_FIX", "measurement_equivalent", "Full directed 4-GPU topology coverage; performance change below 1%.", "p3/p2p-cuda/result/p2p-cuda_results.csv"))

    data = rows_csv(ROOT / "p3/pingpong-cuda/result/pingpong-cuda_results.csv")
    if data:
        # Baseline invalid; final accepted metrics only.
        vals = [
            fnum(r["metric_value"]) for r in data
            if r["metric_name"] == "bandwidth" and "size_bytes=1073741824" in r["case"] and "method=MPI" in r["case"]
        ]
        rows.append(add("P3", "pingpong-cuda", "invalid baseline", mean(vals), "GB/s MPI at 1GiB", "PASS MPI/NCCL full sweep after fix", "MEASURE_FIX", "success_no_speedup_claim", "Baseline NCCL executable missing; P3 correctly reports measurement recovery and avoids speedup claim.", "p3/pingpong-cuda/result/pingpong-cuda_results.csv"))

    text = read(ROOT / "p3/softmax-cuda/result/agent_summary.md")
    m = re.search(r"slice 784 implementation 1:\s*([0-9.]+) ms.*?slice 784.*?improved from [0-9.]+ ms to .*?([0-9.]+) ms", text, re.S)
    if m:
        rows.append(add("P3", "softmax-cuda", float(m.group(1)), float(m.group(2)), "ms slice=784 implementation 1, 3-trial mean", "PASS 42/42 final", "KERNEL_OPT", "valid_with_variance_profiler", "Block-per-slice kernel improves large-slice implementation 1; final result includes 3 trials and contradiction check.", "p3/softmax-cuda/result/softmax-cuda_results.csv"))

    text = read(ROOT / "p3/topk-cuda/result/agent_summary.md")
    m = re.search(r"Baseline mean:\s*([0-9.]+) us.*?Final speedup:\s*[0-9.]+x.*?improves the measured mean from [0-9.]+ us to ([0-9.]+) us", text, re.S)
    if m:
        rows.append(add("P3", "topk-cuda", float(m.group(1)), float(m.group(2)), "us mean over 14 cases, 3 final trials", "PASS all final trials", "KERNEL_OPT", "valid_with_variance", "Hybrid workspace/block-size strategy improves mean top-k time with low trial variance; rejected block512 regression excluded.", "p3/topk-cuda/result/topk-cuda_results.csv"))

    text = read(ROOT / "p3/pretch-cuda/result/agent_summary.md")
    final_section = text.split("## Final Candidate", 1)[-1]
    m = re.search(r"repeat=100 \| without_prefetch \|\s*([0-9.]+)\s*\|\s*([0-9.]+)", final_section)
    if m:
        rows.append(add("P3", "prefetch-cuda", float(m.group(1)), float(m.group(2)), "ms repeat=100 without_prefetch", "PASS 40/40 final", "PARAM_TUNE", "valid_with_variance_profiler", "No-prefetch block-size tuning improves demand-paging path; repeat=100 with_prefetch is explicitly measurement-equivalent.", "p3/pretch-cuda/result/prefetch-cuda_results.csv"))

    text = read(ROOT / "p3/simpleMultiDevice-cuda/result/agent_summary.md")
    base_m = re.search(r"Baseline total_us:\s*([0-9.]+)", text)
    final_m = re.search(r"Final accepted candidate total_us.*?- mean:\s*([0-9.]+) us", text, re.S)
    if base_m and final_m:
        rows.append(add("P3", "simpleMultiDevice-cuda", float(base_m.group(1)), float(final_m.group(1)), "us total time over 4 GPUs, 3-trial mean", "PASS all final trials", "KERNEL_OPT", "measurement_equivalent", "Final kernel optimization is real but total-time speedup is only about 1.2% because H2D copy dominates.", "p3/simpleMultiDevice-cuda/result/simpleMultiDevice-cuda_results.csv"))

    text = read(ROOT / "p3/shmembench-cuda/result/agent_summary.md")
    m = re.search(r"avg kernel time \(ms\) \|\s*([0-9.]+)\s*\|\s*([0-9.]+)", text)
    if m:
        rows.append(add("P3", "shmembench-cuda", float(m.group(1)), float(m.group(2)), "ms avg kernel time, 3-trial mean", "PASS final; failed attempt rejected", "KERNEL_OPT", "valid_with_variance", "Removing unneeded synchronization gives a modest 2.85% time improvement; checksum-failing block-size sweep is rejected.", "p3/shmembench-cuda/result/shmembench-cuda_results.csv"))

    return rows


def write_csv(rows):
    fields = ["level", "benchmark", "baseline_metric", "final_metric", "metric_unit", "speedup", "correctness", "result_type", "status", "notes", "source"]
    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def write_report(rows):
    by_level = defaultdict(list)
    by_bench = defaultdict(list)
    for r in rows:
        by_level[r["level"]].append(r)
        by_bench[r["benchmark"]].append(r)

    def valid_speedups(rs):
        vals = []
        for r in rs:
            try:
                x = float(r["speedup"])
                if not math.isnan(x):
                    vals.append(x)
            except Exception:
                pass
        return vals

    lines = []
    lines.append("# Phase 2 P1/P2/P3 測試結果簡易分析報告")
    lines.append("")
    lines.append("本報告彙整 `/home/a/PP/phase2/p1`、`p2`、`p3` 已回傳結果。分析重點是 prompt 層級對結果完整性、正確性、統計品質與效能宣稱可信度的影響。")
    lines.append("")
    lines.append("## 資料範圍")
    lines.append("")
    complete = [b for b in sorted(by_bench) if len(by_bench[b]) == 3]
    incomplete = [b for b in sorted(by_bench) if len(by_bench[b]) != 3]
    lines.append("- 已有三層結果的 benchmark：" + "、".join(f"`{b}`" for b in complete) + "。")
    if incomplete:
        lines.append("- 尚未形成完整 P1/P2/P3 三層結果的 benchmark：" + "、".join(f"`{b}`" for b in incomplete) + "。")
    else:
        lines.append("- 目前摘要表中的 benchmark 均已形成 P1/P2/P3 三層結果。")
    lines.append(f"- 統一摘要表：`reports/{OUT_CSV.name}`，共 {len(rows)} 筆。")
    lines.append("")
    lines.append("## 層級統計")
    lines.append("")
    lines.append("| Level | 筆數 | 可計算 speedup 平均 | 主要觀察 |")
    lines.append("|---|---:|---:|---|")
    for level in ["P1", "P2", "P3"]:
        rs = by_level[level]
        vals = valid_speedups(rs)
        obs = {
            "P1": "可快速找到可行修改，但 baseline/CSV/variance 常不足，審核成本高。",
            "P2": "開始有 rejected/accepted 紀錄，能過濾失敗嘗試，報告可信度明顯提升。",
            "P3": "CSV、三次 trial、profiler/measurement notes 與 contradiction check 最完整，較少誇大 speedup。",
        }[level]
        lines.append(f"| {level} | {len(rs)} | {fmt(mean(vals), 3) if vals else 'n/a'}x | {obs} |")
    lines.append("")
    lines.append("## Benchmark 橫向比較")
    lines.append("")
    for bench in sorted(by_bench):
        lines.append(f"### {bench}")
        lines.append("| Level | Baseline | Final | Unit | Speedup | Correctness | Result type | Status |")
        lines.append("|---|---:|---:|---|---:|---|---|---|")
        for r in sorted(by_bench[bench], key=lambda x: x["level"]):
            lines.append(f"| {r['level']} | {r['baseline_metric']} | {r['final_metric']} | {r['metric_unit']} | {r['speedup']} | {r['correctness']} | {r['result_type']} | {r['status']} |")
        lines.append("")
        for r in sorted(by_bench[bench], key=lambda x: x["level"]):
            lines.append(f"- {r['level']}: {r['notes']}")
        lines.append("")
    lines.append("## 結論")
    lines.append("")
    lines.append("1. P1 對效能探索有幫助，但常缺少足夠的 baseline、CSV、variance 與無效嘗試紀錄；例如 `moe-align-cuda` 沒有可審核 baseline，`p2p-cuda` 只有 2 GPU/2 direction 結果，無法完整比較。")
    lines.append("2. P2 已能把多數失敗嘗試標成 rejected，對研究報告較友善；`moe-cuda` 與 `pingpong-cuda` 都明確保留失敗/無效提交，不把它們混入最終成果。")
    lines.append("3. P3 的優勢最明顯：標準 CSV、trial 統計、profiler 或 measurement notes、contradiction check 都讓結果更可審核；它也比較會把 `ENV_FIX`、`MEASURE_FIX`、`MEASUREMENT_EQUIVALENT` 與真正 `KERNEL_OPT` 分開。")
    lines.append("4. 效能上，P1 有些案例看起來 speedup 最大，例如 `pingpong-cuda` NCCL 1GiB 約 2x，但因缺少 variance/CSV，可信度低於 P3。P3 不一定追求最高 speedup，而是更準確地界定結果本質。")
    lines.append("")
    lines.append("## 後續建議")
    lines.append("")
    lines.append("- 後續若新增 benchmark，應維持目前的三層摘要格式，並優先補上 P3 CSV、accepted/rejected attempts、variance 與 contradiction check。")
    lines.append("- 之後每層都要求最少輸出一份統一 schema CSV；P1 可以保持弱約束，但實驗紀錄端仍應另外保存 raw log。")
    lines.append("- 統計分析時應分開計算 `KERNEL_OPT` 與 `ENV_FIX/MEASURE_FIX`，否則環境修復會稀釋或誇大 prompt 對 kernel optimization 的影響。")
    lines.append("- 對 P1 的高 speedup 案例進行 P3 重跑驗證，確認是否為真實加速、測量差異或語意改變。")
    lines.append("")
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    REPORTS.mkdir(parents=True, exist_ok=True)
    rows = parse_p1() + parse_p2() + parse_p3_csv_level()
    write_csv(rows)
    write_report(rows)
    print(f"Wrote {OUT_CSV}")
    print(f"Wrote {OUT_MD}")


if __name__ == "__main__":
    main()
