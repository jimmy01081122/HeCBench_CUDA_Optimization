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
    lines.append("- 已有三層結果的 benchmark：`allreduce-cuda`、`moe-align-cuda`、`moe-cuda`、`p2p-cuda`、`pingpong-cuda`。")
    lines.append("- 尚未看到 p1/p2/p3 結果的 benchmark：`softmax-cuda`、`topk-cuda`、`shmembench-cuda`、`simpleMultiDevice-cuda`、`prefetch-cuda`。")
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
    lines.append("- 補齊尚未跑的 5 個 benchmark，尤其 `softmax-cuda`、`topk-cuda` 這類純 kernel optimization 案例，才能更公平評估 P1/P2/P3 對效能探索的影響。")
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
