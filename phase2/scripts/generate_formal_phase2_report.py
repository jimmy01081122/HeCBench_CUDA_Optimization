#!/usr/bin/env python3
import csv
import math
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
FORMAL = REPORTS / "formal_figures"
SUMMARY_CSV = REPORTS / "phase2_level_summary.csv"
FORMAL_CSV = REPORTS / "phase2_formal_statistics.csv"
FORMAL_MD = REPORTS / "PHASE2_FORMAL_REPORT.md"


LEVELS = ["P1", "P2", "P3"]
LEVEL_COLORS = {
    "P1": "#d97706",
    "P2": "#2563eb",
    "P3": "#059669",
}


def read_rows():
    with SUMMARY_CSV.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def fnum(x):
    try:
        return float(x)
    except Exception:
        return float("nan")


def mean(vals):
    vals = [v for v in vals if not math.isnan(v)]
    return sum(vals) / len(vals) if vals else float("nan")


def median(vals):
    vals = sorted(v for v in vals if not math.isnan(v))
    if not vals:
        return float("nan")
    mid = len(vals) // 2
    return vals[mid] if len(vals) % 2 else (vals[mid - 1] + vals[mid]) / 2


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


def is_numeric(x):
    return not math.isnan(fnum(x))


def audit_score(row):
    score = 0
    if row["correctness"].strip():
        score += 20
    if is_numeric(row["baseline_metric"]):
        score += 20
    if is_numeric(row["speedup"]):
        score += 15
    if row["source"].endswith(".csv"):
        score += 20
    if "variance" in row["status"] or "profiler" in row["status"] or row["level"] == "P3":
        score += 15
    if "weak" not in row["status"] and "caution" not in row["status"]:
        score += 10
    return score


def result_family(result_type):
    if "KERNEL_OPT" in result_type:
        return "KERNEL_OPT"
    if "ENV_FIX" in result_type:
        return "ENV_FIX"
    if "TOPOLOGY" in result_type:
        return "TOPOLOGY_MEASURE"
    if "MEASURE_FIX" in result_type:
        return "MEASURE_FIX"
    if "PARAM_TUNE" in result_type:
        return "PARAM_TUNE"
    return result_type.split("/")[0]


def svg_header(width, height):
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<style>text{font-family:Arial,Helvetica,sans-serif;fill:#111827}.axis{stroke:#6b7280;stroke-width:1}.grid{stroke:#e5e7eb;stroke-width:1}.label{font-size:12px}.title{font-size:18px;font-weight:700}.note{font-size:11px;fill:#4b5563}</style>',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
    ]


def save_svg(path, lines):
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def chart_speedup(rows):
    data = []
    for r in rows:
        sp = fnum(r["speedup"])
        if not math.isnan(sp):
            data.append((r["benchmark"], r["level"], sp))
    width, height = 980, 520
    left, top, bottom = 80, 58, 120
    plot_w, plot_h = 840, 330
    max_v = max(sp for _, _, sp in data)
    y_max = max(3.0, math.ceil(max_v * 10) / 10)
    benches = sorted(set(b for b, _, _ in data))
    group_w = plot_w / len(benches)
    bar_w = group_w / 5
    lines = svg_header(width, height)
    lines.append(f'<text x="{left}" y="30" class="title">Figure 1. Speedup by Prompt Level</text>')
    lines.append(f'<text x="{left}" y="48" class="note">Only rows with measured baseline and final metric are included. Higher is better.</text>')
    for i in range(7):
        val = y_max * i / 6
        y = top + plot_h - (val / y_max) * plot_h
        lines.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left+plot_w}" y2="{y:.1f}" class="grid"/>')
        lines.append(f'<text x="{left-8}" y="{y+4:.1f}" text-anchor="end" class="label">{val:.1f}x</text>')
    lines.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_h}" class="axis"/>')
    lines.append(f'<line x1="{left}" y1="{top+plot_h}" x2="{left+plot_w}" y2="{top+plot_h}" class="axis"/>')
    by = {(b, l): s for b, l, s in data}
    for bi, bench in enumerate(benches):
        gx = left + bi * group_w
        for li, level in enumerate(LEVELS):
            if (bench, level) not in by:
                continue
            sp = by[(bench, level)]
            x = gx + group_w / 2 - 1.5 * bar_w + li * bar_w
            h = (sp / y_max) * plot_h
            y = top + plot_h - h
            lines.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w*0.82:.1f}" height="{h:.1f}" fill="{LEVEL_COLORS[level]}"/>')
            lines.append(f'<text x="{x+bar_w*0.41:.1f}" y="{y-4:.1f}" text-anchor="middle" class="note">{sp:.2f}</text>')
        lines.append(f'<text x="{gx+group_w/2:.1f}" y="{top+plot_h+18}" text-anchor="end" transform="rotate(-32 {gx+group_w/2:.1f},{top+plot_h+18})" class="label">{bench}</text>')
    lx = left + 640
    for i, level in enumerate(LEVELS):
        lines.append(f'<rect x="{lx+i*80}" y="22" width="14" height="14" fill="{LEVEL_COLORS[level]}"/>')
        lines.append(f'<text x="{lx+i*80+20}" y="34" class="label">{level}</text>')
    lines.append("</svg>")
    save_svg(FORMAL / "figure1_speedup_by_level.svg", lines)


def chart_auditability(rows):
    by_level = defaultdict(list)
    for r in rows:
        by_level[r["level"]].append(audit_score(r))
    data = [(l, mean(by_level[l])) for l in LEVELS]
    width, height = 720, 420
    left, top, plot_w, plot_h = 80, 58, 560, 270
    lines = svg_header(width, height)
    lines.append(f'<text x="{left}" y="30" class="title">Figure 2. Auditability Score by Prompt Level</text>')
    lines.append(f'<text x="{left}" y="48" class="note">Score combines correctness, measured baseline, speedup, CSV, variance/profiler, and caution flags.</text>')
    for i in range(6):
        val = 100 * i / 5
        y = top + plot_h - (val / 100) * plot_h
        lines.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left+plot_w}" y2="{y:.1f}" class="grid"/>')
        lines.append(f'<text x="{left-8}" y="{y+4:.1f}" text-anchor="end" class="label">{val:.0f}</text>')
    bar_gap = 70
    bar_w = 110
    for i, (level, val) in enumerate(data):
        x = left + 80 + i * (bar_w + bar_gap)
        h = (val / 100) * plot_h
        y = top + plot_h - h
        lines.append(f'<rect x="{x}" y="{y:.1f}" width="{bar_w}" height="{h:.1f}" fill="{LEVEL_COLORS[level]}"/>')
        lines.append(f'<text x="{x+bar_w/2}" y="{y-8:.1f}" text-anchor="middle" class="label">{val:.1f}</text>')
        lines.append(f'<text x="{x+bar_w/2}" y="{top+plot_h+28}" text-anchor="middle" class="label">{level}</text>')
    lines.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_h}" class="axis"/>')
    lines.append(f'<line x1="{left}" y1="{top+plot_h}" x2="{left+plot_w}" y2="{top+plot_h}" class="axis"/>')
    lines.append("</svg>")
    save_svg(FORMAL / "figure2_auditability_by_level.svg", lines)


def chart_result_types(rows):
    families = ["KERNEL_OPT", "PARAM_TUNE", "MEASURE_FIX", "ENV_FIX", "TOPOLOGY_MEASURE"]
    colors = {
        "KERNEL_OPT": "#7c3aed",
        "PARAM_TUNE": "#0891b2",
        "MEASURE_FIX": "#ea580c",
        "ENV_FIX": "#16a34a",
        "TOPOLOGY_MEASURE": "#64748b",
    }
    counts = {level: Counter(result_family(r["result_type"]) for r in rows if r["level"] == level) for level in LEVELS}
    width, height = 820, 420
    left, top, plot_w, plot_h = 90, 58, 570, 260
    lines = svg_header(width, height)
    lines.append(f'<text x="{left}" y="30" class="title">Figure 3. Result Type Distribution</text>')
    lines.append(f'<text x="{left}" y="48" class="note">Counts final result family per prompt level.</text>')
    bar_w = 120
    for i, level in enumerate(LEVELS):
        x = left + 45 + i * 170
        y_cursor = top + plot_h
        total = sum(counts[level].values()) or 1
        for fam in families:
            h = counts[level][fam] / total * plot_h
            if h <= 0:
                continue
            y_cursor -= h
            lines.append(f'<rect x="{x}" y="{y_cursor:.1f}" width="{bar_w}" height="{h:.1f}" fill="{colors[fam]}"/>')
            if h > 18:
                lines.append(f'<text x="{x+bar_w/2}" y="{y_cursor+h/2+4:.1f}" text-anchor="middle" class="label" fill="#fff">{counts[level][fam]}</text>')
        lines.append(f'<text x="{x+bar_w/2}" y="{top+plot_h+28}" text-anchor="middle" class="label">{level}</text>')
    lx, ly = left + plot_w + 30, top
    for i, fam in enumerate(families):
        lines.append(f'<rect x="{lx}" y="{ly+i*26}" width="14" height="14" fill="{colors[fam]}"/>')
        lines.append(f'<text x="{lx+22}" y="{ly+i*26+12}" class="label">{fam}</text>')
    lines.append(f'<line x1="{left}" y1="{top+plot_h}" x2="{left+plot_w}" y2="{top+plot_h}" class="axis"/>')
    lines.append("</svg>")
    save_svg(FORMAL / "figure3_result_type_distribution.svg", lines)


def write_formal_stats(rows):
    out = []
    for level in LEVELS:
        rs = [r for r in rows if r["level"] == level]
        speeds = [fnum(r["speedup"]) for r in rs if is_number(r["speedup"])]
        out.append({
            "level": level,
            "n": len(rs),
            "speedup_mean": fmt(mean(speeds)),
            "speedup_median": fmt(median(speeds)),
            "numeric_speedup_count": len(speeds),
            "mean_auditability_score": fmt(mean([audit_score(r) for r in rs]), 1),
            "csv_source_count": sum(1 for r in rs if r["source"].endswith(".csv")),
            "missing_or_invalid_baseline_count": sum(1 for r in rs if not is_numeric_baseline(r)),
            "measurement_equivalent_or_no_speedup_count": sum(1 for r in rs if "measurement_equivalent" in r["status"] or "no_speedup" in r["status"]),
        })
    with FORMAL_CSV.open("w", newline="", encoding="utf-8") as f:
        fields = list(out[0].keys())
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(out)


def is_number(x):
    return not math.isnan(fnum(x))


def is_numeric_baseline(row):
    return is_number(row["baseline_metric"])


def write_report(rows):
    by_level = defaultdict(list)
    by_bench = defaultdict(list)
    for r in rows:
        by_level[r["level"]].append(r)
        by_bench[r["benchmark"]].append(r)
    complete_benches = [b for b in sorted(by_bench) if len(by_bench[b]) == len(LEVELS)]
    incomplete_benches = [b for b in sorted(by_bench) if len(by_bench[b]) != len(LEVELS)]
    bench_list = "、".join(f"`{b}`" for b in complete_benches)

    def speeds(level):
        return [fnum(r["speedup"]) for r in by_level[level] if is_number(r["speedup"])]

    lines = []
    lines.append("# Phase 2 正式研究報告：Prompt 約束層級對 AI 輔助 CUDA 優化之影響")
    lines.append("")
    lines.append("## 摘要")
    lines.append("")
    lines.append("本研究比較 P1、P2、P3 三種 prompt 約束層級在 HeCBench CUDA benchmark 優化任務中的表現。P1 代表弱約束、接近日常對話式要求；P2 代表具備基本工程規格的中約束 prompt；P3 則是包含 baseline、correctness gate、CSV schema、variance/profiler、contradiction check 與 result-type classification 的強約束實驗 protocol。")
    lines.append("")
    lines.append(f"已完成分析的 benchmark 包含 {bench_list}，共 {len(rows)} 筆 final-level 摘要。結果顯示：P1 有較高機率產生看似亮眼的效能數字，但資料完整性與可審核性不足；P2 能明顯改善 rejected attempt 紀錄與 baseline 對照；P3 不一定追求最高 speedup，但最能產出可重現、可審核、可分類的研究資料。")
    lines.append("")
    lines.append("## 研究問題")
    lines.append("")
    lines.append("- RQ2-1：prompt 約束強度是否影響 AI agent 的正確性、效能與可審核性？")
    lines.append("- RQ2-2：強約束 prompt 是否能降低偽加速、錯誤報告與不可重現結果？")
    lines.append("- RQ2-3：哪些 prompt 條款最關鍵？")
    lines.append("- RQ2-4：prompt.md 是否比一般網頁對話更適合作為工程協作介面？")
    lines.append("")
    lines.append("## 方法")
    lines.append("")
    lines.append("本報告使用 `/home/a/PP/phase2/reports/phase2_level_summary.csv` 作為主要資料來源。效能指標依 benchmark 類型分為 latency/time 與 bandwidth/throughput：latency 類以 `baseline / final` 作為 improvement ratio；bandwidth 類以 `final / baseline` 作為 improvement ratio。對於 baseline invalid 或缺失的案例，不計算 speedup。")
    lines.append("")
    lines.append("可審核性分數由六項構成：correctness 是否記錄、是否有 measured baseline、是否能計算 speedup、是否有 CSV source、是否有 variance/profiler 或 P3 protocol 紀錄、是否沒有 caution/weak auditability 標記。此分數不是效能分數，而是研究資料品質指標。")
    lines.append("")
    lines.append("## 圖表")
    lines.append("")
    lines.append("![Figure 1](formal_figures/figure1_speedup_by_level.svg)")
    lines.append("")
    lines.append("![Figure 2](formal_figures/figure2_auditability_by_level.svg)")
    lines.append("")
    lines.append("![Figure 3](formal_figures/figure3_result_type_distribution.svg)")
    lines.append("")
    lines.append("## 統計摘要")
    lines.append("")
    lines.append("| Level | N | Speedup mean | Speedup median | Numeric speedups | Mean auditability | CSV source count | Missing/invalid baseline | Measurement-equivalent/no-speedup |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for level in LEVELS:
        rs = by_level[level]
        sp = speeds(level)
        lines.append(
            f"| {level} | {len(rs)} | {fmt(mean(sp))} | {fmt(median(sp))} | {len(sp)} | "
            f"{fmt(mean([audit_score(r) for r in rs]), 1)} | "
            f"{sum(1 for r in rs if r['source'].endswith('.csv'))} | "
            f"{sum(1 for r in rs if not is_numeric_baseline(r))} | "
            f"{sum(1 for r in rs if 'measurement_equivalent' in r['status'] or 'no_speedup' in r['status'])} |"
        )
    lines.append("")
    lines.append("## 主要發現")
    lines.append("")
    lines.append("### Finding 1：P1 提升探索性，但也提高審核風險")
    lines.append("")
    lines.append("P1 的平均 speedup 受到 `pingpong-cuda` NCCL 1GiB 約 2x 的結果拉高，但 P1 常缺少 CSV、variance、完整 baseline 或完整 case coverage。例如 `moe-align-cuda` 只有 final latency，無 measured baseline；`p2p-cuda` 僅有 2 GPU / 2 direction 結果，不能代表完整 4-GPU topology matrix。這表示 P1 適合探索可能方向，但不適合直接作為正式研究結論。")
    lines.append("")
    lines.append("### Finding 2：P2 是工程可用性的最低門檻")
    lines.append("")
    lines.append("P2 的報告普遍包含 baseline、accepted/rejected attempts 與 final decision。`moe-cuda` 清楚排除 correctness fail 或 timeout 的嘗試；`pingpong-cuda` 保留 invalid setup attempts 並把最後結果標示為 noise-level/measurement-equivalent。這使 P2 已經能支撐基本工程報告，但仍缺少 P3 所要求的統一 CSV、variance/profiler 與 contradiction check。")
    lines.append("")
    lines.append("### Finding 3：P3 強化研究可審核性，且能抑制過度宣稱")
    lines.append("")
    lines.append("P3 的平均可審核性最高。`allreduce-cuda` 在 P3 中被明確分類為 `ENV_FIX`，不宣稱 kernel speedup；`pingpong-cuda` 因 baseline NCCL executable missing 而不計算 speedup，只回報 measurement recovery；`p2p-cuda` 則將 full directed topology sweep 標記為 `MEASURE_FIX` 與 measurement-equivalent。這些案例顯示 P3 的核心價值是避免把環境修復、測量修復或 topology coverage 誤寫成實際效能優化。")
    lines.append("")
    lines.append("### Finding 4：Prompt 條款中最關鍵的是 correctness gate、baseline、CSV schema 與 contradiction check")
    lines.append("")
    lines.append("結果顯示，單純要求「保持 correctness」不足以產生可審核資料；必須進一步要求 measured baseline、raw output、CSV schema、accepted/rejected 分類、variance/trials 與 contradiction check。尤其在 `pingpong-cuda`、`allreduce-cuda` 這類環境與 launcher 敏感的 benchmark 中，result type classification 是避免錯誤結論的關鍵。")
    lines.append("")
    lines.append("## Benchmark 細部結論")
    lines.append("")
    for bench in sorted(by_bench):
        lines.append(f"### {bench}")
        for r in sorted(by_bench[bench], key=lambda x: x["level"]):
            lines.append(f"- {r['level']}: speedup `{r['speedup']}`，status `{r['status']}`，type `{r['result_type']}`。{r['notes']}")
        lines.append("")
    lines.append("## 後續優化建議")
    lines.append("")
    lines.append("- `allreduce-cuda`：後續應分離 launcher/environment repair 與 collective algorithm/kernel optimization，並固定 NCCL、MPI、GPU topology 與 buffer-size sweep 條件。若最大 size 未改善，報告應避免只引用 geomean。")
    lines.append("- `moe-align-cuda`：目前最穩定的方向是 workspace/cache 與參數調校。下一輪可加入不同 token/expert 分佈，檢查優化是否只對單一 workload 有效。")
    lines.append("- `moe-cuda`：P1/P2/P3 都顯示 topk=1/2/4 較有優化空間，topk=8 可能接近 memory 或 algorithm bottleneck。建議分別記錄 per-topk speedup，不只看 arithmetic mean。")
    lines.append("- `p2p-cuda`：效能變化接近 measurement-equivalent，研究價值主要在完整 topology coverage。後續應將 GPU pair、方向、NUMA/PCIe/NVLink 資訊納入輸出欄位。")
    lines.append("- `pingpong-cuda`：P1 的高 speedup 需要以完整 baseline 與重複試驗重新驗證。下一輪應固定 MPI/NCCL executable、message-size sweep、warmup、iteration count，並把 invalid setup 與 performance result 分開統計。")
    lines.append("- `softmax-cuda`：P2/P3 證明大 slice 的 implementation 1 可由 block-per-slice reduction 受益。後續應分 slice 分組報告，避免小 slice measurement-equivalent 掩蓋大 slice speedup。")
    lines.append("- `topk-cuda`：workspace reuse 是穩定收益來源；後續可把 allocation time、kernel time 與 workspace size 分欄，確認 speedup 來自 timed-loop allocation removal 還是真正 kernel improvement。")
    lines.append("- `prefetch-cuda`：P2/P3 指向不同 primary metric，後續應明確指定 with_prefetch 與 without_prefetch 的主結論是否分開，並把 prefetch API cost 與 steady-state kernel time 分開統計。")
    lines.append("- `simpleMultiDevice-cuda`：總時間受 H2D copy 主導，kernel speedup 會被 total_us 稀釋。後續應同時報告 total/h2d/kernel/d2h，並禁止將改變測量範圍的結果列入 speedup。")
    lines.append("- `shmembench-cuda`：同步移除可帶來小幅但穩定改善；後續需補 Nsight Compute 的 bank conflict、occupancy 與 instruction mix，並嚴格排除 checksum failed 的較快結果。")
    lines.append("")
    lines.append("## Prompt 改善建議")
    lines.append("")
    lines.append("- 保留 P3 作為正式實驗主 prompt，並把 `result_type` 設為必填欄位，例如 `KERNEL_OPT`、`PARAM_TUNE`、`ENV_FIX`、`MEASURE_FIX`、`TOPOLOGY_MEASURE`、`NO_VALID_SPEEDUP`。")
    lines.append("- 要求每個 benchmark 都輸出同一組檔案：raw log、baseline CSV、final CSV、accepted/rejected attempts、summary.md。缺任一檔案時，必須在 summary 中標記 `incomplete_audit_trail`。")
    lines.append("- 在 prompt 中明確禁止只回報最佳單點數字；必須同時回報 full sweep、mean/median、重複次數、是否有 outlier，以及 largest-size 或主要 workload 的單獨結果。")
    lines.append("- 加入 contradiction check：若 correctness fail、baseline invalid、缺少 baseline、或 metric direction 不一致，agent 必須輸出 `no_speedup_claim`，不能宣稱優化成功。")
    lines.append("- 對 P1/P2 對照組可保留較少限制，但仍建議最低限度加入運行指令、correctness command、benchmark command 與 output path，避免資料無法重現。")
    lines.append("")
    lines.append("## 威脅與限制")
    lines.append("")
    if incomplete_benches:
        lines.append("- 目前仍有 benchmark 未形成完整 P1/P2/P3 三層分析：" + "、".join(f"`{b}`" for b in incomplete_benches) + "。")
    else:
        lines.append(f"- 目前摘要表已涵蓋 {len(complete_benches)} 個 benchmark 的 P1/P2/P3 三層分析；後續限制主要來自各 benchmark metric 不同與部分 P1 資料缺少結構化紀錄。")
    lines.append("- 不同 benchmark 的 metric 單位不同，跨 benchmark 的平均 speedup 只能作為方向性指標，不應視為統一排行榜。")
    lines.append("- P1 缺少結構化資料，部分數值需由 summary 或 raw log 解析，存在較高整理誤差風險。")
    lines.append("- 部分 P3 結果刻意不計算 speedup，因其任務本質是環境修復或測量恢復；這會降低表面效能平均，但提高研究可信度。")
    lines.append("")
    lines.append("## 研究結論")
    lines.append("")
    lines.append("Phase 2 結果支持以下結論：prompt 約束層級不只影響 agent 是否能優化程式，也深刻影響結果是否可驗證、可重現與可用於研究。P1 適合探索，P2 適合工程協作，P3 最適合作為正式實驗 protocol。若目標是撰寫研究報告或比較 AI agent 優化能力，建議以 P3 作為主實驗條件，並將 P1/P2 作為對照組，用來量化 prompt 約束不足時產生的偽加速、缺 baseline、缺 raw output 與過度宣稱問題。")
    lines.append("")
    FORMAL_MD.write_text("\n".join(lines), encoding="utf-8")


def main():
    FORMAL.mkdir(parents=True, exist_ok=True)
    rows = read_rows()
    chart_speedup(rows)
    chart_auditability(rows)
    chart_result_types(rows)
    write_formal_stats(rows)
    write_report(rows)
    print(f"Wrote {FORMAL_MD}")
    print(f"Wrote {FORMAL_CSV}")
    print(f"Wrote figures to {FORMAL}")


if __name__ == "__main__":
    main()
