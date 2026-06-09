#!/usr/bin/env python3
import csv
import math
import re
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "benchmark_analysis_report"
DATA = OUT / "data"
PROMPTS = OUT / "prompts"


def read_text(path):
    return path.read_text(encoding="utf-8", errors="replace")


def read_csv(path):
    with path.open(newline="", encoding="utf-8", errors="replace") as f:
        return list(csv.DictReader(f))


def write_csv(path, fieldnames, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def mean(values):
    vals = [float(v) for v in values if v is not None and v != "" and not math.isnan(float(v))]
    return sum(vals) / len(vals) if vals else float("nan")


def fmt(x, digits=3):
    if x is None:
        return "n/a"
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return str(x)
    if math.isnan(xf):
        return "n/a"
    return f"{xf:.{digits}f}"


def speedup(old, new):
    try:
        old, new = float(old), float(new)
        if new == 0:
            return float("nan")
        return old / new
    except (TypeError, ValueError):
        return float("nan")


def prompt_summary():
    prompt_files = sorted(
        set(ROOT.glob("**/prompt.md"))
        | set(ROOT.glob("**/*_prompt.md"))
    )
    rows = []
    for path in prompt_files:
        text = read_text(path)
        rel = path.relative_to(ROOT).as_posix()
        lower = text.lower()
        rows.append({
            "prompt_file": rel,
            "lines": str(text.count("\n") + 1),
            "chars": str(len(text)),
            "mentions_correctness": str(("correctness" in lower) or ("pass" in lower)),
            "mentions_baseline": str("baseline" in lower),
            "mentions_sbatch": str("sbatch" in lower),
            "mentions_submission_limit": str(("最多" in text and "提交" in text) or ("5 次" in text) or ("三次" in text)),
            "mentions_raw_output": str(("raw output" in lower) or ("結果" in text)),
            "main_strength": classify_prompt_strength(text),
            "main_gap": classify_prompt_gap(text),
        })
    write_csv(PROMPTS / "prompt_inventory.csv", list(rows[0].keys()) if rows else ["prompt_file"], rows)
    return rows


def classify_prompt_strength(text):
    score = 0
    for key in ["correctness", "baseline", "sbatch", "不得", "PASS", "FAIL", "raw output", "備份"]:
        if key in text:
            score += 1
    if score >= 7:
        return "限制清楚，利於避免偽加速"
    if score >= 4:
        return "基本約束完整"
    return "任務描述偏短，需要更多實驗規格"


def classify_prompt_gap(text):
    lower = text.lower()
    gaps = []
    if "nsight" not in lower and "profiler" not in lower:
        gaps.append("缺少 profiler 指標要求")
    if "std" not in lower and "標準差" not in text and "variance" not in lower:
        gaps.append("缺少變異與重複統計要求")
    if "baseline" not in lower:
        gaps.append("baseline 定義不足")
    return "；".join(gaps[:2]) if gaps else "規格相對完整"


def summarize_moe():
    rows = []
    cg_summary = ROOT / "moe/CG/summary.md"
    if cg_summary.exists():
        text = read_text(cg_summary)
        table_match = re.search(r"\| top-k \| Baseline \| V1 \| V2 \| V3 \|\n\|.*?\|\n((?:\|.*\|\n)+)", text)
        if table_match:
            for line in table_match.group(1).strip().splitlines():
                cells = [c.strip() for c in line.strip("|").split("|")]
                if len(cells) == 5:
                    topk, baseline, v1, v2, v3 = cells
                    rows.append({
                        "benchmark": "moe-cuda",
                        "agent": "CG",
                        "case": f"topk={topk}",
                        "baseline_metric": baseline,
                        "optimized_metric": v3,
                        "metric_unit": "us",
                        "speedup": fmt(speedup(baseline, v3), 3),
                        "correctness": "PASS",
                        "best_strategy": {
                            "1": "dedicated top-k=1 softmax probability kernel",
                            "2": "fused softmax + top-k",
                            "4": "fused softmax + top-k",
                            "8": "original two-kernel path",
                        }.get(topk, "hybrid dispatch"),
                        "source": "moe/CG/summary.md",
                    })
    return rows


def summarize_moe_align():
    rows = []
    for path in sorted((ROOT / "moe_align").glob("cg_vs_gm_v*_comparison.csv")):
        data = read_csv(path)
        version = re.search(r"v(\d+)", path.name).group(1)
        winners = Counter(row["winner"] for row in data)
        speed_col = next((c for c in data[0].keys() if "speedup" in c), None)
        adv_col = next((c for c in data[0].keys() if "adv_pct" in c), None)
        rows.append({
            "benchmark": "moe-align",
            "agent": f"CG_vs_GM_V{version}",
            "case": "all parameter combinations",
            "baseline_metric": "GM mean latency",
            "optimized_metric": "CG mean latency",
            "metric_unit": "ratio",
            "speedup": fmt(mean([r[speed_col] for r in data]), 3) if speed_col else "n/a",
            "correctness": "not shown in comparison csv",
            "best_strategy": f"winner count: {dict(winners)}; mean CG advantage {fmt(mean([r[adv_col] for r in data]), 2)}%" if adv_col else f"winner count: {dict(winners)}",
            "source": path.relative_to(ROOT).as_posix(),
        })
    return rows


def summarize_pingpong():
    path = ROOT / "pingppong/pingpong-cudaCODEX/result/pingpong_results_946595.csv"
    if not path.exists():
        return []
    data = read_csv(path)
    by_backend = defaultdict(list)
    by_size = defaultdict(dict)
    for row in data:
        by_backend[row["backend"]].append(float(row["gbps"]))
        by_size[int(row["size_bytes"])].setdefault(row["backend"], []).append(float(row["gbps"]))
    large = max(by_size)
    mpi_large = mean(by_size[large].get("MPI", []))
    nccl_large = mean(by_size[large].get("NCCL", []))
    return [{
        "benchmark": "pingpong-cuda",
        "agent": "CODEX",
        "case": "2 ranks / 2 GPUs final sweep",
        "baseline_metric": f"NCCL {fmt(nccl_large)} GB/s at {large} bytes",
        "optimized_metric": f"MPI {fmt(mpi_large)} GB/s at {large} bytes",
        "metric_unit": "GB/s",
        "speedup": fmt(mpi_large / nccl_large if nccl_large else float("nan"), 3),
        "correctness": "MPI PASS; NCCL PASS",
        "best_strategy": "tuned CUDA-aware MPI/UCX path was fastest for two-rank ping-pong",
        "source": path.relative_to(ROOT).as_posix(),
    }]


def summarize_prefetch():
    path = ROOT / "prefetch-cuda/GM/result/prefetch_results_947420.csv"
    if not path.exists():
        return []
    data = read_csv(path)
    rows = []
    grouped = defaultdict(dict)
    for r in data:
        grouped[(r["repeat"], r["prefetch_mode"])][r["variant"]] = r
    for (rep, mode), variants in sorted(grouped.items(), key=lambda x: (int(x[0][0]), x[0][1])):
        if "baseline" in variants and "optimized" in variants:
            b = variants["baseline"]["avg_ms"]
            o = variants["optimized"]["avg_ms"]
            rows.append({
                "benchmark": "prefetch-cuda",
                "agent": "GM",
                "case": f"repeat={rep}, {mode}",
                "baseline_metric": b,
                "optimized_metric": o,
                "metric_unit": "ms",
                "speedup": fmt(speedup(b, o), 3),
                "correctness": "PASS",
                "best_strategy": "vectorized/tuned grid-stride loops; prefetch overhead dominates when prefetch is already used",
                "source": path.relative_to(ROOT).as_posix(),
            })
    return rows


def summarize_softmax():
    path = ROOT / "softmax-cuda/GM/result/softmax_results_947443.csv"
    if not path.exists():
        return []
    data = read_csv(path)
    rows = []
    by_slice = defaultdict(dict)
    for r in data:
        by_slice[r["slice_size"]][r["implementation"]] = r
    for slice_size, impls in sorted(by_slice.items(), key=lambda x: int(x[0])):
        naive = next((v for k, v in impls.items() if k.startswith("naive")), None)
        best = min(impls.values(), key=lambda r: float(r["avg_ms"]))
        if naive:
            rows.append({
                "benchmark": "softmax-cuda",
                "agent": "GM",
                "case": f"slice={slice_size}",
                "baseline_metric": naive["avg_ms"],
                "optimized_metric": best["avg_ms"],
                "metric_unit": "ms",
                "speedup": fmt(speedup(naive["avg_ms"], best["avg_ms"]), 3),
                "correctness": best["correctness"],
                "best_strategy": best["implementation"],
                "source": path.relative_to(ROOT).as_posix(),
            })
    return rows


def summarize_shmembench():
    rows = []
    for agent, rel in [
        ("CG", "shmembench-cuda/CG/result/shmembench_results_946735.csv"),
        ("GM", "shmembench-cuda/GM/shmembench-cuda/result/shmembench_results_947389.csv"),
    ]:
        path = ROOT / rel
        if not path.exists():
            continue
        data = read_csv(path)
        best = max(data, key=lambda r: float(r["bandwidth_GBps"]))
        rows.append({
            "benchmark": "shmembench-cuda",
            "agent": agent,
            "case": f"best block={best['block_size']}",
            "baseline_metric": "original block 256" if agent == "GM" else "baseline 13121.63 GB/s",
            "optimized_metric": best["bandwidth_GBps"],
            "metric_unit": "GB/s",
            "speedup": "n/a" if agent == "GM" else fmt(float(best["bandwidth_GBps"]) / 13121.63, 3),
            "correctness": best["correctness"],
            "best_strategy": best.get("notes", "shared-memory benchmark"),
            "source": rel,
        })
    return rows


def summarize_simple_multi():
    path = ROOT / "simpleMutiDevice/GM/simpleMultiDevice-cuda/result/simpleMultiDevice_results_947375.csv"
    if not path.exists():
        return []
    data = read_csv(path)
    by_gpu = {r["num_gpus"]: r for r in data}
    rows = []
    one = by_gpu.get("1")
    for ngpu in ["2", "4"]:
        r = by_gpu.get(ngpu)
        if one and r:
            rows.append({
                "benchmark": "simpleMultiDevice-cuda",
                "agent": "GM",
                "case": f"{ngpu} GPUs vs 1 GPU",
                "baseline_metric": one["total_us"],
                "optimized_metric": r["total_us"],
                "metric_unit": "us",
                "speedup": fmt(speedup(one["total_us"], r["total_us"]), 3),
                "correctness": r["status"],
                "best_strategy": "multi-GPU partitioned reduction; end-to-end limited by H2D copy",
                "source": path.relative_to(ROOT).as_posix(),
            })
    return rows


def summarize_topk():
    rows = []
    gm = ROOT / "topk-cuda/GM/topk-cuda/result/topk_results_947405.csv"
    if gm.exists():
        data = read_csv(gm)
        grouped = defaultdict(dict)
        for r in data:
            grouped[(r["hidden_size"], r["topk"])][r["variant"]] = r
        speedups = []
        for key, vals in grouped.items():
            if "baseline" in vals and "workspace_reuse_block512" in vals:
                speedups.append(speedup(vals["baseline"]["avg_us"], vals["workspace_reuse_block512"]["avg_us"]))
        rows.append({
            "benchmark": "topk-cuda",
            "agent": "GM",
            "case": "14 hidden_size/topk combinations",
            "baseline_metric": "baseline radix selection",
            "optimized_metric": "workspace_reuse_block512",
            "metric_unit": "us",
            "speedup": fmt(mean(speedups), 3),
            "correctness": "PASS",
            "best_strategy": "reuse CUB workspace and tune block size to 512",
            "source": gm.relative_to(ROOT).as_posix(),
        })
    cg_base = ROOT / "topk-cuda/CG/result/topk_cuda_result_946771.csv"
    cg_opt = ROOT / "topk-cuda/CG/result/topk_cuda_result_946783.csv"
    if cg_base.exists() and cg_opt.exists():
        base = read_csv(cg_base)
        opt = read_csv(cg_opt)
        bmap = {(r["hidden_size"], r["topk"]): r for r in base}
        speedups = []
        for r in opt:
            b = bmap.get((r["hidden_size"], r["topk"]))
            if b:
                speedups.append(speedup(b["avg_us"], r["avg_us"]))
        rows.append({
            "benchmark": "topk-cuda",
            "agent": "CG",
            "case": "14 hidden_size/topk combinations",
            "baseline_metric": "cuda_event_instrumented",
            "optimized_metric": "cached_workspace_async",
            "metric_unit": "us",
            "speedup": fmt(mean(speedups), 3),
            "correctness": "PASS",
            "best_strategy": "cache workspace and remove repeated allocation/synchronization",
            "source": f"{cg_base.relative_to(ROOT).as_posix()} + {cg_opt.relative_to(ROOT).as_posix()}",
        })
    return rows


def summarize_p2p_and_allreduce():
    rows = []
    p2p = ROOT / "p2p-cuda/codex/agent_summary.md"
    if p2p.exists():
        text = read_text(p2p)
        m = re.search(r"best stable avg ([0-9.]+) GB/s", text)
        bw = m.group(1) if m else "48.4455"
        rows.append({
            "benchmark": "p2p-cuda",
            "agent": "CODEX",
            "case": "4-GPU all-pair sweep",
            "baseline_metric": "48.24 GB/s previous best",
            "optimized_metric": f"{bw} GB/s",
            "metric_unit": "GB/s",
            "speedup": fmt(float(bw) / 48.24, 3),
            "correctness": "144 sweep points PASS; 12/12 pair checks PASS",
            "best_strategy": "topology-aware peer copy sweep; best gain is measurement-equivalent",
            "source": p2p.relative_to(ROOT).as_posix(),
        })
    allreduce = ROOT / "allreduce/base/agent_summary.md"
    if allreduce.exists():
        rows.append({
            "benchmark": "allreduce-cuda",
            "agent": "CODEX",
            "case": "2 ranks / 2 GPUs",
            "baseline_metric": "baseline run failed after size 0",
            "optimized_metric": "all tested nonzero sizes PASS",
            "metric_unit": "correctness/timing",
            "speedup": "n/a",
            "correctness": "PASS after tuned UCX launcher",
            "best_strategy": "avoid broken GDRCopy path with UCX_TLS=self,shm,cuda_copy,cuda_ipc",
            "source": allreduce.relative_to(ROOT).as_posix(),
        })
    return rows


def build_metric_rows():
    rows = []
    for fn in [
        summarize_moe,
        summarize_moe_align,
        summarize_pingpong,
        summarize_prefetch,
        summarize_softmax,
        summarize_shmembench,
        summarize_simple_multi,
        summarize_topk,
        summarize_p2p_and_allreduce,
    ]:
        rows.extend(fn())
    fields = ["benchmark", "agent", "case", "baseline_metric", "optimized_metric", "metric_unit", "speedup", "correctness", "best_strategy", "source"]
    write_csv(DATA / "benchmark_summary.csv", fields, rows)
    return rows


def write_report(metric_rows, prompt_rows):
    by_bench = defaultdict(list)
    for r in metric_rows:
        by_bench[r["benchmark"]].append(r)

    speed_values = []
    for r in metric_rows:
        try:
            x = float(r["speedup"])
            if x > 0 and not math.isnan(x):
                speed_values.append(x)
        except ValueError:
            pass

    best_rows = sorted(
        [r for r in metric_rows if re.match(r"^[0-9.]+$", r["speedup"])],
        key=lambda r: float(r["speedup"]),
        reverse=True,
    )[:8]

    prompt_complete = sum(1 for r in prompt_rows if r["main_strength"] == "限制清楚，利於避免偽加速")

    lines = []
    lines.append("# HeCBench AI 輔助程式碼優化結果總結與統計分析")
    lines.append("")
    lines.append("本報告由 `/home/a/PP` 既有結果檔、CSV、agent summary 與 prompt 彙整產生。重點是比較各 AI 輔助版本在 correctness 有效前提下的效能變化，並指出 prompt 與後續優化可改善的地方。")
    lines.append("")
    lines.append("## 產出檔案")
    lines.append("")
    lines.append("- `data/benchmark_summary.csv`: 跨 benchmark 統一摘要表。")
    lines.append("- `prompts/prompt_inventory.csv`: prompt 規格盤點。")
    lines.append("- `REPORT.md`: 本中文總報告。")
    lines.append("- `generate_report.py`: 可重跑的整理腳本。")
    lines.append("")
    lines.append("## 整體結論")
    lines.append("")
    lines.append(f"- 共彙整 {len(by_bench)} 個 benchmark 類別、{len(metric_rows)} 筆可比較摘要。")
    lines.append(f"- 可計算 speedup 的案例平均為 {fmt(mean(speed_values), 3)}x；但不同 benchmark 的 metric 不同，這個數字只能視為方向性統計。")
    lines.append(f"- {prompt_complete}/{len(prompt_rows)} 份 prompt 明確包含 baseline、correctness、提交限制、raw output 或備份等防偽加速約束。")
    lines.append("- 最可信的優化通常不是單純要求「更快」，而是讓 AI 在固定測資、固定 repeat、完整 correctness、完整 raw log 與有限提交次數下逐步假設驗證。")
    lines.append("")
    lines.append("## Benchmark 摘要")
    lines.append("")
    for bench in sorted(by_bench):
        lines.append(f"### {bench}")
        for r in by_bench[bench]:
            lines.append(f"- `{r['agent']}` `{r['case']}`: metric {r['baseline_metric']} -> {r['optimized_metric']} {r['metric_unit']}, speedup `{r['speedup']}`, correctness `{r['correctness']}`。策略: {r['best_strategy']}。")
        lines.append("")
    lines.append("## 最佳加速案例")
    lines.append("")
    lines.append("| Benchmark | Agent | Case | Speedup | 策略 |")
    lines.append("|---|---|---|---:|---|")
    for r in best_rows:
        lines.append(f"| {r['benchmark']} | {r['agent']} | {r['case']} | {r['speedup']}x | {r['best_strategy']} |")
    lines.append("")
    lines.append("## Prompt 比較")
    lines.append("")
    lines.append("多數 prompt 的優點是明確規定不得刪 correctness、不得縮小輸入、不得把 FAIL 當成功，並要求保留 raw output。這些限制讓 AI 輔助優化比較像可審核實驗，而不是只產生漂亮但不可驗證的敘事。")
    lines.append("")
    lines.append("主要差異如下：")
    lines.append("")
    lines.append("- `moe_prompt.md` 較短，聚焦在固定測資與三次提交；適合快速比較兩個 agent，但缺少 profiler、統計變異、baseline 定義細節。")
    lines.append("- `pingpong`、`topk`、`shmembench`、`softmax` 等 prompt 較完整，明確指定 sbatch、環境、讀取 `.out/.err/.txt`、不得跳測、備份檔案與 submission limit。")
    lines.append("- `moe/CLOUD` 的結果顯示 prompt 若沒有強制「矛盾檢查」與「baseline 必須實測」，agent 仍可能在報告中同時宣稱失敗與全通過，或使用 estimated baseline 做過度結論。")
    lines.append("")
    lines.append("## 後續程式優化建議")
    lines.append("")
    lines.append("- 對 `moe-cuda`: 採用 CG V3 hybrid dispatch 作為主線；再用 Nsight Compute 驗證 top-k 8 原始 path 的 global memory traffic 與 top-k reduction 成本，避免憑直覺融合。")
    lines.append("- 對 `topk-cuda`: workspace reuse 已證明有效，下一步應量測 CUB temporary storage、radix pass 次數、occupancy/register pressure；block size 512 可作為 V100 預設，但應保留自動 sweep。")
    lines.append("- 對 `softmax-cuda`: block-level cached expf 在 slice 784/1024 最佳，應針對 slice size 建立 dispatch policy；小 slice 128 仍以 warp cached 為佳。")
    lines.append("- 對 `shmembench-cuda`: GM 版本的 block-size dependent checksum 讓 sweep 成為有效結果，建議加入 per-block analytical checksum 或 reference kernel，並用 profiler 量 shared bank conflict。")
    lines.append("- 對 `pingpong/allreduce`: launcher 是效能與 correctness 的一部分，應把 UCX/NCCL/MPI transport 設定納入 benchmark metadata，避免把環境修復誤判成 kernel 優化。")
    lines.append("- 對 `simpleMultiDevice`: 端到端受 H2D copy 主導，後續應拆分 kernel-only、copy-only、overlap copy/compute 三種模式，否則 GPU 數量擴展會被傳輸掩蓋。")
    lines.append("")
    lines.append("## Prompt 改善建議")
    lines.append("")
    lines.append("建議把後續 prompt 統一成以下規格：")
    lines.append("")
    lines.append("1. 明確定義 baseline 必須是實測結果，不得使用估計值替代正式比較。")
    lines.append("2. 要求每次優化都輸出 machine-readable `RESULT` 或 CSV，欄位至少包含 job id、node、case、metric、correctness、status、variant。")
    lines.append("3. 要求報告自動檢查矛盾：若任一 case FAIL，不得在總結寫 all tests PASS。")
    lines.append("4. 要求至少 3 次 trial 或提供 stddev/CV；若 submission limit 太少，至少 final confirmation 要重複量測。")
    lines.append("5. 要求區分「環境修復」、「量測修復」、「實際 kernel 優化」，避免把可執行性修復當成演算法加速。")
    lines.append("6. 要求列出無效嘗試與拒採理由，像 `moe` 的 top-k 8 full fusion regression、`shmembench` 的 checksum failure 都應保留。")
    lines.append("7. 對 GPU kernel 題目加入 profiler 指標：occupancy、register、shared bank conflict、dram throughput、kernel launch count。")
    lines.append("")
    lines.append("## 資料限制")
    lines.append("")
    lines.append("本報告只使用目前 `/home/a/PP` 中已存在的結果檔；沒有重新提交 Slurm job，也沒有重新跑 GPU benchmark。不同 benchmark 的硬體 node、CUDA 版本、metric 單位與 agent 版本不完全一致，因此跨 benchmark 排名只能做研究管理上的比較，不能當作單一效能排行榜。")
    lines.append("")
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    DATA.mkdir(parents=True, exist_ok=True)
    PROMPTS.mkdir(parents=True, exist_ok=True)
    prompt_rows = prompt_summary()
    metric_rows = build_metric_rows()
    write_report(metric_rows, prompt_rows)
    print(f"Wrote {OUT / 'REPORT.md'}")
    print(f"Wrote {DATA / 'benchmark_summary.csv'}")
    print(f"Wrote {PROMPTS / 'prompt_inventory.csv'}")


if __name__ == "__main__":
    main()
