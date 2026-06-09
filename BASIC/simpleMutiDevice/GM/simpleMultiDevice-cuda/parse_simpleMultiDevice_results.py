#!/usr/bin/env python3
import csv
import os
import sys


FIELDS = [
    "job_id", "node", "num_gpus", "repeat", "total_us", "h2d_us",
    "kernel_us", "d2h_us", "gpu_sum", "cpu_sum", "relative_diff", "status"
]


def parse_kv(line):
    values = {}
    for part in line.strip().split(",")[1:]:
        if "=" in part:
            key, value = part.split("=", 1)
            values[key] = value
    return values


def main():
    if len(sys.argv) != 5:
        print("usage: parse_simpleMultiDevice_results.py <input> <output.csv> <job_id> <node>", file=sys.stderr)
        return 2

    input_path, output_path, job_id, node = sys.argv[1:5]
    correctness = {}
    rows = []

    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("CORRECTNESS,"):
                fields = parse_kv(line)
                correctness[fields.get("num_gpus", "")] = fields
            elif line.startswith("RESULT,"):
                fields = parse_kv(line)
                corr = correctness.get(fields.get("num_gpus", ""), {})
                rows.append({
                    "job_id": job_id,
                    "node": node,
                    "num_gpus": fields.get("num_gpus", ""),
                    "repeat": fields.get("repeat", ""),
                    "total_us": fields.get("total_us", ""),
                    "h2d_us": fields.get("h2d_us", ""),
                    "kernel_us": fields.get("kernel_us", ""),
                    "d2h_us": fields.get("d2h_us", ""),
                    "gpu_sum": corr.get("gpu_sum", ""),
                    "cpu_sum": corr.get("cpu_sum", ""),
                    "relative_diff": fields.get("diff", corr.get("relative_diff", "")),
                    "status": fields.get("status", corr.get("status", "")),
                })

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    return 0 if rows and all(row["status"] == "PASS" for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
