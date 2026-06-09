#!/usr/bin/env python3
import csv
import os
import re
import sys


def parse_kv(line):
    fields = {}
    for part in line.strip().split(",")[1:]:
        if "=" in part:
            key, value = part.split("=", 1)
            fields[key] = value
    return fields


def main():
    if len(sys.argv) != 5:
        print("usage: parse_pingpong_results.py <input> <output.csv> <job_id> <node>", file=sys.stderr)
        return 2

    input_path, output_path, job_id, node = sys.argv[1:5]
    with open(input_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    correctness = "FAIL" if any(re.search(r"ERROR|Waiving|failed", line) for line in lines) else "PASS"
    rows = []
    for line in lines:
        if not line.startswith("RESULT,"):
            continue
        fields = parse_kv(line)
        rows.append({
            "job_id": job_id,
            "node": node,
            "backend": fields.get("backend", ""),
            "size_bytes": fields.get("size_bytes", ""),
            "loop_count": fields.get("loop_count", ""),
            "trial": fields.get("trial", ""),
            "avg_time_s": fields.get("avg_time_s", fields.get("time_s", "")),
            "gbps": fields.get("gbps", ""),
            "correctness": correctness,
        })

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "job_id", "node", "backend", "size_bytes", "loop_count",
            "trial", "avg_time_s", "gbps", "correctness"
        ])
        writer.writeheader()
        writer.writerows(rows)

    return 0 if correctness == "PASS" and rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
