#!/usr/bin/env python3
import csv
import os
import sys

FIELDS = [
    "job_id", "node", "test_name", "repeat", "block_size", "grid_size",
    "shared_bytes", "avg_us", "min_us", "max_us", "bandwidth_GBps",
    "correctness", "status", "notes"
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
        print("usage: parse_shmembench_results.py <input> <output.csv> <job_id> <node>", file=sys.stderr)
        return 2

    input_path, output_path, job_id, node = sys.argv[1:5]
    rows = []

    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("RESULT,"):
                fields = parse_kv(line)
                block_size = int(fields.get("block", "256"))
                size = 1024 * 1024
                grid_size = (size // block_size) // 4
                rows.append({
                    "job_id": job_id,
                    "node": node,
                    "test_name": fields.get("test", "shmembench"),
                    "repeat": fields.get("repeat", "1000"),
                    "block_size": str(block_size),
                    "grid_size": str(grid_size),
                    "shared_bytes": fields.get("shared_bytes", ""),
                    "avg_us": fields.get("avg_us", ""),
                    "min_us": fields.get("min_us", ""),
                    "max_us": fields.get("max_us", ""),
                    "bandwidth_GBps": fields.get("bandwidth_GBps", ""),
                    "correctness": fields.get("correctness", ""),
                    "status": fields.get("status", ""),
                    "notes": "dynamic_shared_memory_volta_opt"
                })

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    return 0 if rows and all(row["status"] == "PASS" for row in rows) else 1

if __name__ == "__main__":
    raise SystemExit(main())
