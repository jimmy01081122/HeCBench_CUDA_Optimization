#!/usr/bin/env python3
import csv
import os
import sys


FIELDS = [
    "job_id", "node", "test_name", "repeat", "block_size", "grid_size",
    "shared_bytes", "avg_us", "min_us", "max_us", "bandwidth_GBps",
    "correctness", "status", "notes",
]


def parse_kv(line):
    result = {}
    for part in line.strip().split(",")[1:]:
        if "=" in part:
            key, value = part.split("=", 1)
            result[key] = value
    return result


def main():
    if len(sys.argv) != 5:
        print("usage: parse_shmembench_results.py <input> <output.csv> <job_id> <node>", file=sys.stderr)
        return 2

    input_path, output_path, job_id, node = sys.argv[1:5]
    rows = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.startswith("RESULT,"):
                continue
            fields = parse_kv(line)
            rows.append({
                "job_id": job_id,
                "node": node,
                "test_name": fields.get("test", ""),
                "repeat": fields.get("repeat", ""),
                "block_size": fields.get("block", ""),
                "grid_size": fields.get("grid", ""),
                "shared_bytes": fields.get("shared_bytes", ""),
                "avg_us": fields.get("avg_us", ""),
                "min_us": fields.get("min_us", ""),
                "max_us": fields.get("max_us", ""),
                "bandwidth_GBps": fields.get("bandwidth_GBps", ""),
                "correctness": fields.get("correctness", ""),
                "status": fields.get("status", ""),
                "notes": "barriered float4 shared-memory swap microbenchmark",
            })

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    return 0 if rows and all(row["status"] == "PASS" for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
