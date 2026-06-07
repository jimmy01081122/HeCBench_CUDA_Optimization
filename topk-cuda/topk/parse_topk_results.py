#!/usr/bin/env python3
import csv
import sys


FIELDS = [
    "job_id",
    "node",
    "batch_size",
    "hidden_size",
    "topk",
    "repeat",
    "warmup",
    "avg_us",
    "cpu_chrono_us",
    "correctness",
    "status",
    "variant",
    "source",
]


def parse_result_line(line):
    row = {}
    for item in line.strip().split(",")[1:]:
        if "=" not in item:
            continue
        key, value = item.split("=", 1)
        row[key] = value
    return row


def parse_node(line):
    prefix = "hostname:"
    if not line.startswith(prefix):
        return ""
    return line[len(prefix):].strip()


def main():
    if len(sys.argv) not in (4, 5):
        print("Usage: parse_topk_results.py <input_txt> <output_csv> <job_id> [variant]", file=sys.stderr)
        return 2

    input_path, output_path, job_id = sys.argv[1:4]
    variant = sys.argv[4] if len(sys.argv) == 5 else "cuda_event_instrumented"
    node = ""
    rows = []

    with open(input_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("hostname:"):
                node = parse_node(line)
            elif line.startswith("RESULT,"):
                row = parse_result_line(line)
                row["job_id"] = job_id
                row["node"] = node
                row["variant"] = variant
                row["source"] = input_path
                rows.append(row)

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"Parsed {len(rows)} RESULT rows into {output_path}")
    return 0 if rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
