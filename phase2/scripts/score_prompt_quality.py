#!/usr/bin/env python3
import re
from pathlib import Path

root = Path(__file__).resolve().parents[1]

def score(text):
    checks = [
        (10, bool(re.search(r"benchmark|Goal|expected metric", text, re.I))),
        (10, "Environment" in text and "V100" in text),
        (10, "baseline" in text.lower()),
        (15, "correctness" in text.lower() and ("FAIL" in text or "invalid" in text.lower())),
        (10, "raw output" in text.lower() or ".out" in text),
        (10, ("submission" in text.lower() or "submissions" in text.lower()) and ("limit" in text.lower() or "at most" in text.lower() or "最多" in text)),
        (10, "csv" in text.lower()),
        (10, "contradiction" in text.lower()),
        (10, "trial" in text.lower() or "stddev" in text.lower() or "variance" in text.lower()),
        (5, "profiler" in text.lower() or "nsight" in text.lower()),
    ]
    return sum(points for points, ok in checks if ok)

for path in sorted((root / "prompts").glob("*/*_prompt.md")):
    text = path.read_text(encoding="utf-8", errors="replace")
    print(f"{path.relative_to(root)},{score(text)}")
