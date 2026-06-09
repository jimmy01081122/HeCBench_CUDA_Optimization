#!/usr/bin/env python3
from pathlib import Path

root = Path(__file__).resolve().parents[1]
required = ["P1_prompt.md", "P2_prompt.md", "P3_prompt.md"]
missing = []
for bench_dir in sorted((root / "prompts").iterdir()):
    if not bench_dir.is_dir():
        continue
    for name in required:
        path = bench_dir / name
        if not path.exists() or path.stat().st_size == 0:
            missing.append(str(path.relative_to(root)))
if missing:
    print("Missing prompt files:")
    for item in missing:
        print(item)
    raise SystemExit(1)
print("All benchmark prompt files exist.")
