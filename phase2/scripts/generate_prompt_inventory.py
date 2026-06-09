#!/usr/bin/env python3
import csv
import subprocess
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
generator = root.parent / "docs" / "generate_phase2_prompts.py"
if generator.exists():
    subprocess.check_call([sys.executable, str(generator)])
    print("Regenerated phase2 metadata.")
else:
    print("Generator not found; use existing metadata.")
