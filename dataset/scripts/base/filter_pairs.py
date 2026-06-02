"""Remove noisy preferred captions from base m2t pair files."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset.paths import DATA_ROOT

input_file = DATA_ROOT / "base/m2t/threshold/CPpair-ld-mnt-muqmulan.jsonl"
output_file = DATA_ROOT / "base/m2t/threshold/CPpair-ld-mnt-muqmulan-cleaned.jsonl"

total_count = deleted_count = 0
with open(input_file, "r", encoding="utf-8") as f_in, open(output_file, "w", encoding="utf-8") as f_out:
    for line in f_in:
        total_count += 1
        data = json.loads(line)
        preferred = data.get("preferred", "")
        if "' is a" in preferred or "Human:" in preferred or "<!DOCTYPE html><html>" in preferred:
            deleted_count += 1
            continue
        f_out.write(json.dumps(data, ensure_ascii=False) + "\n")

print(f"Filtered {deleted_count}/{total_count} rows.")
