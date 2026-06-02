"""Remove noisy preferred captions from DPO pair files."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset.paths import DATA_ROOT

input_file = DATA_ROOT / "CPPair-m2t/dpo/higher/CPpair-ld-mnm-muqmulan.jsonl"
output_file = DATA_ROOT / "CPPair-m2t/dpo/higher/CPpair-ld-mnm-muqmulan-cleaned.jsonl"

total_count = deleted_count = 0
with open(input_file, "r", encoding="utf-8") as f_in, open(output_file, "w", encoding="utf-8") as f_out:
    for line in f_in:
        total_count += 1
        data = json.loads(line)
        preferred = data.get("preferred", "")
        if "' is a" in preferred or "Human:" in preferred:
            deleted_count += 1
            continue
        f_out.write(json.dumps(data, ensure_ascii=False) + "\n")

print("Filtering complete.")
print(f"Original rows : {total_count}")
print(f"Removed rows  : {deleted_count}")
print(f"Remaining rows: {total_count - deleted_count}")
print(f"Removal ratio : {(deleted_count / total_count) * 100:.2f}%")
