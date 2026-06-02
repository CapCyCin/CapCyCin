"""Split thresholded m2t pair files into train/val/test."""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset.paths import DATA_ROOT
from dataset.scripts.split import split_jsonl


def main():
    datasets_dir = DATA_ROOT / "base/m2t/threshold"
    for filename in os.listdir(datasets_dir):
        if not filename.endswith(".jsonl") or filename.endswith("-cleaned.jsonl"):
            continue
        input_jsonl = datasets_dir / filename
        split_jsonl(
            input_jsonl,
            datasets_dir / "train" / filename,
            datasets_dir / "val" / filename,
            datasets_dir / "test" / filename,
        )


if __name__ == "__main__":
    main()
