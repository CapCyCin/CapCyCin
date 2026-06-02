"""Split a JSONL file into train/val/test subsets."""

from __future__ import annotations

import os
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset.paths import DATA_ROOT


def split_jsonl(input_path, train_path, val_path, test_path, val_ratio=0.1, test_ratio=0.1, seed=42):
    assert val_ratio + test_ratio < 1.0, "val_ratio + test_ratio must be < 1"
    random.seed(seed)

    with open(input_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    print(f"Total examples: {len(lines)}")

    n = len(lines)
    n_test = int(n * test_ratio)
    n_val = int(n * val_ratio)
    test_lines = lines[:n_test]
    val_lines = lines[n_test : n_test + n_val]
    train_lines = lines[n_test + n_val :]

    for path in (train_path, val_path, test_path):
        os.makedirs(os.path.dirname(path), exist_ok=True)

    def write(path, data):
        with open(path, "w", encoding="utf-8") as f:
            for line in data:
                f.write(line.strip() + "\n")

    write(train_path, train_lines)
    write(val_path, val_lines)
    write(test_path, test_lines)
    print(f"Train: {len(train_lines)}")
    print(f"Val:   {len(val_lines)}")
    print(f"Test:  {len(test_lines)}")


def main():
    datasets_dir = DATA_ROOT / "CPPair-m2t/dpo/higher"
    for filename in os.listdir(datasets_dir):
        if not filename.endswith(".jsonl"):
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
