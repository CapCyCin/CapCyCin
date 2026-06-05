import json
import random
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

def split_jsonl(
    input_path,
    train_path,
    val_path,
    test_path,
    val_ratio=0.1,
    test_ratio=0.1,
    seed=42,
):
    assert val_ratio + test_ratio < 1.0, "val_ratio + test_ratio must be < 1"

    random.seed(seed)

    with open(input_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    print(f"Total records: {len(lines)}")

    n = len(lines)
    n_test = int(n * test_ratio)
    n_val  = int(n * val_ratio)

    test_lines = lines[:n_test]
    val_lines  = lines[n_test : n_test + n_val]
    train_lines = lines[n_test + n_val :]

    for p in [train_path, val_path, test_path]:
        os.makedirs(os.path.dirname(p), exist_ok=True)

    def _write(path, data):
        with open(path, "w", encoding="utf-8") as f:
            for l in data:
                f.write(l.strip() + "\n")

    _write(train_path, train_lines)
    _write(val_path,   val_lines)
    _write(test_path,  test_lines)

    print(f"Train records: {len(train_lines)}")
    print(f"Val records:   {len(val_lines)}")
    print(f"Test records:  {len(test_lines)}")


if __name__ == "__main__":
    datasets_dir = PROJECT_ROOT / "generate_dataset/CPPair-m2t/addition"

    for f in os.listdir(datasets_dir):
        if f.endswith(".jsonl"):
            input_jsonl = os.path.join(datasets_dir, f)

            train_jsonl = os.path.join(datasets_dir, "train", f)
            val_jsonl   = os.path.join(datasets_dir, "val", f)
            test_jsonl  = os.path.join(datasets_dir, "test", f)

            split_jsonl(
                input_jsonl,
                train_jsonl,
                val_jsonl,
                test_jsonl,
                val_ratio=0,
                test_ratio=0.1,
            )
