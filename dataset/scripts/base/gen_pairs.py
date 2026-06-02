"""Convert ranked m2t candidates into preferred/rejected pairs."""

from __future__ import annotations

import itertools
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset.scripts.build_pairs import build_pairwise_dataset
from dataset.paths import DATA_ROOT


def main():
    input_jsonl = DATA_ROOT / "base/m2t/ld-mnt-muqmulan.jsonl"
    output_jsonl = DATA_ROOT / "base/m2t/harder/CPpair-ld-mnt-muqmulan.jsonl"
    build_pairwise_dataset(input_jsonl, output_jsonl, thresh_similar=0.2, thresh_negative=0.6)


if __name__ == "__main__":
    main()
