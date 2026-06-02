"""Generate captions on MusicCaps and save JSONL outputs."""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets import load_dataset

from dpo.eval_common import load_dpo_model, run_over_items
from dpo.utils import CHECKPOINT_DIR, OUTPUT_DIR

CSV_PATH = REPO_ROOT / "musiccaps/musiccaps_eval.csv"
AUDIO_DIR = REPO_ROOT / "musiccaps/musiccaps_eval"
OUTPUT_PATH = OUTPUT_DIR / "musiccaps-dpo.jsonl"
DPO_CKPT = CHECKPOINT_DIR / "comp-dpo/best-476.pt"


def main():
    processor, model = load_dpo_model(DPO_CKPT)
    dataset = load_dataset("csv", data_files=str(CSV_PATH), split="train")
    items = []
    for item in dataset:
        audio_path = os.path.join(AUDIO_DIR, f"{item['ytid']}.wav")
        items.append(
            {
                "audio_path": audio_path,
                "ground_truth": item.get("caption", ""),
                "metadata": {"ytid": item["ytid"]},
            }
        )
    run_over_items(processor, model, items, OUTPUT_PATH)


if __name__ == "__main__":
    main()
