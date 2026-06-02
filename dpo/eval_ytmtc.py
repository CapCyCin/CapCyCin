"""Generate captions on YouTube8B-MusicTextClips and save JSONL outputs."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dpo.eval_common import load_dpo_model, run_over_items
from dpo.utils import CHECKPOINT_DIR, OUTPUT_DIR

JSONL_PATH = REPO_ROOT / "YouTube8B-MusicTextClips/test.jsonl"
AUDIO_DIR = REPO_ROOT / "YouTube8B-MusicTextClips/test_audio"
OUTPUT_PATH = OUTPUT_DIR / "ytmtc-dpo.jsonl"
DPO_CKPT = CHECKPOINT_DIR / "qwen-dpo/best-500.pt"


def main():
    processor, model = load_dpo_model(DPO_CKPT)
    items = []
    with open(JSONL_PATH, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            audio_path = os.path.join(AUDIO_DIR, f"{item['video_id']}.wav")
            items.append(
                {
                    "audio_path": audio_path,
                    "ground_truth": item.get("caption", ""),
                    "metadata": {"video_id": item["video_id"]},
                }
            )
    run_over_items(processor, model, items, OUTPUT_PATH)


if __name__ == "__main__":
    main()
