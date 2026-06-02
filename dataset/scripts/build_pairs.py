"""Convert ranked candidates into preferred/rejected DPO pairs."""

from __future__ import annotations

import itertools
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset.paths import DATA_ROOT

def normalize_music_path(path: str) -> str:
    marker = "/laion-disco-10s"
    if marker in path:
        return "." + path[path.index(marker) :]
    return path


def build_pairwise_dataset(input_jsonl, output_jsonl, thresh_similar=0.05, thresh_negative=0.7):
    kept = skipped_similar = skipped_negative = 0
    os.makedirs(os.path.dirname(output_jsonl), exist_ok=True)

    with open(input_jsonl, "r", encoding="utf-8") as fin, open(output_jsonl, "w", encoding="utf-8") as fout:
        for line in fin:
            data = json.loads(line)
            data["music"] = normalize_music_path(data["music"])
            music_path = data["music"]
            generations = data["generations"]
            scores = data["scores"]

            for i, j in itertools.combinations(range(len(generations)), 2):
                score_i, score_j = scores[i], scores[j]
                if abs(score_i - score_j) < thresh_similar:
                    skipped_similar += 1
                    continue
                if score_i > score_j:
                    pref, rej, score_pref = generations[i], generations[j], score_i
                else:
                    pref, rej, score_pref = generations[j], generations[i], score_j
                if score_pref < thresh_negative:
                    skipped_negative += 1
                    continue
                fout.write(
                    json.dumps(
                        {"music": music_path, "preferred": pref, "rejected": rej},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                kept += 1

    print("[INFO] Finished.")
    print(f"kept pairs       : {kept}")
    print(f"skipped_similar  : {skipped_similar}")
    print(f"skipped_negative : {skipped_negative}")


def main():
    datasets_dir = DATA_ROOT / "CPPair-m2t/datasets"
    input_jsonl = datasets_dir / "ld-mnm-clap.jsonl"
    output_jsonl = DATA_ROOT / "CPPair-m2t/dpo/CPpair-ld-mnm-clap.jsonl"
    build_pairwise_dataset(input_jsonl, output_jsonl, thresh_similar=0.2, thresh_negative=0.6)


if __name__ == "__main__":
    main()
