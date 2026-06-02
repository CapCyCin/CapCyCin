"""Build candidate caption lists ranked by cycle-consistency scores."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd

from dataset.paths import CAPTION_DATA, DATA_ROOT


def load_similarity_json(path):
    with open(path, "r", encoding="utf-8") as f:
        similarity_dict = json.load(f)
    rows = [{"ytid": ytid, **scores} for ytid, scores in similarity_dict.items()]
    df = pd.DataFrame(rows)
    print(f"[INFO] Loaded similarity keys: {df.columns.tolist()}")
    return df


def load_caption_json(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return pd.DataFrame(data)


def detect_caption_column(df):
    for col in df.columns:
        if col.endswith("caption"):
            return col
    raise ValueError(f"No caption column found in {df.columns}")


def build_dpo_dataset(similarity_path, caption_paths_dict, mc_csv_path, output_path):
    print("[INFO] Loading similarity file...")
    df = load_similarity_json(similarity_path)
    methods = list(caption_paths_dict.keys())
    caption_dict = {}

    for method, path in caption_paths_dict.items():
        if not os.path.exists(path):
            print(f"[WARNING] Caption file not found: {path}")
            continue
        cap_df = load_caption_json(path)
        cap_col = detect_caption_column(cap_df)
        print(f"[INFO] {method}: caption column detected -> {cap_col}")
        caption_dict[method] = dict(zip(cap_df["audio_path"], cap_df[cap_col]))

    pd.read_csv(mc_csv_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(f"[INFO] Writing dataset -> {output_path}")
    with open(output_path, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            ytid = row["ytid"]
            music_path = f"./laion-disco-10s/{ytid}.wav"
            candidates = []
            for method in methods:
                if method not in df.columns or method not in caption_dict:
                    continue
                if ytid not in caption_dict[method]:
                    continue
                candidates.append(
                    {
                        "caption": caption_dict[method][ytid],
                        "score": row[method],
                        "method": method,
                    }
                )
            if not candidates:
                continue
            candidates = sorted(candidates, key=lambda x: x["score"], reverse=True)
            item = {
                "music": str(music_path),
                "generations": [c["caption"] for c in candidates],
                "reconstructions": [
                    f"./dataset/recon/{c['method']}/musicgen-small/0/{ytid}.wav"
                    for c in candidates
                ],
                "scores": [float(c["score"]) for c in candidates],
                "methods": [c["method"] for c in candidates],
                "ytid": ytid,
            }
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print("[INFO] Dataset creation completed.")


def main():
    similarity_path = DATA_ROOT / "CPPair-m2t/datasets/laion-disco_mulan.json"
    caption_paths_dict = {
        "qwen": CAPTION_DATA / "qwen_laion-disco-10s.json",
        "qwen2": CAPTION_DATA / "qwen2_laion-disco-10s.json",
        "qwen2.5-7B": CAPTION_DATA / "qwen-omni-7b_laion-disco-10s.json",
        "qwen2.5-3B": CAPTION_DATA / "qwen-omni-3b_laion-disco-10s.json",
        "salmonn": CAPTION_DATA / "salmonn_laion-disco-10s.json",
        "lpmc": CAPTION_DATA / "lpmc_laion-disco-10s.json",
    }
    mc_csv_path = CAPTION_DATA / "laion-disco-10s-ytid.csv"
    output_path = DATA_ROOT / "CPPair-m2t/ld-mnm-muqmulan.jsonl"
    build_dpo_dataset(similarity_path, caption_paths_dict, mc_csv_path, output_path)


if __name__ == "__main__":
    main()
