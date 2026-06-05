import pandas as pd
import json
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ----------------------------------------
# 1. Load similarity JSON
# ----------------------------------------
def load_similarity_json(path):
    with open(path, "r") as f:
        similarity_dict = json.load(f)

    rows = []
    for ytid, scores in similarity_dict.items():
        row = {"ytid": ytid}
        row.update(scores)
        rows.append(row)

    df = pd.DataFrame(rows)
    print(f"[INFO] Loaded similarity keys: {df.columns.tolist()}")
    return df


# ----------------------------------------
# 2. Load caption JSON
# ----------------------------------------
def load_caption_json(path):
    with open(path, "r") as f:
        data = json.load(f)
    return pd.DataFrame(data)


def detect_caption_column(df):
    """Auto-detect the caption column."""
    for col in df.columns:
        if col.endswith("caption"):
            return col
    raise ValueError(f"No caption column found in {df.columns}")


# ----------------------------------------
# 3. Main DPO Dataset Builder
# ----------------------------------------
def build_dpo_dataset(similarity_path, caption_paths_dict, mc_csv_path, output_path):

    print("[INFO] Loading similarity file...")
    df = load_similarity_json(similarity_path)

    methods = list(caption_paths_dict.keys())

    # -------------------------
    # Load captions
    # -------------------------
    caption_dict = {}

    for method, path in caption_paths_dict.items():
        if not os.path.exists(path):
            print(f"[WARNING] Caption file not found → {path}")
            continue

        cap_df = load_caption_json(path)
        cap_col = detect_caption_column(cap_df)

        print(f"[INFO] {method}: caption column detected → {cap_col}")

        caption_dict[method] = dict(zip(cap_df["audio_path"], cap_df[cap_col]))

    # -------------------------
    # Load musiccaps original captions
    # -------------------------
    mc_df = pd.read_csv(mc_csv_path)
    #ytid2caption = dict(zip(mc_df["ytid"], mc_df["caption"]))

    # -------------------------
    # Create output file
    # -------------------------
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(f"[INFO] Writing dataset → {output_path}")
    with open(output_path, "w", encoding="utf-8") as f:

        for _, row in df.iterrows():
            ytid = row["ytid"]

            music_path = f"data/audio/laion-disco-10s/{ytid}.wav"

            #if ytid not in ytid2caption:
            #    print(f"[WARNING] original caption not found → {ytid}")
            #    continue

            #original_caption = ytid2caption[ytid]

            # ---------------------------------
            # Build candidate generations
            # ---------------------------------
            candidates = []
            for method in methods:
                if method not in df.columns:
                    #print("method not in df.columns")
                    continue
                if method not in caption_dict:
                    #print("method not in caption_dict")
                    continue
                if ytid not in caption_dict[method]:
                    #print("ytid not in caption_dict[method]")
                    continue

                candidates.append({
                    "caption": caption_dict[method][ytid],
                    "score": row[method],
                    "method": method
                })

            if len(candidates) == 0:
                continue

            candidates = sorted(candidates, key=lambda x: x["score"], reverse=True)

            # ---------------------------------
            # Construct json line
            # ---------------------------------
            item = {
                "music": music_path,
                #"caption": original_caption,
                "generations": [c["caption"] for c in candidates],
                "reconstructions": [
                    f"data/audio/m2t_recon"
                    f"/{c['method']}/musicgen-small/0/{ytid}.wav"
                    for c in candidates
                ],
                "scores": [float(c["score"]) for c in candidates],
                "methods": [c["method"] for c in candidates],
                "ytid": ytid
            }

            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print("[INFO] Dataset creation completed.")


if __name__ == "__main__":

    similarity_path = "data/laion-disco_mulan.json"
    #similarity_path = "data/laion-disco_clap.json"

    caption_paths_dict = {
        "qwen": f"data/captioning_data/qwen_laion-disco-10s.json",
        "qwen2": f"data/captioning_data/qwen2_laion-disco-10s.json",
        "qwen2.5-7B": f"data/captioning_data/qwen-omni-7b_laion-disco-10s.json", 
        "qwen2.5-3B": f"data/captioning_data/qwen-omni-3b_laion-disco-10s.json",
        "salmonn": f"data/captioning_data/salmonn_laion-disco-10s.json",
        "lpmc": f"data/captioning_data/lpmc_laion-disco-10s.json"
    }

    mc_csv_path = "data/laion-disco-10s-ytid.csv"

    output_path = str(PROJECT_ROOT / "generate_dataset/CPPair-m2t/ld-mnm-mulan.jsonl")
    #output_path = "generate_dataset/CPPair-m2t/ld-mnm-clap.jsonl"



    build_dpo_dataset(similarity_path, caption_paths_dict, mc_csv_path, output_path)
