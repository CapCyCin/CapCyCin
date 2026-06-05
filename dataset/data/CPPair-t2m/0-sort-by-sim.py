import pandas as pd
import json
import os
from pathlib import Path
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ----------------------------------------
# Utils
# ----------------------------------------

def load_caption_json(path):
    with open(path, "r") as f:
        return pd.DataFrame(json.load(f))


def detect_caption_column(df, method, seed):
    col = f"{method}_seed{seed}_caption"
    if col in df.columns:
        return col
    for c in df.columns:
        if c.endswith("_caption"):
            return c
    raise ValueError(f"No caption column for {method} seed{seed}")


def load_multi_seed_similarity(similarity_paths):
    """
    similarity_paths: {seed: path}
    return: merged_sim[ytid][method_seed] = score
    """
    merged = {}

    for seed, path in similarity_paths.items():
        with open(path, "r") as f:
            sim = json.load(f)

        for ytid, scores in sim.items():
            if ytid not in merged:
                merged[ytid] = {}
            for method, score in scores.items():
                key = f"{method}_seed{seed}"
                merged[ytid][key] = float(score)

    print(f"[INFO] Loaded multi-seed similarity | #ytid={len(merged)}")
    return merged


def build_audio_dict(audio_base_dirs, seeds):
    """
    return audio_dict[method_seed][ytid] = wav_path
    """
    audio_dict = {}

    for seed in seeds:
        for method, base_dir in audio_base_dirs.items():
            dir_path = os.path.join(base_dir, str(seed))
            if not os.path.exists(dir_path):
                continue

            key = f"{method}_seed{seed}"
            audio_dict[key] = {
                f[:-4]: os.path.join(dir_path, f)
                for f in os.listdir(dir_path)
                if f.endswith(".wav")
            }

    return audio_dict


def build_recon_caps(regen_paths, seeds):
    """
    return recon_caps[method_seed][ytid] = caption
    """
    recon_caps = {}

    for seed in seeds:
        for method, path_tpl in regen_paths.items():
            path = path_tpl.format(seed=seed)
            if not os.path.exists(path):
                continue

            df = load_caption_json(path)
            cap_col = detect_caption_column(df, method, seed)
            key = f"{method}_seed{seed}"

            recon_caps[key] = dict(
                zip(df["audio_path"], df[cap_col])
            )

    return recon_caps


# ----------------------------------------
# Main builder
# ----------------------------------------

def build_dpo_dataset_t2m_multiseed(
    similarity_paths,
    audio_base_dirs,
    regen_paths,
    seeds,
    mc_csv_path,
    output_path
):
    sim_dict = load_multi_seed_similarity(similarity_paths)

    mc_df = pd.read_csv(mc_csv_path)
    ytid2caption = dict(zip(mc_df["track_id"], mc_df["caption_writing"])) #msd format

    audio_dict = build_audio_dict(audio_base_dirs, seeds)
    recon_caps = build_recon_caps(regen_paths, seeds)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for ytid, score_dict in tqdm(sim_dict.items()):

            if ytid not in ytid2caption:
                continue

            candidates = []
            for key, score in score_dict.items():
                if ytid not in audio_dict.get(key, {}):
                    continue
                if ytid not in recon_caps.get(key, {}):
                    continue

                candidates.append({
                    "path": audio_dict[key][ytid],
                    "score": score,
                    "method": key,  # method_seed
                    "recon_caption": recon_caps[key][ytid],
                })

            if len(candidates) < 2:
                continue

            candidates.sort(key=lambda x: x["score"], reverse=True)

            item = {
                "caption": ytid2caption[ytid],
                "generations": [c["path"] for c in candidates],
                "reconstructions": [c["recon_caption"] for c in candidates],
                "scores": [c["score"] for c in candidates],
                "methods": [c["method"] for c in candidates],
                "ytid": ytid,
            }

            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"[DONE] Multi-seed T2M dataset written to:\n{output_path}")


# ----------------------------------------
# Run
# ----------------------------------------

if __name__ == "__main__":

    seeds = [0, 1, 2]

    #similarity_paths = {
    #    seed: f"data/msd_clap_{seed}.json"
    #    for seed in seeds
    #}
    similarity_paths = {
        seed: f"data/msd_sbert_{seed}.json"
        for seed in seeds
    }

    audio_base_dirs = {
        "audioldm2": "data/audio/texttomusic/audioldm2",
        "musicgen-medium": "data/audio/texttomusic/musicgen-medium",
        "musicgen-small": "data/audio/texttomusic/musicgen-small",
        "riffusion": "data/audio/texttomusic/riffusion",
    }

    regen_paths = {
        "audioldm2": "data/t2m_recon/audioldm2_seed{seed}_msd.json",
        "musicgen-medium": "data/t2m_recon/musicgen-medium_seed{seed}_msd.json",
        "musicgen-small": "data/t2m_recon/musicgen-small_seed{seed}_msd.json",
        "riffusion": "data/t2m_recon/riffusion_seed{seed}_msd.json",
    }

    mc_csv_path = "data/lpmusiccaps-msd-8k-trunc.csv"

    output_path = str(PROJECT_ROOT / "generate_dataset/CPPair-t2m/datasets/msd-tnt-sbert.jsonl")

    build_dpo_dataset_t2m_multiseed(
        similarity_paths,
        audio_base_dirs,
        regen_paths,
        seeds,
        mc_csv_path,
        output_path
    )
