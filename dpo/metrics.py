"""Caption evaluation metrics: BLEU, SPICE, CIDEr, FENSE, SBERT."""

from __future__ import annotations

import json
import os
from pathlib import Path

import evaluate
import numpy as np
from fense.evaluator import Evaluator
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.spice.spice import Spice
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm

os.environ.setdefault("TMPDIR", os.path.expanduser("~/tmp"))


def compute_bleu(preds, refs):
    bleu = evaluate.load("bleu")
    res = bleu.compute(predictions=preds, references=refs)
    return {
        "BLEU@1": res["precisions"][0],
        "BLEU@2": res["precisions"][1],
        "BLEU@3": res["precisions"][2],
        "BLEU@4": res["precisions"][3],
    }


def compute_spice_cider_spider(preds, refs):
    spice = Spice()
    cider = Cider()
    gts = {i: refs[i] for i in range(len(refs))}
    res = {i: [preds[i]] for i in range(len(preds))}
    spice_score, _ = spice.compute_score(gts, res)
    cider_score, _ = cider.compute_score(gts, res)
    return {
        "SPICE": spice_score,
        "CIDEr": cider_score,
        "SPIDEr": (cider_score + spice_score) / 2,
    }


def compute_fense(preds, refs):
    fense_evaluator = Evaluator(device="cuda")
    score = fense_evaluator.corpus_score(preds, refs)
    return {"FENSE": score}


def compute_sbert(preds, refs):
    model = SentenceTransformer("all-mpnet-base-v2")
    scores = []
    for pred, ref_list in tqdm(zip(preds, refs), total=len(preds)):
        ref = ref_list[0]
        emb_p = model.encode(pred, convert_to_tensor=True, batch_size=32)
        emb_r = model.encode(ref, convert_to_tensor=True, batch_size=32)
        scores.append(util.cos_sim(emb_p, emb_r).item())
    return {"SBERT": float(np.mean(scores))}


def evaluate_all(preds, refs):
    results = {}
    print("BLEU...")
    results.update(compute_bleu(preds, refs))
    print("SPICE / CIDEr / SPIDEr...")
    results.update(compute_spice_cider_spider(preds, refs))
    print("FENSE...")
    results.update(compute_fense(preds, refs))
    print("SBERT...")
    results.update(compute_sbert(preds, refs))
    return results


def load_jsonl(path, use_dpo_caption=True):
    preds, refs = [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            preds.append(data["dpo_caption"] if use_dpo_caption else data["base_caption"])
            refs.append(data["ground_truth"])
    return preds, refs


def main():
    repo_root = Path(__file__).resolve().parents[1]
    jsonl_path = repo_root / "dpo/outputs/inference/16-qwen/qwen-audio-ytmtc-dpo.jsonl"
    if not jsonl_path.exists():
        jsonl_path = repo_root / "dpo/outputs/ytmtc-dpo.jsonl"
    preds, refs = load_jsonl(jsonl_path, use_dpo_caption=True)
    results = evaluate_all(preds, refs)
    print("\n===== FINAL RESULTS =====")
    for key, value in results.items():
        print(f"{key}: {value:.4f}")


if __name__ == "__main__":
    main()
