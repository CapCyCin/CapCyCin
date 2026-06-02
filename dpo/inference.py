"""Run validation accuracy on a saved DPO checkpoint."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration

from dpo.train import collate_fn, evaluate
from dpo.utils import CHECKPOINT_DIR, DATA_ROOT, MODEL_ID

BATCH_SIZE = 4
VAL_JSONL = DATA_ROOT / "CPPair-m2t/dpo/high/val/CPpair-ld-mnm-muqmulan-cleaned.jsonl"
DPO_CKPT = CHECKPOINT_DIR / "comp-dpo/best-2398.pt"


def main():
    AutoProcessor.from_pretrained(MODEL_ID)
    policy_model = Qwen2AudioForConditionalGeneration.from_pretrained(
        MODEL_ID, device_map="auto"
    )
    ref_model = Qwen2AudioForConditionalGeneration.from_pretrained(
        MODEL_ID, device_map="auto"
    ).eval()

    if DPO_CKPT.exists():
        state_dict = torch.load(DPO_CKPT, map_location="cpu")
        policy_model.load_state_dict(state_dict, strict=False)
    policy_model.eval()

    val = load_dataset("json", data_files=str(VAL_JSONL), split="train")
    val_loader = DataLoader(val, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=4)
    val_acc = evaluate(policy_model, ref_model, val_loader)
    print(f"Validation accuracy: {val_acc:.4f}")


if __name__ == "__main__":
    main()
