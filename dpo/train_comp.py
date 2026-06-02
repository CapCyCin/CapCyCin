"""DPO training with cycle-consistency preference data (comp variant)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration, get_cosine_schedule_with_warmup

from dpo.train import collate_fn, compute_loss, evaluate
from dpo.utils import CHECKPOINT_DIR, DATA_ROOT, MODEL_ID, finish_wandb, init_wandb, log_metrics, set_seed

set_seed(42)

BATCH_SIZE = 4
GRAD_ACCUM = 8
EPOCHS = 1
LR = 5e-6
EVAL_INTERVAL = 500
SAVE_DIR = CHECKPOINT_DIR / "comp-dpo"
TRAIN_JSONL = DATA_ROOT / "base/m2t/threshold/train/CPpair-ld-mnt-muqmulan-cleaned.jsonl"
VAL_JSONL = DATA_ROOT / "base/m2t/threshold/val/CPpair-ld-mnt-muqmulan-cleaned.jsonl"

device = torch.device("cuda")


def main():
    wandb_run = init_wandb("music-captioning-dpo", "comp-dpo")
    os.makedirs(SAVE_DIR, exist_ok=True)

    policy_model = Qwen2AudioForConditionalGeneration.from_pretrained(MODEL_ID, trust_remote_code=True).to(device)
    ref_model = Qwen2AudioForConditionalGeneration.from_pretrained(MODEL_ID, trust_remote_code=True).to(device)
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False

    AutoProcessor.from_pretrained(MODEL_ID)

    train = load_dataset("json", data_files=str(TRAIN_JSONL), split="train")
    val = load_dataset("json", data_files=str(VAL_JSONL), split="train")
    train_loader = DataLoader(train, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=4)
    val_loader = DataLoader(val, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=4)
    print("Data loaded.")

    optimizer = torch.optim.AdamW(policy_model.parameters(), lr=LR)
    total_steps = (len(train_loader) // GRAD_ACCUM) * EPOCHS
    scheduler = get_cosine_schedule_with_warmup(optimizer, int(0.1 * total_steps), total_steps)

    best_acc = 0.0
    patience = 3
    patience_counter = 0
    global_step = 0

    for epoch in range(EPOCHS):
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}")
        for step, batch in enumerate(pbar):
            loss, margin = compute_loss(batch, policy_model, ref_model)
            loss = loss / GRAD_ACCUM
            loss.backward()

            if (step + 1) % GRAD_ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(policy_model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                margin_detached = margin.detach()
                acc = (margin_detached > 0).float().mean().item()
                true_loss = loss.item() * GRAD_ACCUM
                pbar.set_postfix({"loss": f"{true_loss:.4f}", "acc": f"{acc:.3f}"})
                log_metrics(
                    wandb_run,
                    {
                        "loss": true_loss,
                        "margin": margin_detached.mean().item(),
                        "margin_std": margin_detached.std().item(),
                        "acc": acc,
                    },
                )

                if global_step % EVAL_INTERVAL == 0:
                    val_acc = evaluate(policy_model, ref_model, val_loader)
                    print(f"[VAL] acc: {val_acc:.4f}")
                    if val_acc > best_acc:
                        best_acc = val_acc
                        patience_counter = 0
                        torch.save(policy_model.state_dict(), SAVE_DIR / f"best-{global_step}.pt")
                    else:
                        patience_counter += 1
                        if patience_counter >= patience:
                            print("Early stopping triggered.")
                            break

        if patience_counter >= patience:
            break

    val_acc = evaluate(policy_model, ref_model, val_loader)
    if val_acc > best_acc:
        torch.save(policy_model.state_dict(), SAVE_DIR / f"best-{global_step}.pt")

    finish_wandb(wandb_run)


if __name__ == "__main__":
    main()
