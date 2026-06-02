"""Full fine-tuning DPO training for Qwen2-Audio."""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import librosa
import torch
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration, get_cosine_schedule_with_warmup

from dpo.utils import (
    CHECKPOINT_DIR,
    DATA_ROOT,
    MODEL_ID,
    PROMPT,
    finish_wandb,
    init_wandb,
    log_metrics,
    set_seed,
)

set_seed(42)

BATCH_SIZE = 4
GRAD_ACCUM = 8
EPOCHS = 1
LR = 5e-6
BETA = 0.5
EVAL_INTERVAL = 500
SAVE_DIR = CHECKPOINT_DIR / "qwen-dpo"
TRAIN_JSONL = DATA_ROOT / "CPPair-m2t/dpo/train/CPpair-ld-mnm-muqmulan-cleaned.jsonl"
VAL_JSONL = DATA_ROOT / "CPPair-m2t/dpo/val/CPpair-ld-mnm-muqmulan-cleaned.jsonl"

device = torch.device("cuda")
processor = AutoProcessor.from_pretrained(MODEL_ID)
processor.tokenizer.padding_side = "right"


def get_logps(model, input_ids, attention_mask, input_features, feature_attention_mask, labels):
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        input_features=input_features,
        feature_attention_mask=feature_attention_mask,
        return_dict=True,
    )
    logits = outputs.logits[:, :-1, :]
    labels_shift = labels[:, 1:]
    log_probs = F.log_softmax(logits, dim=-1)
    token_logps = torch.gather(log_probs, -1, input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)
    loss_mask = (labels_shift != -100).float()
    token_logps = token_logps * loss_mask
    seq_logp = token_logps.sum(dim=-1)
    lengths = loss_mask.sum(dim=-1).clamp(min=1)
    return seq_logp / lengths


def compute_loss(batch, policy_model, ref_model, beta=BETA):
    def run(model, ids, mask, feats, feat_mask, labels):
        return get_logps(
            model,
            ids.to(device),
            mask.to(device),
            feats.to(device),
            feat_mask.to(device),
            labels.to(device),
        )

    logp_c = run(
        policy_model,
        batch["c_ids"],
        batch["c_mask"],
        batch["c_features"],
        batch["c_feature_mask"],
        batch["c_labels"],
    )
    logp_r = run(
        policy_model,
        batch["r_ids"],
        batch["r_mask"],
        batch["r_features"],
        batch["r_feature_mask"],
        batch["r_labels"],
    )

    with torch.no_grad():
        logp_c_ref = run(
            ref_model,
            batch["c_ids"],
            batch["c_mask"],
            batch["c_features"],
            batch["c_feature_mask"],
            batch["c_labels"],
        )
        logp_r_ref = run(
            ref_model,
            batch["r_ids"],
            batch["r_mask"],
            batch["r_features"],
            batch["r_feature_mask"],
            batch["r_labels"],
        )

    margin = (logp_c - logp_r) - (logp_c_ref - logp_r_ref)
    loss = -F.logsigmoid(beta * margin).mean()
    return loss, margin


@torch.no_grad()
def evaluate(policy_model, ref_model, dataloader):
    policy_model.eval()
    ref_model.eval()
    total, correct = 0, 0

    for batch in dataloader:
        logp_c = get_logps(
            policy_model,
            batch["c_ids"].to(device),
            batch["c_mask"].to(device),
            batch["c_features"].to(device),
            batch["c_feature_mask"].to(device),
            batch["c_labels"].to(device),
        )
        logp_r = get_logps(
            policy_model,
            batch["r_ids"].to(device),
            batch["r_mask"].to(device),
            batch["r_features"].to(device),
            batch["r_feature_mask"].to(device),
            batch["r_labels"].to(device),
        )
        logp_c_ref = get_logps(
            ref_model,
            batch["c_ids"].to(device),
            batch["c_mask"].to(device),
            batch["c_features"].to(device),
            batch["c_feature_mask"].to(device),
            batch["c_labels"].to(device),
        )
        logp_r_ref = get_logps(
            ref_model,
            batch["r_ids"].to(device),
            batch["r_mask"].to(device),
            batch["r_features"].to(device),
            batch["r_feature_mask"].to(device),
            batch["r_labels"].to(device),
        )
        margin = (logp_c - logp_r) - (logp_c_ref - logp_r_ref)
        correct += (margin > 0).sum().item()
        total += margin.size(0)

    policy_model.train()
    return correct / total if total > 0 else 0.0


def collate_fn(batch):
    chosen_texts, rejected_texts = [], []
    chosen_audios, rejected_audios = [], []
    sr = processor.feature_extractor.sampling_rate

    def load_audio(path):
        audio, _ = librosa.load(path, sr=processor.feature_extractor.sampling_rate)
        return audio

    def combined_texts(audio_path, response):
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": audio_path},
                    {"type": "text", "text": PROMPT},
                ],
            },
            {"role": "assistant", "content": [{"type": "text", "text": response}]},
        ]
        return processor.apply_chat_template(conversation, add_generation_prompt=False, tokenize=False)

    for item in batch:
        audio = load_audio(item["music"])
        chosen_audios.append(audio)
        rejected_audios.append(audio)
        chosen_texts.append(combined_texts(item["music"], item["preferred"]))
        rejected_texts.append(combined_texts(item["music"], item["rejected"]))

    c_inputs = processor(
        text=list(chosen_texts),
        audio=list(chosen_audios),
        return_tensors="pt",
        padding=True,
        sampling_rate=sr,
    )
    r_inputs = processor(
        text=list(rejected_texts),
        audio=list(rejected_audios),
        return_tensors="pt",
        padding=True,
        sampling_rate=sr,
    )

    c_labels = c_inputs["input_ids"].clone()
    r_labels = r_inputs["input_ids"].clone()
    assistant_start_tokens = processor.tokenizer.encode("<|im_start|>assistant", add_special_tokens=False)

    def mask_labels(texts, inputs, labels):
        for i in range(len(texts)):
            input_ids_row = inputs["input_ids"][i]
            assistant_start_idx = -1
            for j in range(len(input_ids_row) - len(assistant_start_tokens) + 1):
                if torch.equal(
                    input_ids_row[j : j + len(assistant_start_tokens)],
                    torch.tensor(assistant_start_tokens, device=input_ids_row.device),
                ):
                    assistant_start_idx = j
                    break
            if assistant_start_idx != -1:
                labels[i, : assistant_start_idx + len(assistant_start_tokens)] = -100
            else:
                print("Warning: assistant start token not found; masking entire sequence.")
                labels[i, :] = -100
        return labels

    c_labels = mask_labels(chosen_texts, c_inputs, c_labels)
    r_labels = mask_labels(rejected_texts, r_inputs, r_labels)

    return {
        "c_ids": c_inputs.input_ids,
        "c_mask": c_inputs.attention_mask,
        "c_features": c_inputs.input_features,
        "c_feature_mask": c_inputs.feature_attention_mask,
        "c_labels": c_labels,
        "r_ids": r_inputs.input_ids,
        "r_mask": r_inputs.attention_mask,
        "r_features": r_inputs.input_features,
        "r_feature_mask": r_inputs.feature_attention_mask,
        "r_labels": r_labels,
    }


def main():
    wandb_run = init_wandb("music-captioning-dpo", "qwen-dpo")
    os.makedirs(SAVE_DIR, exist_ok=True)

    policy_model = Qwen2AudioForConditionalGeneration.from_pretrained(MODEL_ID, trust_remote_code=True).to(device)
    trainable = sum(p.numel() for p in policy_model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in policy_model.parameters())
    print(f"Trainable params: {trainable}")
    print(f"Total params: {total}")
    print(f"Trainable %: {trainable / total * 100:.2f}")

    ref_model = Qwen2AudioForConditionalGeneration.from_pretrained(MODEL_ID, trust_remote_code=True).to(device)
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False

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
                pbar.set_postfix(
                    {
                        "loss": f"{true_loss:.4f}",
                        "acc": f"{acc:.3f}",
                        "margin": f"{margin_detached.mean().item():.3f}",
                    }
                )
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
                        print(f"patience: {patience_counter}/{patience}")
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
