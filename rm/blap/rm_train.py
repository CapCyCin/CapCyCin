#!/usr/bin/env python3
import os
import gc
import json
import math
import argparse
import itertools
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.utils.data import DataLoader
#from torch.utils.data.distributed import DistributedSampler
import wandb

from transformers import get_cosine_schedule_with_warmup
from accelerate import dispatch_model,infer_auto_device_map

from blap.model.BLAP2.BLAP2_Pretrain import BLAP2_Stage2



from contextlib import contextmanager

@contextmanager
def force_cpu_load():
    _torch_load = torch.load

    def cpu_load(*args, **kwargs):
        kwargs["map_location"] = "cpu"
        return _torch_load(*args, **kwargs)

    torch.load = cpu_load
    try:
        yield
    finally:
        torch.load = _torch_load


# =========================
# Config / helpers
# =========================
def model_device(model):
    return next(model.parameters()).device

AUDIO_EXTS = (".wav")

DATA_KEY_MAPPING = {
    # A2T: anchor=audio path, candidates=text
    'A2T': {'input_key': 'audio',  'preferred_key': 'preferred_text',  'rejected_key': 'rejected_text'},
    # T2A: anchor=text prompt, candidates=audio path
    'T2A': {'input_key': 'prompt', 'preferred_key': 'preferred_audio', 'rejected_key': 'rejected_audio'}
}

def set_seed(seed: int, deterministic: bool = True):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = bool(deterministic)
    torch.backends.cudnn.benchmark = not bool(deterministic)

def is_combo(model_type: str) -> bool:
    return "combo" in (model_type or "").lower()

def is_a2t(model_type: str) -> bool:
    return "a2t" in (model_type or "").lower()

def is_t2a(model_type: str) -> bool:
    return "t2a" in (model_type or "").lower()

def load_audio_mono(path: str, target_sr: int, target_len: int = 480000):
    if not isinstance(path, str) or not os.path.exists(path):
        return None

    wav, sr = torchaudio.load(path)  # (C, T)

    # mono
    if wav.dim() == 2 and wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    elif wav.dim() == 1:
        wav = wav.unsqueeze(0)

    # resample
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)

    wav = wav.squeeze(0)  # (T,)

    # Match target audio length.
    if wav.numel() > target_len:
        wav = wav[:target_len]              # truncate
    elif wav.numel() < target_len:
        pad = target_len - wav.numel()
        wav = F.pad(wav, (0, pad))           # zero-pad

    return wav #.cpu().numpy()

def move_to_device(d: Dict[str, Dict[str, torch.Tensor]], device: str):
    for sub in d.values():
        for k, v in sub.items():
            sub[k] = v.to(device)

# =========================
# Dataset
# =========================

def load_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
    
    
class Dataset(torch.utils.data.Dataset):
    """
    - a2t_data_path: directory for A2T data (train.json / valid.json / test.json)
    - t2a_data_path: directory for T2A data (train.json / valid.json / test.json)
    - each json item: {"input": <anchor(audio path or text)>, "generations": [...], "scores":[...]}
    """
    def __init__(self, a2t_data_path, t2a_data_path, split, model_type, threshold_similar, threshold_negative,max_samples=None):
        self.model_type = model_type

        if "Combo" in model_type:
            A2T_dataset = load_jsonl(a2t_data_path)
            T2A_dataset = load_jsonl(t2a_data_path)
            
            #threshold similar
            #threshold negative
            A2T_data = self.make_data(A2T_dataset, threshold_similar, threshold_negative, data_type='A2T')
            T2A_data = self.make_data(T2A_dataset, threshold_similar, threshold_negative, data_type='T2A')
            
            if max_samples is not None:
                A2T_data = A2T_data[:max_samples]
                T2A_data = T2A_data[:max_samples]
            
            self.data = combine_datasets(A2T_data, T2A_data) 
        else:
            data_path = a2t_data_path if 'A2T' in model_type else t2a_data_path
            dataset = load_jsonl(data_path)
            data_type = model_type.split('-')[-1]  # 'A2T' or 'T2A'
            self.input_key, self.preferred_key, self.rejected_key = DATA_KEY_MAPPING[data_type].values()
            self.data = self.make_data(dataset, threshold_similar, threshold_negative, data_type)

    def make_data(self, data, threshold_similar=None, threshold_negative=None, data_type='A2T'):
        """
        Version without threshold-based filtering.
        - each item already contains preferred / rejected fields.
        """
        pairs = []

        for item in data:
            music     = item.get("music", "")
            original  = item.get("original", "")
            preferred = item.get("preferred", "")
            rejected  = item.get("rejected", "")

            if not preferred or not rejected:
                continue

            # Convert fields to the expected data types.
            if data_type == 'A2T':
                pairs.append({
                    "audio": music,
                    "preferred_text": preferred.strip(),
                    "rejected_text":  rejected.strip(),
                })

            elif data_type == 'T2A':
                pairs.append({
                    "prompt": original,
                    "preferred_audio": preferred,
                    "rejected_audio":  rejected,
                })

        return pairs


    def __getitem__(self, index):
        item = self.data[index]
        if 'Combo' in self.model_type:
            # Combo mode returns merged A2T/T2A dicts without key collisions.
            return {
                "audio":            item.get('audio', ""),
                "preferred_text":   item.get('preferred_text', ""),
                "rejected_text":    item.get('rejected_text', ""),
                "prompt":           item.get('prompt', ""),
                "preferred_audio":  item.get('preferred_audio', ""),
                "rejected_audio":   item.get('rejected_audio', ""),
            }
        # Single mode (A2T or T2A): use keys configured in __init__.
        return {
            self.input_key:     item[self.input_key],
            self.preferred_key: item[self.preferred_key],
            self.rejected_key:  item[self.rejected_key],
        }
        
    def __len__(self): 
        return len(self.data)


def combine_datasets(list1, list2):
    if len(list1) < len(list2):
        list1, list2 = list2, list1
    rep2 = itertools.cycle(list2)
    # Use distinct keys for A2T and T2A to avoid collisions.
    return [{**d1, **next(rep2)} for d1 in list1]

# =========================
# BLAP-native DataCollator
# =========================
@dataclass
class DataCollator:
    def __init__(self, model_type='CycleReward-Combo', target_sr=48000):
        self.model_type = model_type
        self.target_sr = target_sr

    def _load_audio(self, paths: List[str]) -> Optional[torch.Tensor]:
        audios = []
        for p in paths:
            a = load_audio_mono(p, self.target_sr)
            if a is None:
                return None
            audios.append(torch.tensor(a, dtype=torch.float))
        return torch.stack(audios, dim=0)  # (B, T)

    def _process_combo(self, feats: List[Dict[str, Any]]) -> Dict[str, Any]:
        # A2T
        a2t_audio = self._load_audio([f["audio"] for f in feats])
        a2t_pref_text = [f["preferred_text"] for f in feats]
        a2t_rej_text  = [f["rejected_text"]  for f in feats]

        # T2A
        t2a_text = [f["prompt"] for f in feats]
        t2a_pref_audio = self._load_audio([f["preferred_audio"] for f in feats])
        t2a_rej_audio  = self._load_audio([f["rejected_audio"]  for f in feats])

        return {
            "text": {
                "audio": a2t_audio,
                "pref_text": a2t_pref_text,
                "rej_text":  a2t_rej_text,
            },
            "music": {
                "text": t2a_text,
                "pref_audio": t2a_pref_audio,
                "rej_audio":  t2a_rej_audio,
            }
        }

    def _process_a2t(self, feats):
        return {
            "text": {
                "audio": self._load_audio([f["audio"] for f in feats]),
                "pref_text": [f["preferred_text"] for f in feats],
                "rej_text":  [f["rejected_text"]  for f in feats],
            },
            "music": None
        }

    def _process_t2a(self, feats):
        return {
            "text": None,
            "music": {
                "text": [f["prompt"] for f in feats],
                "pref_audio": self._load_audio([f["preferred_audio"] for f in feats]),
                "rej_audio":  self._load_audio([f["rejected_audio"]  for f in feats]),
            }
        }

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        if "Combo" in self.model_type:
            return self._process_combo(features)
        elif "A2T" in self.model_type:
            return self._process_a2t(features)
        else:
            return self._process_t2a(features)

# ======
# loss
# =======

def loss_text(model, text_batch):
    if text_batch is None:
        return None

    dev = model_device(model)
    audios = text_batch["audio"].to(dev)
    pref = text_batch["pref_text"]
    rej  = text_batch["rej_text"]

    r_pref = model(audios, pref)
    r_rej  = model(audios, rej)

    return -torch.log(torch.sigmoid(r_pref - r_rej)).mean()


def loss_music(model, music_batch):
    if music_batch is None:
        return None

    dev = model_device(model)
    text = music_batch["text"]
    pref_audio = music_batch["pref_audio"].to(dev)
    rej_audio  = music_batch["rej_audio"].to(dev)

    r_pref = model(pref_audio, text)
    r_rej  = model(rej_audio,  text)

    return -torch.log(torch.sigmoid(r_pref - r_rej)).mean()

def combined_loss(model, batch, w_text=1.0, w_music=1.0):
    """
    Implements: L = L_text + λ L_img
    Expectation is handled by DataLoader sampling, NOT by dividing losses.
    """
    Lt = loss_text(model, batch["text"])    # text-to-image (Eq.6)
    Lm = loss_music(model, batch["music"])  # image-to-text (Eq.5)

    if Lt is None and Lm is None:
        return None, (None, None)

    total = 0.0
    if Lt is not None:
        total = total + w_text * Lt
    if Lm is not None:
        total = total + w_music * Lm

    return total, (
        float(Lt.item()) if Lt is not None else None,
        float(Lm.item()) if Lm is not None else None,
    )



# =========================
# model (Stage2 ckpt → RM)
# =========================
class CycleRewardBLAP(nn.Module):
    def __init__(
        self,
        blap_ckpt: str,
        blap_stage2_cfg: str,
        freeze_audio: bool = True,
        freeze_qformer_ratio: float = 0.7,
    ):
        super().__init__()

        # =================================================
        # 1. Load Stage2 checkpoint (OOM-safe).
        # =================================================
        self.blap = BLAP2_Stage2.from_checkpoint(
            blap_ckpt,
            blap_stage2_cfg,
            map_location="cpu"
        )
        # =================================================
        # 2. Fully freeze audio encoder (as in paper).
        # =================================================
        if freeze_audio:
            self.blap.audio_encoder.eval()
            for p in self.blap.audio_encoder.parameters():
                p.requires_grad = False

        # =================================================
        # 3. Q-Former partial freeze (70%)
        # =================================================
        qformer_layers = self.blap.qformer.bert.encoder.layer
        num_layers = len(qformer_layers)
        freeze_until = int(num_layers * freeze_qformer_ratio)

        for i, layer in enumerate(qformer_layers):
            trainable = i >= freeze_until
            for p in layer.parameters():
                p.requires_grad = trainable

        print(
            f"[Q-Former] freeze 0~{freeze_until-1}, "
            f"train {freeze_until}~{num_layers-1}"
        )

        # =================================================
        # 4. Reward head (CLS → scalar)
        # =================================================
        self.reward_head = nn.Linear(
            self.blap.qformer.config.hidden_size, 1
        )
        nn.init.normal_(self.reward_head.weight, std=0.02)
        nn.init.zeros_(self.reward_head.bias)

    # =================================================
    # RM score
    # =================================================
    def score(self, audios, captions):
        """
        audios   : (B, T)
        captions : List[str]
        """
        device = audios.device

        # -------- Audio encoder (no grad!) --------
        with torch.no_grad():
            audio_feats = self.blap.audio_encoder(audios)

            audio_embeds = self.blap.ln_audio(
                F.avg_pool2d(
                    audio_feats[2]["fine_grained_embedding"],
                    (32, 1)
                )
            )

        audio_atts = torch.ones(
            audio_embeds.size()[:-1],
            dtype=torch.long,
            device=device,
        )

        # -------- Query tokens --------
        query_tokens = self.blap.query_tokens.expand(
            audio_embeds.size(0), -1, -1
        )

        # -------- Audio-conditioned Q-Former --------
        query_output = self.blap.qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=audio_embeds,
            encoder_attention_mask=audio_atts,
            return_dict=True,
        )

        # -------- Text tokenize --------
        text_tokens = self.blap.tokenizer(
            captions,
            padding="max_length",
            truncation=True,
            max_length=self.blap.max_txt_len,
            return_tensors="pt",
        ).to(device)

        # -------- Text encoding --------
        text_output = self.blap.qformer.bert(
            input_ids=text_tokens.input_ids,
            attention_mask=text_tokens.attention_mask,
            past_key_values=query_output.past_key_values,
            return_dict=True,
        )

        # -------- CLS → reward --------
        cls = text_output.last_hidden_state[:, 0, :]
        reward = self.reward_head(cls).squeeze(-1)

        return reward

    def forward(self, audios, captions):
        return self.score(audios, captions)



@torch.no_grad()
def evaluate(model, dataloader, device="cuda"):
    model.eval()

    def _acc_text(tb):
        if tb is None:
            return None
        audios = tb["audio"].to(device)
        pref = tb["pref_text"]
        rej  = tb["rej_text"]

        rp = model(audios, pref)
        rr = model(audios, rej)
        return (rp > rr).float().mean().item()

    def _acc_music(mb):
        if mb is None:
            return None
        text = mb["text"]
        pref_audio = mb["pref_audio"].to(device)
        rej_audio  = mb["rej_audio"].to(device)

        rp = model(pref_audio, text)
        rr = model(rej_audio,  text)
        return (rp > rr).float().mean().item()

    a_texts, a_musics = [], []
    for batch in dataloader:
        if batch["text"] is not None:
            a_texts.append(_acc_text(batch["text"]))
        if batch["music"] is not None:
            a_musics.append(_acc_music(batch["music"]))

    acc_text  = sum(a_texts)/len(a_texts) if a_texts else None
    acc_music = sum(a_musics)/len(a_musics) if a_musics else None
    acc_combo = (
        0.5*(acc_text+acc_music)
        if acc_text is not None and acc_music is not None
        else acc_text or acc_music
    )

    return {"acc_text": acc_text, "acc_music": acc_music, "acc_combo": acc_combo}

# =========================
# Train
# =========================
def main():
    parser = argparse.ArgumentParser("CycleReward-BLAP (pairwise RM)")

    # ======================
    # Data paths
    # ======================
    parser.add_argument("--a2t_train_path", type=str, default=None)
    parser.add_argument("--t2a_train_path", type=str, default=None)
    parser.add_argument("--a2t_test_path",  type=str, default=None)
    parser.add_argument("--t2a_test_path",  type=str, default=None)
    parser.add_argument("--checkpoint_dir", type=str, default="rm-checkpoint")

    # ======================
    # Model
    # ======================
    parser.add_argument(
        "--model_type",
        type=str,
        default="CycleReward-Combo",
        choices=["CycleReward-Combo", "CycleReward-A2T", "CycleReward-T2A"],
    )
    parser.add_argument("--freeze_qformer", action="store_true")
    parser.add_argument("--no-freeze_qformer", dest="freeze_qformer", action="store_false")
    parser.set_defaults(freeze_qformer=True)
    # Layer-wise GPU split when model does not fit on one GPU.
    parser.add_argument(
        "--device_map",
        type=str,
        default=None,
        choices=["auto"],
        help="'auto' spreads layers across GPUs (model parallelism). Use when spectrogram OOM occurs.",
    )
    parser.add_argument(
        "--max_memory_per_gpu",
        type=str,
        default="10GiB",
        help="Max memory per GPU when device_map=auto (e.g. 10GiB on a 12GB GPU).",
    )

    # ======================
    # Training
    # ======================
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument("--w_text",  type=float, default=1.0)
    parser.add_argument("--w_music", type=float, default=1.0)

    # ======================
    # Misc
    # ======================
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deterministic", action="store_true")
    # ======================
    # Logging
    # ======================
    parser.add_argument("--project", type=str, default="cycleconsistency-rm")
    parser.add_argument("--run_name", type=str, default="blap-reward")
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--save_name", type=str, default="cycle_reward_blap")
    parser.add_argument("--id", type=str, default=None)

    args = parser.parse_args()
    set_seed(args.seed, args.deterministic)


    rank = 0

    # ======================
    # Dataset / Loader
    # ======================
    train_ds = Dataset(
        a2t_data_path=args.a2t_train_path,
        t2a_data_path=args.t2a_train_path,
        split="train",
        model_type=args.model_type,
        threshold_similar=None,
        threshold_negative=None,
        max_samples=160000  # Optional sample limit.
    )

    val_ds = Dataset(
        a2t_data_path=args.a2t_test_path,
        t2a_data_path=args.t2a_test_path,
        split="valid",
        model_type=args.model_type,
        threshold_similar=None,
        threshold_negative=None,
        max_samples=20000  # Optional sample limit.
    )

    collate = DataCollator(
        model_type=args.model_type,
        target_sr=48000, #blap
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate,
    )

    # ======================
    # Model
    # ======================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Avoid OOM: load on CPU and clear cache before moving to GPU.
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    model = CycleRewardBLAP(
        blap_ckpt="data/blap-ckpt/checkpoint.ckpt",
        blap_stage2_cfg="data/blap-ckpt/config.json",
        freeze_qformer_ratio=0.7
    )
    num_gpus = torch.cuda.device_count()

    # When spectrogram does not fit: layer-wise GPU split (model parallel only, no DDP/DataParallel).
    if num_gpus > 1 and args.device_map == "auto":
        max_memory = {i: args.max_memory_per_gpu for i in range(num_gpus)}
        device_map = infer_auto_device_map(
            model,
            max_memory=max_memory,
            no_split_module_classes=["BertLayer"]  # Prevent splitting Q-Former layers
        )

        
        model = dispatch_model(model, device_map)
        print(f"[Model Parallel] Layer split across {num_gpus} GPUs, max_memory={args.max_memory_per_gpu}")
    else:
        model = model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    # ======================
    # W&B
    # ======================
    if not args.no_wandb and rank == 0:
        wandb.init(
            project=args.project,
            name=args.run_name,
            resume="allow",
            id=args.id,
        )
        wandb.config.update(vars(args))

    # ======================
    # Train loop
    # ======================
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    save_path = os.path.join(args.checkpoint_dir, args.save_name)

    from tqdm import tqdm

    grad_accum_steps = max(1, 2048 // args.batch_size)
    total_steps = (
        len(train_loader) * args.epochs
    ) // grad_accum_steps
    pbar = tqdm(total=total_steps, desc="Training", dynamic_ncols=True)

    best = -1.0
    patience = 3
    counter = 0
    global_step = 0

    model.train()
    optimizer.zero_grad()

    for epoch in range(args.epochs):

        for step, batch in enumerate(train_loader):

            loss, (lt, lm) = combined_loss(
                model, batch,
                w_text=args.w_text,
                w_music=args.w_music,
            )

            if loss is None:
                continue

            # -------------------------
            # gradient accumulation
            # -------------------------
            loss_scaled = loss / grad_accum_steps
            loss_scaled.backward()

            if (step + 1) % grad_accum_steps == 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

                # ---- wandb (per optimizer step)
                if not args.no_wandb and rank == 0 and global_step % 50 == 0:
                    wandb.log(
                        {
                            "train/loss": loss.item(),  # Unscaled loss
                            "train/loss_text": lt,
                            "train/loss_music": lm,
                        },
                        step=global_step
                    )

                pbar.update(1)
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})


        # ---- eval
        train_metrics = evaluate(model, train_loader, device)
        val_metrics   = evaluate(model, val_loader, device)

        if rank == 0:
            print(
                f"[Epoch {epoch}] "
                f"train_acc={train_metrics['acc_combo']} "
                f"val_acc={val_metrics['acc_combo']}"
            )

            if not args.no_wandb:
                wandb.log(
                    {
                        **{f"train/{k}": v for k, v in train_metrics.items()},
                        **{f"val/{k}": v for k, v in val_metrics.items()},
                    },
                    step=global_step
                )

            score = val_metrics["acc_combo"]
            if score is not None and score > best:
                best = score
                counter = 0
                state_to_save = model.state_dict()
                torch.save(
                    {"state_dict": state_to_save, "best": best},
                    save_path,
                )
                print(f"Best updated: {best:.4f}")
            else:
                counter += 1
                if counter >= patience:
                    print("Early stopping.")
                    break

    pbar.close()
    if not args.no_wandb and rank == 0:
        wandb.finish()

if __name__ == "__main__":
    main()
