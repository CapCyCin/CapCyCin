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

import librosa

from muq import MuQMuLan

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


def load_audio_mono(path, target_sr, target_len=480000):
    wav, _ = librosa.load(path, sr=target_sr, mono=True)
    wav = torch.from_numpy(wav)

    if wav.numel() > target_len:
        wav = wav[:target_len]
    else:
        wav = F.pad(wav, (0, target_len - wav.numel()))

    return wav

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
# DataCollator
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
            audios.append(a.float())
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
        return None, None, None  # Return the expected number of values.

    dev = model_device(model)
    audios = text_batch["audio"].to(dev)
    pref = text_batch["pref_text"]
    rej  = text_batch["rej_text"]

    r_pref = model(audios, pref)
    r_rej  = model(audios, rej)

    loss = -torch.log(torch.sigmoid(r_pref - r_rej)).mean()
    
    # Return scalar values extracted from tensors.
    return loss, r_pref.mean().item(), r_rej.mean().item()


def loss_music(model, music_batch):
    if music_batch is None:
        return None, None, None

    dev = model_device(model)
    text = music_batch["text"]
    pref_audio = music_batch["pref_audio"].to(dev)
    rej_audio  = music_batch["rej_audio"].to(dev)

    r_pref = model(pref_audio, text)
    r_rej  = model(rej_audio,  text)

    loss = -torch.log(torch.sigmoid(r_pref - r_rej)).mean()
    return loss, r_pref.mean().item(), r_rej.mean().item()

def combined_loss(model, batch, w_text=1.0, w_music=1.0):
    # 1. Compute each loss and reward.
    # Suffix variables with _t / _m to avoid collisions.
    Lt, pref_t, rej_t = loss_text(model, batch["text"])
    Lm, pref_m, rej_m = loss_music(model, batch["music"])

    if Lt is None and Lm is None:
        return None, {}

    total = 0.0
    metrics = {}

    # 2. Aggregate text-to-audio metrics.
    if Lt is not None:
        total += w_text * Lt
        metrics.update({
            "loss_text": Lt.item(),
            "pref_text": pref_t,
            "rej_text": rej_t,
            "margin_text": pref_t - rej_t  # Track whether the margin is widening.
        })

    # 3. Aggregate audio-to-text metrics.
    if Lm is not None:
        total += w_music * Lm
        metrics.update({
            "loss_music": Lm.item(),
            "pref_music": pref_m,
            "rej_music": rej_m,
            "margin_music": pref_m - rej_m
        })

    return total, metrics


# ====

#RM 

'''
self.layers = nn.Sequential(
            
            nn.Linear(input_size, 1024),
            nn.GELU(),

            nn.Linear(1024, 128),
            nn.GELU(),
            
            nn.Linear(128, 64),
            nn.GELU(),
            
            nn.Linear(64, 16),
            nn.GELU(),

            nn.Linear(16, 1),
        )
'''

'''
nn.Linear(input_size, 512),
            nn.GELU(),

            nn.Linear(512, 64),
            nn.GELU(),
            
            nn.Linear(64, 8),
            nn.GELU(),
            
            nn.Linear(8, 1),
'''

# ==== 

class RewardMLP(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        
        self.layers = nn.Sequential(
            # 1. Combine normalized input features.
            nn.Linear(input_size, 1024),
            nn.GELU(),

            nn.Linear(1024, 128),
            nn.GELU(),
            
            nn.Linear(128, 64),
            nn.GELU(),
            
            nn.Linear(64, 16),
            nn.GELU(),

            nn.Linear(16, 1),
        )
        
        

        # Initialize like ImageReward.
        def init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        self.layers.apply(init_weights)

    def forward(self, x):
        return self.layers(x)

import torch
import torch.nn as nn

class MuLANReward(nn.Module):
    def __init__(self, ft_audio_ratio=0.0):
        super().__init__()

        # 1. Load MuQ-MuLAN backbone.
        self.mulan_wrapper = MuQMuLan.from_pretrained("OpenMuQ/MuQ-MuLan-large")
        
        # Underlying MuLanModel instance.
        mulan_model = self.mulan_wrapper.mulan_module

        # 2. Freeze all parameters first.
        for p in self.mulan_wrapper.parameters():
            p.requires_grad = False

        # 3. Enable audio/text projection (to latents) layers.
        # mulan_model.audio_to_latents and text_to_latents handle this.
        if hasattr(mulan_model, 'audio_to_latents'):
            for p in mulan_model.audio_to_latents.parameters():
                p.requires_grad = True
        
        if hasattr(mulan_model, 'text_to_latents'):
            for p in mulan_model.text_to_latents.parameters():
                p.requires_grad = True

        # 4. Optional audio encoder layer fine-tuning.
        if ft_audio_ratio > 0:
            # mulan_model.audio is an AudioSpectrogramTransformer.
            audio_encoder = mulan_model.audio
            
            # AST models typically use transformer.blocks or encoder.layer.
            # Inspect the loaded audio_encoder to find the layer container.
            layers = None
            if hasattr(audio_encoder, 'blocks'):  # ViT-style structure
                layers = audio_encoder.blocks
            elif hasattr(audio_encoder, 'encoder') and hasattr(audio_encoder.encoder, 'layer'):
                layers = audio_encoder.encoder.layer
            
            if layers:
                num_layers = len(layers)
                num_ft = int(num_layers * ft_audio_ratio)
                # Enable only the top (final) layers.
                for i in range(num_layers - num_ft, num_layers):
                    for p in layers[i].parameters():
                        p.requires_grad = True
                print(f"🔥 Fine-tuning {num_ft}/{num_layers} audio layers.")

        # 5. Configure reward head.
        embed_dim = mulan_model.dim_latent  # Use dim_latent from MuLanModel
        self.reward_head = RewardMLP(embed_dim * 4)

    def forward(self, audio, text):
        """
        audio: (B, T) tensor
        text: List[str]
        """
        # MuQMuLan forward must be called with wavs=... or texts=...
        # Internally clips audio longer than 10s and averages embeddings
        tokenizer = self.mulan_wrapper.mulan_module.text.tokenizer
        
        # 2. Truncate text with the tokenizer.
        # Word-boundary truncation keeps sequences within max_length.
        encoded = tokenizer(
            text,
            truncation=True,
            max_length=256,  # Match model config max_seq_len
            padding=False,  # Wrapper handles padding later; truncate only here
            return_tensors=None  # Return lists for decoding back to strings
        )
        
        # 3. Decode truncated tokens back to strings.
        # Ensures model input strings are within safe length.
        text = tokenizer.batch_decode(encoded['input_ids'], skip_special_tokens=True)
  
        a = self.mulan_wrapper(wavs=audio) # (B, D)
        t = self.mulan_wrapper(texts=text) # (B, D)
        
        # Normalize.
        a = F.normalize(a, p=2, dim=-1)
        t = F.normalize(t, p=2, dim=-1)
        
        # Combine features.
        diff = torch.abs(a - t)
        prod = a * t
        joint = torch.cat([a, t, diff, prod], dim=-1)

        # Compute reward.
        reward = self.reward_head(joint).squeeze(-1)
        return reward
    
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

@torch.no_grad()
def evaluate_limited(model, dataloader, device, max_batches=100):
    model.eval()
    
    a_texts, a_musics = [], []

    # Show progress with tqdm during validation.
    from tqdm import tqdm
    pbar = tqdm(total=max_batches if max_batches else len(dataloader), 
                desc="Evaluating", leave=False, dynamic_ncols=True)

    for i, batch in enumerate(dataloader):
        if max_batches and i >= max_batches:
            break
            
        # 1. Text-to-Audio Accuracy (A2T)
        if batch["text"] is not None:
            tb = batch["text"]
            audios = tb["audio"].to(device)
            pref = tb["pref_text"]
            rej  = tb["rej_text"]

            rp = model(audios, pref)
            rr = model(audios, rej)
            # 1 when preferred score is higher, else 0.
            acc = (rp > rr).float().mean().item()
            a_texts.append(acc)

        # 2. Audio-to-Text Accuracy (T2A)
        if batch["music"] is not None:
            mb = batch["music"]
            text = mb["text"]
            pref_audio = mb["pref_audio"].to(device)
            rej_audio  = mb["rej_audio"].to(device)

            rp = model(pref_audio, text)
            rr = model(rej_audio,  text)
            acc = (rp > rr).float().mean().item()
            a_musics.append(acc)
            
        pbar.update(1)
    
    pbar.close()

    # Aggregate results.
    acc_text  = sum(a_texts) / len(a_texts) if a_texts else None
    acc_music = sum(a_musics) / len(a_musics) if a_musics else None
    
    # Combined score (handles missing branches).
    if acc_text is not None and acc_music is not None:
        acc_combo = (acc_text + acc_music) / 2
    else:
        acc_combo = acc_text or acc_music or 0.0

    metrics = {
        "acc_text": acc_text,
        "acc_music": acc_music,
        "acc_combo": acc_combo
    }

    model.train()  # Restore train mode.
    return metrics


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
        max_samples=None  # Optional sample limit.
    )

    val_ds = Dataset(
        a2t_data_path=args.a2t_test_path,
        t2a_data_path=args.t2a_test_path,
        split="valid",
        model_type=args.model_type,
        threshold_similar=None,
        threshold_negative=None,
        max_samples=None  # Optional sample limit.
    )

    collate = DataCollator(
        model_type=args.model_type,
        target_sr=24000, 
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
        
    model = MuLANReward(
        ft_audio_ratio=0  # Fraction of audio layers to fine-tune.
    ).to(device)
    
    num_gpus = torch.cuda.device_count()

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay
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

    grad_accum_steps = max(1, 256 // args.batch_size)
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
    
    # Validate every N steps (e.g. 500-1000 for large datasets).
    eval_interval = 1000
    # Optionally validate only the first N batches.
    max_eval_batches = None

    model.train()
    optimizer.zero_grad()
    global_step = 0

    for epoch in range(args.epochs):
        for step, batch in enumerate(train_loader):
            # 1. Forward pass and loss.
            loss, metrics = combined_loss(model, batch, w_text=args.w_text, w_music=args.w_music)

            if loss is None:
                continue

            # 2. Gradient Accumulation
            loss_scaled = loss / grad_accum_steps
            loss_scaled.backward()

            # 3. Optimizer Step
            if (step + 1) % grad_accum_steps == 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

                # [Log] Record training metrics.
                if not args.no_wandb and rank == 0:
                    log_data = {f"train/{k}": v for k, v in metrics.items()}
                    log_data["train/total_loss"] = loss.item()
                    # Add margin metrics when available.
                    if "pref_text" in metrics and "rej_text" in metrics:
                        log_data["train/margin_text"] = metrics["pref_text"] - metrics["rej_text"]
                    
                    wandb.log(log_data, step=global_step)

                pbar.update(1)
                pbar.set_postfix({
                    "loss": f"{loss.item():.4f}", 
                    "acc": f"{metrics.get('acc_avg', 0):.2f}"
                })
            
                # [Eval] Run validation at fixed step intervals.
                if global_step % eval_interval == 0:
                    print(f"\n[Step {global_step}] Evaluating...")
                    
                    # Call validation (optionally on a subset of val_loader).
                    val_metrics = evaluate_limited(model, val_loader, device, max_batches=max_eval_batches)
                    
                    if rank == 0:
                        print(f"Step {global_step} | Val Acc: {val_metrics['acc_combo']:.4f}")
                        
                        if not args.no_wandb:
                            wandb.log({f"val/{k}": v for k, v in val_metrics.items()}, step=global_step)

                        # Save best checkpoint and check early stopping.
                        score = val_metrics["acc_combo"]
                        if score is not None and score > best:
                            best = score
                            counter = 0
                            torch.save({
                                "step": global_step,
                                "state_dict": model.state_dict(),
                                "best_acc": best
                            }, save_path)
                            print(f"🔥 New Best Model saved at step {global_step} (Acc: {best:.4f})")
                        else:
                            counter += 1
                            if counter >= patience:
                                print("Stop! No improvement detected, ending training early.")
                                pbar.close()
                                return  # Exit training.

        # Optional full validation at epoch end.
        print(f"\n--- Epoch {epoch} Finished ---")
        
        # [Eval] Run validation at fixed step intervals.
        print(f"\n[Step {global_step}] Evaluating...")
        
        # Call validation (optionally on a subset of val_loader).
        val_metrics = evaluate_limited(model, val_loader, device, max_batches=max_eval_batches)
        
        if rank == 0:
            print(f"Step {global_step} | Val Acc: {val_metrics['acc_combo']:.4f}")
            
            if not args.no_wandb:
                wandb.log({f"val/{k}": v for k, v in val_metrics.items()}, step=global_step)

            # Save best checkpoint and check early stopping.
            score = val_metrics["acc_combo"]
            if score is not None and score > best:
                best = score
                counter = 0
                torch.save({
                    "step": global_step,
                    "state_dict": model.state_dict(),
                    "best_acc": best
                }, save_path)
                print(f"🔥 New Best Model saved at step {global_step} (Acc: {best:.4f})")
            else:
                counter += 1
                if counter >= patience:
                    print("Stop! No improvement detected, ending training early.")
                    pbar.close()
                    return  # Exit training.


    pbar.close()
    if not args.no_wandb and rank == 0:
        wandb.finish()

if __name__ == "__main__":
    main()
