"""Shared helpers for DPO training and evaluation."""

from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO_ROOT / "dataset" / "data" / "generate_dataset"
CHECKPOINT_DIR = REPO_ROOT / "dpo" / "checkpoints"
OUTPUT_DIR = REPO_ROOT / "dpo" / "outputs"

MODEL_ID = "Qwen/Qwen2-Audio-7B-Instruct"
PROMPT = "Describe the given music in several sentences with no imagined elements."


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def use_wandb() -> bool:
    return os.environ.get("USE_WANDB", "0").lower() in {"1", "true", "yes"}


def init_wandb(project: str, name: str):
    if not use_wandb():
        return None
    import wandb

    return wandb.init(project=project, name=name)


def log_metrics(run, metrics: dict) -> None:
    if run is not None:
        run.log(metrics)


def finish_wandb(run) -> None:
    if run is not None:
        run.finish()
