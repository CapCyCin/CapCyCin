"""Shared paths for dataset construction scripts."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO_ROOT / "dataset" / "data" / "generate_dataset"
RESOURCES = REPO_ROOT / "dataset" / "resources"
CAPTION_DATA = RESOURCES / "captioning_data"
AUDIO_ROOT = REPO_ROOT / "laion-disco-10s"
