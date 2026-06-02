# Music Captioning DPO

Anonymous release of code for Direct Preference Optimization (DPO) on music captioning with Qwen2-Audio, using cycle-consistency preference pairs.

This repository is currently being updated.
Code and documentation will be available soon.

## Repository Layout

```
.
├── dpo/                         # Training, inference, and evaluation
│   ├── train.py                 # Full fine-tuning DPO (CPPair dataset)
│   ├── train_comp.py            # DPO with cycle-consistency threshold data
│   ├── inference.py             # Validation accuracy on a saved checkpoint
│   ├── metrics.py               # BLEU, SPICE, CIDEr, FENSE, SBERT
│   ├── eval_musiccaps.py
│   ├── eval_sdd.py
│   ├── eval_ytmtc.py
│   ├── checkpoints/             # Saved model weights (not tracked by git)
│   └── outputs/                 # Generated caption JSONL files
├── dataset/
│   ├── data/generate_dataset/   # Preference-pair JSONL files
│   ├── resources/               # Caption JSON files and metadata CSV
│   └── scripts/                 # Dataset construction pipeline
├── laion-disco-10s/             # Audio clips (download separately)
├── musiccaps/                   # MusicCaps eval split (download separately)
├── SongDescriberDataset/        # SDD eval split (download separately)
└── YouTube8B-MusicTextClips/    # YTMTC eval split (download separately)
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Place audio and benchmark files locally before running training or evaluation:

- `laion-disco-10s/{ytid}.wav` — training audio referenced in preference JSONL
- Benchmark folders for MusicCaps, Song Describer Dataset, and YouTube8B-MusicTextClips

Download DPO checkpoints separately and place them under `dpo/checkpoints/`.

## Dataset Construction

Run from the repository root:

```bash
# Step 1: build ranked candidate lists from similarity scores and model captions
python dataset/scripts/build_candidates.py

# Step 2: convert candidates into preferred/rejected pairs
python dataset/scripts/build_pairs.py

# Step 3: remove noisy pairs
python dataset/scripts/filter_pairs.py

# Step 4: split into train/val/test
python dataset/scripts/split.py
```

The music-to-text (`base/m2t`) variant follows the same pattern under `dataset/scripts/base/`.

## DPO Training

WandB is **disabled by default** for anonymous release. Enable it only if you want logging:

```bash
export USE_WANDB=1
python dpo/train.py
python dpo/train_comp.py
```

Checkpoints are written to:

- `dpo/checkpoints/qwen-dpo/`
- `dpo/checkpoints/comp-dpo/`

## Inference and Metrics

Generate captions on a benchmark:

```bash
python dpo/eval_musiccaps.py
python dpo/eval_sdd.py
python dpo/eval_ytmtc.py
```

Compute automatic metrics from a saved JSONL file:

```bash
python dpo/metrics.py
```

SPICE requires Java and [Stanford CoreNLP 4.2.2](https://stanfordnlp.github.io/CoreNLP/). Download the models jar and set `CLASSPATH` before running `metrics.py`.

## Notes

- Preference JSONL files use relative audio paths such as `./laion-disco-10s/{ytid}.wav`.
- Large artifacts (`.pt`, `.wav`, model caches) are excluded via `.gitignore`.
- LoRA training code is intentionally omitted from this release.

## Citation

If you use this code, please cite the corresponding paper (details to be added upon publication).
