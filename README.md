# CapCyCin

Anonymous release of code for **cycle-consistency preference learning** on music captioning and text-to-music generation.

This repository covers the full pipeline:

1. **Dataset** — build cycle-consistency preference pairs (CPPair) from similarity scores and multi-model generations
2. **Reward Model (RM)** — train cycle-consistency reward models on CLAP, MuQ-MuLAN, and BLAP backbones
3. **DPO** — fine-tune Qwen2-Audio with Direct Preference Optimization on CPPair data

---

## Repository Layout

```
.
├── dataset/
│   ├── data/generate_dataset/       # Preference-pair JSONL (CPPair)
│   │   ├── CPPair-m2t/              # Music → Text (audio anchor, text candidates)
│   │   │   ├── datasets/            # Ranked candidate lists
│   │   │   ├── addition/            # Pairwise prefs (addition setting)
│   │   │   └── threshold/           # Pairwise prefs (threshold setting)
│   │   ├── CPPair-t2m/              # Text → Music (caption anchor, audio candidates)
│   │   └── baseline/                # Baseline pairs (single similarity model)
│   ├── resources/                   # Similarity JSON, caption JSON, metadata CSV
│   └── scripts/                     # Dataset construction pipeline
│       ├── build_candidates.py      # Step 1: rank candidates by similarity
│       ├── build_pairs.py           # Step 2: preferred/rejected pairs
│       ├── filter_pairs.py          # Step 3: remove noisy pairs (optional)
│       └── split.py                 # Step 4: train/val/test split
├── rm/                              # Cycle-consistency reward models
│   ├── clap_rm_concat.py            # CLAP backbone RM (train)
│   ├── clap_add.py                  # CLAP RM variant (addition data)
│   ├── mulan_rm_concat.py           # MuQ-MuLAN backbone RM (train)
│   ├── clap-rm-test.py              # Evaluate trained CLAP RM
│   ├── clap-baseline-test.py        # Pure CLAP baseline (no RM head)
│   ├── muq-baseline-test.py         # Pure MuQ-MuLAN baseline
│   └── blap/                        # BLAP-based combo RM
│       └── rm_train.py
├── dpo/                             # Qwen2-Audio DPO training & evaluation
│   ├── train.py                     # Full fine-tuning DPO (CPPair dataset)
│   ├── train_comp.py                # DPO with cycle-consistency threshold data
│   ├── inference.py                 # Validation accuracy on a saved checkpoint
│   ├── metrics.py                   # BLEU, SPICE, CIDEr, FENSE, SBERT
│   ├── eval_musiccaps.py
│   ├── eval_sdd.py
│   ├── eval_ytmtc.py
│   ├── checkpoints/                 # Saved model weights (not tracked by git)
│   └── outputs/                     # Generated caption JSONL files
├── data/                            # External data (download separately, not in git)
│   ├── audio/
│   │   ├── laion-disco-10s/        # M2T training audio clips
│   │   └── texttomusic/             # T2M generated audio (per model/seed)
│   ├── captioning_data/             # Multi-model caption JSON files
│   ├── evaluation/                  # RM evaluation sets
│   └── *.json / *.csv               # Similarity scores, metadata
├── checkpoints/                     # Pretrained backbone weights (not in git)
│   ├── laion_clap/
│   └── rm/                          # Trained RM checkpoints
├── laion-disco-10s/                 # Symlink or copy of data/audio/laion-disco-10s
├── musiccaps/                       # MusicCaps eval split (download separately)
├── SongDescriberDataset/            # SDD eval split (download separately)
└── YouTube8B-MusicTextClips/        # YTMTC eval split (download separately)
```

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### External data

Place the following locally before running any script:

| Path | Description |
|------|-------------|
| `data/audio/laion-disco-10s/{ytid}.wav` | 10-second audio clips for M2T training |
| `data/audio/texttomusic/{model}/{seed}/{id}.wav` | T2M generated audio (audioldm2, musicgen, riffusion, …) |
| `data/captioning_data/*.json` | Captions from multiple captioning models |
| `data/laion-disco_mulan.json` | Audio–text similarity scores (M2T) |
| `data/laion-disco-10s-ytid.csv` | Audio ID metadata |
| `data/lpmusiccaps-msd-8k-trunc.csv` | MusicCaps captions for T2M |
| `checkpoints/laion_clap/...` | Pretrained CLAP weights |
| `data/blap-ckpt/` | BLAP stage-2 checkpoint (for `rm/blap/`) |

Preference JSONL files use **relative paths** such as `data/audio/laion-disco-10s/{ytid}.wav`.

---

## 1. Dataset Construction

Cycle-consistency preference pairs are built in four steps. Run from the repository root.

### Music → Text (CPPair-m2t)

```bash
# Step 1: rank caption candidates by cycle-consistency similarity
python dataset/scripts/build_candidates.py
# → dataset/data/generate_dataset/CPPair-m2t/datasets/ld-mnm-*.jsonl

# Step 2: convert ranked lists into preferred/rejected pairs
python dataset/scripts/build_pairs.py
# → dataset/data/generate_dataset/CPPair-m2t/CPpair-*.jsonl

# Step 3 (optional): filter noisy pairs by score thresholds
python dataset/scripts/filter_pairs.py

# Step 4: split into train / val / test
python dataset/scripts/split.py
# → dataset/data/generate_dataset/CPPair-m2t/addition/{train,val,test}/
```

Each JSONL record in the final split:

```json
{
  "music": "data/audio/laion-disco-10s/{ytid}.wav",
  "preferred": "caption with higher cycle-consistency score",
  "rejected": "caption with lower cycle-consistency score"
}
```

### Text → Music (CPPair-t2m)

Same pipeline under `dataset/data/generate_dataset/CPPair-t2m/`. Records use a text anchor and audio file paths as preferred/rejected candidates.

### Baseline variants

Single-similarity-model baselines live under `dataset/data/generate_dataset/baseline/` (`m2t/` and `t2m/`). The script layout mirrors the CPPair pipeline above.

---

## 2. Reward Model Training

Reward models score whether a generation is **cycle-consistent** with its anchor (audio or text). Three backbone options are provided.

### Model types

| `model_type` | Task | Anchor | Candidates |
|---|---|---|---|
| `CycleReward-A2T` | Music → Text | audio | text captions |
| `CycleReward-T2A` | Text → Music | text prompt | audio files |
| `CycleReward-Combo` | Both | — | trains A2T + T2A jointly |

### CLAP reward model

```bash
python rm/clap_rm_concat.py \
  --model_type CycleReward-A2T \
  --a2t_train_path dataset/data/generate_dataset/CPPair-m2t/addition/train/CPpair-ld-mnm-clap.jsonl \
  --a2t_test_path  dataset/data/generate_dataset/CPPair-m2t/addition/test/CPpair-ld-mnm-clap.jsonl \
  --epochs 2 --batch_size 256 --lr 2e-5 \
  --checkpoint_dir checkpoints/rm \
  --no_wandb
```

Checkpoints are saved to `checkpoints/rm/{save_name}.pt`.

### MuQ-MuLAN reward model

```bash
python rm/mulan_rm_concat.py \
  --model_type CycleReward-A2T \
  --a2t_train_path dataset/data/generate_dataset/CPPair-m2t/addition/train/CPpair-ld-mnm-clap.jsonl \
  --a2t_test_path  dataset/data/generate_dataset/CPPair-m2t/addition/test/CPpair-ld-mnm-clap.jsonl \
  --epochs 2 --batch_size 256 --lr 2e-5 \
  --checkpoint_dir checkpoints/rm \
  --no_wandb
```

### BLAP combo reward model

```bash
python rm/blap/rm_train.py \
  --model_type CycleReward-Combo \
  --a2t_train_path dataset/data/generate_dataset/CPPair-m2t/threshold/train/CPpair-ld-mnm-mert-v1-330M.jsonl \
  --a2t_test_path  dataset/data/generate_dataset/CPPair-m2t/threshold/val/CPpair-ld-mnm-mert-v1-330M.jsonl \
  --t2a_train_path dataset/data/generate_dataset/CPPair-t2m/threshold/train/CPpair-msd-tnt-sbert.jsonl \
  --t2a_test_path  dataset/data/generate_dataset/CPPair-t2m/threshold/val/CPpair-msd-tnt-sbert.jsonl \
  --epochs 5 --batch_size 16 --lr 2e-5 \
  --no_wandb
```

### RM evaluation

```bash
# Trained CLAP RM
python rm/clap-rm-test.py \
  --model_type CycleReward-T2A \
  --t2a_test_path data/evaluation/rm/musiceval/intergrated/eval_pref_textual_0.jsonl \
  --checkpoint checkpoints/rm/your_checkpoint.pt

# Pure backbone baselines (no RM head)
python rm/clap-baseline-test.py  --t2a_test_path <test.jsonl>
python rm/muq-baseline-test.py   --t2a_test_path <test.jsonl>
```

WandB is **disabled by default** (`--no_wandb`). Omit the flag only if you want experiment logging.

---

## 3. DPO Training

Fine-tune Qwen2-Audio on CPPair preference data with Direct Preference Optimization.

```bash
export USE_WANDB=0   # disabled by default for anonymous release
python dpo/train.py
python dpo/train_comp.py
```

Checkpoints are written to:

- `dpo/checkpoints/qwen-dpo/`
- `dpo/checkpoints/comp-dpo/`

Download pretrained DPO checkpoints separately and place them under `dpo/checkpoints/` before running inference.

---

## 4. Inference and Metrics

Generate captions on evaluation benchmarks:

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

---

## Pipeline Overview

```
Similarity scores + multi-model generations
        │
        ▼
  dataset/scripts/          ← build CPPair JSONL
        │
        ├──────────────────┐
        ▼                  ▼
   rm/ (train RM)     dpo/ (fine-tune Qwen2-Audio)
        │                  │
        ▼                  ▼
  ranking accuracy     caption metrics
  on held-out pairs    (BLEU, SPICE, CIDEr, …)
```

---

## Notes

- All paths in code and JSONL are **relative** to the repository root.
- Large artifacts (`.pt`, `.wav`, model caches, wandb logs) are excluded via `.gitignore`.
- LoRA training code is intentionally omitted from this release.
- WandB is disabled by default across RM and DPO scripts for anonymous release.

---

## Citation

If you use this code, please cite the corresponding paper (details to be added upon publication).
