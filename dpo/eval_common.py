"""Shared benchmark inference utilities."""

from __future__ import annotations

import json
import os
from pathlib import Path

import librosa
import torch
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration

from dpo.utils import MODEL_ID, OUTPUT_DIR, PROMPT

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_dpo_model(checkpoint_path: Path):
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    processor.tokenizer.padding_side = "right"
    model = Qwen2AudioForConditionalGeneration.from_pretrained(MODEL_ID, trust_remote_code=True).to(device)
    if checkpoint_path.exists():
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(state_dict, strict=False)
    model.eval()
    return processor, model


def load_audio(processor, path):
    audio, _ = librosa.load(path, sr=processor.feature_extractor.sampling_rate)
    return audio


@torch.no_grad()
def generate_caption(processor, model, audio_path):
    audio = load_audio(processor, audio_path)
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": audio_path},
                {"type": "text", "text": PROMPT},
            ],
        }
    ]
    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    inputs = processor(
        text=text,
        audio=audio,
        return_tensors="pt",
        sampling_rate=processor.feature_extractor.sampling_rate,
    ).to(device)
    output_ids = model.generate(**inputs, max_new_tokens=256, do_sample=False)
    generated_ids = output_ids[:, inputs.input_ids.shape[1] :]
    return processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()


def write_jsonl_results(output_path: Path, rows):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Saved to {output_path}")


def run_over_items(processor, model, items, output_path: Path):
    rows = []
    for item in tqdm(items, desc="Evaluating"):
        audio_path = item["audio_path"]
        if not os.path.exists(audio_path):
            continue
        try:
            dpo_caption = generate_caption(processor, model, audio_path)
        except Exception as exc:
            print(f"Error: {audio_path} -> {exc}")
            continue
        rows.append(
            {
                **item["metadata"],
                "audio_path": audio_path,
                "dpo_caption": dpo_caption,
                "ground_truth": item["ground_truth"],
            }
        )
    write_jsonl_results(output_path, rows)
