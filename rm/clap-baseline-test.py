#!/usr/bin/env python3
import os
import argparse
import torch
import torch.nn.functional as F
import laion_clap
from torch.utils.data import DataLoader
from tqdm import tqdm
import csv
import numpy as np

# Reuse existing Dataset and DataCollator from the training code.
from clap_rm_concat import (
    Dataset,
    DataCollator,
    set_seed,
)

@torch.no_grad()
def evaluate_pure_clap(clap_model, dataloader, save_csv_path=None, device="cuda"):
    clap_model.eval()

    results = []
    
    if save_csv_path:
        os.makedirs(os.path.dirname(save_csv_path), exist_ok=True)
        csv_file = open(save_csv_path, "w", newline="")
        csv_writer = csv.DictWriter(csv_file, fieldnames=[
            "mode", "text", "sim_pref", "sim_rej", "correct"
        ])
        csv_writer.writeheader()

    pbar = tqdm(dataloader, desc="Evaluating Pure CLAP Score")

    for batch in pbar:
        if batch.get("music") is not None:
            mb = batch["music"]
            # Assume text is provided as a list (tokenized inside laion_clap).
            texts = mb["text"] 
            pref_audio = mb["pref_audio"].to(device)
            rej_audio  = mb["rej_audio"].to(device)

            # 1. Extract audio embeddings.
            p_emb = clap_model.get_audio_embedding_from_data(x=pref_audio, use_tensor=True)
            r_emb = clap_model.get_audio_embedding_from_data(x=rej_audio, use_tensor=True)
            
            # 2. Extract text embeddings.
            t_emb = clap_model.get_text_embedding(texts, use_tensor=True)

            # 3. Compute cosine similarity.
            sim_p = F.cosine_similarity(p_emb, t_emb)
            sim_r = F.cosine_similarity(r_emb, t_emb)

            # 4. Mark correct when preferred similarity is higher.
            corrects = (sim_p > sim_r).float().cpu().numpy().tolist()
            results.extend(corrects)

            if save_csv_path:
                for i in range(len(corrects)):
                    csv_writer.writerow({
                        "mode": "T2A",
                        "text": texts[i],
                        "sim_pref": float(sim_p[i].item()),
                        "sim_rej": float(sim_r[i].item()),
                        "correct": int(corrects[i])
                    })

            pbar.set_postfix({"Pure_CLAP_Acc": f"{np.mean(results):.4f}"})

    if save_csv_path:
        csv_file.close()

    print(f"\n--- Pure CLAP Results ---")
    print(f"Final Accuracy: {np.mean(results):.4f}")
    return np.mean(results)

def main():
    parser = argparse.ArgumentParser("Pure CLAP Baseline Test")
    parser.add_argument("--t2a_test_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--clap_ckpt",
        type=str,
        default="checkpoints/laion_clap/models--lukewys--laion_clap/snapshots/"
                "b3708341862f581175dba5c356a4ebf74a9b6651/music_audioset_epoch_15_esc_90.14.pt",
        help="Path to the pretrained CLAP checkpoint.",
    )
    parser.add_argument("--checkpoint", type=str, help="Optional: kept for script compatibility.")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the pretrained CLAP model.
    model = laion_clap.CLAP_Module(enable_fusion=False, amodel='HTSAT-base')
    model.load_ckpt(args.clap_ckpt)

    # Build dataset and dataloader.
    test_ds = Dataset(
        a2t_data_path="",
        t2a_data_path=args.t2a_test_path,
        split="test",
        model_type="CycleReward-T2A",
        threshold_similar=None,
        threshold_negative=None,
    )
    
    collate = DataCollator(model_type="CycleReward-T2A", target_sr=48000)
    
    test_loader = DataLoader(
        test_ds, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=4, 
        collate_fn=collate
    )

    evaluate_pure_clap(model, test_loader, save_csv_path=None, device=device)

if __name__ == "__main__":
    main()
