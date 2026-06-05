#!/usr/bin/env python3
import os
import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import csv
import numpy as np

from muq import MuQMuLan

# Reuse existing Dataset and DataCollator from the training code.
# Note: set target_sr=24000 inside DataCollator.
from mulan_rm_concat import (
    Dataset,
    DataCollator,
    set_seed,
)

@torch.no_grad()
def evaluate_muq_mulan(mulan_model, dataloader, save_csv_path=None, device="cuda"):
    mulan_model.eval()

    results = []
    
    if save_csv_path:
        os.makedirs(os.path.dirname(save_csv_path), exist_ok=True)
        csv_file = open(save_csv_path, "w", newline="")
        csv_writer = csv.DictWriter(csv_file, fieldnames=[
            "mode", "text", "sim_pref", "sim_rej", "correct"
        ])
        csv_writer.writeheader()

    pbar = tqdm(dataloader, desc="Evaluating MuQ-MuLAN Score")

    for batch in pbar:
        if batch.get("music") is not None:
            mb = batch["music"]
            texts = mb["text"] 
            pref_audio = mb["pref_audio"].to(device)
            rej_audio  = mb["rej_audio"].to(device)

            # Extract embeddings via the MuLAN interface.
            p_emb = mulan_model(wavs=pref_audio)
            r_emb = mulan_model(wavs=rej_audio)
            t_emb = mulan_model(texts=texts)

            sim_p = mulan_model.calc_similarity(p_emb, t_emb).squeeze()
            sim_r = mulan_model.calc_similarity(r_emb, t_emb).squeeze()

            if sim_p.dim() == 0:
                sim_p = sim_p.unsqueeze(0)
                sim_r = sim_r.unsqueeze(0)

            corrects = (sim_p > sim_r).float().cpu().numpy().reshape(-1).tolist()
            results.extend(corrects)

            if save_csv_path:
                for i in range(len(corrects)):
                    csv_writer.writerow({
                        "mode": "T2A",
                        "text": texts[i],
                        "sim_pref": float(sim_p[i].item()) if sim_p.dim() > 0 else float(sim_p.item()),
                        "sim_rej": float(sim_r[i].item()) if sim_r.dim() > 0 else float(sim_r.item()),
                        "correct": int(corrects[i])
                    })

            pbar.set_postfix({"MuLAN_Acc": f"{np.mean(results):.4f}"})
    if save_csv_path:
        csv_file.close()

    print(f"\n--- MuQ-MuLAN Results ---")
    print(f"Final Accuracy: {np.mean(results):.4f}")
    return np.mean(results)

def main():
    parser = argparse.ArgumentParser("MuQ-MuLAN Baseline Test")
    parser.add_argument("--t2a_test_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading MuQ-MuLAN model...")
    model = MuQMuLan.from_pretrained("OpenMuQ/MuQ-MuLan-large")
    model = model.to(device).eval()

    test_ds = Dataset(
        a2t_data_path="",
        t2a_data_path=args.t2a_test_path,
        split="test",
        model_type="CycleReward-T2A",
        threshold_similar=None,
        threshold_negative=None,
    )
    
    collate = DataCollator(model_type="CycleReward-T2A", target_sr=24000)
    
    test_loader = DataLoader(
        test_ds, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=4, 
        collate_fn=collate
    )

    evaluate_muq_mulan(model, test_loader, save_csv_path=None, device=device)

if __name__ == "__main__":
    main()
