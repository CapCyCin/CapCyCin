# =========================
# Test Only
# =========================
#!/usr/bin/env python3
import os
import argparse
import torch
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm
import csv

from clap_rm_concat import (
    Dataset,
    DataCollator,
    CLAPReward,
    #evaluate,
    set_seed,
)
@torch.no_grad()
def evaluate(model, dataloader, save_csv_path=None, device="cuda"):
    model.eval()

    csv_file = None
    csv_writer = None

    if save_csv_path is not None:
        csv_file = open(save_csv_path, "w", newline="")
        csv_writer = csv.DictWriter(csv_file, fieldnames=[
            "mode", "text",
            "pref_path", "rej_path",
            "rp", "rr", "correct"
        ])
        csv_writer.writeheader()
        
    def _get_correct_list(rp, rr):
        # Count correct when rp > rr (ranking accuracy).
        return (rp > rr).float().cpu().numpy().tolist()

    a_texts_results, a_musics_results = [], []
    
    # Configure tqdm progress bar.
    pbar = tqdm(dataloader, desc="Evaluating")
    
    for batch in pbar:
        current_metrics = {}
        
        # 1. Compute music-to-text (A2T) accuracy.
        if batch.get("text") is not None:
            tb = batch["text"]
            audio = tb["audio"].to(device)
            pref_text = tb["pref_text"]
            rej_text  = tb["rej_text"]

            rp = model(audio, pref_text)
            rr = model(audio, rej_text)
            
            corrects = _get_correct_list(rp, rr)
            a_texts_results.extend(corrects)
            
            if csv_writer is not None:
                pref_path = tb.get("pref_path", ["N/A"] * len(pref_text))
                rej_path  = tb.get("rej_path",  ["N/A"] * len(pref_text))
                for i in range(len(pref_text)):
                    csv_writer.writerow({
                        "mode": "A2T", "text": pref_text[i],
                        "pref_path": pref_path[i], "rej_path": rej_path[i],
                        "rp": float(rp[i].item()), "rr": float(rr[i].item()),
                        "correct": int(corrects[i])
                    })
            
            current_metrics["acc_A2T"] = sum(a_texts_results) / len(a_texts_results)

        # 2. Compute text-to-music (T2A) accuracy.
        if batch.get("music") is not None:
            mb = batch["music"]
            text = mb["text"]
            pref_audio = mb["pref_audio"].to(device)
            rej_audio  = mb["rej_audio"].to(device)

            rp = model(pref_audio, text)
            rr = model(rej_audio,  text)
            
            corrects = _get_correct_list(rp, rr)
            a_musics_results.extend(corrects)

            if csv_writer is not None:
                pref_path = mb.get("pref_path", ["N/A"] * len(text))
                rej_path  = mb.get("rej_path",  ["N/A"] * len(text))
                for i in range(len(text)):
                    csv_writer.writerow({
                        "mode": "T2A", "text": text[i],
                        "pref_path": pref_path[i], "rej_path": rej_path[i],
                        "rp": float(rp[i].item()), "rr": float(rr[i].item()),
                        "correct": int(corrects[i])
                    })
            
            current_metrics["acc_T2A"] = sum(a_musics_results) / len(a_musics_results)

        # Show live accuracy in the tqdm postfix.
        pbar.set_postfix(current_metrics)

    if csv_file is not None:
        csv_file.close()

    final_acc_text  = sum(a_texts_results)/len(a_texts_results) if a_texts_results else None
    final_acc_music = sum(a_musics_results)/len(a_musics_results) if a_musics_results else None
    
    # Compute combined score.
    if final_acc_text is not None and final_acc_music is not None:
        final_acc_combo = 0.5 * (final_acc_text + final_acc_music)
    else:
        final_acc_combo = final_acc_text if final_acc_text is not None else final_acc_music

    print(f"\n--- Final Results ---")
    print(f"Final Acc (A2T - Text): {final_acc_text}")
    print(f"Final Acc (T2A - Music): {final_acc_music}")
    print(f"Final Acc (Combo): {final_acc_combo}")

    return {"acc_text": final_acc_text, "acc_music": final_acc_music, "acc_combo": final_acc_combo}
# =========================

# =========================
# Test + WandB Logging
# =========================
def main():
    parser = argparse.ArgumentParser("CycleReward-CLAP RM Test")

    # ===== data =====
    parser.add_argument("--a2t_test_path", type=str, required=True)
    parser.add_argument("--t2a_test_path", type=str, required=True)
    parser.add_argument("--model_type", type=str, default="CycleReward-Combo")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)

    # ===== checkpoint =====
    parser.add_argument("--checkpoint", type=str, required=True)

    # ===== misc =====
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_csv", type=str, default=None, help="Optional CSV output path.")

    args = parser.parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ======================
    # Dataset
    # ======================
    test_ds = Dataset(
        a2t_data_path=args.a2t_test_path,
        t2a_data_path=args.t2a_test_path,
        split="test",
        model_type=args.model_type,
        threshold_similar=None,
        threshold_negative=None,
    )

    collate = DataCollator(
        model_type=args.model_type,
        target_sr=48000,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate,
    )

    # ======================
    # Model
    # ======================
    model = CLAPReward(ft_ratio=0).to(device)
   

    print("Loading checkpoint:", args.checkpoint)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()

    # ======================
    # Evaluate
    # ======================
    evaluate(model, test_loader, args.output_csv)
    


if __name__ == "__main__":
    main()