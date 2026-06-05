import json
import itertools
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]

def build_pairwise_dataset(
    input_jsonl,
    output_jsonl,
    thresh_similar=0.05,
    thresh_negative=0.7
):
    kept = 0
    skipped_similar = 0
    skipped_negative = 0
    
    os.makedirs(os.path.dirname(output_jsonl), exist_ok=True)

    with open(input_jsonl, "r", encoding="utf-8") as fin, \
         open(output_jsonl, "w", encoding="utf-8") as fout:

        for line in fin:
            data = json.loads(line)

            music_path = data["music"]
            generations = data["generations"]
            scores = data["scores"]

            for i, j in itertools.combinations(range(len(generations)), 2):
                score_i, score_j = scores[i], scores[j]

                if abs(score_i - score_j) < thresh_similar:
                    skipped_similar += 1
                    continue

                if score_i > score_j:
                    pref, rej, score_pref = generations[i], generations[j], score_i
                else:
                    pref, rej, score_pref = generations[j], generations[i], score_j

                if score_pref < thresh_negative:
                    skipped_negative += 1
                    continue

                pref_item = {
                    "music": music_path,
                    "preferred": pref,
                    "rejected": rej
                }

                fout.write(json.dumps(pref_item, ensure_ascii=False) + "\n")
                kept += 1

    print(f"[INFO] Finished!")
    print(f"kept pairs       : {kept}")
    print(f"skipped_similar  : {skipped_similar}")
    print(f"skipped_negative : {skipped_negative}")


if __name__ == "__main__":
    datasets_dir = PROJECT_ROOT / "generate_dataset/baseline/m2t"
        
    for f in os.listdir(datasets_dir):
        if f.endswith(".jsonl"): 
            input_jsonl = os.path.join(datasets_dir, f)
            output_jsonl = str(PROJECT_ROOT / f"generate_dataset/baseline/m2t/add/CPpair-{f}")

            THRESH_SIMILAR = 0.17
            THRESH_NEGATIVE = 0.39

            build_pairwise_dataset(
                input_jsonl,
                output_jsonl,
                thresh_similar=THRESH_SIMILAR,
                thresh_negative=THRESH_NEGATIVE
            )
