import csv
import json
import os
from collections import defaultdict

import numpy as np

# ===== Path Setup =====
base_dir = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(base_dir, "..", "..", "ai_module", "dataset", "asl_landmark_data.csv")
out_path = os.path.join(base_dir, "..", "frontend", "assets", "letter_references.json")

# ===== Aggregate mean normalized landmark vector per (letter, hand) =====
# Powers js/coaching.js (per-finger corrective hints) and, later, a ghost-hand
# overlay -- both compare the player's live landmarks against this reference.
#
# Left- and right-hand samples for the same letter are mirror images of each
# other (roughly opposite sign on x), so they're kept as separate references
# rather than averaged together -- averaging mirrored geometry would cancel
# out exactly the asymmetric detail (e.g. thumb position) coaching needs.
rows_by_label_hand = defaultdict(list)

with open(dataset_path, newline="") as f:
    reader = csv.DictReader(f)
    feature_cols = [c for c in reader.fieldnames if c not in ("label", "hand")]
    for row in reader:
        hand = row["hand"].strip().lower()
        rows_by_label_hand[(row["label"], hand)].append([float(row[c]) for c in feature_cols])

references = defaultdict(dict)
for (label, hand), rows in rows_by_label_hand.items():
    mean_vec = np.mean(np.array(rows, dtype=np.float64), axis=0)
    references[label][hand] = [round(v, 5) for v in mean_vec.tolist()]

os.makedirs(os.path.dirname(out_path), exist_ok=True)
with open(out_path, "w") as f:
    json.dump(references, f, indent=1, sort_keys=True)

print("Wrote", out_path)
print("Letters:", sorted(references.keys()))
one_letter = next(iter(references.values()))
print("Hands per letter:", sorted(one_letter.keys()))
print("Vector length:", len(next(iter(one_letter.values()))))
