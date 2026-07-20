"""
threshold_sweep.py

Sweeps decision thresholds for the fine-tuned DistilBERT checkpoint against
the validation split, to check whether a threshold other than the default
argmax (0.5) improves PII F1 / recall, matching the validation-threshold
approach Alan and Fil used for the classical models and NN.

Run from the repo root:
    python threshold_sweep.py
"""

import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import precision_recall_fscore_support, classification_report
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ---------------------------------------------------------------------------
# Setup — mirrors the notebook's repo_root resolution
# ---------------------------------------------------------------------------

repo_root = subprocess.run(
    ["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True
).stdout.strip()
repo_root = Path(repo_root)

CHECKPOINT_DIR = repo_root / "results" / "distilbert" / "checkpoint-24408"
TOKENIZER_NAME = "distilbert-base-multilingual-cased"
MAX_LENGTH = 256
VAL_PATH = repo_root / "data_splits" / "val.parquet"

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading tokenizer ({TOKENIZER_NAME})...")
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

print(f"Loading fine-tuned checkpoint from {CHECKPOINT_DIR}...")
model = AutoModelForSequenceClassification.from_pretrained(str(CHECKPOINT_DIR)).to(device)
model.eval()

print(f"Loading validation split from {VAL_PATH}...")
val = pd.read_parquet(VAL_PATH)
print(f"Validation set size: {len(val)}")

# ---------------------------------------------------------------------------
# Get PII-class probabilities for every validation example
# ---------------------------------------------------------------------------

BATCH_SIZE = 64
all_probs = []

texts = val["text"].tolist()
true_labels = val["label"].tolist()  # assumes column is named "label"; adjust if different

print("Running inference on validation set...")
with torch.no_grad():
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]
        inputs = tokenizer(
            batch,
            truncation=True,
            padding=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        ).to(device)
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[:, 1]  # probability of PII class (label=1)
        all_probs.extend(probs.cpu().numpy().tolist())

        if i % (BATCH_SIZE * 20) == 0:
            print(f"  {i}/{len(texts)} processed")

all_probs = np.array(all_probs)
true_labels = np.array(true_labels)

# ---------------------------------------------------------------------------
# Sweep thresholds, report PII precision/recall/F1 at each
# ---------------------------------------------------------------------------

print("\n" + "=" * 70)
print("Threshold sweep (validation set)")
print("=" * 70)
print(f"{'Threshold':>10} {'PII Precision':>15} {'PII Recall':>12} {'PII F1':>10}")

results = []
for threshold in np.arange(0.30, 0.71, 0.05):
    preds = (all_probs >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, preds, labels=[1], average=None, zero_division=0
    )
    results.append((threshold, precision[0], recall[0], f1[0]))
    print(f"{threshold:>10.2f} {precision[0]:>15.4f} {recall[0]:>12.4f} {f1[0]:>10.4f}")

best = max(results, key=lambda r: r[3])
default = next(r for r in results if abs(r[0] - 0.50) < 1e-6)

print("\n" + "=" * 70)
print(f"Default threshold (0.50, argmax-equivalent): PII F1 = {default[3]:.4f}")
print(f"Best threshold found ({best[0]:.2f}): PII F1 = {best[3]:.4f}")
print(f"Difference: {best[3] - default[3]:+.4f}")
print("=" * 70)

if best[3] - default[3] > 0.005:
    print("\nA non-default threshold measurably improves PII F1.")
    print("Worth considering for the final reported configuration.")
else:
    print("\nDefault argmax (0.5) is already at or near optimal.")
    print("Worth stating in the write-up as a deliberate, validated choice.")
