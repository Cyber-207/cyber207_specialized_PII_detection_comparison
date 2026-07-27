"""
distilbert_on_phi4_subset.py

Evaluates the fine-tuned DistilBERT checkpoint on the exact same 3,000-row
stratified subset used for the Phi-4 eval (data_splits/llm_test_subset.parquet),
so the two models can be compared on identical data. Marked non-canonical
since DistilBERT's real reported metrics come from the full test set.

Run from the repo root:
    python distilbert_on_phi4_subset.py
"""

import subprocess
from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import confusion_matrix
from transformers import AutoTokenizer, AutoModelForSequenceClassification

repo_root = subprocess.run(
    ["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True
).stdout.strip()
repo_root = Path(repo_root)

CHECKPOINT_DIR = repo_root / "results" / "distilbert" / "checkpoint-24408"
TOKENIZER_NAME = "distilbert-base-multilingual-cased"
MAX_LENGTH = 256
SUBSET_PATH = repo_root / "data_splits" / "llm_test_subset.parquet"

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading tokenizer ({TOKENIZER_NAME})...")
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

print(f"Loading fine-tuned checkpoint from {CHECKPOINT_DIR}...")
model = AutoModelForSequenceClassification.from_pretrained(str(CHECKPOINT_DIR)).to(device)
model.eval()

print(f"Loading Phi-4's exact subset from {SUBSET_PATH}...")
subset = pd.read_parquet(SUBSET_PATH)
print(f"Subset size: {len(subset)}, PII={sum(subset['label']==1)}, Safe={sum(subset['label']==0)}")

texts = subset["text"].tolist()
true_labels = subset["label"].tolist()

BATCH_SIZE = 64
preds = []

print("Running inference...")
with torch.no_grad():
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]
        inputs = tokenizer(
            batch, truncation=True, padding=True, max_length=MAX_LENGTH, return_tensors="pt"
        ).to(device)
        logits = model(**inputs).logits
        batch_preds = torch.argmax(logits, dim=-1).cpu().numpy().tolist()
        preds.extend(batch_preds)

tn, fp, fn, tp = confusion_matrix(true_labels, preds, labels=[0, 1]).ravel()

print("\n" + "=" * 60)
print("DistilBERT on Phi-4's exact 3,000-row subset (non-canonical)")
print("=" * 60)
print(f"TN={tn}  FP={fp}  FN={fn}  TP={tp}  N={tn+fp+fn+tp}")

accuracy = (tn + tp) / (tn + fp + fn + tp)
precision_pii = tp / (tp + fp) if (tp + fp) > 0 else 0
recall_pii = tp / (tp + fn) if (tp + fn) > 0 else 0
f1_pii = 2 * precision_pii * recall_pii / (precision_pii + recall_pii) if (precision_pii + recall_pii) > 0 else 0

print(f"Accuracy: {accuracy:.4f}")
print(f"PII Precision: {precision_pii:.4f}")
print(f"PII Recall: {recall_pii:.4f}")
print(f"PII F1: {f1_pii:.4f}")

# Save as a row matching model_metrics_full.csv format for easy pasting
result_row = {
    "Model": "DistilBERT (Phi-4 subset)",
    "Family": "Transformer",
    "Features": "sub-word tokens",
    "Eval set": "test-subset (Phi-4's exact 3,000 rows)",
    "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
    "N": int(tn + fp + fn + tp),
    "canonical": False,
}
out_path = repo_root / "results" / "distilbert_on_phi4_subset.csv"
pd.DataFrame([result_row]).to_csv(out_path, index=False)
print(f"\nSaved to {out_path}")
