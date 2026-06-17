"""
CYBER 207 - Final Project
Data Preparation and Split Script
Team: Brandon Shumack, Fil Dziembowski, Aura Gaines, Alan Jiang

Seed: 42 (fixed for reproducibility)
Run once, commit outputs to GitHub. Both teams evaluate on the same test set.
"""

import pandas as pd
import numpy as np
from datasets import load_dataset
from sklearn.model_selection import train_test_split
import os
import json

SEED = 42
OUTPUT_DIR = "data_splits"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Loading AI4Privacy dataset...")
dataset = load_dataset("ai4privacy/pii-masking-400k", split="train")
df = dataset.to_pandas()
print(f"  Loaded {len(df):,} examples")
print(f"  Columns: {list(df.columns)}\n")

def is_sensitive(row):
    tags = row.get('mbert_token_classes', None)
    if tags is not None:
        if isinstance(tags, (list, tuple, np.ndarray)):
            return int(any(str(t).strip() != 'O' for t in tags))
    mask = row.get('privacy_mask', None)
    if mask is not None:
        if isinstance(mask, (list, tuple)):
            return int(len(mask) > 0)
        if isinstance(mask, str):
            stripped = mask.strip()
            if stripped in ['[]', '', 'null', 'None']:
                return 0
            try:
                return int(len(json.loads(stripped)) > 0)
            except Exception:
                return 1
    return 0

print("Converting token-level NER labels to binary document-level labels...")
df['label'] = df.apply(is_sensitive, axis=1)

sensitive_count = int(df['label'].sum())
safe_count = len(df) - sensitive_count
print(f"  Sensitive (1): {sensitive_count:,} ({sensitive_count/len(df)*100:.1f}%)")
print(f"  Safe      (0): {safe_count:,} ({safe_count/len(df)*100:.1f}%)")
print()

if sensitive_count == 0:
    print("WARNING: No sensitive examples found. Check mbert_token_classes parsing.")
    raise SystemExit(1)

df_core = df[['source_text', 'label']].copy()
df_core.columns = ['text', 'label']
df_core['original_index'] = df.index
df_core = df_core.dropna(subset=['text'])
print(f"{len(df_core):,} examples after null drop.\n")

print("Splitting data (stratified, seed=42)...")
train_df, temp_df = train_test_split(
    df_core, test_size=0.20, random_state=SEED, stratify=df_core['label']
)
val_df, test_df = train_test_split(
    temp_df, test_size=0.50, random_state=SEED, stratify=temp_df['label']
)

print(f"  Train: {len(train_df):,} ({len(train_df)/len(df_core)*100:.1f}%)")
print(f"  Val:   {len(val_df):,} ({len(val_df)/len(df_core)*100:.1f}%)")
print(f"  Test:  {len(test_df):,} ({len(test_df)/len(df_core)*100:.1f}%)")
print()

print("Class balance verification:")
for name, split in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
    s = int(split['label'].sum())
    total = len(split)
    print(f"  {name}: {s:,} sensitive ({s/total*100:.1f}%), {total-s:,} safe ({(total-s)/total*100:.1f}%)")
print()

print("Saving splits...")
train_df.to_parquet(f"{OUTPUT_DIR}/train.parquet", index=False)
val_df.to_parquet(f"{OUTPUT_DIR}/val.parquet", index=False)
test_df.to_parquet(f"{OUTPUT_DIR}/test.parquet", index=False)
test_df.to_csv(f"{OUTPUT_DIR}/test.csv", index=False)
val_df.to_csv(f"{OUTPUT_DIR}/val.csv", index=False)

metadata = {
    "seed": SEED,
    "split_ratios": {"train": 0.80, "val": 0.10, "test": 0.10},
    "total_examples": len(df_core),
    "train_size": len(train_df),
    "val_size": len(val_df),
    "test_size": len(test_df),
    "label_logic": "sensitive=1 if any mbert_token_classes tag != O, else safe=0",
    "class_balance": {
        "train": {"sensitive": int(train_df['label'].sum()), "safe": int((train_df['label']==0).sum())},
        "val":   {"sensitive": int(val_df['label'].sum()),   "safe": int((val_df['label']==0).sum())},
        "test":  {"sensitive": int(test_df['label'].sum()),  "safe": int((test_df['label']==0).sum())},
    }
}
with open(f"{OUTPUT_DIR}/split_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print(f"\nAll files saved to ./{OUTPUT_DIR}/")
print("  train.parquet")
print("  val.parquet")
print("  test.parquet")
print("  test.csv       (for cross-team LLM benchmarking)")
print("  val.csv")
print("  split_metadata.json")
print("\nDone. Commit data_splits/ to GitHub before either team begins modeling.")
