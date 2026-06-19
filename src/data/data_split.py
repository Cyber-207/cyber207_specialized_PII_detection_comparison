"""
CYBER 207: AI and ML in Cybersecurity, Summer 2026
Final Project - Data Preparation and Split Script

Team: Alan Jiang, Aura Gaines, Brandon Shumack, Fil Dziembowski

What this script does:
    1. Downloads the AI4Privacy PII-Masking-400k dataset from Hugging Face
    2. Converts token-level NER labels into binary document-level labels:
           sensitive (1) = the text contains at least one PII token
           safe      (0) = the text contains no PII tokens
    3. Creates a stratified 80/10/10 train/validation/test split
    4. Saves the splits as parquet files (fast, efficient) and CSV files
       (human-readable, for cross-team use and LLM benchmarking)
    5. Saves a metadata JSON file documenting the split configuration

Why we use a frozen split:
    All models (classical ML, neural network, DistilBERT, local LLM) must
    be evaluated on the exact same test examples so results are comparable.
    Run this script ONCE, commit split_metadata.json, and everyone uses the
    same data.

How to run:
    python3 src/data/data_split.py

Output files (saved to ./data_splits/):
    train.parquet          - 80% of data, used to train models
    val.parquet            - 10% of data, used to tune hyperparameters
    test.parquet           - 10% of data, used for final evaluation only
    test.csv               - same as test.parquet but in CSV format
    val.csv                - same as val.parquet but in CSV format
    split_metadata.json    - documents the split config and class balance

Note: parquet and CSV files are git-ignored (too large to commit).
      Only split_metadata.json is committed to the repo.
      Anyone who needs the split files should run this script locally.
"""

import pandas as pd        # for dataframe manipulation
import numpy as np         # for array type checking (numpy.ndarray)
from datasets import load_dataset   # Hugging Face datasets library
from sklearn.model_selection import train_test_split  # for splitting data
import os                  # for creating output directories
import json                # for saving metadata as JSON

# =============================================================================
# Configuration
# =============================================================================

# Fixed random seed ensures the same split every time the script is run.
# Everyone on the team must use this seed to get identical splits.
SEED = 42

# Output directory for all split files
OUTPUT_DIR = "data_splits"

# Create the output directory if it doesn't already exist
os.makedirs(OUTPUT_DIR, exist_ok=True)


# =============================================================================
# Step 1: Load the dataset from Hugging Face
# =============================================================================
# The AI4Privacy PII-Masking-400k dataset contains text examples where
# each token (word) has been labeled with a PII category or marked as 'O'
# (meaning it is not PII). We use the 'train' split, which contains all
# 325,517 examples. We will create our own train/val/test split below.

print("Loading AI4Privacy dataset...")
dataset = load_dataset("ai4privacy/pii-masking-400k", split="train")

# Convert the Hugging Face dataset to a pandas DataFrame for easier processing
df = dataset.to_pandas()

print(f"  Loaded {len(df):,} examples")
print(f"  Columns: {list(df.columns)}\n")

# Key columns in this dataset:
#   source_text         - the original text prompt (what we classify)
#   mbert_token_classes - list of NER tags for each token, e.g.:
#                         ['O', 'O', 'B-USERNAME', 'I-USERNAME', 'O', ...]
#                         'O' means the token is NOT PII
#                         anything else (B-EMAIL, B-PHONE, etc.) means PII
#   privacy_mask        - list of dicts identifying PII spans (fallback)
#   masked_text         - version of source_text with PII replaced by labels


# =============================================================================
# Step 2: Convert token-level NER labels to binary document-level labels
# =============================================================================
# The dataset labels individual tokens (words), but we need one label per
# document (text example). We collapse the token labels using this rule:
#
#   IF any token in the text has a label other than 'O' -> label = 1 (sensitive)
#   IF all tokens have label 'O'                        -> label = 0 (safe)
#
# This is a conservative approach: any presence of PII = sensitive.
# This makes sense for a DLP (Data Loss Prevention) use case where we
# want to minimize false negatives (missing PII is riskier than over-flagging).
#
# Primary source:  mbert_token_classes (confirmed column in this dataset)
# Fallback source: privacy_mask (used if mbert_token_classes is unavailable)

def is_sensitive(row):
    """
    Returns 1 if the text contains any PII token, 0 if it is clean.

    Parameters:
        row: a single row from the pandas DataFrame

    Returns:
        int: 1 (sensitive / contains PII) or 0 (safe / no PII)
    """

    # --- Primary method: check mbert_token_classes ---
    # This column contains a numpy array of NER tag strings for each token.
    # Example: ['O', 'O', 'B-USERNAME', 'I-USERNAME', 'O', 'B-EMAIL', ...]
    # We check if any tag is NOT 'O'. If so, PII is present.
    tags = row.get('mbert_token_classes', None)
    if tags is not None:
        # The column stores values as numpy.ndarray (not a regular Python list)
        # so we explicitly check for that type
        if isinstance(tags, (list, tuple, np.ndarray)):
            return int(any(str(t).strip() != 'O' for t in tags))

    # --- Fallback method: check privacy_mask ---
    # privacy_mask is a list of dicts identifying PII spans.
    # Example: [{'label': 'USERNAME', 'start': 12, 'end': 25}, ...]
    # An empty list means no PII. A non-empty list means PII is present.
    mask = row.get('privacy_mask', None)
    if mask is not None:
        if isinstance(mask, (list, tuple)):
            # Non-empty list = PII present
            return int(len(mask) > 0)
        if isinstance(mask, str):
            # Sometimes stored as a JSON string -- parse it
            stripped = mask.strip()
            if stripped in ['[]', '', 'null', 'None']:
                return 0
            try:
                return int(len(json.loads(stripped)) > 0)
            except Exception:
                # If we can't parse it and it's not empty, assume sensitive
                return 1

    # If neither column is available, default to safe (0)
    return 0


print("Converting token-level NER labels to binary document-level labels...")
df['label'] = df.apply(is_sensitive, axis=1)

# Report class distribution
sensitive_count = int(df['label'].sum())
safe_count = len(df) - sensitive_count
print(f"  Sensitive (1): {sensitive_count:,} ({sensitive_count/len(df)*100:.1f}%)")
print(f"  Safe      (0): {safe_count:,} ({safe_count/len(df)*100:.1f}%)")
print()

# Safety check: if no sensitive examples were found, something went wrong
if sensitive_count == 0:
    print("WARNING: No sensitive examples found. Check mbert_token_classes parsing.")
    raise SystemExit(1)


# =============================================================================
# Step 3: Select and clean the columns we need
# =============================================================================
# We only keep three columns for the split files:
#   text           - the original prompt text (renamed from source_text)
#   label          - our binary label (0 = safe, 1 = sensitive)
#   original_index - the row number in the original dataset
#
# original_index is important: it lets us re-join back to mbert_token_classes
# later for EDA and error analysis (e.g. which PII categories were missed).

df_core = df[['source_text', 'label']].copy()
df_core.columns = ['text', 'label']
df_core['original_index'] = df.index  # preserve link back to original dataset

# Drop any rows where the text is null/missing
before = len(df_core)
df_core = df_core.dropna(subset=['text'])
dropped = before - len(df_core)
if dropped > 0:
    print(f"Dropped {dropped} rows with null text.")
print(f"{len(df_core):,} examples remaining after null check.\n")
# Remove blank text rows and exact duplicate text rows before splitting
# This cleanup comes from the EDA recommendation and reduces duplicate-text leakage risk

before_cleaning = len(df_core)

blank_text_rows_removed = int((df_core["text"].fillna("").str.strip() == "").sum())
duplicate_text_rows_before_blank_removal = int(df_core["text"].duplicated().sum())

# Remove blank text rows first
df_core = df_core[df_core["text"].fillna("").str.strip() != ""].copy()

# Then remove exact duplicate text rows, keeping the first copy
duplicate_text_rows_removed_after_blank_removal = int(df_core["text"].duplicated().sum())
df_core = df_core.drop_duplicates(subset=["text"], keep="first").copy()

total_rows_removed_by_cleaning = before_cleaning - len(df_core)

print("Cleaning before split:")
print(f"  Blank text rows removed: {blank_text_rows_removed}")
print(f"  Duplicate text rows before blank removal: {duplicate_text_rows_before_blank_removal}")
print(f"  Duplicate text rows removed after blank removal: {duplicate_text_rows_removed_after_blank_removal}")
print(f"  Total rows removed by cleaning: {total_rows_removed_by_cleaning}")
print(f"  Examples remaining after cleaning: {len(df_core):,}\n")

# =============================================================================
# Step 4: Create the stratified 80/10/10 train/validation/test split
# =============================================================================
# Stratified split means the class balance (67.4% sensitive / 32.6% safe)
# is preserved in each split. This prevents any one split from being
# accidentally dominated by one class.
#
# We do this in two steps:
#   Step A: split into 80% train and 20% temp
#   Step B: split the 20% temp evenly into 10% val and 10% test

print("Splitting data (stratified, seed=42)...")

# Step A: 80% train, 20% temp
train_df, temp_df = train_test_split(
    df_core,
    test_size=0.20,            # 20% goes to temp
    random_state=SEED,         # fixed seed for reproducibility
    stratify=df_core['label']  # preserve class balance
)

# Step B: split temp evenly into val and test (50/50 of the 20% = 10% each)
val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,            # 50% of temp = 10% of total
    random_state=SEED,
    stratify=temp_df['label']
)

print(f"  Train: {len(train_df):,} examples  ({len(train_df)/len(df_core)*100:.1f}%)")
print(f"  Val:   {len(val_df):,} examples  ({len(val_df)/len(df_core)*100:.1f}%)")
print(f"  Test:  {len(test_df):,} examples  ({len(test_df)/len(df_core)*100:.1f}%)")
print()


# =============================================================================
# Step 5: Verify class balance across all three splits
# =============================================================================
# This confirms the stratification worked correctly.
# All three splits should show approximately 67.4% sensitive / 32.6% safe.

print("Class balance verification:")
for name, split in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
    s = int(split['label'].sum())
    total = len(split)
    print(f"  {name}: {s:,} sensitive ({s/total*100:.1f}%), "
          f"{total-s:,} safe ({(total-s)/total*100:.1f}%)")
print()
# =============================================================================
# Extra validation: check for duplicate text overlap across split boundaries
# =============================================================================

# turn each split's text column into a set so we can compare exact text values
train_texts = set(train_df["text"])
val_texts = set(val_df["text"])
test_texts = set(test_df["text"])

# count exact text overlap between each pair of splits
split_overlap = {
    "train_val": len(train_texts.intersection(val_texts)),
    "train_test": len(train_texts.intersection(test_texts)),
    "val_test": len(val_texts.intersection(test_texts)),
}

# print the overlap counts so we can confirm the split is clean
print("Duplicate text overlap check:")
print(f"  Train/Val overlap:  {split_overlap['train_val']}")
print(f"  Train/Test overlap: {split_overlap['train_test']}")
print(f"  Val/Test overlap:   {split_overlap['val_test']}")
print()

# stop the script if the same exact text appears in more than one split
if any(count > 0 for count in split_overlap.values()):
    raise ValueError("Duplicate text overlap found across split boundaries.")

# =============================================================================
# Step 6: Save the split files
# =============================================================================
# We save each split in two formats:
#   parquet - fast and efficient, used for model training
#   CSV     - human-readable, shared with LLM benchmarking lane
#
# IMPORTANT: These files are git-ignored. Do not commit them.
# Only split_metadata.json is committed to the repo.

print("Saving splits...")

# Parquet files (all three splits)
train_df.to_parquet(f"{OUTPUT_DIR}/train.parquet", index=False)
val_df.to_parquet(f"{OUTPUT_DIR}/val.parquet", index=False)
test_df.to_parquet(f"{OUTPUT_DIR}/test.parquet", index=False)

# CSV files (test and val only -- for cross-team access and LLM benchmarking)
test_df.to_csv(f"{OUTPUT_DIR}/test.csv", index=False)
val_df.to_csv(f"{OUTPUT_DIR}/val.csv", index=False)


# =============================================================================
# Step 7: Save split metadata
# =============================================================================
# This JSON file documents exactly how the split was created.
# It is the ONLY split-related file that gets committed to GitHub.
# Anyone on the team can use it to verify their local splits match.

metadata = {
    "seed": SEED,
    "split_ratios": {
        "train": 0.80,
        "val": 0.10,
        "test": 0.10
    },
    "total_examples": len(df_core),
    "cleaning": {
        "null_text_rows_removed": int(dropped),
        "blank_text_rows_removed": int(blank_text_rows_removed),
        "duplicate_text_rows_before_blank_removal": int(duplicate_text_rows_before_blank_removal),
        "duplicate_text_rows_removed_after_blank_removal": int(duplicate_text_rows_removed_after_blank_removal),
        "total_rows_removed_by_cleaning": int(total_rows_removed_by_cleaning)
    },
    "train_size": len(train_df),
    "val_size": len(val_df),
    "test_size": len(test_df),
    "label_logic": (
        "sensitive=1 if any token in mbert_token_classes != 'O', "
        "safe=0 if all tokens are 'O'"
    ),
    "class_balance": {
        "train": {
            "sensitive": int(train_df['label'].sum()),
            "safe": int((train_df['label'] == 0).sum())
        },
        "val": {
            "sensitive": int(val_df['label'].sum()),
            "safe": int((val_df['label'] == 0).sum())
        },
        "test": {
            "sensitive": int(test_df['label'].sum()),
            "safe": int((test_df['label'] == 0).sum())
        }
    },
    "columns_saved": ["text", "label", "original_index"],
    "note": (
        "original_index links back to the full dataset row in "
        "mbert_token_classes for EDA and error analysis by PII category."
    ),
    "split_overlap_check": split_overlap,
    "feature_restrictions": [
        "Do not use token-level labels as model features.",
        "Do not use privacy masks as model features.",
        "Do not use original PII annotations as model features."
    ]
}

with open(f"{OUTPUT_DIR}/split_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)


# =============================================================================
# Done
# =============================================================================

print(f"\nAll files saved to ./{OUTPUT_DIR}/")
print("  train.parquet")
print("  val.parquet")
print("  test.parquet")
print("  test.csv          (for cross-team LLM benchmarking)")
print("  val.csv")
print("  split_metadata.json  (only this file gets committed to GitHub)")
print()
print("Done. Commit data_splits/split_metadata.json before either team begins modeling.")
