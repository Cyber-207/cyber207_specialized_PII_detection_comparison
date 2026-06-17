# CYBER 207: Specialized PII Detection Comparison
Alan Jiang, Aura Gaines, Brandon Shumack, Fil Dziembowski

## Setup
```bash
python3 -m venv cyber207_env
source cyber207_env/bin/activate
pip install datasets scikit-learn pandas pyarrow torch transformers ollama jupyter nbstripout
```

## Data
Dataset: AI4Privacy PII-Masking-400k (Hugging Face)
Never commit raw data. Run the split script once to generate local splits:
```bash
python3 src/data/data_split.py
```
Split: 80/10/10 train/val/test, stratified, seed=42.
Both teams evaluate on the same frozen test set.

## Repo Structure
- `notebooks/` — phase notebooks and final
- `src/data/` — data ingestion and split scripts
- `src/features/` — feature engineering modules
- `src/eval/` — evaluation and metrics modules
- `data_splits/` — split metadata only (data files git-ignored)
- `results/` — metrics CSV/JSON and figures
- `docs/` — writeup mirror
