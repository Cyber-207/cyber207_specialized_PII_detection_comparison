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


- [ ] **Repo layout:** \`notebooks/\` (\`01\`…\`06\` \+ final), \`src/\` (data, features, eval modules), \`data/\` (git-ignored), \`results/\` (metrics CSV/JSON \+ figures), \`docs/\`, \`[readme.md](http://readme.md)\`. [https://github.com/Goat20726/cyber207\_specilized\_PII\_detection\_comparison](https://github.com/Goat20726/cyber207_specilized_PII_detection_comparison)  
- [ ]  **Git**: \`main\` protected; one branch per phase; PR \+ one review before merge; small, frequent commits.  
- [ ]  **Notebook hygiene**: one **canonical** final notebook; phase notebooks stay modular; **runs top-to-bottom with no hidden state**; **set all seeds** (numpy / torch / sklearn); run \`nbstripout\` so output diffs don't pollute git.  
- [ ] **Data**: never commit the dataset; document download steps; save the fixed split \+ seed.  
- [ ]  **Secrets**: API keys via env var / Colab secret — **never in the notebook**; share via a password manager, not git.  
- [ ] **Cadence**: 2 syncs/week **(Mon plan, Thu review)** \+ async standup; the PM role owns the schedule. (what works best Mon after class 30 min??)  
- [ ] **Write-up**: lives inline in the final notebook (markdown cells), mirrored in \`docs/\`.  
- [ ] And Fil use comments\!

|  |  | W1  15 JUN | W2 22 JUN | W3 29JUN | W4 6JUL | W5 13JUL | W6 20JUL | W7 27JUL |
| ----- | :---- | ----- | ----- | ----- | ----- | ----- | ----- | ----- |
| Data  |  |  |  |  |  |  |  |  |
|  | Setup / Data |  |  |  |  |  |  |  |
|  | EDA |  |  |  |  |  |  |  |
|  | Feature Engineering |  |  |  |  |  |  |  |
| AI/ML |  |  |  |  |  |  |  |  |
|  | Classical ML |  |  |  |  |  |  |  |
|  | Neural Net |  |  |  |  |  |  |  |
| LLM  |  |  |  |  |  |  |  |  |
|  | DistilBERT |  |  |  |  |  |  |  |
|  | LLM Baseline |  |  |  |  |  |  |  |
|  | Comparison harness |  |  |  |  |  |  |  |
| Eval |  |  |  |  |  |  |  |  |
|  | Eval & Error |  |  |  |  |  |  |  |
|  | Write Up |  |  |  |  |  |  |  |
|  | Presentation |  |  |  |  |  |  |  |
|  | Schedule |  |  |  |  |  |  |  |