# CYBER 207 FINAL PROJECT Decision Log

> **Format (team convention):** every protocol decision gets a short entry —
> **Chose / Rejected / Why** — so the final paper can reconstruct *why* the channel
> looks the way it does. Keep each entry tight. Add a one-line meta header per entry.
>
> Project: *Specialized PII Detection Comparison* (CYBER 207, Summer 2026).
> Channel: **Lightweight ML models using TF-IDF and privacy-pattern features can detect sensitive prompts with strong recall and may provide a practical, lower-cost screening layer compared to an LLM-only classifier**
> Legend: ✅ decided · 🔁 revisit later · 🧪 stretch/optional.



## DistilBERT (Transformer)

### D001 **Model selection** ✅
- **Chose**: `distilbert-base-multilingual-cased`
- **Rejected**: `distilbert-base-uncased`
- **Why**:  Dataset is 79% non-English (Italian, French, German, Spanish, Dutch). Token length analysis showed mean 52.5 tokens, std 24.9. Multilingual model improved test PII F1 from 0.973 to 0.983 and reduced false negatives from 624 to 447.

### D002 **Best model metric** ✅
- **Chose**: `f1_pii` as the primary metric for checkpoint selection
- **Rejected**: accuracy
- **Why**: Class imbalance (67/33) makes accuracy misleading. False negatives (sensitive prompts missed) are the primary risk in a PII detector.

### D003 **Sequence length** ✅
- **Chose**: max_length=256 tokens
- **Rejected**: 128 (truncates 1.12% of examples), 512 (only 0.02% of examples exceed 256)
- **Why**: Token analysis showed mean=52.5, std=24.9. 256 covers 99.98% of examples with lower compute cost than 512.

### D004 **Mixed precision** ✅
- **Chose**: `fp16=torch.cuda.is_available()`
- **Rejected**: `fp16=True` (hardcoded)
- **Why**: Hardcoded fp16 fails on CPU. Auto-detection ensures reproducibility across machines.

## Core Model Design

### D005 - Per-language stats breakout add "locale" to dataset 🧪 
- **Chose**: Push decision as a stretch goal for decision in **Week 5** 
- **Rejected**: adding "locale" to dataset from hugginface 
- **Why**: Through adding per-language breakdown would strengthen the final evalutation espcially using multilingual model; breaking down the model now would distract for current scheduled task of features work.
