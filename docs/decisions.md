# CYBER 207 FINAL PROJECT Decision Log

> **Format (team convention):** every protocol decision gets a short entry —
> **Chose / Rejected / Why** — so the final paper can reconstruct *why* the channel
> looks the way it does. Keep each entry tight. Add a one-line meta header per entry.
>
> Project: *Specialized PII Detection Comparison* (CYBER 207, Summer 2026).
> Channel: **Lightweight ML models using TF-IDF and privacy-pattern features can detect sensitive prompts with strong recall and may provide a practical, lower-cost screening layer compared to an LLM-only classifier**
> Legend: ✅ decided · 🔁 revisit later · 🧪 stretch/optional.

## Core Model Design

## DistilBERT (Transformer)

**Model selection** ✅
Chose: `distilbert-base-uncased`
Rejected: `distilbert-base-multilingual-cased`
Why: Dataset is predominantly English; base model is faster to fine-tune and easier to reproduce. Multilingual support documented as a limitation.

**Best model metric** ✅
Chose: `f1_pii` as the primary metric for checkpoint selection
Rejected: accuracy
Why: Class imbalance (67/33) makes accuracy misleading. False negatives (sensitive prompts missed) are the primary risk in a PII detector.

**Sequence length** ✅
Chose: max_length=128 tokens
Rejected: 256 or 512
Why: EDA showed average prompt length ~157 chars. 128 covers the vast majority of examples with lower memory and compute cost.

**Mixed precision** ✅
Chose: `fp16=torch.cuda.is_available()`
Rejected: `fp16=True` (hardcoded)
Why: Hardcoded fp16 fails on CPU. Auto-detection ensures reproducibility across machines.
