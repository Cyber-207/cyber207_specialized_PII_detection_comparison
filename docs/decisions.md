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

## LLM Baseline (Phi-4)

### D006 **Model selection** ✅
- **Chose**: `phi4:14b` via Ollama
- **Rejected**: API-based LLMs (GPT-4, Claude)
- **Why**: Local inference preserves data privacy, no API costs, and the RTX 4090 handles 14B parameters comfortably. Ollama provides a simple REST interface consistent with the project's local-first approach.

### D007 **Inference approach** ✅
- **Chose**: Structured JSON prompt with entity-level output and self-reported confidence scores
- **Rejected**: Binary yes/no prompt
- **Why**: Entity-level output provides richer comparison data (per-category breakdowns, confidence scores) without meaningful added complexity. Self-reported confidence scores give a numeric metric for comparison without requiring API log probabilities.

### D008 **Temperature** ✅
- **Chose**: `temperature=0.0`
- **Rejected**: Default temperature
- **Why**: Deterministic output ensures reproducibility across runs and eliminates variance from sampling as a confound in evaluation results.

### D009 **Eval sample size** ✅
- **Chose**: 500-sample stratified subset (pending team decision to increase to 2,000-3,000)
- **Rejected**: Full 32,552-sample test set
- **Why**: LLM inference is significantly slower per sample than fine-tuned transformer inference. 500 stratified samples preserve the 67/33 PII/safe class distribution and are statistically defensible for a course project. Final sample size pending team alignment.

### D010 **Label set** ✅
- **Chose**: Expanded label set aligned to AI4Privacy PII taxonomy
- **Rejected**: Minimal label set (name, address, dob, phone, email, ssn, mrn, credit_card)
- **Why**: AI4Privacy contains additional PII categories (passport, license, bank_account, routing_number, tax_id, national_id, student_id, ip_address, url) not covered by the minimal set. Expanding improves coverage against the actual evaluation data.
