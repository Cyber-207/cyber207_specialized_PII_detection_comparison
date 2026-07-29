# CYBER 207 FINAL PROJECT Decision Log

> **Format (team convention):** every protocol decision gets a short entry —
> **Chose / Rejected / Why** — so the final paper can reconstruct *why* the channel
> looks the way it does. Keep each entry tight. Add a one-line meta header per entry.
>
> Project: *Specialized PII Detection Comparison* (CYBER 207, Summer 2026).
> Channel: **Lightweight ML models using TF-IDF and privacy-pattern features can detect sensitive prompts with strong recall and may provide a practical, lower-cost screening layer compared to an LLM-only classifier**
> Legend: ✅ decided · 🔁 revisit later · 🧪 stretch/optional.

## Project-Wide Data and Evaluation Design

### D000a **Label conversion** ✅
- **Chose**: Convert token-level PII annotations into a single document-level label (privacy-sensitive if the prompt contains at least one annotated PII token)
- **Rejected**: Token-level (span) classification as the primary task
- **Why**: Matches the intended real-world screening decision, allow the prompt through or flag it, rather than a span-tagging task the project wasn't scoped to evaluate.

### D000b **Data cleaning before split** ✅
- **Chose**: Remove blank and duplicate text before creating the final stratified split
- **Rejected**: Splitting on the raw dataset as downloaded
- **Why**: Prevents duplicate examples from leaking across training and evaluation data, which would inflate reported performance.

### D000c **One frozen split for all models** ✅
- **Chose**: A single seed-42 train/validation/test split shared by every specialized model; Phi-4's stratified subset drawn from that same frozen test set
- **Rejected**: Per-model splits or re-splitting for each notebook
- **Why**: Keeps every model comparison apples-to-apples. Without a shared frozen split, differences in reported metrics could reflect different evaluation data rather than different model quality.

### D000d **Exclude answer-derived features** ✅
- **Chose**: Drop token labels, masking annotations, and original PII category fields from model inputs
- **Rejected**: Using any of the above as engineered features
- **Why**: None of these would be available at real screening time; including them would be data leakage and would overstate every model's real-world performance.

### D000e **Fit transforms on training data only** ✅
- **Chose**: Learn TF-IDF vocabularies and numeric feature scaling from the training split only
- **Rejected**: Fitting on the full dataset before splitting
- **Why**: Standard leakage prevention, validation and test data must stay unseen until evaluation.

### D000f **Evaluation metric priorities** ✅
- **Chose**: Emphasize PII recall and F1, alongside explicit false negative and false positive counts, over raw accuracy
- **Rejected**: Accuracy as the primary reported metric
- **Why**: With a 67/33 class split, accuracy alone conceals both missed PII (the costlier error for a privacy screen) and excessive false-flagging.

### D000g **Operating point selection** ✅
- **Chose**: Select classical-model and MLP decision thresholds via validation-set analysis; retain DistilBERT's default argmax threshold after a validation sweep showed negligible change
- **Rejected**: Using the default 0.5 threshold uniformly without checking sensitivity
- **Why**: Lets each model be compared at its own best operating point rather than an arbitrary shared cutoff. See D004-classical (threshold ≈0.42 for tuned LR) and the MLP's recall-first t=0.02 deployed threshold.

### D000h **Latency measurement scope** ✅
- **Chose**: Measure complete per-prompt response time, including feature extraction or tokenization plus model inference
- **Rejected**: Measuring raw model forward-pass time only
- **Why**: Reflects the delay a user would actually experience during screening, not just an isolated model-inference number that ignores the featurization cost sitting in front of it.

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

### D005 **Per-language leak-rate breakdown** ✅
- **Chose**: Add a heuristic per-language leak-rate table (Section 13.2 of notebook 00) covering French, English, German, Spanish, Italian, Dutch, and other/unknown
- **Rejected**: Reporting only aggregate leak rate across all languages
- **Why**: Strengthens the multilingual-model justification by showing DistilBERT stays under 3% leak rate in every language group, versus 5-13% for the classical models and MLP. Originally logged as a Week 5 stretch goal; completed and merged.

## Core Model Design (Classical and Neural Network)

### D011 **Feature-set ablation** ✅
- **Chose**: Run a controlled ablation isolating the engineered privacy-pattern features from the TF-IDF vocabulary change (results in `results/feature_ablation.csv`)
- **Rejected**: Leaving the "combined beats raw TF-IDF" claim as an unconfirmed future-work item
- **Why**: The original raw-vs-combined comparison confounded a vocabulary change (100k unigram vs 50k uni+bigram) with the added engineered columns. The isolated ablation shows the engineered columns alone add +0.013 test PII F1 over the same 50k TF-IDF baseline (FN 2,066 to 1,841), confirming their contribution independent of the vocabulary change.

### D012 **MLP operating point** ✅
- **Chose**: Deploy the MLP at threshold t=0.02 as a recall-first screen (recall 0.938, 1,350 false negatives)
- **Rejected**: Using the MLP's default t=0.5 threshold (recall 0.863, 3,005 false negatives)
- **Why**: On the identical combined feature space, tuned Logistic Regression already beats the MLP on F1 and precision (0.881 vs 0.871). The MLP's value is the recall-first operating point, cutting missed PII by 27% versus tuned LR at the cost of precision and additional false alarms, a genuinely different point on the tradeoff curve rather than a strictly better model.

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
- **Chose**: 3,000-sample stratified subset (2,021 PII / 979 safe), drawn from the frozen test set
- **Rejected**: 500-sample subset (initial pilot); full 32,543-sample test set
- **Why**: LLM inference is significantly slower per sample than fine-tuned transformer inference, making a full-test-set run impractical (10+ hour estimate). The 500-sample pilot and the 3,000-sample final run produced consistent results (PII F1 0.80 both times), confirming 500 would have been statistically sufficient, but 3,000 gives stronger confidence for the final paper. DistilBERT was subsequently evaluated on this exact same 3,000-row subset (`results/distilbert_on_phi4_subset.csv`) to provide a controlled, same-example comparison rather than a cross-evaluation-size one.

### D010 **Label set** ✅
- **Chose**: Expanded label set aligned to AI4Privacy PII taxonomy
- **Rejected**: Minimal label set (name, address, dob, phone, email, ssn, mrn, credit_card)
- **Why**: AI4Privacy contains additional PII categories (passport, license, bank_account, routing_number, tax_id, national_id, student_id, ip_address, url) not covered by the minimal set. Expanding improves coverage against the actual evaluation data.

### D013 **Placeholder false-positive prompt fix** ✅
- **Chose**: Add a targeted instruction and worked example clarifying that bracketed masking placeholders (e.g. `[IPV4_1]`) represent already-redacted data, not real PII
- **Rejected**: A general "be more conservative" instruction (tried first; moved PII F1 from 0.777 to 0.800 but did not resolve the placeholder pattern specifically)
- **Why**: Error analysis showed placeholder tokens were being misread as real PII, driving a large share of false positives. The targeted fix reduced false positives from 547 to 382 and raised macro F1 from approximately 0.65 to 0.70, at the cost of a false-negative increase (326 to 406) from the model becoming more conservative generally. Because the fix was developed after reviewing errors from the same 3,000-row evaluation subset, the post-fix result is reported with that limitation rather than treated as performance on a fully untouched test set.
