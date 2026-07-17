# CYBER 207: Transformer/LLM Lead - Full Portion Analysis

**Author:** Brandon Shumack
**Role:** Transformer/LLM Lead
**Project:** Specialized PII Detection Comparison

This document consolidates all work completed to date across both models
owned in this role: the fine-tuned DistilBERT classifier and the Phi-4
zero-shot LLM baseline, including design decisions, results, and error
analysis for each.

---

## 1. DistilBERT (Fine-Tuned Transformer)

### 1.1 Design Decisions

| Decision | Chose | Rejected | Why |
|---|---|---|---|
| Model selection | `distilbert-base-multilingual-cased` | `distilbert-base-uncased` | Dataset is 79% non-English (Italian, French, German, Spanish, Dutch). Multilingual model improved test PII F1 from 0.973 to 0.983 and reduced false negatives from 624 to 447. |
| Checkpoint metric | `f1_pii` | accuracy | Class imbalance (67/33) makes accuracy misleading; false negatives are the primary risk in a PII detector. |
| Sequence length | max_length=256 | 128 / 512 | Token analysis showed mean=52.5, std=24.9. 256 covers 99.98% of examples at lower compute cost than 512. |
| Mixed precision | `fp16=torch.cuda.is_available()` | Hardcoded `fp16=True` | Auto-detection avoids failures on CPU-only environments. |

### 1.2 Results

Evaluated on the full cleaned frozen test split (32,543 rows):

| Metric | Safe | PII | Macro |
|---|---|---|---|
| Precision | 0.96 | 0.98 | — |
| Recall | 0.96 | 0.98 | — |
| F1 | 0.96 | 0.98 | 0.97 |

Overall accuracy: 0.97. False positives: 395. False negatives: 447.

### 1.2.5 Threshold Selection

A validation-set threshold sweep was run (0.30 to 0.70 in steps of 0.05,
32,542 validation examples) to check whether a decision threshold other
than the default argmax (0.5) improves PII F1, matching the
validation-threshold approach used for the classical models and neural
network.

| Threshold | PII Precision | PII Recall | PII F1 |
|---|---|---|---|
| 0.30 | 0.9831 | 0.9826 | 0.9829 |
| 0.40 | 0.9836 | 0.9819 | 0.9828 |
| 0.50 (default) | 0.9839 | 0.9814 | 0.9826 |
| 0.60 | 0.9843 | 0.9806 | 0.9825 |
| 0.70 | 0.9850 | 0.9796 | 0.9823 |

PII F1 varies by only 0.0006 across the full sweep range, meaning the
decision boundary is not sensitive to threshold choice in this range. The
best threshold found (0.30) improves PII F1 over default by only 0.0002,
which is within noise. As expected, precision and recall trade off in the
usual direction as threshold increases (recall drops from 0.9826 to
0.9796, precision rises from 0.9839 to 0.9850).

**Decision:** the default argmax threshold (0.5) is retained as the final,
deliberate choice, validated rather than assumed. Given false negatives
are the primary risk for a pre-submission PII screen, a lower threshold
(e.g. 0.30-0.35) would be a defensible alternative if maximizing recall
specifically were prioritized over F1, since it nudges recall up
marginally at a small precision cost, but this was not adopted since the
F1 difference does not justify moving off the standard default.

### 1.3 Error Analysis

**False positives (395):** Manual review shows many contain what look like
genuinely sensitive identifiers (real-looking account numbers, VIN numbers,
bank routing details) that ground truth marks as safe, most often in
business-context messages. This points more toward ambiguity in the ground
truth labeling for financial/vehicle identifiers in business contexts than
a model weakness.

**False negatives (447):** A recurring pattern is masking placeholder
tokens (e.g. `[IPV6_1]`) being under-flagged, along with dense numeric
strings (long account-like numbers, OTP codes) embedded in business or
technical narrative text without an explicit nearby label cue.

---

## 2. Phi-4 (Zero-Shot LLM Baseline)

### 2.1 Design Decisions

| Decision | Chose | Rejected | Why |
|---|---|---|---|
| Model selection | `phi4:14b` via Ollama | API-based LLMs (GPT-4, Claude) | Local inference preserves data privacy, no API costs, RTX 4090 handles 14B comfortably. |
| Inference approach | Structured JSON prompt, entity-level output, self-reported confidence | Binary yes/no prompt | Entity-level output gives richer comparison data (per-category breakdowns, confidence) at little added complexity. |
| Temperature | 0.0 | Default | Deterministic, reproducible results; removes sampling variance as a confound. |
| Eval sample size | 3,000-sample stratified subset | Full 32,543-sample test set | LLM inference is significantly slower per sample; 3,000 stratified samples (2,021 PII / 979 safe) preserve the 67/33 split and give strong statistical confidence without a 10+ hour run. |
| Label set | Expanded to match AI4Privacy taxonomy | Minimal starter label set | Minimal set omitted categories (passport, bank_account, tax_id, etc.) present in the actual evaluation data. |

Prompt iteration: an initial version was refined with a conservative
instruction and a non-PII example to try to reduce over-flagging. This
moved PII F1 from 0.777 to 0.800 and accuracy from 0.69 to 0.71, but did not
meaningfully change the underlying over-flagging behavior (see error
analysis below) — the fix that would actually move the needle is a
targeted, root-cause-specific prompt change rather than a general
conservatism instruction.

### 2.2 Results

Evaluated on 3,000-sample stratified subset (2,021 PII / 979 safe), test
subset of the cleaned frozen split:

| Metric | Safe | PII | Macro |
|---|---|---|---|
| Precision | 0.57 | 0.76 | — |
| Recall | 0.44 | 0.84 | — |
| F1 | 0.50 | 0.80 | 0.65 |

Overall accuracy: 0.71. False positives: 547. False negatives: 326. Zero
failed API calls across all runs (500-sample and 3,000-sample).

Results were consistent between the initial 500-sample run and the final
3,000-sample run (PII F1 0.80 both times, macro F1 0.65 both times),
validating that 500 samples would have been statistically sufficient, but
3,000 gives stronger confidence for the paper.

### 2.3 Error Analysis

**False positives (547):** Dominated by two shape-based over-triggering
patterns:
1. Masking placeholder tokens (e.g. `[IPV4_1]`) misread as real PII — the
   model sees "IP" and a bracketed placeholder shape and flags it, even
   though the ground truth treats already-masked placeholders as non-PII.
   This alone accounts for a meaningful share of the `ip_address` and `url`
   false positive volume (142 and 123 occurrences respectively across all
   false positive detections).
2. Generic business/organizational URLs (receipt links, company sites)
   flagged as personal, without reasoning about whether the URL actually
   identifies an individual.

**False negatives (326):** Cluster around:
1. Titled/prefixed names (e.g. "Dr Ilkorkor") not recognized as PII.
2. Subtle personal disclosures embedded in narrative text (e.g. a job
   promotion tied to a specific city).
3. Structured IDs and password-like strings that were either outside the
   dataset taxonomy or not represented clearly enough in the prompt label
   definitions.
4. A noticeable skew toward non-English text among reviewed false
   negatives.
5. Ambiguous numeric codes (IBAN-style numbers, zip codes) embedded in
   dense informal text.

---

## 3. Cross-Model Comparison

| | DistilBERT (fine-tuned) | Phi-4 (zero-shot) |
|---|---|---|
| Macro F1 | 0.97 | 0.65 |
| Accuracy | 0.97 | 0.71 |
| Eval size | Full test set (32,543) | Stratified subset (3,000) |
| False positive driver | Ambiguous ground truth on financial/vehicle IDs | Shape-only pattern matching (URLs, IPs, placeholders) |
| False negative driver | Placeholder tokens, dense numeric strings | Label gaps (passwords, contract numbers), placeholder tokens, non-English context |
| Shared weakness | Masking placeholder tokens (opposite direction: under-flagged) | Masking placeholder tokens (opposite direction: over-flagged) |

**Key cross-model finding:** both models show sensitivity to masking
placeholder artifacts, but in opposite ways. Phi-4 often over-flags
placeholders such as `[IPV4_1]` as if they were real PII, creating false
positives when the ground truth treats already-masked placeholders as
safe. DistilBERT's placeholder-related errors appear in cases where
placeholder-like or dense structured tokens occur in examples labeled as
PII but lack enough surrounding context for the classifier to detect them
reliably. This suggests the placeholder issue is a dataset/preprocessing
artifact, not a weakness unique to one model (independently identified by
Fil's EDA as a P1 team-wide fix).

**Note on comparability:** because DistilBERT was evaluated on the full
cleaned test split and Phi-4 was evaluated on a 3,000-row stratified
subset, the comparison above should be interpreted directionally unless we
also report DistilBERT on the same Phi-4 subset.

**Supporting the project thesis:** these results support the project's
core direction — for this narrow PII-detection task, a fine-tuned
specialized model substantially outperforms a zero-shot general-purpose
LLM baseline on classification metrics. Broader claims about tradeoff
should also consider runtime, cost, privacy, and deployment complexity.
The two models' distinct error signatures (label ambiguity vs. shape-based
over-triggering) also make for a substantive error analysis section rather
than a simple "one model is better" comparison.

---

## 4. Current Status and Open Items

**Completed and merged:**
- DistilBERT fine-tuning, evaluation, and decision log entries (D001-D004)
- `src/models/llm_baseline.py` module, calibrated prompt, warmup, raw
  response logging (PR #22, merged)
- 3,000-sample stratified LLM eval and results (PR #24, merged)
- Phi-4 decision log entries (D006-D010)
- PR reviews for Aura's feature engineering notebook (#26) and numeric
  feature scaling follow-up (#28), both approved

**Open / in progress:**
- Team decision pending on whether to pursue a targeted Phi-4 prompt fix
  for the placeholder/URL false positive pattern, or leave results as final
  given a weaker LLM baseline supports the project thesis
- Stretch comparison against a frontier hosted model (Claude Opus 4.8) via
  Fil's self-hosted wrapper, subset and prompt already prepared and shared
  with Fil
- Possible smaller general-purpose LLM baseline (Francisco's suggestion) to
  add a model-size dimension to the comparison, not yet started
- `week4-integration` branch is live; local repo not yet switched over
