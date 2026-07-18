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
| 0.35 | 0.9834 | 0.9823 | 0.9828 |
| 0.40 | 0.9836 | 0.9819 | 0.9828 |
| 0.45 | 0.9838 | 0.9816 | 0.9827 |
| 0.50 (default) | 0.9839 | 0.9814 | 0.9826 |
| 0.55 | 0.9841 | 0.9809 | 0.9825 |
| 0.60 | 0.9843 | 0.9806 | 0.9825 |
| 0.65 | 0.9847 | 0.9802 | 0.9824 |
| 0.70 | 0.9850 | 0.9796 | 0.9823 |

PII F1 varies by only 0.0006 across the full sweep range, meaning the
decision boundary is not sensitive to threshold choice in this range. The
best threshold found (0.30) improves PII F1 over default by only 0.0003,
which is within noise. As expected, precision and recall trade off in the
usual direction as threshold increases (recall drops from 0.9826 to
0.9796, precision rises from 0.9831 to 0.9850).

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
| Precision | 0.60 | 0.81 | — |
| Recall | 0.61 | 0.80 | — |
| F1 | 0.60 | 0.80 | 0.70 |

Overall accuracy: 0.74. False positives: 382. False negatives: 406. Zero
failed API calls across all runs (500-sample, 3,000-sample, and the
post-fix 3,000-sample rerun below).

Results were consistent between the initial 500-sample run and the
3,000-sample run at the original prompt (PII F1 0.80 both times, macro F1
0.65 both times), validating that 500 samples would have been
statistically sufficient, but 3,000 gives stronger confidence for the
paper.

**Placeholder-fix rerun (2026-07-08):** after identifying the masking
placeholder false-positive pattern below, the prompt was updated with an
explicit instruction that bracketed placeholder tokens (e.g. `[IPV4_1]`)
represent already-redacted data, not real PII, plus a worked example
demonstrating the correct empty-array output. This is a targeted,
root-cause-specific fix, distinct from the earlier general "be more
conservative" instruction that did not resolve this pattern (see prompt
iteration note below). The fix was validated on isolated placeholder
examples (`[IPV4_1]`, `[EMAIL_1]`, `[URL_1]`, `[PHONE_1]`) before being
rolled into a full 3,000-sample rerun.

| Metric | Before fix | After fix | Change |
|---|---|---|---|
| Safe F1 | 0.50 | 0.60 | +0.10 |
| Safe recall | 0.44 | 0.61 | +0.17 |
| PII F1 | 0.80 | 0.80 | unchanged |
| PII precision | 0.76 | 0.81 | +0.05 |
| Macro F1 | 0.65 | 0.70 | +0.05 |
| Accuracy | 0.71 | 0.74 | +0.03 |
| False positives | 547 | 382 | -165 |
| False negatives | 326 | 406 | +80 |

The fix substantially reduced false positives, directly addressing the
placeholder over-triggering pattern, at the cost of a moderate increase in
false negatives, consistent with the model becoming somewhat more
conservative overall rather than purely more accurate on placeholders
specifically. Net effect is a meaningful macro F1 and accuracy gain. All
results and figures elsewhere in this document reflect the post-fix
numbers unless otherwise noted.

### 2.3 Error Analysis

**False positives (382, post-fix):** Prior to the placeholder fix, false
positives were dominated by masking placeholder tokens (e.g. `[IPV4_1]`)
misread as real PII — the model saw "IP" and a bracketed placeholder shape
and flagged it, even though the ground truth treats already-masked
placeholders as non-PII. This alone accounted for a meaningful share of
the `ip_address` and `url` false positive volume (142 and 123 occurrences
respectively across all false positive detections at the original
prompt). The prompt fix described above largely resolved this pattern.
Remaining false positives are dominated by generic business/organizational
URLs (receipt links, company sites) flagged as personal, without reasoning
about whether the URL actually identifies an individual.

**False negatives (406, post-fix):** Cluster around:
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
6. A modest increase in missed cases overall, attributable to the
   placeholder fix making the model somewhat more conservative in
   borderline cases, not limited to placeholder tokens specifically.

---

## 3. Cross-Model Comparison

| | DistilBERT (fine-tuned) | Phi-4 (zero-shot, post-fix) |
|---|---|---|
| Macro F1 | 0.97 | 0.70 |
| Accuracy | 0.97 | 0.74 |
| Eval size | Full test set (32,543) | Stratified subset (3,000) |
| False positive driver | Ambiguous ground truth on financial/vehicle IDs | Generic business URLs (placeholder pattern resolved via prompt fix) |
| False negative driver | Placeholder tokens, dense numeric strings | Label gaps (passwords, contract numbers), non-English context |
| Shared weakness | Masking placeholder tokens (opposite direction: under-flagged) | Resolved for Phi-4 via targeted prompt fix (see §2.2); DistilBERT's placeholder handling unchanged |

**Key cross-model finding:** both models initially showed sensitivity to
masking placeholder artifacts, but in opposite ways. Phi-4 over-flagged
placeholders such as `[IPV4_1]` as if they were real PII, creating false
positives when the ground truth treats already-masked placeholders as
safe. DistilBERT's placeholder-related errors appear in cases where
placeholder-like or dense structured tokens occur in examples labeled as
PII but lack enough surrounding context for the classifier to detect them
reliably. This confirmed the placeholder issue was a genuine
dataset/preprocessing artifact rather than a weakness unique to one model
(independently identified by Fil's EDA as a P1 team-wide fix). Phi-4's
side of this was resolved with a targeted prompt fix (§2.2); DistilBERT's
side remains an open dataset-level consideration, since it is a training
data pattern rather than an inference-time prompt adjustment.

**Note on comparability:** because DistilBERT was evaluated on the full
cleaned test split and Phi-4 was evaluated on a 3,000-row stratified
subset, the comparison above should be interpreted directionally unless we
also report DistilBERT on the same Phi-4 subset.

**Latency comparison:** average per-prompt inference time was measured
over a 100-sample draw from the frozen test set (seed=42), timing single
predictions one at a time to match real single-prompt usage rather than
batched throughput.

| Model | Mean | Median | Min | Max |
|---|---|---|---|---|
| DistilBERT | 2.4 ms | 2.3 ms | 2.1 ms | 7.5 ms |
| Phi-4 | 688.8 ms | 736.0 ms | 174.3 ms | 1564.9 ms |

DistilBERT is roughly 287x faster on average. Phi-4's latency is also
substantially more variable (174-1565 ms range vs. DistilBERT's tight
2.1-7.5 ms range), reflecting inherent unpredictability in local LLM
inference that a lightweight transformer does not share. This latency gap
is a meaningful part of the practical case for a specialized screening
model, independent of the accuracy gap already discussed above.

**Supporting the project thesis:** these results support the project's
core direction — for this narrow PII-detection task, a fine-tuned
specialized model substantially outperforms a zero-shot general-purpose
LLM baseline on classification metrics, even after a targeted, good-faith
prompt fix narrowed the gap from 0.65 to 0.70 macro F1 against
DistilBERT's 0.97. Broader claims about tradeoff should also consider
runtime, cost, privacy, and deployment complexity. The two models'
distinct error signatures (label ambiguity vs. shape-based
over-triggering, now partially resolved) also make for a substantive error
analysis section rather than a simple "one model is better" comparison.

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
- Targeted Phi-4 prompt fix for the placeholder/URL false positive
  pattern (2026-07-08), reran full 3,000-sample eval: macro F1 0.65 →
  0.70, false positives 547 → 382

**Open / in progress:**
- Stretch comparison against a frontier hosted model (Claude Opus 4.8) via
  Fil's self-hosted wrapper, subset and prompt already prepared and shared
  with Fil
- Possible smaller general-purpose LLM baseline (Francisco's suggestion) to
  add a model-size dimension to the comparison, not yet started
- `week4-integration` branch is live; local repo not yet switched over
