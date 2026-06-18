# CYBER 207 Final Project Plan

Alan Jiang, Aura Gaines, Brandon Shumack, and Fil Dziembowski

## 1. Project Overview

We are building and comparing models that classify user-written LLM prompts as either safe or privacy-sensitive before they are submitted to an external LLM.

This is a binary classification project:

0 = safe / no PII
1 = privacy-sensitive / contains PII

The security motivation is that users may accidentally paste sensitive personal information into external LLM tools. A screening model could help detect risky prompts before that data leaves the user or organization.

## 2. Research Question and Hypothesis

Research question: Can machine learning models detect privacy-sensitive prompts well enough to act as a pre-submission screening layer before external LLM use?

Hypothesis: Lightweight ML models using TF-IDF and privacy-pattern features can detect sensitive prompts with strong recall and may provide a practical, lower-cost screening layer compared to an LLM-only classifier.

## 3. Dataset

Dataset: AI4Privacy PII-Masking-400k
Source: Hugging Face
Total examples: 325,517

We are using the Hugging Face version because it loads directly through the Python `datasets` library and avoids manual download steps. This is the same AI4Privacy dataset family originally discussed by the group.

The dataset contains text examples with token-level privacy/PII labels. For this project, we collapse those token-level labels into a document-level binary label so the task becomes safe vs privacy-sensitive prompt classification.

Frozen split details:

- Train/validation/test split: 80/10/10
- Random seed: 42
- Stratified split: yes
- Class balance: 67.4% sensitive / 32.6% safe
- Split metadata file: `data_splits/split_metadata.json`

All model work should use the same frozen split so our results are comparable.

## 4. Labeling Plan

We use binary labels:

0 = safe / no PII
1 = privacy-sensitive / contains PII

The label-collapse logic is already implemented in `src/data/data_split.py`.

That script uses the dataset’s `mbert_token_classes` field to collapse token-level NER labels into a document-level binary label:

If an example contains any PII token label, it is labeled sensitive.
If an example does not contain PII token labels, it is labeled safe.

This keeps the final task simple and aligned with our project goal: detecting whether a prompt should be flagged before being submitted to an external LLM.

## 5. Grading Rubric Alignment

10% Initial setup, background, hypothesis: explain the LLM privacy risk, dataset, and hypothesis.

10% Problem description: define the task as binary classification of prompts into safe vs privacy-sensitive.

25% Sensible methods: compare classical ML, neural network, DistilBERT, and local LLM baseline using the same frozen train/validation/test split.

20% Feature engineering: define and compare TF-IDF features, basic text features, privacy-pattern regex features, and combined feature sets.

15% Error analysis: analyze false positives and false negatives, with special attention to false negatives because missed PII creates privacy risk.

10% Write-up: final notebook markdown plus supporting documentation in docs/.

10% Overall results: final metrics table, confusion matrices, model comparison, and conclusion.

## 6. Roles and Responsibilities

Aura: Data ingest, label verification, EDA, feature engineering, feature pipeline.

Alan: Classical ML and neural network models.

Brandon: Transformer / LLM benchmark and local LLM comparison.

Fil: Evaluation, write-up, project management, schedule, and final results organization.

## 7. Methods and Models

We plan to compare the following approaches:

Logistic Regression
Naive Bayes
Linear SVM
Feed-forward neural network
DistilBERT
Local LLM baseline using Ollama/Phi

All models should use the same frozen train/validation/test split when possible.

Classical ML models will use engineered features such as TF-IDF and privacy-pattern features. DistilBERT and the local LLM baseline will provide comparison points against the lighter-weight ML approaches.

## 8. Feature Engineering Plan

Because feature engineering is 20% of the grade, we will define feature sets clearly.

Feature Set 1: TF-IDF baseline
Turns prompt text into word/term features for classical ML models.

Feature Set 2: Basic text features
Examples: character length, word count, digit count, uppercase count, special character count.

Feature Set 3: Privacy-pattern features
Examples: email pattern, phone-number pattern, URL pattern, IP address pattern, ID-like number pattern.

Feature Set 4: Combined features
TF-IDF plus basic text features plus privacy-pattern features.

Important: features should be created from the prompt text only, not from the original PII entity labels, to avoid data leakage.

## 9. Evaluation Plan

We will evaluate models using:

Accuracy
Precision
Recall
F1 score
Confusion matrix

For this project, recall for the sensitive class is especially important because a false negative means the model marked a sensitive prompt as safe.

## 10. Error Analysis Plan

We will review examples where models make mistakes.

Important error types:
False negative: sensitive prompt predicted as safe. This is the highest-risk privacy error.

False positive: safe prompt predicted as sensitive. This may create user friction but is usually less harmful than missing PII.

We will look for patterns in missed examples, such as names, emails, phone numbers, addresses, IDs, short prompts, or unusual formatting.

## 11. Weekly Timeline

W1: setup, repo structure, dataset access, frozen split, project plan

W2: EDA and feature engineering

W3: classical ML baselines

W4: neural network, DistilBERT, and LLM baseline

W5: evaluation and error analysis

W6: final notebook/write-up and presentation draft

W7: final presentation and rehearsal


## 12. GitHub / Repo Workflow

Do not commit raw data.

Use the frozen train/validation/test split.

Use `notebooks/` for notebooks.

Use `src/data/` for data scripts.

Use `src/features/` for feature engineering code.

Use `results/` for metrics, figures, and model comparison outputs.

Use `docs/` for project planning and write-up support.

Do not commit API keys, tokens, secrets, or local environment files.

## 13. Final Deliverables

Final deliverables:

8–10 minute in-class presentation
Up to 5 minutes Q&A
Final Colab/Jupyter notebook with explanatory write-up
Results and analysis included inline or attached
