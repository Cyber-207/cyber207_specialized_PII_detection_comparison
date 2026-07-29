"""
latency_benchmark.py

Measures average per-prompt inference latency for DistilBERT and Phi-4,
for Fil's cross-model eval notebook. Runs over a sample of the actual
test set rather than a handful of hand-picked examples, and reports
mean/median/min/max so the numbers are defensible, not illustrative.

Run from the repo root:
    python latency_benchmark.py
"""

import json
import re
import subprocess
import time
from pathlib import Path
from statistics import mean, median

import pandas as pd
import requests
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

repo_root = subprocess.run(
    ["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True
).stdout.strip()
repo_root = Path(repo_root)

TEST_PATH = repo_root / "data_splits" / "test.parquet"
CHECKPOINT_DIR = repo_root / "results" / "distilbert" / "checkpoint-24408"
TOKENIZER_NAME = "distilbert-base-multilingual-cased"
MAX_LENGTH = 256

OLLAMA_URL = "http://localhost:11434/api/generate"
PHI4_MODEL_NAME = "phi4:14b"
OLLAMA_TIMEOUT = 60

N_SAMPLES = 100  # sample size for the timing run
SEED = 42

PHI4_SYSTEM_PROMPT = """\
You are a PII detection assistant evaluating text for the AI4Privacy benchmark dataset.

Identify every piece of personally identifiable information in the input text and emit a JSON array. \
Respond in English only — every label MUST be a lowercase ASCII English word from the allowed list below.

Output format (no commentary, no markdown fences, JSON only):
[{"label": "<english_label>", "value": "<exact_substring_from_input>", "confidence_score": "<0.00-1.00>"}]

If no PII is detected, output an empty array: []

Allowed labels:
name, username, email, phone, address, dob, ssn, passport, license,
credit_card, bank_account, routing_number, tax_id, national_id,
student_id, ip_address, url, mrn

Important: text may contain masking placeholder tokens in square brackets, \
such as [IPV4_1], [EMAIL_1], [URL_2], or [PHONE_1]. These placeholders \
represent data that has ALREADY been redacted upstream. Do NOT treat a \
placeholder token itself as PII, since it is not the original sensitive \
value, it is a stand-in marker. Only flag PII that appears as actual, \
unmasked text in the input.

Context:
You are evaluating text samples for PII content. Each sample is a short \
prompt or message that may or may not contain sensitive personal information. \
Confidence scores reflect your certainty that the identified substring is \
genuinely PII of the labeled type, not a general measure of detection quality.

Examples
--------
Input:
Patient Jane Doe, DOB 1978-03-12, lives at 100 Main St, phone 555-0100.

Output:
[
  {"label": "name",    "value": "Jane Doe",    "confidence_score": "0.95"},
  {"label": "dob",     "value": "1978-03-12",  "confidence_score": "0.99"},
  {"label": "address", "value": "100 Main St", "confidence_score": "0.72"},
  {"label": "phone",   "value": "555-0100",    "confidence_score": "0.88"}
]

Input:
The quarterly earnings report showed a 12% increase in revenue.

Output:
[]

Input:
The server IP is [IPV4_1] and should be reachable from the VPN.

Output:
[]\
"""


def _call_ollama(text: str) -> str:
    payload = {
        "model": PHI4_MODEL_NAME,
        "prompt": f"Input:\n{text}\n\nOutput:",
        "system": PHI4_SYSTEM_PROMPT,
        "stream": False,
        "options": {"temperature": 0.0, "num_predict": 512},
    }
    resp = requests.post(OLLAMA_URL, json=payload, timeout=OLLAMA_TIMEOUT)
    resp.raise_for_status()
    return resp.json().get("response", "").strip()


def report(name, timings):
    print(f"\n{name}")
    print(f"  n={len(timings)}")
    print(f"  mean:   {mean(timings):.1f} ms")
    print(f"  median: {median(timings):.1f} ms")
    print(f"  min:    {min(timings):.1f} ms")
    print(f"  max:    {max(timings):.1f} ms")
    return {
        "model": name,
        "n": len(timings),
        "mean_ms": round(mean(timings), 1),
        "median_ms": round(median(timings), 1),
        "min_ms": round(min(timings), 1),
        "max_ms": round(max(timings), 1),
    }


# ---------------------------------------------------------------------------
# Load sample
# ---------------------------------------------------------------------------

print(f"Loading test set and sampling {N_SAMPLES} rows (seed={SEED})...")
test = pd.read_parquet(TEST_PATH)
sample = test.sample(n=N_SAMPLES, random_state=SEED)
texts = sample["text"].tolist()

# ---------------------------------------------------------------------------
# DistilBERT timing
# ---------------------------------------------------------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\nLoading DistilBERT checkpoint ({device})...")
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
model = AutoModelForSequenceClassification.from_pretrained(str(CHECKPOINT_DIR)).to(device)
model.eval()

# warmup (exclude first-call overhead from the timed sample)
with torch.no_grad():
    warm_inputs = tokenizer("warmup", return_tensors="pt").to(device)
    model(**warm_inputs)

print(f"Timing DistilBERT on {N_SAMPLES} prompts (one at a time, matching real usage)...")
distilbert_timings = []
with torch.no_grad():
    for text in texts:
        start = time.perf_counter()
        inputs = tokenizer(
            text, truncation=True, padding=True, max_length=MAX_LENGTH, return_tensors="pt"
        ).to(device)
        model(**inputs)
        distilbert_timings.append((time.perf_counter() - start) * 1000)

distilbert_stats = report("DistilBERT", distilbert_timings)

# ---------------------------------------------------------------------------
# Phi-4 timing
# ---------------------------------------------------------------------------

print(f"\nWarming up Phi-4 ({PHI4_MODEL_NAME})...")
_call_ollama("ping")

print(f"Timing Phi-4 on {N_SAMPLES} prompts...")
phi4_timings = []
for i, text in enumerate(texts):
    start = time.perf_counter()
    _call_ollama(text)
    phi4_timings.append((time.perf_counter() - start) * 1000)
    if (i + 1) % 20 == 0:
        print(f"  {i + 1}/{N_SAMPLES} done")

phi4_stats = report("Phi-4", phi4_timings)

# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------

out_path = repo_root / "results" / "latency_benchmark.csv"
new_rows = pd.DataFrame([distilbert_stats, phi4_stats])
if out_path.exists():
    prev = pd.read_csv(out_path)
    prev = prev[~prev["model"].isin(new_rows["model"])]   
    new_rows = pd.concat([prev, new_rows], ignore_index=True)
new_rows.to_csv(out_path, index=False)
print(f"\nSaved to {out_path}")
print(new_rows.to_string(index=False))
