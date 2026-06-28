"""
llm_baseline.py
LLM-based PII detection baseline using Phi-4 via Ollama.
Usage: from src.models.llm_baseline import predict_llm_baseline
"""

import json
import re
import requests
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "phi4:14b"
TIMEOUT = 60

SYSTEM_PROMPT = """\
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

Context:
You are evaluating text samples for PII content. Each sample is a short \
prompt or message that may or may not contain sensitive personal information. \
Confidence scores reflect your certainty that the identified substring is \
genuinely PII of the labeled type, not a general measure of detection quality.

Example
-------
Input:
Patient Jane Doe, DOB 1978-03-12, lives at 100 Main St, phone 555-0100.

Output:
[
  {"label": "name",    "value": "Jane Doe",    "confidence_score": "0.95"},
  {"label": "dob",     "value": "1978-03-12",  "confidence_score": "0.99"},
  {"label": "address", "value": "100 Main St", "confidence_score": "0.72"},
  {"label": "phone",   "value": "555-0100",    "confidence_score": "0.88"}
]\
"""


def _build_prompt(text: str) -> str:
    return f"Input:\n{text}\n\nOutput:"


def _call_ollama(text: str) -> str:
    payload = {
        "model": MODEL_NAME,
        "prompt": _build_prompt(text),
        "system": SYSTEM_PROMPT,
        "stream": False,
        "options": {
            "temperature": 0.0,
            "num_predict": 512,
        },
    }
    resp = requests.post(OLLAMA_URL, json=payload, timeout=TIMEOUT)
    resp.raise_for_status()
    return resp.json().get("response", "").strip()


def _parse_entities(raw: str) -> list[dict]:
    cleaned = re.sub(r"```(?:json)?", "", raw).strip()
    match = re.search(r"\[.*\]", cleaned, re.DOTALL)
    if not match:
        return []
    try:
        entities = json.loads(match.group())
        if not isinstance(entities, list):
            return []
        return entities
    except json.JSONDecodeError:
        return []


def _entities_to_record(text: str, entities: list[dict]) -> dict:
    if not entities:
        return {
            "text": text,
            "document_label": 0,
            "max_confidence": 0.0,
            "entity_count": 0,
            "entities_json": "[]",
        }

    confidences = []
    for e in entities:
        try:
            confidences.append(float(e.get("confidence_score", 0)))
        except (ValueError, TypeError):
            confidences.append(0.0)

    return {
        "text": text,
        "document_label": 1,
        "max_confidence": max(confidences),
        "entity_count": len(entities),
        "entities_json": json.dumps(entities, ensure_ascii=False),
    }


def predict_llm_baseline(texts, verbose: bool = False) -> pd.DataFrame:
    """
    Run Phi-4 PII detection over a list or Series of texts.

    Parameters
    ----------
    texts : list or pd.Series
        Raw prompt texts to evaluate.
    verbose : bool
        If True, prints progress every 50 samples.

    Returns
    -------
    pd.DataFrame
        One row per input text with columns:
        - text            : original input
        - document_label  : 1 = PII detected, 0 = safe (binary)
        - max_confidence  : highest entity-level confidence score
        - entity_count    : number of PII spans found
        - entities_json   : full JSON string of detected entities
        - error           : non-empty string if the API call failed
    """
    if isinstance(texts, pd.Series):
        texts = texts.tolist()

    records = []

    for i, text in enumerate(texts):
        if verbose and i > 0 and i % 50 == 0:
            print(f"  llm_baseline: {i}/{len(texts)} processed")

        if pd.isna(text):
            text = ""
        else:
            text = str(text)

        try:
            raw = _call_ollama(text)
            entities = _parse_entities(raw)
            record = _entities_to_record(text, entities)
            record["error"] = ""
            record["raw_response"] = raw
        except Exception as exc:
            record = {
                "text": text,
                "document_label": -1,
                "max_confidence": None,
                "entity_count": None,
                "entities_json": "[]",
                "error": str(exc),
		"raw_response": "",
            }

        records.append(record)

    return pd.DataFrame(records)
