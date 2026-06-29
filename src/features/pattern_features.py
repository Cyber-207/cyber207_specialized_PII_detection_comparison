"""
pattern_features.py
Privacy-pattern regex features for PII detection.
Usage: from src.features.pattern_features import extract_pattern_features
"""

import re
import pandas as pd

EMAIL = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
PHONE = re.compile(r"(\+?\d[\d\s\-().]{7,}\d)")
URL = re.compile(r"https?://\S+|www\.\S+")
IP = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
LONG_NUMBER = re.compile(r"\b\d{8,}\b")
DATE = re.compile(r"\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{1,2}[/-]\d{1,2})\b")

ADDRESS_WORDS = re.compile(
    r"\b(street|st|avenue|ave|boulevard|blvd|road|rd|lane|ln|drive|dr|court|ct|zip|postal)\b",
    re.IGNORECASE,
)

ID_WORDS = re.compile(
    r"\b(ssn|passport|license|dob|date of birth|id number|employee id|account number|"
    r"credit card|card number|routing number|bank account|tax id|national id|student id|"
    r"driver license|drivers license)\b",
    re.IGNORECASE,
)


def extract_pattern_features(texts):
    """
    Extract privacy-pattern binary flags from a list of strings.

    Parameters
    ----------
    texts : list or pd.Series
        Raw prompt texts.

    Returns
    -------
    pd.DataFrame
        One row per input text, one column per feature.
    """
    records = []

    for text in texts:
        if pd.isna(text):
            text = ""
        else:
            text = str(text)

        records.append({
            "has_email": 1 if EMAIL.search(text) else 0,
            "has_phone": 1 if PHONE.search(text) else 0,
            "has_url": 1 if URL.search(text) else 0,
            "has_ip": 1 if IP.search(text) else 0,
            "has_long_number": 1 if LONG_NUMBER.search(text) else 0,
            "has_date": 1 if DATE.search(text) else 0,
            "has_address_words": 1 if ADDRESS_WORDS.search(text) else 0,
            "has_id_words": 1 if ID_WORDS.search(text) else 0,
        })

    return pd.DataFrame(records)
