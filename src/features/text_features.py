"""
text_features.py
Basic text shape features for PII detection.
Usage: from src.features.text_features import extract_text_features
"""

import re
import pandas as pd


def extract_text_features(texts):
    """
    Extract basic text shape features from a list of strings.

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

        char_count = len(text)
        word_count = len(text.split())
        digit_count = sum(c.isdigit() for c in text)
        upper_count = sum(c.isupper() for c in text)
        punct_count = sum(not c.isalnum() and not c.isspace() for c in text)
        whitespace_count = sum(c.isspace() for c in text)

        # count symbols commonly seen in PII patterns like emails, URLs, dates, IDs, and codes
        special_char_count = sum(c in "@.-/+\\#_:" for c in text)

        digit_ratio = digit_count / char_count if char_count > 0 else 0
        digit_runs = re.findall(r"\d+", text)
        max_digit_run = max((len(r) for r in digit_runs), default=0)
        has_html = 1 if re.search(r"<[^>]+>", text) else 0

        records.append({
            "char_count": char_count,
            "word_count": word_count,
            "digit_count": digit_count,
            "upper_count": upper_count,
            "punct_count": punct_count,
            "whitespace_count": whitespace_count,
            "special_char_count": special_char_count,
            "digit_ratio": digit_ratio,
            "max_digit_run": max_digit_run,
            "has_html": has_html,
        })

    return pd.DataFrame(records)
