"""Scoring normalization only; never used to filter answers."""
import re
import unicodedata


def normalize_answer(text, remove_article=True):
    text = "".join(c for c in text.casefold() if not unicodedata.category(c).startswith("P"))
    text = " ".join(text.split())
    if remove_article:
        text = re.sub(r"^(?:the|a|an)\s+", "", text)
    return text
