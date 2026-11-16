"""Conservative text cleaning and validation. No tokenization."""
import hashlib
import html
import re
from html.parser import HTMLParser

PIPELINE_VERSION = "1"
MEDIA_PATTERN = re.compile(r"</?(?:a|iframe|video|audio|img|embed|object)\b|href\s*=|https?://|www\.", re.I)


class _TextExtractor(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []

    def handle_data(self, data):
        self.parts.append(data)

    def handle_starttag(self, tag, attrs):
        if tag in {"br", "p", "div", "li"}:
            self.parts.append(" ")

    def handle_endtag(self, tag):
        if tag in {"p", "div", "li"}:
            self.parts.append(" ")


def clean_text(text):
    parser = _TextExtractor()
    parser.feed(html.unescape(text))
    parser.close()
    text = "".join(parser.parts).replace("\\'", "'").strip()
    if len(text) >= 2 and text.startswith("'") and text.endswith("'"):
        text = text[1:-1]
    return " ".join(text.split())


def clue_group(clue):
    return hashlib.sha256(" ".join(clue.casefold().split()).encode("utf-8")).hexdigest()


def prepare_record(raw, source_index):
    """Return (record, None) or (None, exclusive rejection reason)."""
    if not isinstance(raw, dict) or any(not isinstance(raw.get(f), str) for f in ("category", "question", "answer")):
        return None, "missing_or_invalid_required_text"
    if MEDIA_PATTERN.search(html.unescape(raw["question"])):
        return None, "media_or_link"
    category, clue, answer = (clean_text(raw[f]) for f in ("category", "question", "answer"))
    if not all((category, clue, answer)):
        return None, "empty_cleaned_text"
    return {
        "id": f"jeopardy-{source_index:06d}", "source_index": source_index,
        "raw": raw, "category": category, "clue": clue, "answer": answer,
        "air_date": raw.get("air_date"), "show_number": raw.get("show_number"),
        "round": raw.get("round"), "value": raw.get("value"),
        "duplicate_group": clue_group(clue),
    }, None
