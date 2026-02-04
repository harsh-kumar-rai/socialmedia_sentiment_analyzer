"""Text preprocessing helpers for sentiment analysis."""

import re
from typing import Iterable, List

DEFAULT_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "he",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "that",
    "the",
    "to",
    "was",
    "were",
    "will",
    "with",
}


def normalize_text(text: str) -> str:
    """Normalize social media text for vectorization."""
    text = text.lower()
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"@[\w_]+", "", text)
    text = re.sub(r"#[\w_]+", "", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str, stopwords: Iterable[str] = DEFAULT_STOPWORDS) -> List[str]:
    """Split text into tokens, removing stopwords."""
    normalized = normalize_text(text)
    tokens = [token for token in normalized.split(" ") if token]
    return [token for token in tokens if token not in stopwords]
