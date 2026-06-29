"""Text preprocessing and search utilities for the recommendation pipeline."""

import re
from pathlib import Path

from nltk.stem import PorterStemmer

DATA_DIR = Path(__file__).parent
TOKEN_PATTERN = re.compile(r"[a-zA-Z]+(?:[-'][a-zA-Z]+)?")
_STEMMER = PorterStemmer()

with open(DATA_DIR / "stopwords_en.txt", encoding="utf-8") as f:
    STOPWORDS = set(f.read().splitlines())


def preprocess(text: str) -> str:
    """Normalize raw text into a token string for vectorization."""
    if not isinstance(text, str):
        return ""
    tokens = TOKEN_PATTERN.findall(text.lower())
    return " ".join(t for t in tokens if len(t) > 1 and t not in STOPWORDS)


def stem_word(word: str) -> str:
    """Stem a single token to reduce vocabulary sparsity in search."""
    return _STEMMER.stem(word.lower())


def stem_query(query: str) -> list[str]:
    """Tokenize and stem a user search query."""
    return [stem_word(t) for t in TOKEN_PATTERN.findall(query.lower())]


def search_item_score(row, stemmed_query: list[str]) -> int:
    """Count how many stemmed query tokens appear in item metadata."""
    if not stemmed_query:
        return 0
    text = " ".join([
        str(row["clothes_title"]),
        str(row["clothes_desc"]),
        str(row["class_name"]),
        str(row["department"]),
    ]).lower()
    words = [stem_word(w) for w in re.findall(r"[a-z]+", text)]
    return sum(1 for t in stemmed_query if t in words)
