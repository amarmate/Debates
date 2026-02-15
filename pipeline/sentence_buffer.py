"""
Sentence segmentation and buffering logic for live and batch processing.

Includes post-processing to rejoin fragments that NLTK splits on
Portuguese abbreviations (Sr., Sra., Dr., Dra., Prof., Eng., etc.).
"""
import re
from collections import deque
from typing import Iterator

import nltk

# Ensure punkt is available for sentence tokenization
nltk.download("punkt_tab", quiet=True)

_LANGUAGE_MAP = {
    "pt": "portuguese",
    "en": "english",
    "es": "spanish",
    "fr": "french",
    "de": "german",
    "it": "italian",
    "nl": "dutch",
    "pl": "polish",
    "ru": "russian",
    "ja": "japanese",
    "zh": "chinese",
}

# Common Portuguese abbreviations that NLTK's sent_tokenize incorrectly
# treats as sentence boundaries.  Pattern matches the abbreviation at the
# very end of a candidate sentence.
_PT_ABBREVIATIONS = re.compile(
    r"\b(?:Sr|Sra|Dr|Dra|Prof|Eng|Arq|Adv|Exm[oa]?|Ilm[oa]?|V\.Ex[ªa]?|D)\.\s*$",
    re.IGNORECASE,
)


def _nltk_language(lang: str | None) -> str:
    """Map client language code to nltk language name."""
    if not lang:
        return "portuguese"
    return _LANGUAGE_MAP.get(lang.lower(), "portuguese")


def _rejoin_abbreviation_splits(sentences: list[str]) -> list[str]:
    """
    Rejoin consecutive sentences where the first ends with a Portuguese
    abbreviation (e.g. "Boa noite, Sr." + "Deputado André Ventura.").
    """
    if len(sentences) < 2:
        return sentences
    merged: list[str] = []
    carry = ""
    for s in sentences:
        if carry:
            s = f"{carry} {s}"
            carry = ""
        if _PT_ABBREVIATIONS.search(s):
            carry = s
        else:
            merged.append(s)
    if carry:
        merged.append(carry)
    return merged


def _is_valid_sentence(s: str) -> bool:
    """
    Filter out fragments that are not complete sentences:
    - Very short or punctuation-only
    - Do not end with sentence-ending punctuation
    - Start with lowercase conjunction/preposition (orphaned clause)
    """
    if len(s) < 5:
        return False
    if re.match(r"^[\s.,;:!?\-]+$", s):
        return False
    stripped = s.strip()
    if not re.search(r"[.!?]$", stripped) and not stripped.endswith("..."):
        return False
    # Reject orphaned clauses that start with lowercase conjunction/preposition
    # (e.g. "e amamentação.", "que alarga os contratos...").
    # These are fragments from bad sentence splits, not standalone facts.
    if re.match(r"^(e|ou|que|mas|nem|de|do|da|dos|das|no|na|nos|nas|ao|à|aos|às)\s", stripped):
        return False
    return True


def segment_sentences(text: str, language: str = "portuguese") -> list[str]:
    """
    Segment full text into sentences using nltk.
    Rejoins abbreviation-split fragments, then filters invalid ones.
    """
    lang = _nltk_language(language) if isinstance(language, str) and len(language) <= 3 else language
    raw_sentences = nltk.sent_tokenize(text, language=lang)
    rejoined = _rejoin_abbreviation_splits(raw_sentences)
    return [s.strip() for s in rejoined if _is_valid_sentence(s.strip())]


class SentenceBuffer:
    """
    Accumulates deduplicated text fragments and yields complete sentences as they become available.
    Used for live streaming: on each append, tokenize buffer and emit all but last (last may be incomplete).
    Filters incomplete fragments, rejoins abbreviation splits, and deduplicates.
    """

    def __init__(self, language: str | None = None, dedup_size: int = 15) -> None:
        self._language = _nltk_language(language) if language else "portuguese"
        self._buffer = ""
        self._dedup_size = dedup_size
        self._recent: deque[str] = deque(maxlen=dedup_size)

    def append(self, fragment: str) -> list[str]:
        """
        Append a new deduplicated text fragment. Returns list of newly complete sentences.
        Filters incomplete fragments and skips duplicates.
        """
        fragment = fragment.strip()
        if not fragment:
            return []

        self._buffer = f"{self._buffer} {fragment}".strip() if self._buffer else fragment
        raw = nltk.sent_tokenize(self._buffer, language=self._language)

        if not raw:
            return []

        # Keep last element as the incomplete buffer.
        candidates = raw[:-1]
        self._buffer = raw[-1].strip()

        # Rejoin abbreviation-split fragments before validating.
        rejoined = _rejoin_abbreviation_splits(candidates)

        # If the last rejoined candidate ends with an abbreviation, push it
        # back into the buffer so it can join with the next fragment.
        if rejoined and _PT_ABBREVIATIONS.search(rejoined[-1]):
            self._buffer = f"{rejoined.pop()} {self._buffer}".strip()

        valid = [s.strip() for s in rejoined if _is_valid_sentence(s.strip())]

        result: list[str] = []
        for s in valid:
            if s not in self._recent:
                result.append(s)
                self._recent.append(s)
        return result

    def flush(self) -> list[str]:
        """
        Flush any remaining buffer as final sentence(s). Call when stream ends.
        """
        if not self._buffer.strip():
            return []
        raw = nltk.sent_tokenize(self._buffer, language=self._language)
        self._buffer = ""
        rejoined = _rejoin_abbreviation_splits(raw)
        result: list[str] = []
        for s in rejoined:
            s = s.strip()
            if _is_valid_sentence(s) and s not in self._recent:
                result.append(s)
                self._recent.append(s)
        return result

    def iter_sentences(self, fragments: Iterator[str]) -> Iterator[str]:
        """
        Convenience: iterate over fragments and yield complete sentences.
        Does not call flush at end; use for infinite streams.
        """
        for frag in fragments:
            for s in self.append(frag):
                yield s
