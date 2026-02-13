"""
Shared deduplication and sentence segmentation logic for live and batch processing.
"""
import re
from typing import Iterator

import nltk

# Ensure punkt is available for sentence tokenization
nltk.download("punkt_tab", quiet=True)


def extract_new_suffix(last_sent: str, text_new: str) -> str:
    """
    Extract only the new portion of transcript to avoid duplicate output.
    For rolling buffer: Whisper returns full transcript per window; windows overlap.
    Find overlap at boundary (end of last_sent matches start of text_new) and return remainder.
    """
    if not last_sent:
        return text_new.strip()
    text_new = text_new.strip()
    if not text_new:
        return ""
    # Cumulative case: new text extends last_sent
    if text_new.startswith(last_sent) and len(text_new) > len(last_sent):
        return text_new[len(last_sent) :].lstrip()
    last_stripped = last_sent.rstrip()
    if text_new.startswith(last_stripped) and len(text_new) > len(last_stripped):
        return text_new[len(last_stripped) :].lstrip()
    # Overlapping windows: find longest overlap at boundary
    max_overlap = min(len(last_sent), len(text_new))
    for i in range(max_overlap, 0, -1):
        if last_sent[-i:] == text_new[:i]:
            suffix = text_new[i:].strip()
            return suffix if suffix else text_new
    return text_new


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


def _nltk_language(lang: str | None) -> str:
    """Map client language code to nltk language name."""
    if not lang:
        return "portuguese"
    return _LANGUAGE_MAP.get(lang.lower(), "portuguese")


def _is_valid_sentence(s: str) -> bool:
    """Filter out very short or punctuation-only fragments."""
    if len(s) < 3:
        return False
    if re.match(r"^[\s.,;:!?\-]+$", s):
        return False
    return True


def segment_sentences(text: str, language: str = "portuguese") -> list[str]:
    """
    Segment full text into sentences using nltk.
    Filters out very short fragments.
    """
    lang = _nltk_language(language) if isinstance(language, str) and len(language) <= 3 else language
    raw_sentences = nltk.sent_tokenize(text, language=lang)
    return [s.strip() for s in raw_sentences if _is_valid_sentence(s.strip())]


class SentenceBuffer:
    """
    Accumulates deduplicated text fragments and yields complete sentences as they become available.
    Used for live streaming: on each append, tokenize buffer and emit all but last (last may be incomplete).
    """

    def __init__(self, language: str | None = None) -> None:
        self._language = _nltk_language(language) if language else "portuguese"
        self._buffer = ""

    def append(self, fragment: str) -> list[str]:
        """
        Append a new deduplicated text fragment. Returns list of newly complete sentences.
        """
        fragment = fragment.strip()
        if not fragment:
            return []

        self._buffer = f"{self._buffer} {fragment}".strip() if self._buffer else fragment
        raw = nltk.sent_tokenize(self._buffer, language=self._language)

        if not raw:
            return []

        # All but last are complete; last may be incomplete (no trailing punctuation yet)
        complete = [s.strip() for s in raw[:-1] if _is_valid_sentence(s.strip())]
        self._buffer = raw[-1].strip()

        return complete

    def flush(self) -> list[str]:
        """
        Flush any remaining buffer as final sentence(s). Call when stream ends.
        """
        if not self._buffer.strip():
            return []
        sentences = [s.strip() for s in nltk.sent_tokenize(self._buffer, language=self._language) if _is_valid_sentence(s.strip())]
        self._buffer = ""
        return sentences

    def iter_sentences(self, fragments: Iterator[str]) -> Iterator[str]:
        """
        Convenience: iterate over fragments and yield complete sentences.
        Does not call flush at end; use for infinite streams.
        """
        for frag in fragments:
            for s in self.append(frag):
                yield s
