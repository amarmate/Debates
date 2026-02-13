"""
Shared deduplication and sentence segmentation logic for live and batch processing.
"""
import difflib
import re
from collections import deque
from typing import Iterator

import nltk

# Ensure punkt is available for sentence tokenization
nltk.download("punkt_tab", quiet=True)


def _normalize_for_overlap(s: str) -> str:
    """Normalize whitespace around punctuation for more robust overlap detection."""
    s = re.sub(r"\s+", " ", s.strip())
    s = re.sub(r"\s+([.,;:!?])", r"\1", s)  # remove space before punctuation
    return s


def merge_chunks(
    accumulated_text: str,
    new_chunk: str,
    search_window: int = 200,
    min_overlap: int = 10,
) -> str:
    """
    Find longest overlap between end of accumulated_text and start of new_chunk,
    append only the non-overlapping suffix. Uses difflib for fuzzy matching
    when Whisper transcription varies slightly between chunks.
    """
    new_chunk = new_chunk.strip()
    if not new_chunk:
        return accumulated_text
    if not accumulated_text:
        return new_chunk

    look_back = (
        accumulated_text[-search_window:]
        if len(accumulated_text) > search_window
        else accumulated_text
    )
    look_back_norm = _normalize_for_overlap(look_back)
    new_chunk_norm = _normalize_for_overlap(new_chunk)
    s = difflib.SequenceMatcher(None, look_back_norm, new_chunk_norm)
    match = s.find_longest_match(0, len(look_back_norm), 0, len(new_chunk_norm))

    # Match must be at boundary: end of look_back (accumulated) and start of new_chunk
    at_end_of_accumulated = match.a + match.size == len(look_back_norm)
    at_start_of_chunk = match.b == 0
    if match.size >= min_overlap and at_end_of_accumulated and at_start_of_chunk:
        new_content_start = match.b + match.size
        suffix = new_chunk[new_content_start:].strip()
        return accumulated_text + (" " + suffix if suffix else "")
    # Fallback: exact boundary match with normalized comparison for punctuation variance
    suffix = _extract_new_suffix_normalized(accumulated_text, new_chunk).strip()
    return accumulated_text + (" " + suffix if suffix else "")


def _extract_new_suffix_normalized(last_sent: str, text_new: str) -> str:
    """Like extract_new_suffix but uses normalized comparison for punctuation variance."""
    if not last_sent:
        return text_new.strip()
    text_new = text_new.strip()
    if not text_new:
        return ""
    max_overlap = min(len(last_sent), len(text_new))
    for i in range(max_overlap, 0, -1):
        if _normalize_for_overlap(last_sent[-i:]) == _normalize_for_overlap(text_new[:i]):
            return text_new[i:].lstrip()
    return text_new


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
    """
    Filter out fragments that are not complete sentences:
    - Very short or punctuation-only
    - Do not end with sentence-ending punctuation

    Note: Do not filter ellipsis; long factual sentences may span chunks or use "..." rhetorically.
    """
    if len(s) < 5:
        return False
    if re.match(r"^[\s.,;:!?\-]+$", s):
        return False
    stripped = s.strip()
    if not re.search(r"[.!?]$", stripped) and not stripped.endswith("..."):
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
    Filters incomplete fragments and deduplicates.
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

        valid = [s.strip() for s in raw[:-1] if _is_valid_sentence(s.strip())]
        self._buffer = raw[-1].strip()

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
        result: list[str] = []
        for s in raw:
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
