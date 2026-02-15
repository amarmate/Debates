"""
Shared deduplication and sentence segmentation logic for live and batch processing.
"""
import difflib
import logging
import re
from collections import deque
from typing import Iterator, Optional

import nltk

# Ensure punkt is available for sentence tokenization
nltk.download("punkt_tab", quiet=True)

logger = logging.getLogger(__name__)


def _normalize_for_overlap(s: str) -> str:
    """Normalize whitespace around punctuation for more robust overlap detection."""
    s = re.sub(r"\s+", " ", s.strip())
    s = re.sub(r"\s+([.,;:!?])", r"\1", s)  # remove space before punctuation
    return s


def merge_chunks(
    accumulated_text: str,
    new_chunk: str,
    search_window: int = 100,
    min_overlap: int = 10,
) -> str:
    """
    Merge overlapping chunks using anchor-anywhere fuzzy algorithm.
    Uses difflib.SequenceMatcher to locate the anchor (last ~100 chars of committed
    text) anywhere inside new_chunk. Discards everything before the match end
    (removes hallucinations like "O deputado prévio" when real text is "Um sorteio prévio"),
    appends only the remainder.
    """
    merged, _ = merge_chunks_with_meta(
        accumulated_text, new_chunk, search_window, min_overlap
    )
    return merged


# Minimum confidence (match_size / anchor_len) to accept a match; below this, discard to prevent duplication
_MERGE_CONFIDENCE_THRESHOLD = 0.6
# Characters of accumulated text to check for "already said" duplicates
_ALREADY_SAID_WINDOW = 1000


def merge_chunks_with_meta(
    accumulated_text: str,
    new_chunk: str,
    search_window: int = 100,
    min_overlap: int = 10,
) -> tuple[str, dict]:
    """
    Anchor-anywhere merge: find anchor (last ~100 chars of committed text) anywhere
    in new_chunk via SequenceMatcher. Crop everything before match end, append remainder.
    Handles hallucinations where Whisper produces different start text.
    Returns (merged_text, metadata) with debug fields for verification.
    """
    new_chunk = new_chunk.strip()
    meta: dict = {
        "anchor": "",
        "match_size": 0,
        "match_at": -1,
        "best_match": "",
        "text_removed": "",
        "new_content": "",
        "raw_chunk": new_chunk,
    }

    if not new_chunk:
        return accumulated_text, meta
    if not accumulated_text:
        return new_chunk, meta

    # Safety 1: "Already Said" check — discard only if entire new_chunk is in recent text (true duplicate).
    # Do NOT check "significant substring": overlapping windows always share a prefix; merge will extract new tail.
    last_recent = accumulated_text[-_ALREADY_SAID_WINDOW:]
    if new_chunk in last_recent:
        logger.info("Merge: chunk already in transcript (last %d chars), discarding duplicate.", _ALREADY_SAID_WINDOW)
        return accumulated_text, meta

    # 1. Anchor: last ~100 chars of committed text
    anchor_len = min(search_window, max(50, len(accumulated_text)))
    anchor = accumulated_text[-anchor_len:]
    meta["anchor"] = anchor

    # 2. Fuzzy search: find anchor (or its suffix) anywhere in new_chunk
    matcher = difflib.SequenceMatcher(None, anchor, new_chunk, autojunk=False)
    blocks = matcher.get_matching_blocks()

    # Find block that spans anchor end (a+size==len(anchor)) with sufficient overlap
    best_block = None
    best_ref_len: Optional[int] = None  # length of pattern we matched against (anchor or suffix)
    for block in blocks:
        if block.size >= min_overlap and block.a + block.size == len(anchor):
            if best_block is None or block.size > best_block.size:
                best_block = block
                best_ref_len = len(anchor)

    # Fallback: try anchor suffixes (handles cases where full anchor tail doesn't match)
    if best_block is None:
        for suffix_len in range(len(anchor), min_overlap - 1, -1):
            suffix = anchor[-suffix_len:]
            sub_matcher = difflib.SequenceMatcher(None, suffix, new_chunk, autojunk=False)
            sub_blocks = sub_matcher.get_matching_blocks()
            for block in sub_blocks:
                if block.size >= min_overlap and block.a + block.size == len(suffix):
                    best_block = block
                    best_ref_len = suffix_len
                    break
            if best_block is not None:
                break

    # Safety 2: "Strict Match" policy — require confidence > 0.6 (match_size / ref_len)
    if best_block is not None:
        ref_len = best_ref_len if best_ref_len is not None else len(anchor)
        confidence = best_block.size / ref_len if ref_len > 0 else 0.0
        if confidence <= _MERGE_CONFIDENCE_THRESHOLD:
            best_block = None  # Fall through to Step B (Tail-Anchor Fallback)

    # Step A success: use difflib result
    if best_block is not None:
        overlap_end_in_new = best_block.b + best_block.size
        text_discarded = new_chunk[:overlap_end_in_new]
        new_text = new_chunk[overlap_end_in_new:].strip()
        best_match = new_chunk[best_block.b : overlap_end_in_new]

        meta["match_size"] = best_block.size
        meta["match_at"] = best_block.b
        meta["best_match"] = best_match
        meta["text_removed"] = text_discarded
        meta["new_content"] = new_text

        _log_merge_debug(meta, accumulated_text, new_chunk)

        return accumulated_text + (" " + new_text if new_text else ""), meta

    # Step B: Tail-Anchor Fallback — when difflib fails (e.g. hallucinated intro),
    # search for last 3 words of anchor in new_chunk. If found, crop before them and append.
    # Try last 3, then 2, then 1 word (handles "Bem-vindos" appearing alone).
    # Normalize punctuation so "Boa noite." matches "boa noite," in chunk.
    words = anchor.split()
    for n in (3, 2, 1):
        if len(words) < n:
            continue
        tail_phrase = " ".join(words[-n:])
        if len(tail_phrase) < 5:
            continue
        tail_stripped = tail_phrase.rstrip(".,;:!?")
        if len(tail_stripped) < 5:
            continue
        idx = new_chunk.find(tail_phrase)
        if idx >= 0:
            match_len = len(tail_phrase)
        else:
            idx = new_chunk.lower().find(tail_phrase.lower())
            if idx >= 0:
                match_len = len(tail_phrase)
        if idx < 0:
            idx = new_chunk.find(tail_stripped)
            if idx >= 0:
                match_len = len(tail_stripped)
        if idx < 0:
            idx = new_chunk.lower().find(tail_stripped.lower())
            if idx >= 0:
                match_len = len(tail_stripped)
        if idx >= 0:
            overlap_end = idx + match_len
            new_text = new_chunk[overlap_end:].strip()
            meta["match_size"] = match_len
            meta["match_at"] = idx
            meta["best_match"] = tail_phrase
            meta["text_removed"] = new_chunk[:overlap_end]
            meta["new_content"] = new_text
            logger.info(
                "Merge: Tail-Anchor Fallback used (last 3 words at %d), appending %d chars",
                idx,
                len(new_text),
            )
            _log_merge_debug(meta, accumulated_text, new_chunk)
            return accumulated_text + (" " + new_text if new_text else ""), meta

    # Step C: When both Step A and Step B fail, append the new chunk (no overlap found = likely new content).
    # We already returned early if new_chunk was a full duplicate, so appending is safe.
    meta["new_content"] = new_chunk
    meta["text_removed"] = ""
    logger.info("Merge: No overlap found, appending full chunk (%d chars) as new content.", len(new_chunk))
    _log_merge_debug(meta, accumulated_text, new_chunk)
    return accumulated_text + (" " + new_chunk if new_chunk else ""), meta


def _log_merge_debug(meta: dict, accumulated_text: str, new_chunk: str) -> None:
    """Log debug info for every chunk: Prompt, Anchor, Match At, Text Discarded."""
    anchor = meta.get("anchor", "")
    match_at = meta.get("match_at", -1)
    text_discarded = meta.get("text_removed", "")
    logger.info(
        "Merge: Anchor Text=%r | Match Found At=%d | Text Discarded=%r",
        anchor[-80:] + ("..." if len(anchor) > 80 else ""),
        match_at,
        text_discarded[:100] + ("..." if len(text_discarded) > 100 else ""),
    )


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
