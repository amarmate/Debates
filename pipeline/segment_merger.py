"""
Timestamp-based segment deduplication for rolling-window transcription.

Includes two safety layers beyond pure timestamp filtering:
1. Text-overlap boundary guard — catches word duplicates caused by Whisper
   assigning slightly different timestamps to the same word across windows.
2. Trailing-ellipsis strip — removes "..." that Whisper adds when audio is
   cut at a chunk boundary (next window will have the complete text).
"""
import logging
import re
from collections import deque
from dataclasses import dataclass

from pipeline.src.transcriber import TranscribedSegment

logger = logging.getLogger(__name__)

_EPSILON_SEC = 0.05
# How many tail words to remember for the text-overlap boundary guard.
_BOUNDARY_GUARD_WORDS = 3


def _join_words(words: list[str]) -> str:
    """Join word tokens into readable text and normalize punctuation spacing."""
    text = " ".join(w.strip() for w in words if w and w.strip())
    if not text:
        return ""
    text = re.sub(r"\s+([.,;:!?])", r"\1", text)
    return re.sub(r"\s+", " ", text).strip()


def _normalize_word(w: str) -> str:
    """Lowercase and strip punctuation for comparison."""
    return re.sub(r"[.,;:!?\-\"'…]+", "", w.strip().lower())


def _strip_trailing_ellipsis(text: str) -> str:
    """Remove trailing '...' (chunk-boundary artifact, not real speech)."""
    return re.sub(r"\.\.\.\s*$", "", text).rstrip()


def _strip_trailing_period(text: str) -> str:
    """
    Remove a trailing period from fragment text.

    Whisper adds '.' at the end of every chunk because it thinks the audio
    ended.  These are chunk-boundary artifacts — the next window will
    provide the continuation.  Real sentence-ending periods are preserved
    because they appear *within* the fragment, not at the trailing edge.

    We only strip '.' (not '!' or '?') because exclamation/question marks
    are almost always intentional.
    """
    stripped = text.rstrip()
    if stripped.endswith(".") and not stripped.endswith("..."):
        return stripped[:-1].rstrip()
    return text


@dataclass(frozen=True)
class MergeMetadata:
    """Debug information for UI/diagnostics."""
    window_start: float
    kept_units: int
    dropped_units: int
    boundary_dropped: int
    last_committed_before: float
    last_committed_after: float


class SegmentMerger:
    """
    Merge overlapping rolling windows using absolute timestamps.

    Strategy:
    - Convert local segment/word times to absolute times with window_start.
    - Keep only content ending after last committed timestamp.
    - Prefer word-level filtering when available for partial segment overlap.
    - Text-overlap boundary guard: skip leading words that duplicate the
      tail of already-committed text (handles Whisper timestamp drift).
    - Strip trailing ellipsis before committing (chunk-boundary artifact).
    """

    def __init__(self, epsilon_sec: float = _EPSILON_SEC) -> None:
        self._epsilon_sec = epsilon_sec
        self._last_committed_end_sec = 0.0
        # Ring buffer of the last N normalized words committed, for boundary guard.
        self._committed_tail: deque[str] = deque(maxlen=_BOUNDARY_GUARD_WORDS)

    @property
    def last_committed_end_sec(self) -> float:
        return self._last_committed_end_sec

    def merge(
        self, segments: list[TranscribedSegment], window_start: float
    ) -> tuple[str, MergeMetadata]:
        """
        Return only new text not already committed in prior windows.
        """
        last_before = self._last_committed_end_sec
        kept_units = 0
        dropped_units = 0
        kept_pieces: list[str] = []

        for seg in segments:
            seg_text = (seg.text or "").strip()
            if not seg_text:
                continue

            if seg.words:
                kept_words: list[str] = []
                keeping = False  # once True, keep ALL remaining words in segment
                for word in seg.words:
                    abs_word_end = window_start + float(word.end)
                    if not keeping:
                        if abs_word_end > (self._last_committed_end_sec + self._epsilon_sec):
                            keeping = True
                        else:
                            dropped_units += 1
                            continue
                    # keeping == True: accept this and all subsequent words
                    kept_words.append(word.word)
                    kept_units += 1
                    self._last_committed_end_sec = max(
                        self._last_committed_end_sec, abs_word_end
                    )
                piece = _join_words(kept_words)
                if piece:
                    kept_pieces.append(piece)
                continue

            abs_seg_end = window_start + float(seg.end)
            if abs_seg_end > (self._last_committed_end_sec + self._epsilon_sec):
                kept_pieces.append(seg_text)
                kept_units += 1
                self._last_committed_end_sec = max(self._last_committed_end_sec, abs_seg_end)
            else:
                dropped_units += 1

        merged_new = " ".join(kept_pieces).strip()

        # --- Safety layer 1: text-overlap boundary guard ---
        # Strip leading words that duplicate the tail of committed text.
        merged_new, boundary_dropped = self._strip_boundary_overlap(merged_new)

        # --- Safety layer 2: strip trailing chunk-boundary artifacts ---
        # First ellipsis ("..."), then the period Whisper adds at chunk ends.
        merged_new = _strip_trailing_ellipsis(merged_new)
        merged_new = _strip_trailing_period(merged_new)

        # Update committed tail words for next merge's boundary guard.
        if merged_new:
            new_words = merged_new.split()
            for w in new_words:
                nw = _normalize_word(w)
                if nw:
                    self._committed_tail.append(nw)

        meta = MergeMetadata(
            window_start=float(window_start),
            kept_units=kept_units,
            dropped_units=dropped_units,
            boundary_dropped=boundary_dropped,
            last_committed_before=last_before,
            last_committed_after=self._last_committed_end_sec,
        )
        logger.debug(
            "Segment merge: window_start=%.2f kept=%d dropped=%d boundary_dropped=%d"
            " last_before=%.2f last_after=%.2f",
            meta.window_start,
            meta.kept_units,
            meta.dropped_units,
            meta.boundary_dropped,
            meta.last_committed_before,
            meta.last_committed_after,
        )
        return merged_new, meta

    def _strip_boundary_overlap(self, text: str) -> tuple[str, int]:
        """
        Remove leading words of *text* that duplicate the tail of committed text.

        Whisper can assign slightly different timestamps to the same spoken word
        across overlapping windows. Pure timestamp filtering lets these through.
        This catches them by comparing normalized word forms.

        Returns (cleaned_text, number_of_words_stripped).
        """
        if not text or not self._committed_tail:
            return text, 0

        words = text.split()
        if not words:
            return text, 0

        tail = list(self._committed_tail)  # last N committed words (normalized)

        # Find how many leading words of `text` match the end of committed tail.
        # Check up to min(len(tail), len(words)) leading words.
        overlap = 0
        max_check = min(len(tail), len(words))
        for n in range(1, max_check + 1):
            # Do the last n words of tail match the first n words of text?
            candidate = [_normalize_word(w) for w in words[:n]]
            if candidate == tail[-n:]:
                overlap = n

        if overlap > 0:
            logger.debug(
                "Boundary guard: stripped %d duplicate leading word(s): %s",
                overlap,
                words[:overlap],
            )
            cleaned = " ".join(words[overlap:]).strip()
            return cleaned, overlap

        return text, 0
