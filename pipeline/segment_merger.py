"""
Timestamp-based segment deduplication for rolling-window transcription.
"""
import logging
import re
from dataclasses import dataclass

from pipeline.src.transcriber import TranscribedSegment

logger = logging.getLogger(__name__)

_EPSILON_SEC = 0.05


def _join_words(words: list[str]) -> str:
    """Join word tokens into readable text and normalize punctuation spacing."""
    text = " ".join(w.strip() for w in words if w and w.strip())
    if not text:
        return ""
    text = re.sub(r"\s+([.,;:!?])", r"\1", text)
    return re.sub(r"\s+", " ", text).strip()


@dataclass(frozen=True)
class MergeMetadata:
    """Debug information for UI/diagnostics."""
    window_start: float
    kept_units: int
    dropped_units: int
    last_committed_before: float
    last_committed_after: float


class SegmentMerger:
    """
    Merge overlapping rolling windows using absolute timestamps.

    Strategy:
    - Convert local segment/word times to absolute times with window_start.
    - Keep only content ending after last committed timestamp.
    - Prefer word-level filtering when available for partial segment overlap.
    """

    def __init__(self, epsilon_sec: float = _EPSILON_SEC) -> None:
        self._epsilon_sec = epsilon_sec
        self._last_committed_end_sec = 0.0

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
                for word in seg.words:
                    abs_word_end = window_start + float(word.end)
                    if abs_word_end > (self._last_committed_end_sec + self._epsilon_sec):
                        kept_words.append(word.word)
                        kept_units += 1
                        self._last_committed_end_sec = max(
                            self._last_committed_end_sec, abs_word_end
                        )
                    else:
                        dropped_units += 1
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
        meta = MergeMetadata(
            window_start=float(window_start),
            kept_units=kept_units,
            dropped_units=dropped_units,
            last_committed_before=last_before,
            last_committed_after=self._last_committed_end_sec,
        )
        logger.debug(
            "Segment merge: window_start=%.2f kept=%d dropped=%d last_before=%.2f last_after=%.2f",
            meta.window_start,
            meta.kept_units,
            meta.dropped_units,
            meta.last_committed_before,
            meta.last_committed_after,
        )
        return merged_new, meta
