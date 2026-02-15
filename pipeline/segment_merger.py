"""
Timestamp-based segment deduplication for rolling-window transcription.

Includes:
1. Confirmation buffer — words from the trailing edge of each window are
   held back until the next window re-transcribes them with more forward
   context.  This fixes inaccurate words (e.g. "parado" -> "prazo") and
   artificial punctuation at chunk boundaries.
2. Text-overlap boundary guard — catches word duplicates caused by Whisper
   assigning slightly different timestamps to the same word across windows.
3. Trailing-ellipsis strip — removes "..." that Whisper adds when audio is
   cut at a chunk boundary.
"""
import logging
import re
from collections import deque
from dataclasses import dataclass

from pipeline.src.transcriber import TranscribedSegment

logger = logging.getLogger(__name__)

_EPSILON_SEC = 0.05
_BOUNDARY_GUARD_WORDS = 3
_DEFAULT_CONFIRMATION_MARGIN_SEC = 2.0


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


@dataclass(frozen=True)
class MergeMetadata:
    """Debug information for UI/diagnostics."""
    window_start: float
    window_end: float
    confirmed_units: int
    tentative_units: int
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
    - Within a segment, once the first word passes the threshold, keep ALL
      remaining words (prevents garbling from non-monotonic Whisper timestamps).
    - Confirmation buffer: words in the last `confirmation_margin_sec` of each
      window are held back as tentative.  The next window re-transcribes them
      with more forward context, producing better word choices and punctuation.
    - Text-overlap boundary guard: skip leading words that duplicate the
      tail of already-committed text (handles cross-window timestamp drift).
    - Strip trailing ellipsis (chunk-boundary artifact).
    """

    def __init__(
        self,
        epsilon_sec: float = _EPSILON_SEC,
        confirmation_margin_sec: float = _DEFAULT_CONFIRMATION_MARGIN_SEC,
    ) -> None:
        self._epsilon_sec = epsilon_sec
        self._confirmation_margin_sec = confirmation_margin_sec
        self._last_committed_end_sec = 0.0
        self._committed_tail: deque[str] = deque(maxlen=_BOUNDARY_GUARD_WORDS)
        self._pending_text = ""  # tentative text held back for confirmation

    @property
    def last_committed_end_sec(self) -> float:
        return self._last_committed_end_sec

    def merge(
        self,
        segments: list[TranscribedSegment],
        window_start: float,
        window_end: float,
    ) -> tuple[str, MergeMetadata]:
        """
        Return only new *confirmed* text.

        Words in the last `confirmation_margin_sec` of the window are held
        back as tentative.  The next window will re-transcribe them with more
        forward context (better word accuracy and natural punctuation).
        """
        last_before = self._last_committed_end_sec
        cutoff = window_end - self._confirmation_margin_sec

        # Previous tentative text is discarded — the current window
        # re-transcribes that time range with better context.
        self._pending_text = ""

        # --- Step 1: Timestamp-filter words/segments ---
        # Collect (word_text, abs_end_time) tuples for all new content.
        all_kept: list[tuple[str, float]] = []
        dropped_units = 0

        for seg in segments:
            seg_text = (seg.text or "").strip()
            if not seg_text:
                continue

            if seg.words:
                keeping = False
                for word in seg.words:
                    abs_word_end = window_start + float(word.end)
                    if not keeping:
                        if abs_word_end > (self._last_committed_end_sec + self._epsilon_sec):
                            keeping = True
                        else:
                            dropped_units += 1
                            continue
                    # Once first word passes threshold, keep ALL remaining
                    # words in this segment (prevents non-monotonic garbling).
                    all_kept.append((word.word, abs_word_end))
                continue

            # Segment-level fallback (no word timestamps).
            abs_seg_end = window_start + float(seg.end)
            if abs_seg_end > (self._last_committed_end_sec + self._epsilon_sec):
                all_kept.append((seg_text, abs_seg_end))
            else:
                dropped_units += 1

        # --- Step 2: Split into confirmed and tentative ---
        confirmed: list[tuple[str, float]] = []
        tentative: list[tuple[str, float]] = []
        for word_text, abs_end in all_kept:
            if abs_end <= cutoff + self._epsilon_sec:
                confirmed.append((word_text, abs_end))
            else:
                tentative.append((word_text, abs_end))

        # Update last_committed_end_sec for confirmed words only.
        for _, t in confirmed:
            self._last_committed_end_sec = max(self._last_committed_end_sec, t)

        confirmed_text = _join_words([w for w, _ in confirmed])
        tentative_text = _join_words([w for w, _ in tentative])

        # --- Step 3: Safety layers on confirmed text ---
        confirmed_text, boundary_dropped = self._strip_boundary_overlap(confirmed_text)
        confirmed_text = _strip_trailing_ellipsis(confirmed_text)

        # Store tentative for potential flush at stream end.
        self._pending_text = tentative_text

        # Update committed tail for next merge's boundary guard.
        if confirmed_text:
            for w in confirmed_text.split():
                nw = _normalize_word(w)
                if nw:
                    self._committed_tail.append(nw)

        meta = MergeMetadata(
            window_start=float(window_start),
            window_end=float(window_end),
            confirmed_units=len(confirmed),
            tentative_units=len(tentative),
            dropped_units=dropped_units,
            boundary_dropped=boundary_dropped,
            last_committed_before=last_before,
            last_committed_after=self._last_committed_end_sec,
        )
        logger.debug(
            "Segment merge: window=[%.2f, %.2f] cutoff=%.2f confirmed=%d tentative=%d"
            " dropped=%d boundary=%d last=%.2f->%.2f",
            meta.window_start, meta.window_end, cutoff,
            meta.confirmed_units, meta.tentative_units,
            meta.dropped_units, meta.boundary_dropped,
            meta.last_committed_before, meta.last_committed_after,
        )
        return confirmed_text, meta

    def flush(self) -> str:
        """
        Emit any remaining tentative text.  Call when the stream ends so
        the last window's trailing content is not lost.
        """
        text = self._pending_text
        self._pending_text = ""
        if text:
            text, _ = self._strip_boundary_overlap(text)
            text = _strip_trailing_ellipsis(text)
            for w in text.split():
                nw = _normalize_word(w)
                if nw:
                    self._committed_tail.append(nw)
        return text

    def _strip_boundary_overlap(self, text: str) -> tuple[str, int]:
        """
        Remove leading words of *text* that duplicate the tail of committed text.
        """
        if not text or not self._committed_tail:
            return text, 0

        words = text.split()
        if not words:
            return text, 0

        tail = list(self._committed_tail)

        overlap = 0
        max_check = min(len(tail), len(words))
        for n in range(1, max_check + 1):
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
