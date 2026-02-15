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
_RECENT_TEXT_CHARS = 500  # chars of committed text to remember for hallucination check

# Pattern: sentence-end punctuation, then a proper noun phrase (2+ capitalized
# words, possibly with accents) followed by a comma.  This is the typical
# shape of a Whisper "speaker label" hallucination at speaker transitions.
_SPEAKER_LABEL_RE = re.compile(
    r"(?<=[.?!])"                                          # after sentence end
    r"(\s+)"                                               # whitespace
    r"([A-ZÀ-Ú][a-zà-ú]+(?:\s+[A-ZÀ-Ú][a-zà-ú]+)+)"   # ProperName ProperName+
    r",\s*"                                                # comma + optional space
)


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
    pending_recovered: int
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
        self._recent_committed = ""  # last N chars for hallucination detection

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

        # --- Step 1b: Recover pending words the current window skipped ---
        recovered_text = ""
        preserve_pending = False
        if all_kept:
            recovered_text = self._recover_lost_pending(all_kept)
            self._pending_text = ""
        elif self._pending_text:
            # Window produced nothing — preserve old tentative text so the
            # next window (with better coverage) can still recover it.
            preserve_pending = True
        else:
            self._pending_text = ""

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

        # Prepend any recovered pending words that the window skipped.
        if recovered_text:
            confirmed_text = (
                (recovered_text + " " + confirmed_text).strip()
                if confirmed_text
                else recovered_text
            )

        # --- Step 3: Safety layers on confirmed text ---
        confirmed_text, boundary_dropped = self._strip_boundary_overlap(confirmed_text)
        confirmed_text = _strip_trailing_ellipsis(confirmed_text)
        confirmed_text = self._strip_hallucinated_speaker_labels(confirmed_text)

        # Store tentative for potential flush at stream end.
        if tentative_text:
            self._pending_text = tentative_text
        elif not preserve_pending:
            self._pending_text = ""
        # else: keep old _pending_text — this window had no content

        # Update committed tail and recent text buffer.
        if confirmed_text:
            for w in confirmed_text.split():
                nw = _normalize_word(w)
                if nw:
                    self._committed_tail.append(nw)
            self._recent_committed = (
                self._recent_committed + " " + confirmed_text
            )[-_RECENT_TEXT_CHARS:]

        pending_recovered = len(recovered_text.split()) if recovered_text else 0
        meta = MergeMetadata(
            window_start=float(window_start),
            window_end=float(window_end),
            confirmed_units=len(confirmed),
            tentative_units=len(tentative),
            dropped_units=dropped_units,
            boundary_dropped=boundary_dropped,
            pending_recovered=pending_recovered,
            last_committed_before=last_before,
            last_committed_after=self._last_committed_end_sec,
        )
        logger.debug(
            "Segment merge: window=[%.2f, %.2f] cutoff=%.2f confirmed=%d tentative=%d"
            " dropped=%d boundary=%d recovered=%d last=%.2f->%.2f",
            meta.window_start, meta.window_end, cutoff,
            meta.confirmed_units, meta.tentative_units,
            meta.dropped_units, meta.boundary_dropped,
            meta.pending_recovered,
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
            text = self._strip_hallucinated_speaker_labels(text)
            for w in text.split():
                nw = _normalize_word(w)
                if nw:
                    self._committed_tail.append(nw)
            self._recent_committed = (
                self._recent_committed + " " + text
            )[-_RECENT_TEXT_CHARS:]
        return text

    def _recover_lost_pending(
        self, all_kept: list[tuple[str, float]]
    ) -> str:
        """
        Check if previous tentative words were skipped by the current window.

        When Whisper assigns slightly different timestamps across overlapping
        windows, tentative words from the previous window may be dropped in
        the current window (their re-transcribed timestamps fall at or below
        ``last_committed_end_sec + epsilon``).  Detect this by checking
        whether the current window's kept words cover the pending text, and
        recover any uncovered prefix so those words are not permanently lost.
        """
        if not self._pending_text:
            return ""

        pending_words = self._pending_text.split()
        if not pending_words or not all_kept:
            return ""

        pending_norm = [_normalize_word(w) for w in pending_words]
        kept_norm = [
            _normalize_word(w)
            for w, _ in all_kept[: len(pending_words) + _BOUNDARY_GUARD_WORDS]
        ]

        if not kept_norm:
            return ""

        # If the first kept word matches the first pending word, the window IS
        # re-transcribing the pending region — no recovery needed.
        if pending_norm[0] == kept_norm[0]:
            return ""

        # Find the longest suffix of pending that matches a prefix of kept.
        # Words before that suffix are "lost" (not re-transcribed).
        best_overlap_start = len(pending_words)  # default: all lost
        for i in range(1, len(pending_norm) + 1):
            suffix = pending_norm[i:]
            if not suffix:
                # Reached the end without a match — all pending words are lost.
                break
            if len(suffix) > len(kept_norm):
                continue
            if suffix == kept_norm[: len(suffix)]:
                best_overlap_start = i
                break

        if best_overlap_start == 0:
            return ""

        lost_words = pending_words[:best_overlap_start]
        lost_text = _join_words(lost_words)
        if lost_text:
            logger.info(
                "Pending recovery: %d word(s) not re-transcribed, committing: %s",
                len(lost_words),
                lost_text,
            )
        return lost_text

    def _strip_hallucinated_speaker_labels(self, text: str) -> str:
        """
        Remove hallucinated speaker labels from confirmed text.

        Whisper sometimes inserts the speaker's name (from prompt context) at
        speaker transitions — e.g. "apreciar? António José Seguro, o país
        não precisa".  The name is hallucinated, not in the audio.

        Detection: a proper noun phrase (2+ capitalized words + comma) right
        after sentence-ending punctuation, where the same phrase already
        appeared in recently committed text.
        """
        if not text or not self._recent_committed:
            return text

        recent_lower = self._recent_committed.lower()

        for m in _SPEAKER_LABEL_RE.finditer(text):
            name = m.group(2)
            if name.lower() in recent_lower:
                # Strip "Name Name, " but keep the sentence-end and whitespace.
                before = text[: m.start() + len(m.group(1))]
                after = text[m.end():]
                # Capitalize first letter of continuation.
                if after and after[0].islower():
                    after = after[0].upper() + after[1:]
                text = before + after
                logger.debug(
                    "Speaker-label guard: stripped hallucinated '%s' from confirmed text",
                    name,
                )
                break  # one per merge is enough

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
