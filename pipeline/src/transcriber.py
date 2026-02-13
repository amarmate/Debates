"""
Transcription using faster-whisper with context injection for continuity.
"""
import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Whisper's initial_prompt is limited to 224 tokens. Use ~4 chars/token heuristic for Portuguese.
WHISPER_PROMPT_MAX_TOKENS = 224
CHARS_PER_TOKEN_ESTIMATE = 4


def _truncate_context(text: str, max_chars: int) -> str:
    """Return the last max_chars of text, preserving word boundaries if possible."""
    if not text or max_chars <= 0:
        return ""
    text = text.strip()
    if len(text) <= max_chars:
        return text
    truncated = text[-max_chars:]
    # Prefer starting at a word boundary
    first_space = truncated.find(" ")
    if first_space > 0:
        return truncated[first_space + 1 :].strip()
    return truncated


def _estimate_tokens(text: str) -> int:
    """Rough token estimate for Portuguese (~4 chars/token)."""
    if not text:
        return 0
    return max(1, len(text.strip()) // CHARS_PER_TOKEN_ESTIMATE)


def _truncate_to_token_limit(
    initial_prompt: str,
    previous_context: str,
    max_tokens: int = WHISPER_PROMPT_MAX_TOKENS,
) -> str:
    """
    Build combined prompt and truncate to fit Whisper's 224-token limit.
    Never truncates initial_prompt; only truncates previous_context.
    """
    sep = " ... "
    if not initial_prompt and not previous_context:
        return ""
    if not previous_context:
        return initial_prompt.strip()
    if not initial_prompt:
        max_chars = max_tokens * CHARS_PER_TOKEN_ESTIMATE
        return _truncate_context(previous_context, max_chars)
    # Both present: keep initial_prompt intact, truncate previous_context as needed
    prefix = initial_prompt.strip()
    prefix_tokens = _estimate_tokens(prefix)
    sep_tokens = _estimate_tokens(sep)
    suffix_max_tokens = max(0, max_tokens - prefix_tokens - sep_tokens)
    suffix_max_chars = suffix_max_tokens * CHARS_PER_TOKEN_ESTIMATE
    truncated_suffix = _truncate_context(previous_context, suffix_max_chars)
    return f"{prefix}{sep}{truncated_suffix}" if truncated_suffix else prefix


class Transcriber:
    """
    Wraps faster-whisper and provides transcribe_chunk with previous-context injection.
    """

    def __init__(
        self,
        model_size: str = "small",
        device: str = "cpu",
        compute_type: str = "int8",
        context_window_size: int = 200,
        vad_filter: bool = False,
        repetition_penalty: float = 1.1,
        compression_ratio_threshold: float = 2.6,
    ):
        self._model_size = model_size
        self._device = device
        self._compute_type = compute_type
        self._context_window_size = context_window_size
        self._vad_filter = vad_filter
        self._repetition_penalty = repetition_penalty
        self._compression_ratio_threshold = compression_ratio_threshold
        self._model = None

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        try:
            from faster_whisper import WhisperModel
        except ImportError as e:
            raise ImportError(
                "faster-whisper is required for the pipeline. Install with: pip install faster-whisper"
            ) from e

        logger.info(
            "Loading faster-whisper model: size=%s, device=%s, compute_type=%s",
            self._model_size,
            self._device,
            self._compute_type,
        )
        self._model = WhisperModel(
            self._model_size,
            device=self._device,
            compute_type=self._compute_type,
        )
        logger.info("Model loaded")

    def transcribe_chunk(
        self,
        audio: np.ndarray,
        previous_context_text: Optional[str] = None,
        initial_prompt: Optional[str] = None,
        sample_rate: int = 16_000,
        language: Optional[str] = None,
    ) -> str:
        """
        Transcribe an audio chunk, priming the model with previous context.

        Args:
            audio: float32 mono audio
            previous_context_text: last N chars of prior transcript for continuity
            initial_prompt: domain prompt for first chunk (e.g. debate metadata)
            sample_rate: audio sample rate (default 16 kHz)
            language: language code (e.g. "en", "pt") or None for auto-detect

        Returns:
            Transcribed text (stripped)
        """
        self._ensure_model()

        # Always combine initial_prompt + previous_context; never use if/elif.
        # Truncate only previous_context to fit token limit; never truncate initial_prompt.
        truncated_context = ""
        if previous_context_text and previous_context_text.strip():
            truncated_context = _truncate_context(
                previous_context_text, self._context_window_size
            )
        effective_prompt = _truncate_to_token_limit(
            (initial_prompt or "").strip(),
            truncated_context,
        )

        if effective_prompt:
            logger.info(
                "Prompt [combined]: %s",
                effective_prompt[:200] + ("..." if len(effective_prompt) > 200 else ""),
            )

        segments, _ = self._model.transcribe(
            audio,
            language=language,
            initial_prompt=effective_prompt if effective_prompt else None,
            vad_filter=self._vad_filter,
            repetition_penalty=self._repetition_penalty,
            compression_ratio_threshold=self._compression_ratio_threshold,
        )

        parts = []
        for seg in segments:
            if seg.text:
                parts.append(seg.text)
        return " ".join(parts).strip()