"""
Handles WebSocket audio: buffer, VAD, resample to 16kHz, and trigger transcription.
"""
import logging
import time
from collections import deque
from typing import Deque, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

TARGET_SAMPLE_RATE = 16_000


def rms_level(audio: np.ndarray) -> float:
    """Compute RMS of audio signal."""
    if audio.size == 0:
        return 0.0
    if audio.dtype in (np.int16, np.int32):
        audio = audio.astype(np.float64) / np.iinfo(audio.dtype).max
    return float(np.sqrt(np.mean(audio**2)))


def trim_silence(
    audio: np.ndarray,
    sample_rate: int,
    threshold: float = 0.01,
    frame_ms: float = 30.0,
    min_speech_ms: float = 250.0,
) -> np.ndarray:
    """Remove leading and trailing silence to reduce hallucinations."""
    if audio.size == 0:
        return audio
    if audio.dtype != np.float32:
        audio = audio.astype(np.float32) / np.iinfo(np.int16).max
    frame = int(sample_rate * frame_ms / 1000)
    min_speech = int(sample_rate * min_speech_ms / 1000)
    if len(audio) < min_speech:
        return audio

    energy = np.abs(audio)
    n_windows = max(1, len(audio) - frame)
    window_rms = np.array(
        [np.sqrt(np.mean(energy[i : i + frame] ** 2)) for i in range(0, n_windows)]
    )
    speech = window_rms > threshold
    if not np.any(speech):
        return audio

    first_speech = int(np.argmax(speech))
    last_speech = len(speech) - 1 - int(np.argmax(speech[::-1]))
    if last_speech < first_speech:
        return audio
    start = max(0, first_speech)
    end = min(len(audio), last_speech + frame)
    if start >= end or (end - start) < min_speech:
        return audio
    return audio[start:end]


def resample_to_16k(audio: np.ndarray, orig_sr: int) -> np.ndarray:
    """Resample audio to 16 kHz mono float32."""
    if orig_sr == TARGET_SAMPLE_RATE:
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32) / np.iinfo(np.int16).max
        return audio

    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0

    num_samples = int(len(audio) * TARGET_SAMPLE_RATE / orig_sr)
    indices = np.linspace(0, len(audio) - 1, num_samples)
    resampled = np.interp(indices, np.arange(len(audio)), audio)
    return resampled.astype(np.float32)


class WebAudioBuffer:
    """
    Buffers incoming WebSocket audio, runs VAD, and provides chunks for transcription.
    """

    def __init__(
        self,
        sample_rate: int,
        silence_threshold: float = 0.01,
        silence_duration_ms: float = 500.0,
        min_chunk_duration: float = 1.0,
        max_chunk_duration: float = 10.0,
    ):
        self._sample_rate = sample_rate
        self._silence_threshold = silence_threshold
        self._silence_duration_s = silence_duration_ms / 1000.0
        self._min_chunk_duration = min_chunk_duration
        self._max_chunk_duration = max_chunk_duration
        self._buffer: Deque[np.ndarray] = deque()
        self._total_samples = 0
        self._silence_start: Optional[float] = None
        max_samples = int(max_chunk_duration * sample_rate) + 4096
        self._max_samples = max_samples

    def append(self, chunk: np.ndarray) -> None:
        ch = chunk.flatten()
        if ch.dtype == np.int16:
            ch = ch.astype(np.float32) / 32768.0
        self._buffer.append(ch)
        self._total_samples += len(ch)
        while self._total_samples > self._max_samples:
            oldest = self._buffer.popleft()
            self._total_samples -= len(oldest)

    def duration_seconds(self) -> float:
        return self._total_samples / self._sample_rate

    def should_extract_chunk(self) -> Tuple[bool, str]:
        duration = self.duration_seconds()
        if duration < self._min_chunk_duration:
            return False, "buffer_too_short"
        if duration >= self._max_chunk_duration:
            return True, "max_duration_reached"

        lookback = min(int(0.2 * self._sample_rate), self._total_samples)
        if lookback <= 0:
            return False, "empty"
        flat = np.concatenate(list(self._buffer))
        tail = flat[-lookback:]
        level = rms_level(tail)
        now = time.monotonic()

        if level < self._silence_threshold:
            if self._silence_start is None:
                self._silence_start = now
            elif (now - self._silence_start) >= self._silence_duration_s:
                return True, "silence_detected"
        else:
            self._silence_start = None
        return False, "waiting"

    def extract_chunk(self) -> Optional[np.ndarray]:
        """Extract buffer as float32 at buffer sample rate."""
        if not self._buffer:
            return None
        self._silence_start = None
        data = np.concatenate(list(self._buffer))
        self._buffer.clear()
        self._total_samples = 0
        return data.astype(np.float32)
