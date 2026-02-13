"""
Thread-safe audio capture with VAD (Voice Activity Detection) and ring buffer.
"""
import logging
import threading
import time
from collections import deque
from typing import Deque, Optional, Tuple

import numpy as np
import sounddevice as sd

logger = logging.getLogger(__name__)


class RingBuffer:
    """Thread-safe ring buffer for audio data."""

    def __init__(self, max_samples: int, sample_rate: int, dtype: np.dtype = np.int16):
        self._max_samples = max_samples
        self._sample_rate = sample_rate
        self._dtype = dtype
        self._buffer: Deque[np.ndarray] = deque()
        self._total_samples = 0
        self._lock = threading.Lock()

    def append(self, chunk: np.ndarray) -> None:
        with self._lock:
            self._buffer.append(chunk.copy())
            self._total_samples += len(chunk)
            while self._total_samples > self._max_samples:
                oldest = self._buffer.popleft()
                self._total_samples -= len(oldest)

    def get_all_and_clear(self) -> Optional[np.ndarray]:
        with self._lock:
            if not self._buffer:
                return None
            data = np.concatenate(list(self._buffer))
            self._buffer.clear()
            self._total_samples = 0
            return data

    def duration_seconds(self) -> float:
        with self._lock:
            return self._total_samples / self._sample_rate

    def sample_count(self) -> int:
        with self._lock:
            return self._total_samples

    def get_tail_samples(self, n: int) -> np.ndarray:
        """Return the last n samples (for VAD). Returns empty array if buffer is empty."""
        with self._lock:
            if not self._buffer or n <= 0:
                return np.array([], dtype=np.int16)
            flat = np.concatenate(list(self._buffer))
            return flat[-n:] if len(flat) >= n else flat


def rms_level(audio: np.ndarray) -> float:
    """Compute RMS (root mean square) of audio signal."""
    if audio.size == 0:
        return 0.0
    if audio.dtype in (np.int16, np.int32):
        audio = audio.astype(np.float64) / np.iinfo(audio.dtype).max
    return float(np.sqrt(np.mean(audio**2)))


class AudioStream:
    """
    Captures audio from the microphone, accumulates it in a ring buffer,
    and provides VAD-based chunk extraction.
    """

    def __init__(
        self,
        sample_rate: int = 16_000,
        channels: int = 1,
        chunk_samples: int = 1024,
        silence_threshold: float = 0.01,
        silence_duration_ms: float = 500.0,
        min_chunk_duration: float = 1.0,
        max_chunk_duration: float = 10.0,
        debug: bool = False,
        debug_dir: Optional[str] = None,
    ):
        self._sample_rate = sample_rate
        self._channels = channels
        self._chunk_samples = chunk_samples
        self._silence_threshold = silence_threshold
        self._silence_duration_s = silence_duration_ms / 1000.0
        self._min_chunk_duration = min_chunk_duration
        self._max_chunk_duration = max_chunk_duration
        self._debug = debug
        self._debug_dir = debug_dir

        max_samples = int(max_chunk_duration * sample_rate) + chunk_samples * 2
        self._buffer = RingBuffer(max_samples, sample_rate)

        self._silence_start: Optional[float] = None
        self._stream: Optional[sd.InputStream] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        """Start capturing audio in a background thread."""
        if self._running:
            logger.warning("Audio stream already running")
            return

        self._running = True
        self._stream = sd.InputStream(
            samplerate=self._sample_rate,
            channels=self._channels,
            dtype=np.int16,
            blocksize=self._chunk_samples,
            callback=self._audio_callback,
        )
        self._stream.start()
        logger.info("Audio stream started (sample_rate=%s, channels=%s)", self._sample_rate, self._channels)

    def stop(self) -> None:
        """Stop capturing audio."""
        self._running = False
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        logger.info("Audio stream stopped")

    def _audio_callback(self, indata: np.ndarray, _frames: int, _time_info, _status) -> None:
        if _status:
            logger.warning("Sounddevice status: %s", _status)
        if self._running and indata is not None:
            chunk = indata.copy().flatten()
            self._buffer.append(chunk)

    def should_extract_chunk(self) -> Tuple[bool, str]:
        """
        Determine if we should extract a chunk for transcription.

        Returns:
            (should_extract, reason)
        """
        duration = self._buffer.duration_seconds()

        if duration < self._min_chunk_duration:
            return False, "buffer_too_short"

        if duration >= self._max_chunk_duration:
            return True, "max_duration_reached"

        # Check recent audio for silence (last ~200ms for VAD)
        lookback_samples = int(0.2 * self._sample_rate)
        tail = self._buffer.get_tail_samples(lookback_samples)
        if tail.size == 0:
            return False, "empty"

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
        """
        Extract and clear the current buffer, returning audio as float32 for Whisper.
        """
        data = self._buffer.get_all_and_clear()
        if data is None:
            return None

        self._silence_start = None

        # Convert to float32 for faster-whisper
        if data.dtype == np.int16:
            data = data.astype(np.float32) / 32768.0

        if self._debug and self._debug_dir:
            try:
                import os

                os.makedirs(self._debug_dir, exist_ok=True)
                path = os.path.join(self._debug_dir, f"chunk_{time.time():.3f}.npy")
                np.save(path, data)
                logger.debug("Saved debug chunk to %s", path)
            except OSError as e:
                logger.warning("Could not save debug chunk: %s", e)

        return data

    def get_sample_rate(self) -> int:
        return self._sample_rate
