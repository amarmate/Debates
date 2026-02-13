"""
Configuration for the Speech-to-Fact real-time transcription pipeline.
"""
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class PipelineConfig:
    """Configuration controlling the sliding-window transcription pipeline."""

    # Voice Activity Detection (VAD)
    SILENCE_THRESHOLD: float = 0.01  # RMS threshold below which audio is "silent"
    SILENCE_DURATION_MS: float = 500.0  # ms of silence before triggering chunk extraction

    # Chunk boundaries
    MIN_CHUNK_DURATION: float = 1.0  # Minimum seconds of audio before transcribing
    MAX_CHUNK_DURATION: float = 10.0  # Maximum seconds before forcing a cut
    FILE_CHUNK_DURATION: float = 4.0  # Seconds per chunk when playing file (configurable)

    # Context injection (sliding window)
    CONTEXT_WINDOW_SIZE: int = 200  # Characters from previous transcript to inject

    # faster-whisper model
    VAD_FILTER: bool = False  # Silero VAD filter (saves resources when disabled)
    MODEL_SIZE: str = "small"  # tiny | base | small | medium | large-v2 | large-v3
    DEVICE: Literal["cuda", "mps", "cpu", "auto"] = "auto"
    COMPUTE_TYPE: Literal["float16", "int8"] = "int8"

    # Debug
    DEBUG_MODE: bool = False  # Save audio chunks to data/raw_audio/

    # Audio capture
    SAMPLE_RATE: int = 16_000  # Whisper expects 16 kHz
    CHANNELS: int = 1
    CHUNK_SAMPLES: int = 1024  # Samples per read
    FORMAT: str = "int16"


DEFAULT_CONFIG = PipelineConfig()
