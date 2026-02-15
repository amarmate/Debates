"""
Configuration for the Speech-to-Fact real-time transcription pipeline.
"""
from dataclasses import asdict, dataclass
from typing import Any, Literal


@dataclass(frozen=True)
class PipelineConfig:
    """Configuration controlling the sliding-window transcription pipeline."""

    # Voice Activity Detection (VAD)
    SILENCE_THRESHOLD: float = 0.01  # RMS threshold below which audio is "silent"
    SILENCE_DURATION_MS: float = 500.0  # ms of silence before triggering chunk extraction

    # Chunk boundaries
    MIN_CHUNK_DURATION: float = 1.0  # Minimum seconds of audio before transcribing
    MAX_CHUNK_DURATION: float = 10.0  # Maximum seconds before forcing a cut

    # Rolling buffer for low-latency live streaming (sub-10s)
    ROLLING_INTERVAL_SEC: float = 2.0  # Process every N seconds
    ROLLING_BUFFER_SEC: float = 14.0  # Keep last N seconds of audio for context

    # Context injection (sliding window)
    CONTEXT_WINDOW_SIZE: int = 450  # Characters from previous transcript to inject
    INITIAL_PROMPT_ENABLED: bool = True  # Use debate metadata / domain prompt for first chunk
    CONTEXT_INJECTION_ENABLED: bool = True  # Pass previous transcript context to each chunk

    # File playback: skip trim_silence on chunks (preserves boundaries, reduces clipping)
    TRIM_SILENCE_FILE_CHUNKS: bool = False

    # faster-whisper model
    VAD_FILTER: bool = False  # Silero VAD filter (saves resources when disabled)
    REPETITION_PENALTY: float = 1.1  # Penalize repeated tokens
    COMPRESSION_RATIO_THRESHOLD: float = 2.6  # Treat highly repetitive output as failed
    MODEL_SIZE: str = "small"  # tiny | base | small | medium | large-v2 | large-v3
    DEVICE: Literal["cuda", "mps", "cpu", "auto"] = "auto"
    COMPUTE_TYPE: Literal["float16", "int8", "auto"] = "auto"

    # Punctuation restoration (respunct)
    PUNCTUATION_RESTORE: bool = True  # Restore punctuation on transcribed chunks

    # Debug
    DEBUG_MODE: bool = False  # Save audio chunks to data/raw_audio/

    # Audio capture
    SAMPLE_RATE: int = 16_000  # Whisper expects 16 kHz
    CHANNELS: int = 1
    CHUNK_SAMPLES: int = 1024  # Samples per read
    FORMAT: str = "int16"


DEFAULT_CONFIG = PipelineConfig()

# Runtime config (mutable) - used by the web app; starts as default
_runtime_config = PipelineConfig()


def config_to_dict(cfg: PipelineConfig) -> dict[str, Any]:
    """Export config to a JSON-serializable dict."""
    return asdict(cfg)


def config_from_dict(d: dict[str, Any]) -> PipelineConfig:
    """Build config from dict; unknown keys are ignored."""
    valid = {f.name for f in PipelineConfig.__dataclass_fields__.values()}
    return PipelineConfig(**{k: v for k, v in d.items() if k in valid})


def get_config() -> PipelineConfig:
    """Return the current runtime config (used by the server)."""
    return _runtime_config


def update_config(**kwargs: Any) -> PipelineConfig:
    """Update runtime config with given fields. Returns the new config."""
    global _runtime_config
    d = config_to_dict(_runtime_config)
    d.update({k: v for k, v in kwargs.items() if k in d})
    _runtime_config = config_from_dict(d)
    return _runtime_config
