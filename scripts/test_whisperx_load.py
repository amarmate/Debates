#!/usr/bin/env python3
"""
Minimal test: load WhisperX HF model and transcribe a short audio clip.
Used to verify torch.load weights_only fix for pyannote/speechbrain.
"""
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def main():
    import tempfile

    import numpy as np
    import soundfile as sf

    # Create ~2 second silent wav (16kHz mono) for a fast test
    sr = 16000
    duration = 2.0
    samples = int(sr * duration)
    wav = np.zeros(samples, dtype=np.float32)
    # Add tiny noise so it's not completely empty
    wav += np.random.randn(samples).astype(np.float32) * 0.001

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        tmp_path = f.name
    try:
        sf.write(tmp_path, wav, sr)
        logger.info("Created test audio: %s (%.1fs)", tmp_path, duration)

        from transcribe_audio import transcribe_audio

        logger.info("Calling transcribe_audio with inesc-id/WhisperLv3-EP-X...")
        transcribe_audio(
            tmp_path,
            language="pt",
            model_size="inesc-id/WhisperLv3-EP-X",
            enable_diarization=False,
            enable_vad=False,
            enable_overlap_detection=False,
        )
        logger.info("SUCCESS: WhisperX model loaded and transcribe completed")
        return 0
    except Exception as e:
        logger.exception("FAILED: %s", e)
        return 1
    finally:
        Path(tmp_path).unlink(missing_ok=True)


if __name__ == "__main__":
    sys.exit(main())
