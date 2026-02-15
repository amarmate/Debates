"""
Speech-to-Fact: real-time transcription main loop.
"""
import logging
import sys
import time

from pipeline.config import DEFAULT_CONFIG
from pipeline.utils import resolve_compute_type, resolve_device
from pipeline.src.audio_stream import AudioStream
from pipeline.src.transcriber import Transcriber, segments_to_text

logger = logging.getLogger(__name__)


def run_pipeline(config=None) -> None:
    """
    Run the continuous transcription loop.

    - Accumulates audio from the microphone
    - On silence (after min duration) or max duration: transcribes and prints
    - Updates context with new transcript for next chunk
    """
    cfg = config or DEFAULT_CONFIG

    device = resolve_device(cfg.DEVICE)
    compute_type = resolve_compute_type(cfg.COMPUTE_TYPE, device)

    debug_dir = "data/raw_audio" if cfg.DEBUG_MODE else None

    stream = AudioStream(
        sample_rate=cfg.SAMPLE_RATE,
        channels=cfg.CHANNELS,
        chunk_samples=cfg.CHUNK_SAMPLES,
        silence_threshold=cfg.SILENCE_THRESHOLD,
        silence_duration_ms=cfg.SILENCE_DURATION_MS,
        min_chunk_duration=cfg.MIN_CHUNK_DURATION,
        max_chunk_duration=cfg.MAX_CHUNK_DURATION,
        debug=cfg.DEBUG_MODE,
        debug_dir=debug_dir,
    )

    transcriber = Transcriber(
        model_size=cfg.MODEL_SIZE,
        device=device,
        compute_type=compute_type,
        context_window_size=cfg.CONTEXT_WINDOW_SIZE,
    )

    last_context = ""
    stream.start()

    try:
        logger.info("Listening. Speak into the microphone. Press Ctrl+C to stop.")
        poll_interval = 0.05  # 50ms

        while True:
            time.sleep(poll_interval)
            should_extract, reason = stream.should_extract_chunk()
            if not should_extract:
                continue

            chunk = stream.extract_chunk()
            if chunk is None or len(chunk) == 0:
                continue

            logger.debug("Transcribing chunk (reason=%s, %.2fs)", reason, len(chunk) / cfg.SAMPLE_RATE)
            try:
                segments = transcriber.transcribe_chunk(
                    chunk,
                    previous_context_text=last_context,
                    sample_rate=cfg.SAMPLE_RATE,
                )
            except Exception:
                logger.exception("Transcription failed")
                continue

            text = segments_to_text(segments)
            if text:
                logger.info("%s", text)
                sys.stdout.flush()
                last_context = text
    except KeyboardInterrupt:
        logger.info("Stopped by user")
    finally:
        stream.stop()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    run_pipeline()
