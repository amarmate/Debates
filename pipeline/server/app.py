"""
FastAPI server: web UI with Start button, audio visualizer, and live transcript.
"""
import asyncio
import logging
import sys
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, FileResponse, Response
from fastapi.staticfiles import StaticFiles

from pipeline.config import DEFAULT_CONFIG
from pipeline.debate_metadata import build_initial_prompt, lookup_debate_metadata
from pipeline.punctuation_restore import cleanup_chunk_artifacts, restore_punctuation
from pipeline.utils import resolve_compute_type, resolve_device
from pipeline.server.audio_handler import (
    TARGET_SAMPLE_RATE,
    WebAudioBuffer,
    resample_to_16k,
    trim_silence,
)
from pipeline.src.transcriber import Transcriber

logger = logging.getLogger(__name__)


def extract_new_suffix(last_sent: str, text_new: str) -> str:
    """
    Extract only the new portion of transcript to avoid duplicate output.
    For rolling buffer: Whisper returns full transcript per window; windows overlap.
    Find overlap at boundary (end of last_sent matches start of text_new) and return remainder.
    """
    if not last_sent:
        return text_new.strip()
    text_new = text_new.strip()
    if not text_new:
        return ""
    # Cumulative case: new text extends last_sent
    if text_new.startswith(last_sent) and len(text_new) > len(last_sent):
        return text_new[len(last_sent) :].lstrip()
    last_stripped = last_sent.rstrip()
    if text_new.startswith(last_stripped) and len(text_new) > len(last_stripped):
        return text_new[len(last_stripped) :].lstrip()
    # Overlapping windows: find longest overlap at boundary
    max_overlap = min(len(last_sent), len(text_new))
    for i in range(max_overlap, 0, -1):
        if last_sent[-i:] == text_new[:i]:
            suffix = text_new[i:].strip()
            return suffix if suffix else text_new
    return text_new


app = FastAPI(title="Speech-to-Fact")

STATIC_DIR = Path(__file__).parent / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Audio files directory (project_root/data/debates)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
AUDIO_DIR = PROJECT_ROOT / "data" / "debates"
AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".ogg", ".flac"}


_transcribers: dict = {}
_transcriber_lock = threading.Lock()


def get_transcriber(model_size: str = "small") -> Transcriber:
    with _transcriber_lock:
        if model_size not in _transcribers:
            cfg = DEFAULT_CONFIG
            device = resolve_device(cfg.DEVICE)
            compute_type = resolve_compute_type(cfg.COMPUTE_TYPE, device)
            _transcribers[model_size] = Transcriber(
                model_size=model_size,
                device=device,
                compute_type=compute_type,
                context_window_size=cfg.CONTEXT_WINDOW_SIZE,
                vad_filter=cfg.VAD_FILTER,
                repetition_penalty=cfg.REPETITION_PENALTY,
                compression_ratio_threshold=cfg.COMPRESSION_RATIO_THRESHOLD,
            )
        return _transcribers[model_size]


@app.get("/favicon.ico")
@app.get("/health.ico")
async def no_icon():
    """Avoid 404 for browser/monitor icon requests."""
    return Response(status_code=204)


@app.get("/api/audio/files")
async def list_audio_files():
    """List audio files available in data/debates on the server."""
    if not AUDIO_DIR.exists():
        return []
    files = []
    for p in sorted(AUDIO_DIR.iterdir()):
        if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS:
            files.append(p.name)
    return files


@app.get("/api/audio/{filename}")
async def get_audio_file(filename: str):
    """Serve a specific audio file from data/debates."""
    if ".." in filename or "/" in filename or "\\" in filename:
        raise HTTPException(400, "Invalid filename")
    path = AUDIO_DIR / filename
    if not path.is_file() or path.suffix.lower() not in AUDIO_EXTENSIONS:
        raise HTTPException(404, "File not found")
    return FileResponse(path, media_type="audio/mpeg" if path.suffix.lower() == ".mp3" else None)


@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = STATIC_DIR / "index.html"
    if html_path.exists():
        return FileResponse(html_path)
    return HTMLResponse("<h1>Speech-to-Fact</h1><p>Place index.html in pipeline/server/static/</p>")


async def file_rolling_worker(
    ws: WebSocket,
    queue: asyncio.Queue,
    sample_rate: int,
    language: Optional[str],
    model_size: str,
    first_prompt_ref: list,
    trim_silence_file: bool,
    debug_frames: bool = False,
) -> None:
    """
    Process file chunks with rolling buffer: append to buffer, transcribe last N sec
    every ROLLING_INTERVAL_SEC. Produces overlapping windows like [0-3], [0-6], [0-9], [3-12].
    """
    cfg = DEFAULT_CONFIG
    transcriber = get_transcriber(model_size)
    buffer = WebAudioBuffer(
        sample_rate=sample_rate,
        max_chunk_duration=cfg.ROLLING_BUFFER_SEC,
        max_buffer_seconds=cfg.ROLLING_BUFFER_SEC,
    )
    last_context = ""
    last_sent = ""
    last_process_sec = 0.0
    min_samples = int(cfg.MIN_CHUNK_DURATION * TARGET_SAMPLE_RATE)

    while True:
        item = await queue.get()
        if item is None:
            break
        audio_arr, sr = item
        buffer.append(audio_arr)
        elapsed_sec = buffer.duration_seconds()

        should_process = (
            elapsed_sec - last_process_sec >= cfg.ROLLING_INTERVAL_SEC
            or (last_process_sec == 0 and elapsed_sec >= cfg.MIN_CHUNK_DURATION)
        )
        if not should_process or elapsed_sec < cfg.MIN_CHUNK_DURATION:
            queue.task_done()
            continue

        audio = buffer.get_rolling_audio(seconds=cfg.ROLLING_BUFFER_SEC)
        if audio is None or len(audio) < min_samples:
            queue.task_done()
            continue

        last_process_sec = elapsed_sec
        audio_16k = resample_to_16k(audio, sample_rate)
        if trim_silence_file:
            audio_16k = trim_silence(audio_16k, TARGET_SAMPLE_RATE)
        if len(audio_16k) < min_samples:
            queue.task_done()
            continue

        initial_prompt = first_prompt_ref[0] if first_prompt_ref else None
        previous_context = last_context or None

        def _transcribe():
            t = transcriber.transcribe_chunk(
                audio_16k,
                previous_context_text=previous_context,
                initial_prompt=initial_prompt,
                language=language,
            )
            if t and cfg.PUNCTUATION_RESTORE:
                t = restore_punctuation(t)
            if t:
                t = cleanup_chunk_artifacts(t)
            return t

        loop = asyncio.get_event_loop()
        try:
            text = await loop.run_in_executor(None, _transcribe)
        except Exception:
            logger.exception("Transcription failed")
            queue.task_done()
            continue

        if debug_frames:
            window_start = max(0.0, elapsed_sec - cfg.ROLLING_BUFFER_SEC)
            try:
                await ws.send_json({
                    "type": "debug_frame",
                    "time_range": [round(window_start, 1), round(elapsed_sec, 1)],
                    "raw": text or "",
                })
            except Exception:
                pass

        if text:
            last_context = text
            to_send = extract_new_suffix(last_sent, text)
            if to_send:
                last_sent = text
                try:
                    await ws.send_json({"type": "transcript", "text": to_send})
                except Exception:
                    pass

        queue.task_done()


async def process_audio_loop(
    ws: WebSocket,
    buffer: WebAudioBuffer,
    sample_rate: int,
    language: Optional[str] = None,
    model_size: str = "small",
    initial_prompt: Optional[str] = None,
    debug_frames: bool = False,
):
    cfg = DEFAULT_CONFIG
    transcriber = get_transcriber(model_size)
    last_context = ""
    last_sent = ""
    last_process = 0.0
    session_start = time.monotonic()
    poll_interval = 0.05
    min_samples = int(cfg.MIN_CHUNK_DURATION * TARGET_SAMPLE_RATE)

    async def _send_debug_frame(elapsed: float, text: str) -> None:
        if not debug_frames:
            return
        window_start = max(0.0, elapsed - cfg.ROLLING_BUFFER_SEC)
        window_end = elapsed
        try:
            await ws.send_json({
                "type": "debug_frame",
                "time_range": [round(window_start, 1), round(window_end, 1)],
                "raw": text or "",
            })
        except Exception:
            pass

    while True:
        await asyncio.sleep(poll_interval)
        now = time.monotonic()
        should_by_time = (now - last_process) >= cfg.ROLLING_INTERVAL_SEC
        should_by_vad, _ = buffer.should_extract_chunk()
        if not should_by_time and not should_by_vad:
            continue

        audio = buffer.get_rolling_audio(seconds=cfg.ROLLING_BUFFER_SEC)
        if audio is None or len(audio) < min_samples:
            continue

        audio_16k = resample_to_16k(audio, sample_rate)
        audio_16k = trim_silence(audio_16k, TARGET_SAMPLE_RATE)
        if len(audio_16k) < min_samples:
            continue

        last_process = now
        elapsed = now - session_start
        loop = asyncio.get_event_loop()
        lang = language

        def _transcribe_mic():
            t = transcriber.transcribe_chunk(
                audio_16k,
                previous_context_text=last_context or None,
                initial_prompt=initial_prompt,
                language=lang,
            )
            if t and cfg.PUNCTUATION_RESTORE:
                t = restore_punctuation(t)
            if t:
                t = cleanup_chunk_artifacts(t)
            return t

        try:
            text = await loop.run_in_executor(None, _transcribe_mic)
        except Exception:
            logger.exception("Transcription failed")
            continue

        await _send_debug_frame(elapsed, text)

        if text:
            last_context = text
            to_send = extract_new_suffix(last_sent, text)
            if to_send:
                last_sent = text
                try:
                    await ws.send_json({"type": "transcript", "text": to_send})
                except Exception:
                    break


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    buffer: Optional[WebAudioBuffer] = None
    sample_rate = DEFAULT_CONFIG.SAMPLE_RATE
    language: Optional[str] = None
    model_size = DEFAULT_CONFIG.MODEL_SIZE
    source_type = "mic"
    debug_frames = False
    last_context_ref: list = [""]
    task: Optional[asyncio.Task] = None
    file_queue: Optional[asyncio.Queue] = None
    file_worker: Optional[asyncio.Task] = None

    try:
        while True:
            try:
                msg = await ws.receive()
            except RuntimeError as e:
                if "disconnect" in str(e).lower():
                    break
                raise
            if "text" in msg:
                import json as _json
                cfg = _json.loads(msg["text"])
                sample_rate = int(cfg.get("sample_rate", DEFAULT_CONFIG.SAMPLE_RATE))
                language = cfg.get("language") or None
                model_size = cfg.get("model_size") or DEFAULT_CONFIG.MODEL_SIZE
                source_type = cfg.get("source") or "mic"
                filename = cfg.get("filename") or None
                debug_frames = bool(cfg.get("debug_frames", False))
                buffer = WebAudioBuffer(
                    sample_rate=sample_rate,
                    silence_threshold=DEFAULT_CONFIG.SILENCE_THRESHOLD,
                    silence_duration_ms=DEFAULT_CONFIG.SILENCE_DURATION_MS,
                    min_chunk_duration=DEFAULT_CONFIG.MIN_CHUNK_DURATION,
                    max_chunk_duration=DEFAULT_CONFIG.MAX_CHUNK_DURATION,
                    max_buffer_seconds=DEFAULT_CONFIG.ROLLING_BUFFER_SEC,
                )
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, lambda: get_transcriber(model_size))
                if source_type == "file":
                    file_queue = asyncio.Queue()
                    audio_path = (AUDIO_DIR / filename) if filename else None
                    metadata = lookup_debate_metadata(audio_path) if audio_path else None
                    initial_prompt = build_initial_prompt(metadata)
                    first_prompt_ref = [initial_prompt] if initial_prompt else [None]
                    file_worker = asyncio.create_task(
                        file_rolling_worker(
                            ws, file_queue, sample_rate, language, model_size,
                            first_prompt_ref,
                            DEFAULT_CONFIG.TRIM_SILENCE_FILE_CHUNKS,
                            debug_frames=debug_frames,
                        )
                    )
                elif task is None:
                    task = asyncio.create_task(
                        process_audio_loop(
                            ws, buffer, sample_rate, language, model_size,
                            debug_frames=debug_frames,
                        )
                    )
                ready_payload = {"type": "ready", "sample_rate": sample_rate}
                if source_type == "file":
                    ready_payload["file_chunk_duration"] = DEFAULT_CONFIG.ROLLING_INTERVAL_SEC
                await ws.send_json(ready_payload)
            elif "bytes" in msg:
                data = msg["bytes"]
                if len(data) < 2:
                    continue
                arr = np.frombuffer(data, dtype=np.int16)
                if source_type == "file" and file_queue is not None:
                    await file_queue.put((arr.copy(), sample_rate))
                else:
                    if buffer is None:
                        buffer = WebAudioBuffer(
                            sample_rate=sample_rate,
                            silence_threshold=DEFAULT_CONFIG.SILENCE_THRESHOLD,
                            silence_duration_ms=DEFAULT_CONFIG.SILENCE_DURATION_MS,
                            min_chunk_duration=DEFAULT_CONFIG.MIN_CHUNK_DURATION,
                            max_chunk_duration=DEFAULT_CONFIG.MAX_CHUNK_DURATION,
                            max_buffer_seconds=DEFAULT_CONFIG.ROLLING_BUFFER_SEC,
                        )
                        task = asyncio.create_task(
                            process_audio_loop(
                                ws, buffer, sample_rate, language, model_size,
                                debug_frames=debug_frames,
                            )
                        )
                    buffer.append(arr)
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception:
        logger.exception("WebSocket error")
    finally:
        if file_queue is not None:
            await file_queue.put(None)
        if file_worker and not file_worker.done():
            await file_worker
        if task and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass


def run_server(host: str = "0.0.0.0", port: int = 8000):
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    run_server()
