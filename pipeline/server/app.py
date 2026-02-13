"""
FastAPI server: web UI with Start button, audio visualizer, and live transcript.
"""
import asyncio
import logging
import sys
import threading
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from pipeline.config import DEFAULT_CONFIG
from pipeline.server.audio_handler import (
    TARGET_SAMPLE_RATE,
    WebAudioBuffer,
    resample_to_16k,
    trim_silence,
)
from pipeline.src.transcriber import Transcriber

logger = logging.getLogger(__name__)

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
            device = cfg.DEVICE
            if device == "mps":
                try:
                    import torch
                    if not torch.backends.mps.is_available():
                        device = "cpu"
                except ImportError:
                    device = "cpu"
            _transcribers[model_size] = Transcriber(
                model_size=model_size,
                device=device,
                compute_type=cfg.COMPUTE_TYPE,
                context_window_size=cfg.CONTEXT_WINDOW_SIZE,
            )
        return _transcribers[model_size]


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


async def _process_file_chunk(
    ws: WebSocket,
    audio_int16: np.ndarray,
    sample_rate: int,
    language: Optional[str],
    model_size: str,
    last_context_ref: list,
) -> None:
    """Transcribe one file chunk and send result."""
    audio_float = audio_int16.astype(np.float32) / 32768.0
    audio_16k = resample_to_16k(audio_float, sample_rate)
    audio_16k = trim_silence(audio_16k, TARGET_SAMPLE_RATE)
    if len(audio_16k) < 100:
        return
    transcriber = get_transcriber(model_size)
    loop = asyncio.get_event_loop()
    try:
        text = await loop.run_in_executor(
            None,
            lambda: transcriber.transcribe_chunk(
                audio_16k,
                previous_context_text=last_context_ref[0] if last_context_ref else "",
                language=language,
            ),
        )
    except Exception:
        logger.exception("Transcription failed")
        return
    if text:
        last_context_ref[0] = text
        try:
            await ws.send_json({"type": "transcript", "text": text})
        except Exception:
            pass


async def file_chunk_worker(
    ws: WebSocket,
    queue: asyncio.Queue,
    sample_rate: int,
    language: Optional[str],
    model_size: str,
    last_context_ref: list,
) -> None:
    """Process file chunks sequentially so transcription streams during playback."""
    while True:
        item = await queue.get()
        if item is None:
            break
        audio_arr, sr = item
        await _process_file_chunk(ws, audio_arr, sr, language, model_size, last_context_ref)
        queue.task_done()


async def process_audio_loop(
    ws: WebSocket,
    buffer: WebAudioBuffer,
    sample_rate: int,
    language: Optional[str] = None,
    model_size: str = "small",
):
    transcriber = get_transcriber(model_size)
    last_context = ""
    poll_interval = 0.05

    while True:
        await asyncio.sleep(poll_interval)
        should_extract, reason = buffer.should_extract_chunk()
        if not should_extract:
            continue

        chunk = buffer.extract_chunk()
        if chunk is None or len(chunk) == 0:
            continue

        audio_16k = resample_to_16k(chunk, sample_rate)
        audio_16k = trim_silence(audio_16k, TARGET_SAMPLE_RATE)
        if len(audio_16k) < 100:
            continue

        loop = asyncio.get_event_loop()
        lang = language
        try:
            text = await loop.run_in_executor(
                None,
                lambda: transcriber.transcribe_chunk(
                    audio_16k,
                    previous_context_text=last_context,
                    language=lang,
                ),
            )
        except Exception:
            logger.exception("Transcription failed")
            continue

        if text:
            last_context = text
            try:
                await ws.send_json({"type": "transcript", "text": text})
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
                buffer = WebAudioBuffer(
                    sample_rate=sample_rate,
                    silence_threshold=DEFAULT_CONFIG.SILENCE_THRESHOLD,
                    silence_duration_ms=DEFAULT_CONFIG.SILENCE_DURATION_MS,
                    min_chunk_duration=DEFAULT_CONFIG.MIN_CHUNK_DURATION,
                    max_chunk_duration=DEFAULT_CONFIG.MAX_CHUNK_DURATION,
                )
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, lambda: get_transcriber(model_size))
                if source_type == "file":
                    file_queue = asyncio.Queue()
                    file_worker = asyncio.create_task(
                        file_chunk_worker(
                            ws, file_queue, sample_rate, language, model_size, last_context_ref
                        )
                    )
                elif task is None:
                    task = asyncio.create_task(
                        process_audio_loop(ws, buffer, sample_rate, language, model_size)
                    )
                await ws.send_json({"type": "ready", "sample_rate": sample_rate})
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
                        )
                        task = asyncio.create_task(
                            process_audio_loop(ws, buffer, sample_rate, language, model_size)
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
