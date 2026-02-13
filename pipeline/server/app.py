"""
FastAPI server: web UI with Start button, audio visualizer, and live transcript.
"""
import asyncio
import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from pipeline.config import DEFAULT_CONFIG
from pipeline.server.audio_handler import WebAudioBuffer, resample_to_16k
from pipeline.src.transcriber import Transcriber

logger = logging.getLogger(__name__)

app = FastAPI(title="Speech-to-Fact")

STATIC_DIR = Path(__file__).parent / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


_transcribers: dict = {}


def get_transcriber(model_size: str = "small") -> Transcriber:
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


@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = STATIC_DIR / "index.html"
    if html_path.exists():
        return FileResponse(html_path)
    return HTMLResponse("<h1>Speech-to-Fact</h1><p>Place index.html in pipeline/server/static/</p>")


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
    task: Optional[asyncio.Task] = None

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
                buffer = WebAudioBuffer(
                    sample_rate=sample_rate,
                    silence_threshold=DEFAULT_CONFIG.SILENCE_THRESHOLD,
                    silence_duration_ms=DEFAULT_CONFIG.SILENCE_DURATION_MS,
                    min_chunk_duration=DEFAULT_CONFIG.MIN_CHUNK_DURATION,
                    max_chunk_duration=DEFAULT_CONFIG.MAX_CHUNK_DURATION,
                )
                if task is None:
                    task = asyncio.create_task(
                        process_audio_loop(ws, buffer, sample_rate, language, model_size)
                    )
                await ws.send_json({"type": "ready", "sample_rate": sample_rate})
            elif "bytes" in msg:
                data = msg["bytes"]
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
                if len(data) >= 2:
                    arr = np.frombuffer(data, dtype=np.int16)
                    buffer.append(arr)
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception:
        logger.exception("WebSocket error")
    finally:
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
