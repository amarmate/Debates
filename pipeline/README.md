# Speech-to-Fact Pipeline

Real-time audio transcription using faster-whisper with a sliding-window context injection mechanism.

## Setup

Install pipeline dependencies:

```powershell
uv pip install -e ".[pipeline]"
```

Or with pip:

```powershell
pip install -r pipeline/requirements.txt
```

## Run

From the project root:

```powershell
uv run python -m pipeline.src.main
```

Or:

```powershell
python -m pipeline.src.main
```

Speak into your microphone. Transcription appears in the console as silence-separated chunks.

Press **Ctrl+C** to stop.

## Web Server (Browser UI)

Run the web app with Start button, live audio visualizer, and transcript:

```powershell
uv pip install -e ".[pipeline,server]"
uv run python -m pipeline.server
```

Open http://localhost:8000 in your browser.

## Configuration

Edit `pipeline/config.py` to adjust:

- `SILENCE_THRESHOLD` – RMS below which audio is considered silent
- `MIN_CHUNK_DURATION` / `MAX_CHUNK_DURATION` – chunk bounds
- `CONTEXT_WINDOW_SIZE` – characters of prior transcript to inject
- `MODEL_SIZE` – `tiny` | `base` | `small` | `medium` | `large-v3`
- `DEVICE` – `cuda` | `mps` (Mac) | `cpu`
- `DEBUG_MODE` – save chunks to `data/raw_audio/`
