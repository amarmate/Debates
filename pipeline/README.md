# Speech-to-Fact Pipeline

Real-time audio transcription using faster-whisper with a sliding-window context injection mechanism.

## Installation

### Web app (recommended)

Install pipeline and server dependencies:

```powershell
uv pip install -e ".[pipeline,server]"
```

Or with pip:

```powershell
pip install -r pipeline/requirements.txt
pip install fastapi uvicorn[standard]
```

### Terminal-only mode

If you only need the console transcription (no web UI):

```powershell
uv pip install -e ".[pipeline]"
```

Or:

```powershell
pip install -r pipeline/requirements.txt
```

## Web App

### Run

From the project root:

```powershell
uv run python -m pipeline.server
```

Or:

```powershell
python -m pipeline.server
```

### Host and port

By default the server binds to **0.0.0.0:8000** (all interfaces, port 8000). Open **http://localhost:8000** in your browser, or use your machine’s IP if accessing from another device.

**Change port:**

```powershell
uv run python -m pipeline.server --port 9000
```

Or set the `PORT` environment variable:

```powershell
$env:PORT = 9000; uv run python -m pipeline.server
```

**Change host (IP address):**

```powershell
# Listen only on localhost (no external access)
uv run python -m pipeline.server --host 127.0.0.1

# Listen on a specific interface (e.g. LAN IP)
uv run python -m pipeline.server --host 192.168.1.100
```

**Combine host and port:**

```powershell
uv run python -m pipeline.server --host 127.0.0.1 --port 3000
```

If you get `ModuleNotFoundError: No module named 'fastapi'`, install with the `server` extra:

```powershell
uv pip install -e ".[pipeline,server]"
```

### Usage

1. **Source** – Choose **Microphone** (live speech) or **Audio file** (pre-recorded).
2. **File** – When using Audio file, select a file from the dropdown. Files are loaded from `data/debates/` (MP3, WAV, M4A, OGG, FLAC).
3. **Model** – Pick a Whisper model: Tiny (fastest), Base, Small, Medium, Turbo, Large v2/v3.
4. **Language** – Auto-detect or force a language (e.g. Portuguese, English).
5. **Debug** – Check “Show frame transcriptions” to see raw per-window output.
6. **Start** – Click to begin. For microphone, allow browser access when prompted.

### Output

- **Transcript** – Live text as you speak or as the file plays.
- **Sentences (for fact-checking)** – Complete sentences, one per line, suitable for fact-checking.
- **Debug frames** – Raw transcription per time window (when enabled).

### Settings tab

Use the **Settings** tab to adjust VAD, rolling buffer, Whisper tuning, and features. Changes are saved to the server and apply to the next session.

## Terminal Mode

From the project root:

```powershell
uv run python -m pipeline.src.main
```

Speak into your microphone. Transcription appears in the console as silence-separated chunks.

Press **Ctrl+C** to stop.

## Configuration

Edit `pipeline/config.py` to adjust defaults:

- `SILENCE_THRESHOLD` – RMS below which audio is considered silent
- `MIN_CHUNK_DURATION` / `MAX_CHUNK_DURATION` – chunk bounds
- `CONTEXT_WINDOW_SIZE` – characters of prior transcript to inject
- `MODEL_SIZE` – `tiny` | `base` | `small` | `medium` | `large-v3`
- `DEVICE` – `cuda` | `mps` (Mac) | `cpu`
- `DEBUG_MODE` – save chunks to `data/raw_audio/`
