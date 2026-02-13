# Debates

Extract, transcribe, and benchmark political debate audio from CNN Portugal videos.

## Setup with uv

```powershell
uv venv
.venv\Scripts\activate
uv pip install -e .
playwright install chromium
```

On Linux or Docker, install Chromium system dependencies first (fixes `libnspr4.so` etc.):

```bash
playwright install-deps chromium
playwright install chromium
```

## Usage

**Download and transcribe all debates** (from `data/links/debates_unified.csv`):
```powershell
uv run python transcribe_all.py
```

**Download only:**
```powershell
uv run python scripts/download_all_debates.py
```

## Audio Transcription with Speaker Diarization

The `scripts/transcribe_audio.py` script transcribes audio files using OpenAI Whisper and automatically identifies speakers.

### Setup for Speaker Diarization

1. **Install dependencies:**
   ```powershell
   uv pip install -e .
   ```
   This will install all dependencies including `pyannote.audio`.

2. **Get a Hugging Face token:**
   - Go to https://huggingface.co/settings/tokens
   - Create a new token (read access is sufficient)

3. **Accept the model terms:**
   - Visit https://huggingface.co/pyannote/speaker-diarization-3.1
   - Accept the model terms and conditions

4. **Authenticate:**
   ```powershell
   huggingface-cli login
   ```
   Enter your token when prompted.

### Usage

**Basic transcription with speaker diarization:**
```powershell
python scripts/transcribe_audio.py audio_file.mp3
```

**Specify ASR model** (Whisper size or Hugging Face path):
```powershell
# Standard Whisper models (tiny, base, small, medium, large)
python scripts/transcribe_audio.py audio_file.mp3 base

# European Portuguese fine-tuned model (requires whisperx)
python scripts/transcribe_audio.py audio_file.mp3 inesc-id/WhisperLv3-EP-X
```

**Specify number of speakers (if known):**
```powershell
python scripts/transcribe_audio.py audio_file.mp3 base 2
```

**Disable speaker diarization:**
```powershell
python scripts/transcribe_audio.py audio_file.mp3 base --no-diarization
```

**Disable overlap detection** (faster; segments where two speakers talk at once won't be flagged):
```powershell
python scripts/transcribe_audio.py audio_file.mp3 --no-overlap-detection
```

**Whisper anti-repetition options** (reduce repetitive hallucinations):
```powershell
python scripts/transcribe_audio.py audio_file.mp3 --condition-on-previous-text false --compression-ratio-threshold 2.0
```

The output will be saved to `data/transcripts/{audio_stem}_{model}.txt` (e.g. `data/transcripts/2025-04-07_AD-vs-CDU_TVI_base.txt`) with speaker annotations in the format:
```
[SPEAKER_00]:
Text spoken by speaker 00...

[SPEAKER_01]:
Text spoken by speaker 01...
```

## Project Structure

```
Debates/
├── transcribe_all.py  # Main entry: download + transcribe all debates (run from root)
├── pipeline/         # Speech-to-Fact: real-time transcription
│   ├── config.py     # Configuration (VAD, chunk bounds, model)
│   ├── src/
│   │   ├── audio_stream.py   # Mic capture, VAD, ring buffer
│   │   ├── transcriber.py    # faster-whisper + context injection
│   │   └── main.py          # Main loop
│   └── requirements.txt
├── scripts/          # Supporting modules and utility scripts
│   ├── debate_downloader.py
│   ├── download_all_debates.py
│   ├── transcribe_audio.py
│   ├── benchmark_transcription.py
│   ├── cut_audio.py
│   └── migrate_links_to_unified.py
└── tests/            # Test scripts
    ├── test_first_debate.py   # End-to-end: first debate
    └── test_whisperx_load.py  # WhisperX model load test
```

## Scripts

| Script | Purpose |
|--------|---------|
| `transcribe_all.py` | **Main entry**: download + transcribe all debates |
| `scripts/benchmark_transcription.py` | Compare transcription to reference, compute WER/CER |
| `scripts/cut_audio.py` | Cut MP3 by start/end or duration |
| `scripts/migrate_links_to_unified.py` | Regenerate `data/links/debates_unified.csv` from per-election CSVs |

**Test first debate** (download + transcribe):
```powershell
uv run python tests/test_first_debate.py
```

**Benchmark** (add refs to `data/benchmark/refs/`):
```powershell
uv run python scripts/benchmark_transcription.py
```

**Regenerate unified links:**
```powershell
uv run python scripts/migrate_links_to_unified.py
```

## Docker

Run the pipeline in a container (no local PyAV/FFmpeg dev libs or Chromium needed). All Docker files live in **`docker/`**—see **[`docker/README.md`](docker/README.md)** for full instructions.

```bash
# Quick start (from project root)
docker build -f docker/Dockerfile -t debates:latest .
docker run --rm -e HF_TOKEN=<token> -v "$(pwd)/data:/app/data" debates:latest
```

## Speech-to-Fact (Real-Time Transcription)

Real-time, low-latency transcription with faster-whisper and sliding-window context injection.

**Setup:**
```powershell
uv pip install -e ".[pipeline]"
```

**Terminal mode:**
```powershell
uv run python -m pipeline.src.main
```

**Web UI (Start button, visualizer, transcript):**
```powershell
uv pip install -e ".[pipeline,server]"
uv run python -m pipeline.server
```
Then open http://localhost:8000 in your browser.

See [pipeline/README.md](pipeline/README.md) for configuration.

## Dependencies

- `playwright` - For browser automation to intercept network requests
- `yt-dlp` - For downloading and converting video streams to audio
- `openai-whisper` - For audio transcription (tiny, base, small, medium, large)
- `whisperx` - For Hugging Face models (e.g. `inesc-id/WhisperLv3-EP-X` for European Portuguese)
- `pyannote.audio` - For speaker diarization and overlapped speech detection
- `librosa` - For audio processing
- `tqdm` - For progress bars
- `jiwer` - For WER/CER benchmarking


