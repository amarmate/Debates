# Docker – Debates Pipeline

Run the debate download and transcription pipeline in a reproducible container. No need to install PyAV, FFmpeg dev libs, Chromium, PyTorch, or other heavy dependencies locally.

---

## Prerequisites

- **Docker** installed and running ([docker.com](https://docs.docker.com/get-docker/))
- **Hugging Face token** (optional but required for speaker diarization):
  1. Create a token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
  2. Accept terms at [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
  3. Use it as `HF_TOKEN` when running the container

---

## Quick Start

From the **project root** (parent of `docker/`):

```bash
# 1. Build the image
docker build -f docker/Dockerfile -t debates:latest .

# 2. Run the full pipeline (download + transcribe all debates from data/links/)
docker run --rm -v "$(pwd)/data:/app/data" debates:latest

# 3. With speaker diarization (requires HF_TOKEN)
docker run --rm -e HF_TOKEN="hf_xxxx" -v "$(pwd)/data:/app/data" debates:latest
```

**PowerShell (Windows):**
```powershell
docker build -f docker/Dockerfile -t debates:latest .
docker run --rm -v "${PWD}/data:/app/data" debates:latest
```

---

## Common Commands

### Test the first debate only
```bash
docker run --rm -v "$(pwd)/data:/app/data" debates:latest \
  uv run python scripts/test_first_debate.py
```

### Use WhisperX European Portuguese model
```bash
docker run --rm -e HF_TOKEN="hf_xxxx" -v "$(pwd)/data:/app/data" debates:latest \
  uv run python process_debates.py --model inesc-id/WhisperLv3-EP-X
```

### Use a specific Whisper model (tiny, base, small, medium, large)
```bash
docker run --rm -e HF_TOKEN="hf_xxxx" -v "$(pwd)/data:/app/data" debates:latest \
  uv run python process_debates.py --model large
```

### Download only (no transcription)
```bash
docker run --rm -v "$(pwd)/data:/app/data" debates:latest \
  uv run python download_all_debates.py
```

### Transcribe a single file
```bash
docker run --rm -e HF_TOKEN="hf_xxxx" -v "$(pwd)/data:/app/data" debates:latest \
  uv run python transcribe_audio.py data/debates/your_file.mp3 base
```

### Disable diarization / VAD / overlap detection
```bash
docker run --rm -v "$(pwd)/data:/app/data" debates:latest \
  uv run python process_debates.py --no-diarization --no-vad --no-overlap-detection
```

---

## Using docker-compose

From the project root:

```bash
# Build
docker compose -f docker/docker-compose.yml build

# Run full pipeline (HF_TOKEN from environment)
export HF_TOKEN="hf_xxxx"
docker compose -f docker/docker-compose.yml run --rm debates

# Test first debate
docker compose -f docker/docker-compose.yml run --rm debates \
  uv run python scripts/test_first_debate.py

# Custom model
docker compose -f docker/docker-compose.yml run --rm debates \
  uv run python process_debates.py --model inesc-id/WhisperLv3-EP-X
```

---

## Smoke Test

Verify the image builds and runs correctly:

```bash
./docker/test.sh
```

This builds the image, checks imports, and prints the `test_first_debate.py` help.

---

## Volumes

The container expects `data/` mounted at `/app/data`. Ensure this structure exists:

```
data/
├── links/          # debates_unified.csv and other CSVs (required)
├── debates/        # Downloaded MP3 files (created on first run)
└── transcripts/    # Output transcripts (created on first run)
```

Your host `data/` is mounted into the container, so downloads and transcripts persist on your machine.

---

## File Layout

| File | Purpose |
|------|---------|
| `docker/Dockerfile` | Image definition |
| `docker/docker-compose.yml` | Compose configuration |
| `docker/test.sh` | Build + smoke test script |
| `docker/README.md` | This file |
| `.dockerignore` | At repo root; excludes `.venv`, outputs, etc. from build context |

---

## Troubleshooting

### Build fails: `pkg-config is required for building PyAV`
Install system deps first (or use the Docker image, which includes them):
```bash
# Ubuntu/Debian
apt-get install -y pkg-config libavformat-dev libavcodec-dev libavutil-dev \
  libswresample-dev libswscale-dev libavdevice-dev
```

### `torchaudio has no attribute 'AudioMetaData'`
Pin `torchaudio<2.9` in `pyproject.toml` (already done in this project).

### Docker daemon not running
Start Docker Desktop (Windows/macOS) or the Docker service (Linux).

### No space left on device
Clean up: `docker system prune -a` (removes unused images and build cache).
