# Docker – Debates Pipeline

Run the debate download and transcription pipeline in a reproducible container. Designed for **build → push → pull on Linux servers**.

---

## Prerequisites

- **Docker** installed and running ([docker.com](https://docs.docker.com/get-docker/))
- **Hugging Face token** (optional but required for speaker diarization):
  1. Create a token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
  2. Accept terms at [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
  3. Use it as `HF_TOKEN` when running the container

---

## Workflow Overview

| Step | Where | Action |
|------|-------|--------|
| 1. Build | CI or dev machine | Build image from source |
| 2. Push | CI or dev machine | Push image to registry (Docker Hub, GHCR, etc.) |
| 3. Pull & Run | Linux server(s) | Pull image, mount data, run pipeline |

---

## 1. Build and Push (CI / dev machine)

From the **project root**:

```bash
# Build
docker build -f docker/Dockerfile -t debates:latest .

# Tag for your registry (replace <registry>/<repo> with your image name)
docker tag debates:latest <registry>/<repo>:latest
# Examples:
#   docker tag debates:latest ghcr.io/YOUR_ORG/debates:latest
#   docker tag debates:latest docker.io/YOUR_USER/debates:latest

# Push (login first: docker login ghcr.io or docker login)
docker push <registry>/<repo>:latest
```

**GitHub Container Registry (ghcr.io):**
```bash
docker tag debates:latest ghcr.io/YOUR_ORG/debates:latest
echo $GITHUB_TOKEN | docker login ghcr.io -u USERNAME --password-stdin
docker push ghcr.io/YOUR_ORG/debates:latest
```

**Docker Hub:**
```bash
docker tag debates:latest docker.io/YOUR_USER/debates:latest
docker login
docker push docker.io/YOUR_USER/debates:latest
```

---

## 2. Pull and Run (Linux server)

On the **server** you only need Docker. No need to clone the repo or have the source code.

### Setup data directory

```bash
# Create data structure
mkdir -p /opt/debates/data/{links,debates,transcripts}

# Copy debates_unified.csv into data/links/ (or sync from your repo/storage)
# Example: scp data/links/debates_unified.csv server:/opt/debates/data/links/
```

### Run the pipeline

```bash
# Pull the image (replace with your registry image)
docker pull <registry>/<repo>:latest

# Run full pipeline with diarization
docker run --rm \
  -e HF_TOKEN="$HF_TOKEN" \
  -v /opt/debates/data:/app/data \
  <registry>/<repo>:latest

# Or: run with token from env file
docker run --rm \
  --env-file /opt/debates/.env \
  -v /opt/debates/data:/app/data \
  <registry>/<repo>:latest
```

Create `/opt/debates/.env`:
```
HF_TOKEN=hf_xxxxxxxxxxxx
```

### Custom commands (model, test run, etc.)

```bash
# Test first debate only
docker run --rm -v /opt/debates/data:/app/data \
  <registry>/<repo>:latest uv run python scripts/test_first_debate.py

# Use WhisperX European Portuguese model
docker run --rm -e HF_TOKEN="$HF_TOKEN" -v /opt/debates/data:/app/data \
  <registry>/<repo>:latest uv run python process_debates.py --model inesc-id/WhisperLv3-EP-X

# Use large Whisper model
docker run --rm -e HF_TOKEN="$HF_TOKEN" -v /opt/debates/data:/app/data \
  <registry>/<repo>:latest uv run python process_debates.py --model large
```

---

## Quick Start (local dev, no registry)

If you're developing locally and not pushing to a registry:

```bash
# From project root
docker build -f docker/Dockerfile -t debates:latest .
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

## Volumes and Data Layout

The container expects a data directory mounted at `/app/data` with this structure:

```
data/
├── links/          # debates_unified.csv and other CSVs (required – copy from repo or sync)
├── debates/        # Downloaded MP3 files (created on first run)
└── transcripts/    # Output transcripts (created on first run)
```

**On the server:** Ensure `data/links/debates_unified.csv` exists before running. You can sync it from the repo, copy via `scp`, or use a config management tool.

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

---

## CI/CD Example (GitHub Actions)

Build and push on every tag:

```yaml
# .github/workflows/docker.yml
name: Build and push Docker image
on:
  push:
    tags: ['v*']
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: docker/login-action@v3
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}
      - uses: docker/build-push-action@v6
        with:
          context: .
          file: docker/Dockerfile
          push: true
          tags: |
            ghcr.io/${{ github.repository_owner }}/debates:${{ github.ref_name }}
            ghcr.io/${{ github.repository_owner }}/debates:latest
```
