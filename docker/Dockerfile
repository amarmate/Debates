# Debates: download and transcribe political debate audio
# Uses Python 3.12 with uv pre-installed (Debian-based)
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# System deps: PyAV build (pkg-config + FFmpeg dev), ffmpeg runtime, Playwright/Chromium
RUN apt-get update && apt-get install -y --no-install-recommends \
    pkg-config \
    ffmpeg \
    libavformat-dev libavcodec-dev libavutil-dev \
    libswresample-dev libswscale-dev libavdevice-dev \
    libnss3 libnspr4 libatk1.0-0 libatk-bridge2.0-0 libcups2 \
    libdrm2 libdbus-1-3 libxkbcommon0 libatspi2.0-0 libxcomposite1 \
    libxdamage1 libxfixes3 libxrandr2 libgbm1 libasound2 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy project files (use .dockerignore at repo root to exclude large/unnecessary files)
COPY pyproject.toml uv.lock ./
COPY data/links data/links/
COPY *.py ./
COPY scripts scripts/

# Create dirs for outputs (mounted as volumes at runtime)
RUN mkdir -p data/debates data/transcripts data/benchmark/refs data/benchmark/results

# Install Python dependencies and project (cache mount speeds up rebuilds)
ENV UV_LINK_MODE=copy
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --no-dev --locked

# Install Playwright Chromium (headless browser for downloads)
RUN uv run playwright install chromium

# Default: run the full pipeline
CMD ["uv", "run", "python", "process_debates.py"]
