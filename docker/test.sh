#!/bin/bash
# Docker build and smoke test for Debates
# Run from project root: ./docker/test.sh

set -e
cd "$(dirname "$0")/.."

echo "=== Building Docker image ==="
docker build -f docker/Dockerfile -t debates:latest .

echo ""
echo "=== Smoke test: verify imports ==="
docker run --rm debates:latest uv run python -c "
from debate_downloader import get_debate_audio
from download_all_debates import create_debate_filename, DOWNLOAD_FOLDER
from transcribe_audio import transcribe_audio, TRANSCRIPTS_FOLDER
print('All imports OK')
"

echo ""
echo "=== Smoke test: help ==="
docker run --rm debates:latest uv run python scripts/test_first_debate.py --help

echo ""
echo "=== Done ==="
echo "See docker/README.md for full usage instructions."
