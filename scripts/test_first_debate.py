#!/usr/bin/env python3
"""
Test script: download the first debate from debates_unified.csv and transcribe it.
Runs the same pipeline as process_debates for a single debate.
"""

import argparse
import csv
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from debate_downloader import get_debate_audio
from download_all_debates import create_debate_title, create_debate_filename, DOWNLOAD_FOLDER

logger = logging.getLogger(__name__)
from transcribe_audio import transcribe_audio

CSV_FILE = Path("data/links/debates_unified.csv")


def main() -> int:
    parser = argparse.ArgumentParser(description="Download and transcribe the first debate.")
    parser.add_argument("--model", default="base", help="Whisper model size (tiny, base, small, medium, large)")
    args = parser.parse_args()
    model_size = args.model
    if not CSV_FILE.exists():
        logger.error("%s not found", CSV_FILE)
        return 1

    with open(CSV_FILE, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = [r for r in reader if (r.get("url") or "").strip().startswith("http")]

    if not rows:
        logger.error("No valid debate rows in CSV")
        return 1

    row = rows[0]
    url = row.get("url", "").strip()
    title = create_debate_title(row)

    logger.info("=" * 60)
    logger.info("TEST: First debate from debates_unified.csv")
    logger.info("=" * 60)
    logger.info("Debate: %s", title)
    logger.info("URL: %s", url)
    logger.info("Model: %s", model_size)

    # Download
    filename_base = create_debate_filename(row)
    logger.info("Step 1: Downloading...")
    try:
        get_debate_audio(
            page_url=url,
            download_audio_only=True,
            audio_format="mp3",
            title=title,
            filename_base=filename_base,
        )
    except Exception as e:
        logger.error("Download failed: %s", e)
        return 1

    # Find the downloaded file: prefer one matching our filename_base, else most recent
    if not DOWNLOAD_FOLDER.exists():
        logger.error("Download folder not found")
        return 1

    matches = list(DOWNLOAD_FOLDER.glob(f"{filename_base}.mp3"))
    if not matches:
        matches = sorted(DOWNLOAD_FOLDER.glob("*.mp3"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not matches:
        logger.error("No MP3 files found in download folder")
        return 1

    audio_path = matches[0]
    logger.info("Downloaded: %s", audio_path)

    # Transcribe
    logger.info("Step 2: Transcribing...")
    try:
        transcribe_audio(
            str(audio_path),
            language="pt",
            model_size=model_size,
            enable_diarization=True,
            enable_vad=True,
            condition_on_previous_text=False,
            compression_ratio_threshold=2.0,
            enable_overlap_detection=True,
        )
    except Exception as e:
        logger.error("Transcription failed: %s", e)
        return 1

    txt_path = audio_path.parent / f"{audio_path.stem}_{model_size}.txt"
    logger.info("=" * 60)
    logger.info("Done!")
    logger.info("Audio: %s", audio_path)
    logger.info("Transcript: %s", txt_path)
    logger.info("=" * 60)
    return 0


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    sys.exit(main())
