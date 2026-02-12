#!/usr/bin/env python3
"""
Test script: download the first debate from debates_unified.csv and transcribe it.
Runs the same pipeline as process_debates for a single debate.
"""

import argparse
import csv
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from debate_downloader import get_debate_audio
from download_all_debates import create_debate_title, create_debate_filename, DOWNLOAD_FOLDER
from transcribe_audio import transcribe_audio

CSV_FILE = Path("data/links/debates_unified.csv")


def main() -> int:
    parser = argparse.ArgumentParser(description="Download and transcribe the first debate.")
    parser.add_argument("--model", default="base", help="Whisper model size (tiny, base, small, medium, large)")
    args = parser.parse_args()
    model_size = args.model
    if not CSV_FILE.exists():
        print(f"Error: {CSV_FILE} not found")
        return 1

    with open(CSV_FILE, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = [r for r in reader if (r.get("url") or "").strip().startswith("http")]

    if not rows:
        print("No valid debate rows in CSV")
        return 1

    row = rows[0]
    url = row.get("url", "").strip()
    title = create_debate_title(row)

    print("=" * 60)
    print("TEST: First debate from debates_unified.csv")
    print("=" * 60)
    print(f"Debate: {title}")
    print(f"URL: {url}")
    print(f"Model: {model_size}")
    print()

    # Download
    filename_base = create_debate_filename(row)
    print("Step 1: Downloading...")
    try:
        get_debate_audio(
            page_url=url,
            download_audio_only=True,
            audio_format="mp3",
            title=title,
            filename_base=filename_base,
        )
    except Exception as e:
        print(f"Download failed: {e}")
        return 1

    # Find the downloaded file: prefer one matching our filename_base, else most recent
    if not DOWNLOAD_FOLDER.exists():
        print("Error: Download folder not found")
        return 1

    matches = list(DOWNLOAD_FOLDER.glob(f"{filename_base}.mp3"))
    if not matches:
        matches = sorted(DOWNLOAD_FOLDER.glob("*.mp3"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not matches:
        print("Error: No MP3 files found in download folder")
        return 1

    audio_path = matches[0]
    print(f"Downloaded: {audio_path}")
    print()

    # Transcribe
    print("Step 2: Transcribing...")
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
        print(f"Transcription failed: {e}")
        return 1

    txt_path = audio_path.parent / f"{audio_path.stem}_{model_size}.txt"
    print()
    print("=" * 60)
    print("Done!")
    print(f"Audio: {audio_path}")
    print(f"Transcript: {txt_path}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
