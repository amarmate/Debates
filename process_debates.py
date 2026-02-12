#!/usr/bin/env python3
"""
Orchestration script to download debates and transcribe them.
"""

import argparse
import logging
import os
import sys
from datetime import datetime

# Import local modules
# Add current directory to path to ensure imports work
sys.path.append(os.getcwd())

try:
    from download_all_debates import download_all_debates, DOWNLOAD_FOLDER
    from transcribe_audio import transcribe_audio
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Make sure download_all_debates.py and transcribe_audio.py are in the current directory.")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("process_debates")

# Error logging configuration
ERROR_LOG_FILE = "error_log.txt"

def log_error(message, error_details=None):
    """Log error to console and file"""
    logger.error(f"❌ {message}")
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(ERROR_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] {message}\n")
        if error_details:
            f.write(f"{error_details}\n")
        f.write("-" * 40 + "\n")

def main():
    parser = argparse.ArgumentParser(description="Download and transcribe debates.")
    parser.add_argument("--model", default="base", help="Whisper model size (tiny, base, small, medium, large)")
    parser.add_argument("--csv", default="data/links/debates_unified.csv", help="Path to CSV file with debate links (default: data/links/debates_unified.csv)")
    parser.add_argument("--no-diarization", action="store_true", help="Disable speaker diarization")
    parser.add_argument("--no-vad", action="store_true", help="Disable Voice Activity Detection (VAD)")
    parser.add_argument("--no-overlap-detection", action="store_true", help="Disable overlapped speech detection")
    parser.add_argument("--prompt", help="Custom initial prompt for transcription (default: Portuguese political debate prompt)")
    parser.add_argument("--condition-on-previous-text", type=str, default="false", help="Whisper condition_on_previous_text (true/false)")
    parser.add_argument("--compression-ratio-threshold", type=float, default=2.0, help="Whisper compression_ratio_threshold")
    parser.add_argument("--chunk-length-s", type=int, default=None, help="Whisper chunk_length_s")
    args = parser.parse_args()

    # Step 1: Download Debates
    logger.info("=" * 80)
    logger.info("🚀 STARTING DEBATE PROCESSING PIPELINE")
    logger.info("=" * 80)
    logger.info("Step 1: Downloading debates...")
    
    try:
        download_all_debates(csv_file=args.csv)
    except Exception as e:
        log_error("Critical error during download phase", str(e))
        # We continue to transcription even if download failed, 
        # to process any files that might have been downloaded or existed previously.

    # Step 2: Transcribe Debates
    logger.info("\n" + "=" * 80)
    logger.info("Step 2: Transcribing debates...")
    logger.info(f"Model: {args.model}")
    logger.info(f"Diarization: {'Disabled' if args.no_diarization else 'Enabled'}")
    logger.info(f"VAD: {'Disabled' if args.no_vad else 'Enabled'}")
    if args.prompt:
        logger.info(f"Custom prompt: {args.prompt[:80]}...")
    logger.info("=" * 80)

    # Find all audio files
    if not DOWNLOAD_FOLDER.exists():
        log_error(f"Download folder not found: {DOWNLOAD_FOLDER}")
        return

    # Look for mp3 files
    audio_files = list(DOWNLOAD_FOLDER.glob("*.mp3"))
    
    if not audio_files:
        logger.info("No audio files found to transcribe.")
        return

    logger.info(f"Found {len(audio_files)} audio files in {DOWNLOAD_FOLDER}")

    success_count = 0
    skipped_count = 0
    error_count = 0

    for i, audio_file in enumerate(audio_files, 1):
        try:
            txt_file = audio_file.with_suffix(".txt")
            
            logger.info(f"\nProcessing {i}/{len(audio_files)}: {audio_file.name}")
            
            if txt_file.exists():
                logger.info(f"⏭️  Transcript already exists, skipping: {txt_file.name}")
                skipped_count += 1
                continue
            
            # Transcribe
            logger.info("🎙️  Transcribing...")
            transcribe_audio(
                str(audio_file),
                language="pt",
                model_size=args.model,
                enable_diarization=not args.no_diarization,
                enable_vad=not args.no_vad,
                initial_prompt=args.prompt,
                condition_on_previous_text=str(args.condition_on_previous_text).lower() in ("1", "true", "yes", "y", "on"),
                compression_ratio_threshold=args.compression_ratio_threshold,
                chunk_length_s=args.chunk_length_s,
                enable_overlap_detection=not args.no_overlap_detection,
            )
            success_count += 1
            logger.info(f"✅ Transcription saved to: {txt_file.name}")
            
        except Exception as e:
            error_count += 1
            log_error(f"Failed to transcribe {audio_file.name}", str(e))
    
    # Final Summary
    logger.info("\n" + "=" * 80)
    logger.info("🎉 PIPELINE COMPLETED")
    logger.info("=" * 80)
    logger.info(f"✅ Transcribed: {success_count}")
    logger.info(f"⏭️  Skipped: {skipped_count}")
    logger.info(f"❌ Errors: {error_count}")
    
    if error_count > 0:
        logger.info(f"Check {ERROR_LOG_FILE} for error details.")

if __name__ == "__main__":
    main()
