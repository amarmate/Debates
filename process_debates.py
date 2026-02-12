#!/usr/bin/env python3
"""
Orchestration script to download debates and transcribe them.
"""

import argparse
import logging
import os
import subprocess
import sys
from datetime import datetime

# Import local modules
# Add current directory to path to ensure imports work
sys.path.append(os.getcwd())

# Configure logging (before imports so logger works in except)
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("process_debates")

try:
    from download_all_debates import download_all_debates, DOWNLOAD_FOLDER
    from transcribe_audio import transcribe_audio, get_transcript_path, TRANSCRIPTS_FOLDER
except ImportError as e:
    logger.error("Error importing modules: %s. Make sure download_all_debates.py and transcribe_audio.py are in the current directory.", e)
    sys.exit(1)

# Error logging configuration
ERROR_LOG_FILE = "error_log.txt"


def log_error(message, error_details=None):
    """Log error to console and file"""
    logger.error("❌ %s", message)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(ERROR_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] {message}\n")
        if error_details:
            f.write(f"{error_details}\n")
        f.write("-" * 40 + "\n")


def _run_transcription_subprocess(
    audio_file: os.PathLike | str,
    args: argparse.Namespace,
) -> None:
    """
    Run transcription for a single audio file in a separate Python process.

    This isolates GPU memory usage per file, ensuring that all CUDA state and
    model weights are fully released when the child process exits. This is
    much more robust against CUDA out-of-memory errors than trying to
    manually free all model references within a long‑running process.
    """
    audio_path = os.path.abspath(str(audio_file))
    script_path = os.path.abspath(os.path.join(os.getcwd(), "transcribe_audio.py"))
    cwd = os.getcwd()

    cmd: list[str] = [
        sys.executable,
        script_path,
        audio_path,
        args.model,
    ]

    # Map CLI flags from this orchestrator to the child process script
    if args.no_diarization:
        cmd.append("--no-diarization")
    if getattr(args, "num_speakers", None) is not None:
        cmd.extend(["--num-speakers", str(args.num_speakers)])
    if args.no_vad:
        cmd.append("--no-vad")
    if args.no_overlap_detection:
        cmd.append("--no-overlap-detection")

    if args.prompt:
        cmd.extend(["--prompt", args.prompt])

    cmd.extend(
        [
            "--condition-on-previous-text",
            str(args.condition_on_previous_text).lower(),
            "--compression-ratio-threshold",
            str(args.compression_ratio_threshold),
        ]
    )

    if args.chunk_length_s is not None:
        cmd.extend(["--chunk-length-s", str(args.chunk_length_s)])

    logger.info("Starting transcription subprocess: %s", " ".join(cmd))

    # Let stdout/stderr stream directly so the user sees detailed logs.
    # cwd and env are inherited so HF token and paths resolve correctly.
    completed = subprocess.run(cmd, cwd=cwd)

    if completed.returncode != 0:
        raise RuntimeError(
            f"Transcription subprocess failed with exit code {completed.returncode} "
            f"for file {audio_path}"
        )

def main():
    parser = argparse.ArgumentParser(description="Download and transcribe debates.")
    parser.add_argument(
        "--model",
        default="base",
        help="ASR model: Whisper size (tiny, base, small, medium, large) or Hugging Face path (e.g. inesc-id/WhisperLv3-EP-X)",
    )
    parser.add_argument("--csv", default="data/links/debates_unified.csv", help="Path to CSV file with debate links (default: data/links/debates_unified.csv)")
    parser.add_argument("--no-diarization", action="store_true", help="Disable speaker diarization")
    parser.add_argument("--num-speakers", type=int, default=3, help="Number of speakers for diarization (default: 3 for debates)")
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
            txt_file = get_transcript_path(str(audio_file), args.model)

            logger.info(f"\nProcessing {i}/{len(audio_files)}: {audio_file.name}")

            if txt_file.exists():
                logger.info(f"⏭️  Transcript already exists, skipping: {txt_file.name}")
                skipped_count += 1
                continue
            
            # Transcribe
            logger.info("🎙️  Transcribing...")
            _run_transcription_subprocess(audio_file, args)
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
