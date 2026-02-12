#!/usr/bin/env python3
"""
Script to cut MP3 audio files.
Supports cutting by start/end time or start time and duration.
"""

import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger(__name__)
from pydub import AudioSegment


def parse_time(time_str):
    """
    Parse time string in format MM:SS or HH:MM:SS or seconds.
    
    Args:
        time_str: Time string (e.g., "1:30", "0:01:30", "90")
    
    Returns:
        Time in milliseconds
    """
    try:
        # Try parsing as seconds (float)
        if ':' not in time_str:
            return int(float(time_str) * 1000)
        
        # Parse as MM:SS or HH:MM:SS
        parts = time_str.split(':')
        if len(parts) == 2:
            # MM:SS format
            minutes, seconds = map(float, parts)
            return int((minutes * 60 + seconds) * 1000)
        elif len(parts) == 3:
            # HH:MM:SS format
            hours, minutes, seconds = map(float, parts)
            return int((hours * 3600 + minutes * 60 + seconds) * 1000)
        else:
            raise ValueError("Invalid time format")
    except ValueError as e:
        raise ValueError(f"Invalid time format '{time_str}'. Use MM:SS, HH:MM:SS, or seconds (e.g., '1:30', '0:01:30', '90')")


def cut_audio(input_path: str, start_time: str = None, end_time: str = None, 
              duration: str = None, output_path: str = None):
    """
    Cut an MP3 audio file.
    
    Args:
        input_path: Path to input MP3 file
        start_time: Start time (MM:SS, HH:MM:SS, or seconds). Default: 0
        end_time: End time (MM:SS, HH:MM:SS, or seconds). If None, uses duration
        duration: Duration from start (MM:SS, HH:MM:SS, or seconds). Used if end_time is None
        output_path: Output file path. If None, auto-generates from input filename
    """
    if not os.path.exists(input_path):
        logger.error("Audio file not found: %s", input_path)
        return

    # Parse times
    start_ms = 0
    if start_time:
        start_ms = parse_time(start_time)
    
    if end_time:
        end_ms = parse_time(end_time)
        if end_ms <= start_ms:
            logger.error("End time must be greater than start time")
            return
    elif duration:
        duration_ms = parse_time(duration)
        end_ms = start_ms + duration_ms
    else:
        logger.error("Must specify either end_time or duration")
        return

    logger.info("Loading audio file: %s", input_path)
    try:
        audio = AudioSegment.from_mp3(input_path)
    except Exception as e:
        logger.error(
            "Error loading audio file: %s. pydub requires ffmpeg for MP3. Install: https://ffmpeg.org/download.html",
            e,
        )
        return

    total_duration_ms = len(audio)
    total_duration_s = total_duration_ms / 1000
    
    logger.info("Audio duration: %.2f seconds (%d ms)", total_duration_s, total_duration_ms)

    # Validate times
    if start_ms >= total_duration_ms:
        logger.error("Start time (%dms) is beyond audio duration (%dms)", start_ms, total_duration_ms)
        return

    if end_ms > total_duration_ms:
        logger.warning("End time (%dms) exceeds audio duration (%dms). Clipping to end.", end_ms, total_duration_ms)
        end_ms = total_duration_ms
    
    # Cut the audio
    logger.info("Cutting audio from %dms to %dms...", start_ms, end_ms)
    cut_audio_segment = audio[start_ms:end_ms]
    
    # Generate output path if not provided
    if output_path is None:
        input_file = Path(input_path)
        start_s = start_ms / 1000
        end_s = end_ms / 1000
        output_path = input_file.parent / f"{input_file.stem}_cut_{start_s:.1f}s-{end_s:.1f}s{input_file.suffix}"
    
    # Export
    logger.info("Exporting to: %s", output_path)
    try:
        cut_audio_segment.export(output_path, format="mp3")
    except Exception as e:
        logger.error("Error exporting audio: %s", e)
        return

    cut_duration_s = len(cut_audio_segment) / 1000
    logger.info("Successfully cut audio! Original: %.2fs, Cut: %.2fs, Output: %s", total_duration_s, cut_duration_s, output_path)


def log_usage():
    """Log usage information."""
    logger.info("=" * 80)
    logger.info("MP3 Audio Cutter")
    logger.info("=" * 80)
    logger.info("Usage: uv run python scripts/cut_audio.py <input_file> [options]")
    logger.info("Options: --start TIME --end TIME --duration TIME --output FILE")
    logger.info("Examples: uv run python scripts/cut_audio.py audio.mp3 --start 1:30 --end 3:45")
    logger.info("=" * 80)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    if len(sys.argv) < 2 or "--help" in sys.argv or "-h" in sys.argv:
        log_usage()
        sys.exit(0)
    
    input_file = sys.argv[1]
    start_time = None
    end_time = None
    duration = None
    output_path = None
    
    # Parse arguments
    i = 2
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg == "--start" and i + 1 < len(sys.argv):
            start_time = sys.argv[i + 1]
            i += 2
        elif arg == "--end" and i + 1 < len(sys.argv):
            end_time = sys.argv[i + 1]
            i += 2
        elif arg == "--duration" and i + 1 < len(sys.argv):
            duration = sys.argv[i + 1]
            i += 2
        elif arg == "--output" and i + 1 < len(sys.argv):
            output_path = sys.argv[i + 1]
            i += 2
        else:
            logger.error("Unknown argument: %s", arg)
            log_usage()
            sys.exit(1)

    if not end_time and not duration:
        logger.error("Must specify either --end or --duration")
        log_usage()
        sys.exit(1)
    
    cut_audio(input_file, start_time, end_time, duration, output_path)
