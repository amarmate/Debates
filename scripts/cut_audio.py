#!/usr/bin/env python3
"""
Script to cut MP3 audio files.
Supports cutting by start/end time or start time and duration.
"""

import sys
import os
from pathlib import Path
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
        print(f"Error: Audio file not found: {input_path}")
        return
    
    # Parse times
    start_ms = 0
    if start_time:
        start_ms = parse_time(start_time)
    
    if end_time:
        end_ms = parse_time(end_time)
        if end_ms <= start_ms:
            print(f"Error: End time must be greater than start time")
            return
    elif duration:
        duration_ms = parse_time(duration)
        end_ms = start_ms + duration_ms
    else:
        print("Error: Must specify either end_time or duration")
        return
    
    print(f"Loading audio file: {input_path}")
    try:
        audio = AudioSegment.from_mp3(input_path)
    except Exception as e:
        print(f"Error loading audio file: {e}")
        print("\nNote: pydub requires ffmpeg to be installed for MP3 support.")
        print("Install ffmpeg: https://ffmpeg.org/download.html")
        return
    
    total_duration_ms = len(audio)
    total_duration_s = total_duration_ms / 1000
    
    print(f"Audio duration: {total_duration_s:.2f} seconds ({total_duration_ms} ms)")
    
    # Validate times
    if start_ms >= total_duration_ms:
        print(f"Error: Start time ({start_ms}ms) is beyond audio duration ({total_duration_ms}ms)")
        return
    
    if end_ms > total_duration_ms:
        print(f"Warning: End time ({end_ms}ms) exceeds audio duration ({total_duration_ms}ms). Clipping to end.")
        end_ms = total_duration_ms
    
    # Cut the audio
    print(f"\nCutting audio from {start_ms}ms to {end_ms}ms...")
    cut_audio_segment = audio[start_ms:end_ms]
    
    # Generate output path if not provided
    if output_path is None:
        input_file = Path(input_path)
        start_s = start_ms / 1000
        end_s = end_ms / 1000
        output_path = input_file.parent / f"{input_file.stem}_cut_{start_s:.1f}s-{end_s:.1f}s{input_file.suffix}"
    
    # Export
    print(f"Exporting to: {output_path}")
    try:
        cut_audio_segment.export(output_path, format="mp3")
    except Exception as e:
        print(f"Error exporting audio: {e}")
        return
    
    cut_duration_s = len(cut_audio_segment) / 1000
    print(f"\n✓ Successfully cut audio!")
    print(f"  Original duration: {total_duration_s:.2f} seconds")
    print(f"  Cut duration: {cut_duration_s:.2f} seconds")
    print(f"  Output file: {output_path}")


def print_usage():
    """Print usage information"""
    print("=" * 80)
    print("MP3 Audio Cutter")
    print("=" * 80)
    print("\nUsage:")
    print("  uv run python scripts/cut_audio.py <input_file> [options]")
    print("\nOptions:")
    print("  --start TIME       Start time (MM:SS, HH:MM:SS, or seconds). Default: 0")
    print("  --end TIME         End time (MM:SS, HH:MM:SS, or seconds)")
    print("  --duration TIME    Duration from start (MM:SS, HH:MM:SS, or seconds)")
    print("  --output FILE      Output file path (default: auto-generated)")
    print("\nExamples:")
    print("  uv run python scripts/cut_audio.py audio.mp3 --start 1:30 --end 3:45")
    print("  uv run python scripts/cut_audio.py audio.mp3 --start 0:01:30 --duration 2:15")
    print("=" * 80)


if __name__ == "__main__":
    if len(sys.argv) < 2 or "--help" in sys.argv or "-h" in sys.argv:
        print_usage()
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
            print(f"Unknown argument: {arg}")
            print_usage()
            sys.exit(1)
    
    if not end_time and not duration:
        print("Error: Must specify either --end or --duration")
        print_usage()
        sys.exit(1)
    
    cut_audio(input_file, start_time, end_time, duration, output_path)
