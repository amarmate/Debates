#!/usr/bin/env python3
"""
Script to transcribe audio files using OpenAI Whisper.
Supports Portuguese language transcription with progress tracking.
"""

import whisper
import sys
import os
from pathlib import Path
from tqdm import tqdm
import librosa
import threading
import time


class TranscriptionProgress:
    """Helper class to track transcription progress with tqdm"""
    
    def __init__(self, audio_duration=None, model_size="base"):
        self.audio_duration = audio_duration
        self.model_size = model_size
        self.start_time = None
        self.pbar = None
        self.done = False
        
    def __enter__(self):
        if self.audio_duration:
            # Estimate: transcription typically takes 0.5-2x audio duration
            # Smaller models are faster
            speed_factor = 1.0 if self.model_size in ['tiny', 'base'] else 1.5
            estimated_total = self.audio_duration * speed_factor
            self.pbar = tqdm(
                total=int(estimated_total),
                unit="s",
                desc="Transcribing",
                bar_format='{l_bar}{bar}| {n:.0f}s/{total:.0f}s [{elapsed}<{remaining}, {rate_fmt}]'
            )
        else:
            self.pbar = tqdm(
                desc="Transcribing",
                unit="s",
                bar_format='{l_bar}{bar}| {elapsed} elapsed'
            )
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.pbar:
            if self.audio_duration:
                # Complete the progress bar
                elapsed = time.time() - self.start_time
                self.pbar.n = min(int(elapsed), self.pbar.total)
            self.pbar.close()
    
    def update(self):
        """Update progress bar based on elapsed time"""
        if self.pbar and not self.done:
            elapsed = time.time() - self.start_time
            if self.audio_duration:
                self.pbar.n = min(int(elapsed), self.pbar.total)
            else:
                self.pbar.n = int(elapsed)
            self.pbar.refresh()


def transcribe_audio(audio_path: str, language: str = "pt", model_size: str = "base"):
    """
    Transcribe an audio file using Whisper with progress tracking.
    
    Args:
        audio_path: Path to the audio file
        language: Language code (default: "pt" for Portuguese)
        model_size: Whisper model size - "tiny", "base", "small", "medium", "large"
                   Larger models are more accurate but slower
    """
    if not os.path.exists(audio_path):
        print(f"Error: Audio file not found: {audio_path}")
        return
    
    # Get audio duration for progress tracking
    print("Loading audio file to get duration...")
    try:
        duration = librosa.get_duration(path=audio_path)
        duration_str = f"{int(duration // 60)}m {int(duration % 60)}s"
        print(f"Audio duration: {duration_str} ({duration:.1f} seconds)")
    except Exception as e:
        print(f"Warning: Could not get audio duration: {e}")
        duration = None
    
    print(f"\nLoading Whisper model '{model_size}'...")
    model = whisper.load_model(model_size)
    
    print(f"\nStarting transcription...")
    
    # Use progress context manager
    with TranscriptionProgress(duration, model_size) as progress:
        # Update progress in a separate thread
        def update_loop():
            while not progress.done:
                progress.update()
                time.sleep(0.5)  # Update every 500ms
        
        update_thread = threading.Thread(target=update_loop, daemon=True)
        update_thread.start()
        
        try:
            # Transcribe with Portuguese language specified
            # verbose=False suppresses Whisper's built-in progress (we use our own)
            result = model.transcribe(audio_path, language=language, verbose=False)
        finally:
            progress.done = True
            time.sleep(0.6)  # Allow final update
    
    # Get the transcribed text
    text = result["text"]
    
    # Save to text file
    output_path = Path(audio_path).with_suffix('.txt')
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(text)
    
    print(f"\nTranscription complete!")
    print(f"Text saved to: {output_path}")
    print(f"\nTranscribed text ({len(text)} characters):")
    print("-" * 80)
    print(text)
    print("-" * 80)
    
    # Also print segments if available
    if "segments" in result:
        print(f"\nSegments with timestamps:")
        for segment in result["segments"]:
            start = segment["start"]
            end = segment["end"]
            print(f"[{start:.1f}s - {end:.1f}s]: {segment['text']}")


if __name__ == "__main__":
    # Default to test_debate.mp3 if no argument provided
    audio_file = sys.argv[1] if len(sys.argv) > 1 else "test_debate.mp3"
    
    # Optional: specify model size (tiny, base, small, medium, large)
    # base is a good balance between speed and accuracy
    model_size = sys.argv[2] if len(sys.argv) > 2 else "base"
    
    print("=" * 80)
    print("Whisper Audio Transcription")
    print("=" * 80)
    print(f"Audio file: {audio_file}")
    print(f"Model: {model_size}")
    print(f"Language: Portuguese (pt)")
    print("=" * 80)
    print()
    
    transcribe_audio(audio_file, language="pt", model_size=model_size)
