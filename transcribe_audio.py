#!/usr/bin/env python3
"""
Script to transcribe audio files using OpenAI Whisper with automatic speaker diarization.
Supports Portuguese language transcription with progress tracking and speaker identification.
"""

import whisper
import sys
import os
from pathlib import Path
from tqdm import tqdm
import librosa
import numpy as np
import threading
import time
import warnings
from typing import List, Dict, Tuple, Optional

# Suppress torchcodec warnings - pyannote.audio will fall back to librosa
warnings.filterwarnings("ignore", message=".*torchcodec.*")
warnings.filterwarnings("ignore", message=".*libtorchcodec.*")
# Suppress std() degrees of freedom warning from pyannote pooling
warnings.filterwarnings("ignore", message=".*std\\(\\): degrees of freedom.*")


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


def perform_speaker_diarization(audio_path: str, num_speakers: Optional[int] = None):
    """
    Perform speaker diarization on an audio file using pyannote.audio.
    
    Args:
        audio_path: Path to the audio file
        num_speakers: Optional number of speakers. If None, will be automatically detected.
    
    Returns:
        List of tuples: [(start_time, end_time, speaker_label), ...]
    """
    try:
        from pyannote.audio import Pipeline
        from pyannote.core import Annotation
    except ImportError:
        print("\nError: pyannote.audio is not installed or not accessible.")
        print("\nIf you're using uv (you have uv.lock file):")
        print("  Run the script with: uv run python transcribe_audio.py <audio_file>")
        print("\nIf you're using pip:")
        print("  Install with: pip install pyannote.audio")
        print("\nYou also need a Hugging Face token to access the models:")
        print("  1. Get one at: https://huggingface.co/settings/tokens")
        print("  2. Authenticate: huggingface-cli login")
        print("  3. Accept model terms: https://huggingface.co/pyannote/speaker-diarization-3.1")
        return None
    
    print("\nLoading speaker diarization model...")
    try:
        # Load the pre-trained speaker diarization pipeline
        # This requires a Hugging Face token - the model will prompt if needed
        try:
            # Try newer API first (token parameter)
            pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                token=True
            )
        except TypeError:
            # Fall back to older API (use_auth_token parameter)
            pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=True
            )
    except Exception as e:
        print(f"Error loading diarization model: {e}")
        print("\nMake sure you have:")
        print("1. Installed pyannote.audio: pip install pyannote.audio")
        print("2. A Hugging Face token (get one at https://huggingface.co/settings/tokens)")
        print("3. Authenticated: huggingface-cli login")
        print("4. Accepted the model terms at https://huggingface.co/pyannote/speaker-diarization-3.1")
        return None
    
    print("Performing speaker diarization...")
    try:
        # Preprocess audio with librosa to ensure consistent sample rate
        # pyannote.audio expects 16kHz mono audio
        print("Preprocessing audio (resampling to 16kHz mono)...")
        audio, original_sr = librosa.load(audio_path, sr=16000, mono=True)
        
        # Convert to the format pyannote expects: dict with waveform and sample_rate
        # Shape needs to be (channels, samples) - mono = 1 channel
        waveform = audio.reshape(1, -1)  # Shape: (1, samples) for mono
        
        audio_dict = {
            "waveform": waveform,
            "sample_rate": 16000
        }
        
        # Run diarization with preprocessed audio
        if num_speakers:
            diarization = pipeline(audio_dict, num_speakers=num_speakers)
        else:
            diarization = pipeline(audio_dict)
        
        # Extract speaker segments
        speaker_segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            speaker_segments.append((turn.start, turn.end, speaker))
        
        print(f"Found {len(set(seg[2] for seg in speaker_segments))} speaker(s)")
        return speaker_segments
        
    except Exception as e:
        print(f"Error during diarization: {e}")
        import traceback
        traceback.print_exc()
        return None


def match_speakers_to_segments(transcription_segments: List[Dict], 
                               speaker_segments: List[Tuple[float, float, str]]) -> List[Dict]:
    """
    Match transcription segments with speaker labels based on timestamps.
    
    Args:
        transcription_segments: List of transcription segments from Whisper
        speaker_segments: List of (start, end, speaker) tuples from diarization
    
    Returns:
        List of segments with added 'speaker' field
    """
    if not speaker_segments:
        return transcription_segments
    
    # Create a list to store matched segments
    matched_segments = []
    
    for trans_seg in transcription_segments:
        trans_start = trans_seg["start"]
        trans_end = trans_seg["end"]
        trans_mid = (trans_start + trans_end) / 2
        
        # Find the speaker segment that overlaps most with this transcription segment
        best_speaker = None
        best_overlap = 0
        
        for spk_start, spk_end, speaker in speaker_segments:
            # Calculate overlap
            overlap_start = max(trans_start, spk_start)
            overlap_end = min(trans_end, spk_end)
            overlap_duration = max(0, overlap_end - overlap_start)
            
            if overlap_duration > best_overlap:
                best_overlap = overlap_duration
                best_speaker = speaker
        
        # If no overlap found, find the closest speaker segment
        if best_speaker is None:
            min_distance = float('inf')
            for spk_start, spk_end, speaker in speaker_segments:
                spk_mid = (spk_start + spk_end) / 2
                distance = abs(trans_mid - spk_mid)
                if distance < min_distance:
                    min_distance = distance
                    best_speaker = speaker
        
        # Add speaker label to segment
        matched_seg = trans_seg.copy()
        matched_seg["speaker"] = best_speaker if best_speaker else "UNKNOWN"
        matched_segments.append(matched_seg)
    
    return matched_segments


def transcribe_audio(audio_path: str, language: str = "pt", model_size: str = "base", 
                     enable_diarization: bool = True, num_speakers: Optional[int] = None):
    """
    Transcribe an audio file using Whisper with progress tracking and optional speaker diarization.
    
    Args:
        audio_path: Path to the audio file
        language: Language code (default: "pt" for Portuguese)
        model_size: Whisper model size - "tiny", "base", "small", "medium", "large"
                   Larger models are more accurate but slower
        enable_diarization: Whether to perform speaker diarization (default: True)
        num_speakers: Optional number of speakers for diarization. If None, auto-detected.
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
    
    # Perform speaker diarization if enabled
    speaker_segments = None
    if enable_diarization:
        speaker_segments = perform_speaker_diarization(audio_path, num_speakers)
        if speaker_segments is None:
            print("\n⚠️  WARNING: Speaker diarization failed or is not available.")
            print("   All segments will be labeled as 'UNKNOWN'.")
            print("   To enable speaker identification:")
            print("   1. Install pyannote.audio: pip install pyannote.audio")
            print("   2. Get a Hugging Face token: https://huggingface.co/settings/tokens")
            print("   3. Authenticate: huggingface-cli login")
            print("   4. Accept model terms: https://huggingface.co/pyannote/speaker-diarization-3.1")
            print("   Or run with --no-diarization to disable speaker identification.\n")
    
    # Match speakers to transcription segments
    segments = result.get("segments", [])
    if speaker_segments and segments:
        segments = match_speakers_to_segments(segments, speaker_segments)
        result["segments"] = segments
    elif enable_diarization and segments:
        # Diarization was enabled but failed - add UNKNOWN speaker labels
        for segment in segments:
            segment["speaker"] = "UNKNOWN"
    
    # Generate annotated text
    if segments and speaker_segments:
        # Build text with speaker annotations
        annotated_lines = []
        current_speaker = None
        
        for segment in segments:
            speaker = segment.get("speaker", "UNKNOWN")
            text = segment["text"].strip()
            
            if speaker != current_speaker:
                if annotated_lines:  # Add blank line between speakers
                    annotated_lines.append("")
                annotated_lines.append(f"[{speaker}]:")
                current_speaker = speaker
            
            annotated_lines.append(text)
        
        annotated_text = "\n".join(annotated_lines)
    else:
        annotated_text = result["text"]
    
    # Save annotated text to file
    output_path = Path(audio_path).with_suffix('.txt')
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(annotated_text)
    
    print(f"\nTranscription complete!")
    print(f"Text saved to: {output_path}")
    
    if speaker_segments:
        num_unique_speakers = len(set(seg.get("speaker", "UNKNOWN") for seg in segments))
        print(f"Identified {num_unique_speakers} speaker(s)")
    
    print(f"\nTranscribed text ({len(annotated_text)} characters):")
    print("-" * 80)
    print(annotated_text)
    print("-" * 80)
    
    # Also print segments with speaker labels if available
    if segments:
        print(f"\nSegments with timestamps and speakers:")
        for segment in segments:
            start = segment["start"]
            end = segment["end"]
            speaker = segment.get("speaker", "UNKNOWN")
            text = segment['text']
            print(f"[{start:.1f}s - {end:.1f}s] [{speaker}]: {text}")


if __name__ == "__main__":
    # Check if pyannote.audio is available (for better error messages)
    pyannote_available = False
    try:
        import pyannote.audio
        pyannote_available = True
    except ImportError:
        pass
    
    # Default to test_debate.mp3 if no argument provided
    audio_file = sys.argv[1] if len(sys.argv) > 1 else "test_debate.mp3"
    
    # Optional: specify model size (tiny, base, small, medium, large)
    # base is a good balance between speed and accuracy
    model_size = sys.argv[2] if len(sys.argv) > 2 else "base"
    
    # Optional: specify number of speakers (if known)
    num_speakers = None
    if len(sys.argv) > 3:
        try:
            num_speakers = int(sys.argv[3])
        except ValueError:
            print(f"Warning: Invalid number of speakers '{sys.argv[3]}', using auto-detection")
    
    # Optional: disable diarization with --no-diarization flag
    enable_diarization = "--no-diarization" not in sys.argv
    
    # Warn if diarization is enabled but pyannote is not available
    if enable_diarization and not pyannote_available:
        uv_lock_exists = Path("uv.lock").exists()
        print("\n⚠️  WARNING: pyannote.audio is not available in the current Python environment.")
        if uv_lock_exists:
            print("   You're using uv - run the script with: uv run python transcribe_audio.py <audio_file>")
        else:
            print("   Install with: pip install pyannote.audio")
        print("   Speaker diarization will be disabled. All segments will be labeled as 'UNKNOWN'.\n")
    
    print("=" * 80)
    print("Whisper Audio Transcription with Speaker Diarization")
    print("=" * 80)
    print(f"Audio file: {audio_file}")
    print(f"Model: {model_size}")
    print(f"Language: Portuguese (pt)")
    print(f"Speaker diarization: {'Enabled' if enable_diarization else 'Disabled'}")
    if num_speakers:
        print(f"Number of speakers: {num_speakers}")
    print("=" * 80)
    print()
    
    transcribe_audio(audio_file, language="pt", model_size=model_size, 
                     enable_diarization=enable_diarization, num_speakers=num_speakers)
