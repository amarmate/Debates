#!/usr/bin/env python3
"""
Script to transcribe audio files using OpenAI Whisper with automatic speaker diarization.
Supports Portuguese language transcription with progress tracking and speaker identification.
"""

import os
# Set PyTorch memory allocation configuration to avoid fragmentation
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import logging
import torch

# PyTorch 2.6+ defaults weights_only=True; pyannote/speechbrain checkpoints use OmegaConf
try:
    from omegaconf import ListConfig, DictConfig
    from omegaconf.base import ContainerMetadata
    torch.serialization.add_safe_globals([ListConfig, DictConfig, ContainerMetadata])
except ImportError:
    pass

import gc
import whisper
import sys
import csv
from pathlib import Path

logger = logging.getLogger(__name__)
from tqdm import tqdm
import librosa
import numpy as np
import threading
import time
import warnings
from typing import List, Dict, Tuple, Optional
from datetime import datetime

# Suppress torchcodec warnings - pyannote.audio will fall back to librosa
warnings.filterwarnings("ignore", message=".*torchcodec.*")
warnings.filterwarnings("ignore", message=".*libtorchcodec.*")
# Suppress std() degrees of freedom warning from pyannote pooling
warnings.filterwarnings("ignore", message=".*std\\(\\): degrees of freedom.*")

# Folder for annotated transcripts (speaker labels, overlap markers)
TRANSCRIPTS_FOLDER = Path("data/transcripts")

def _is_whisperx_model(model_name: str) -> bool:
    """Return True if model should be loaded via WhisperX (Hugging Face path, e.g. inesc-id/WhisperLv3-EP-X)."""
    return "/" in model_name


def _model_name_for_path(model_name: str) -> str:
    """Sanitize model name for use in filenames (replace / and other chars)."""
    return model_name.replace("/", "_").replace("\\", "_")


def get_transcript_path(audio_path: str, model_size: str) -> Path:
    """Return the path where the transcript will be saved for given audio and model."""
    stem = Path(audio_path).stem
    label = _model_name_for_path(model_size)
    return TRANSCRIPTS_FOLDER / f"{stem}_{label}.txt"


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
        logger.error(
            "pyannote.audio is not installed or not accessible. "
            "If using uv: uv run python transcribe_audio.py <audio_file>. "
            "If using pip: pip install pyannote.audio. "
            "Get Hugging Face token at https://huggingface.co/settings/tokens, "
            "run huggingface-cli login, accept terms at "
            "https://huggingface.co/pyannote/speaker-diarization-3.1"
        )
        return None

    logger.info("Loading speaker diarization model...")
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
        
        # Move pipeline to GPU if available for faster processing
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        pipeline.to(device)
        logger.info("Pipeline loaded on device: %s", device)

    except Exception as e:
        logger.error(
            "Error loading diarization model: %s. "
            "Make sure you have: 1) pip install pyannote.audio "
            "2) Hugging Face token 3) huggingface-cli login "
            "4) Accepted model terms at https://huggingface.co/pyannote/speaker-diarization-3.1",
            e,
        )
        return None

    logger.info("Performing speaker diarization...")
    try:
        # Load audio manually to avoid torchcodec/AudioDecoder issues on Windows
        # pyannote.audio 3.1+ can accept memory inputs
        try:
            # OPTIMIZATION: Resample to 16kHz for faster processing
            # pyannote works well with 16kHz, and this reduces data by 3x for 48kHz audio
            target_sr = 16000
            logger.info("Loading audio and resampling to %sHz for faster processing...", target_sr)
            
            # Load with librosa and resample to 16kHz (much faster than 48kHz)
            waveform, original_sr = librosa.load(audio_path, sr=None, mono=True)
            
            # Resample if needed
            if original_sr != target_sr:
                waveform = librosa.resample(waveform, orig_sr=original_sr, target_sr=target_sr)
                sample_rate = target_sr
                logger.info(
                    "Resampled from %sHz to %sHz (reduced by %.1fx)",
                    original_sr, target_sr, original_sr / target_sr,
                )
            else:
                sample_rate = original_sr
            
            # Reshape to (channels, time) -> (1, time) for mono
            if waveform.ndim == 1:
                waveform = waveform[np.newaxis, :]
            
            # Convert to torch tensor and move to same device as pipeline
            # Try to detect device from pipeline, fallback to CUDA if available
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            # Check if pipeline has a device attribute or try to infer from its components
            try:
                if hasattr(pipeline, 'segmentation') and hasattr(pipeline.segmentation, 'device'):
                    device = pipeline.segmentation.device
                elif hasattr(pipeline, 'embedding') and hasattr(pipeline.embedding, 'device'):
                    device = pipeline.embedding.device
            except (AttributeError, RuntimeError):
                pass  # Use default device
            
            waveform_tensor = torch.from_numpy(waveform).to(device)
            
            audio_input = {"waveform": waveform_tensor, "sample_rate": sample_rate}
            duration = waveform_tensor.shape[1] / sample_rate
            logger.info(
                "Loaded audio into memory: shape=%s, sr=%s, duration=%.1fs",
                waveform_tensor.shape, sample_rate, duration,
            )
            
        except Exception as e:
            logger.warning("Failed to load audio into memory: %s. Falling back to file path input.", e)
            audio_input = {"audio": audio_path}

        # OPTIMIZATION: Use min_duration_off to skip very short silence segments (speeds up processing)
        # This parameter filters out silence segments shorter than 0.5 seconds
        diarization_params = {
            "min_duration_off": 0.5  # Ignore silence segments shorter than 0.5s
        }
        
        if num_speakers:
            diarization_params["num_speakers"] = num_speakers
        
        # Show progress indicator for long-running diarization
        logger.info("Processing diarization (this may take a while for long audio files)...")
        start_time = time.time()
        
        # Run diarization in a thread with progress updates
        diarization_result = [None]
        diarization_error = [None]
        
        def run_diarization():
            try:
                diarization_result[0] = pipeline(audio_input, **diarization_params)
            except Exception as e:
                diarization_error[0] = e
        
        diarization_thread = threading.Thread(target=run_diarization)
        diarization_thread.start()
        
        # Show progress while diarization runs
        last_logged = 0
        while diarization_thread.is_alive():
            elapsed = time.time() - start_time
            if int(elapsed) - last_logged >= 10 or int(elapsed) == 0:
                logger.info("  Elapsed: %.1fs...", elapsed)
                last_logged = int(elapsed)
            time.sleep(1)

        diarization_thread.join()
        
        if diarization_error[0]:
            raise diarization_error[0]
        
        diarization = diarization_result[0]
        elapsed_time = time.time() - start_time
        logger.info("Diarization completed in %.1fs", elapsed_time)
        
        # Extract speaker segments
        # Handle different API versions of pyannote.audio
        speaker_segments = []
        
        # Check if it's a DiarizeOutput (newer API) - try to get the annotation
        if hasattr(diarization, 'speaker_diarization'):
            # pyannote 4.x DiarizeOutput
            annotation = diarization.speaker_diarization
            for turn, _, speaker in annotation.itertracks(yield_label=True):
                speaker_segments.append((turn.start, turn.end, speaker))
        elif hasattr(diarization, 'annotation'):
            # DiarizeOutput has an annotation attribute
            annotation = diarization.annotation
            for turn, _, speaker in annotation.itertracks(yield_label=True):
                speaker_segments.append((turn.start, turn.end, speaker))
        elif hasattr(diarization, 'itertracks'):
            # Direct Annotation object (older API)
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                speaker_segments.append((turn.start, turn.end, speaker))
        else:
            # Try to iterate directly - might be iterable
            try:
                for turn, _, speaker in diarization:
                    speaker_segments.append((turn.start, turn.end, speaker))
            except (TypeError, ValueError):
                logger.debug("Diarization type: %s, attributes: %s", type(diarization), dir(diarization))
                raise ValueError(f"Unknown diarization output format: {type(diarization)}. Attributes: {[attr for attr in dir(diarization) if not attr.startswith('_')]}")

        logger.info("Found %d speaker(s)", len(set(seg[2] for seg in speaker_segments)))
        
        # Clean up pipeline and tensors to free VRAM immediately
        if 'pipeline' in locals():
            del pipeline
        if 'waveform_tensor' in locals():
            del waveform_tensor
        if 'diarization' in locals():
            del diarization
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        return speaker_segments
        
    except Exception as e:
        logger.exception("Error during diarization: %s", e)
        return None


def perform_vad(audio_path: str, threshold: float = 0.5):
    """
    Perform Voice Activity Detection (VAD) to identify speech segments.
    Uses Silero VAD model to detect speech and filter out silence/noise.
    
    Args:
        audio_path: Path to the audio file
        threshold: VAD threshold (0.0-1.0), higher = more conservative (default: 0.5)
    
    Returns:
        List of tuples: [(start_time, end_time), ...] for speech segments
    """
    logger.info("Performing Voice Activity Detection (VAD)...")
    try:
        # Load Silero VAD model from PyTorch Hub
        model, utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False,
            onnx=False
        )
        
        # Unpack utilities
        (get_speech_timestamps, _, read_audio, *_) = utils
        
        # Load audio using Silero's read_audio function (resamples to 16kHz)
        sampling_rate = 16000
        wav = read_audio(audio_path, sampling_rate=sampling_rate)
        
        # Get speech timestamps
        speech_timestamps = get_speech_timestamps(
            wav, 
            model, 
            threshold=threshold,
            sampling_rate=sampling_rate
        )
        
        # Convert timestamps to list of (start, end) tuples in seconds
        speech_segments = []
        for ts in speech_timestamps:
            start = ts['start'] / sampling_rate
            end = ts['end'] / sampling_rate
            speech_segments.append((start, end))
        
        total_speech_duration = sum(end - start for start, end in speech_segments)
        total_duration = len(wav) / sampling_rate
        speech_percentage = (total_speech_duration / total_duration * 100) if total_duration > 0 else 0
        
        logger.info(
            "VAD detected %d speech segments. Speech duration: %.1fs / %.1fs (%.1f%%)",
            len(speech_segments), total_speech_duration, total_duration, speech_percentage,
        )
        
        # Clean up
        del model, wav
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return speech_segments
        
    except Exception as e:
        logger.warning("VAD failed: %s. Continuing without VAD filtering...", e, exc_info=True)
        return None


def lookup_debate_metadata(audio_path: str) -> Optional[Dict[str, str]]:
    """
    Look up debate metadata from the unified CSV based on audio filename.

    Args:
        audio_path: Path to the audio file

    Returns:
        Dictionary with metadata (candidate1, candidate2, date, channel) or None
    """
    import unicodedata

    audio_file = Path(audio_path)
    filename = audio_file.stem.lower()

    csv_file = Path("data/links/debates_unified.csv")
    if not csv_file.exists():
        return None

    def normalize_name(name: str) -> str:
        name = unicodedata.normalize("NFD", name.lower())
        name = "".join(c for c in name if unicodedata.category(c) != "Mn")
        return name.replace(" ", "").replace("-", "")

    try:
        with open(csv_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            matches: list[tuple[int, dict[str, str]]] = []

            for row in reader:
                date_str = (row.get("date") or "").strip()
                party1 = (row.get("party1") or "").strip()
                party2 = (row.get("party2") or "").strip()
                candidate1 = (row.get("candidate1") or "").strip()
                candidate2 = (row.get("candidate2") or "").strip()
                channel = (row.get("channel") or "").strip()

                # Display names: candidate if present, else party
                name1 = candidate1 or party1
                name2 = candidate2 or party2

                if not date_str or len(date_str) < 10:
                    continue
                # Match both YYYY-MM-DD (new format) and YYYY_MM_DD (legacy)
                date_prefix = date_str[:10].lower()  # 2025-04-07
                date_prefix_alt = date_str[:10].replace("-", "_").lower()  # 2025_04_07
                if not filename.startswith(date_prefix) and not filename.startswith(date_prefix_alt):
                    continue

                metadata = {
                    "candidate1": name1,
                    "candidate2": name2,
                    "date": date_str,
                    "channel": channel,
                }

                name1_norm = normalize_name(name1) if name1 else ""
                name2_norm = normalize_name(name2) if name2 else ""
                filename_norm = normalize_name(filename)

                match_score = 0
                if name1_norm and name1_norm in filename_norm:
                    match_score += 1
                if name2_norm and name2_norm in filename_norm:
                    match_score += 1

                matches.append((match_score, metadata))

            if matches:
                matches.sort(key=lambda x: x[0], reverse=True)
                return matches[0][1]
    except Exception:
        pass

    return None


def build_initial_prompt(metadata: Optional[Dict[str, str]] = None) -> str:
    """
    Build an initial prompt for Whisper transcription.
    
    Args:
        metadata: Optional dictionary with debate metadata (candidate1, candidate2, date, channel)
    
    Returns:
        Initial prompt string
    """
    base_prompt = "Transcrição de um debate político em Portugal"
    
    if not metadata:
        return base_prompt
    
    # Build enriched prompt with metadata
    parts = [base_prompt]
    
    # Add candidates
    candidate1 = metadata.get('candidate1', '').strip()
    candidate2 = metadata.get('candidate2', '').strip()
    if candidate1 and candidate2:
        parts.append(f"entre {candidate1} e {candidate2}")
    elif candidate1:
        parts.append(f"com {candidate1}")
    elif candidate2:
        parts.append(f"com {candidate2}")
    
    # Add date
    date_str = metadata.get('date', '').strip()
    if date_str:
        try:
            date_obj = datetime.strptime(date_str, '%Y-%m-%d')
            # Format date in Portuguese style: "dia X de mês de YYYY"
            months_pt = [
                'janeiro', 'fevereiro', 'março', 'abril', 'maio', 'junho',
                'julho', 'agosto', 'setembro', 'outubro', 'novembro', 'dezembro'
            ]
            formatted_date = f"{date_obj.day} de {months_pt[date_obj.month - 1]} de {date_obj.year}"
            parts.append(f"realizado em {formatted_date}")
        except ValueError:
            # If date parsing fails, use original date string
            if date_str:
                parts.append(f"realizado em {date_str}")
    
    # Add channel
    channel = metadata.get('channel', '').strip()
    if channel:
        parts.append(f"transmitido pela {channel}")
    
    return ", ".join(parts) + "."


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


def perform_overlap_detection(audio_path: str) -> List[Tuple[float, float]]:
    """
    Detect overlapped speech (two or more speakers talking at once) using pyannote.

    Returns:
        List of (start, end) tuples for overlapped regions in seconds.
    """
    try:
        from pyannote.audio import Pipeline
    except ImportError:
        logger.warning("pyannote.audio not available for overlap detection.")
        return []

    logger.info("Loading overlapped speech detection model...")
    try:
        try:
            pipeline = Pipeline.from_pretrained(
                "pyannote/overlapped-speech-detection",
                token=True,
            )
        except TypeError:
            pipeline = Pipeline.from_pretrained(
                "pyannote/overlapped-speech-detection",
                use_auth_token=True,
            )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        pipeline.to(device)
    except Exception as e:
        logger.warning(
            "Could not load overlap model: %s. Visit "
            "https://huggingface.co/pyannote/overlapped-speech-detection to accept terms.",
            e,
        )
        return []

    try:
        target_sr = 16000
        waveform, original_sr = librosa.load(audio_path, sr=None, mono=True)
        if original_sr != target_sr:
            waveform = librosa.resample(waveform, orig_sr=original_sr, target_sr=target_sr)
        if waveform.ndim == 1:
            waveform = waveform[np.newaxis, :]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        waveform_tensor = torch.from_numpy(waveform).to(device)
        audio_input = {"waveform": waveform_tensor, "sample_rate": target_sr}
    except Exception:
        audio_input = {"audio": audio_path}

    try:
        output = pipeline(audio_input)
        segments = []
        if hasattr(output, "get_timeline"):
            for seg in output.get_timeline().support():
                segments.append((float(seg.start), float(seg.end)))
        elif hasattr(output, "itertracks"):
            for segment, _, _ in output.itertracks(yield_label=True):
                segments.append((float(segment.start), float(segment.end)))
        logger.info("Detected %d overlapped speech region(s)", len(segments))
        del pipeline
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return segments
    except Exception as e:
        logger.warning("Overlap detection failed: %s", e)
        return []


def _segments_overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> bool:
    return max(a_start, b_start) < min(a_end, b_end)


def _merge_overlap_regions(regions: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """Merge overlapping or adjacent regions into a sorted, non-overlapping list."""
    if not regions:
        return []
    sorted_regions = sorted(regions, key=lambda r: r[0])
    merged: List[Tuple[float, float]] = [sorted_regions[0]]
    for start, end in sorted_regions[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def compute_overlap_regions_from_diarization(
    speaker_segments: List[Tuple[float, float, str]]
) -> List[Tuple[float, float]]:
    """
    Derive overlap regions from diarization output by finding intervals where
    2+ different speakers are active at the same time.

    Handles 3-way (or more) overlap when pyannote diarization outputs overlapping
    speaker segments. Returns empty list if diarization produces non-overlapping
    (partition) output.

    Returns:
        List of (start, end) tuples for overlapped regions in seconds.
    """
    if not speaker_segments:
        return []

    events: List[Tuple[float, int, str]] = []  # (time, +1 or -1, speaker)
    for start, end, speaker in speaker_segments:
        events.append((start, 1, speaker))
        events.append((end, -1, speaker))

    # Sort by time; for same time, process +1 before -1 (start before end)
    events.sort(key=lambda x: (x[0], -x[1]))

    active_speakers: set = set()
    overlap_regions: List[Tuple[float, float]] = []
    overlap_start: Optional[float] = None

    for t, delta, speaker in events:
        if delta == 1:
            active_speakers.add(speaker)
        else:
            active_speakers.discard(speaker)

        if len(active_speakers) >= 2:
            if overlap_start is None:
                overlap_start = t
        else:
            if overlap_start is not None:
                overlap_regions.append((overlap_start, t))
                overlap_start = None

    if overlap_regions:
        logger.info("Diarization-based overlap: %d region(s) where 2+ speakers active", len(overlap_regions))
    return overlap_regions


def transcribe_audio(audio_path: str, language: str = "pt", model_size: str = "base", 
                     enable_diarization: bool = True, num_speakers: Optional[int] = None,
                     enable_vad: bool = True, initial_prompt: Optional[str] = None,
                     condition_on_previous_text: bool = False,
                     compression_ratio_threshold: float = 2.0,
                     chunk_length_s: Optional[int] = None,
                     enable_overlap_detection: bool = True):
    """
    Transcribe an audio file using Whisper with progress tracking and optional speaker diarization.
    
    Args:
        audio_path: Path to the audio file
        language: Language code (default: "pt" for Portuguese)
        model_size: Whisper model size - "tiny", "base", "small", "medium", "large"
                   Larger models are more accurate but slower
        enable_diarization: Whether to perform speaker diarization (default: True)
        num_speakers: Optional number of speakers for diarization. If None, auto-detected.
        enable_vad: Whether to use Voice Activity Detection to filter non-speech segments (default: True)
        initial_prompt: Optional initial prompt to guide transcription and reduce hallucinations
    """
    if not os.path.exists(audio_path):
        logger.error("Audio file not found: %s", audio_path)
        return

    # Get audio duration for progress tracking
    logger.info("Loading audio file to get duration...")
    try:
        duration = librosa.get_duration(path=audio_path)
        duration_str = f"{int(duration // 60)}m {int(duration % 60)}s"
        logger.info("Audio duration: %s (%.1f seconds)", duration_str, duration)
    except Exception as e:
        logger.warning("Could not get audio duration: %s", e)
        duration = None
    
    # Perform VAD if enabled
    vad_segments = None
    if enable_vad:
        vad_segments = perform_vad(audio_path)
        if vad_segments is None:
            logger.info("Continuing without VAD filtering...")

    # Look up metadata and build initial prompt before loading model (WhisperX needs it at load time)
    if initial_prompt is None:
        metadata = lookup_debate_metadata(audio_path)
        initial_prompt = build_initial_prompt(metadata)
        if metadata:
            logger.info(
                "Found metadata: %s vs %s (%s)",
                metadata.get("candidate1", ""), metadata.get("candidate2", ""), metadata.get("date", ""),
            )

    use_whisperx = _is_whisperx_model(model_size)
    model_label = _model_name_for_path(model_size)  # for progress bar and output filename

    if use_whisperx:
        try:
            import whisperx
        except ImportError:
            logger.error(
                "WhisperX is required for Hugging Face models like '%s'. "
                "Install with: pip install whisperx",
                model_size,
            )
            return None

        logger.info("Loading WhisperX model '%s'...", model_size)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "float16" if device == "cuda" else "int8"
        asr_options = {}
        if initial_prompt:
            asr_options["initial_prompt"] = initial_prompt
        model = whisperx.load_model(
            model_size,
            device=device,
            compute_type=compute_type,
            language=language,
            task="transcribe",
            asr_options=asr_options,
        )
    else:
        logger.info("Loading Whisper model '%s'...", model_size)
        model = whisper.load_model(model_size)

    logger.info("Starting transcription...")
    if initial_prompt:
        logger.info("Using initial prompt: %s...", initial_prompt[:100])

    # Use progress context manager
    with TranscriptionProgress(duration, model_label) as progress:
        # Update progress in a separate thread
        def update_loop():
            while not progress.done:
                progress.update()
                time.sleep(0.5)  # Update every 500ms

        update_thread = threading.Thread(target=update_loop, daemon=True)
        update_thread.start()

        try:
            if use_whisperx:
                from whisperx.audio import load_audio

                audio_array = load_audio(audio_path)
                transcribe_kwargs = dict(
                    batch_size=16,
                    print_progress=False,
                    verbose=False,
                )
                wx_result = model.transcribe(audio_array, **transcribe_kwargs)
                # Normalize to same format as openai-whisper
                segments = wx_result.get("segments", [])
                result = {
                    "segments": segments,
                    "text": " ".join(s.get("text", "").strip() for s in segments),
                }
            else:
                # Transcribe with Portuguese language specified and initial prompt
                # verbose=False suppresses Whisper's built-in progress (we use our own)
                # initial_prompt helps guide the model and reduce hallucinations
                transcribe_kwargs = dict(
                    language=language,
                    verbose=False,
                    initial_prompt=initial_prompt,
                    condition_on_previous_text=condition_on_previous_text,
                    compression_ratio_threshold=compression_ratio_threshold,
                )
                if chunk_length_s is not None:
                    transcribe_kwargs["chunk_length_s"] = chunk_length_s
                result = model.transcribe(audio_path, **transcribe_kwargs)
        finally:
            progress.done = True
            time.sleep(0.6)  # Allow final update
    
    # Filter transcription segments using VAD if available
    if vad_segments and result.get("segments"):
        logger.info("Filtering transcription segments using VAD...")
        original_count = len(result["segments"])
        filtered_segments = []
        
        for segment in result["segments"]:
            seg_start = segment["start"]
            seg_end = segment["end"]
            seg_mid = (seg_start + seg_end) / 2
            
            # Check if segment overlaps with any VAD-detected speech segment
            overlaps = False
            for vad_start, vad_end in vad_segments:
                if seg_mid >= vad_start and seg_mid <= vad_end:
                    overlaps = True
                    break
            
            if overlaps:
                filtered_segments.append(segment)
        
        result["segments"] = filtered_segments
        filtered_count = len(filtered_segments)
        logger.info("Filtered %d segments to %d speech segments", original_count, filtered_count)
        
        # Rebuild full text from filtered segments
        result["text"] = " ".join(seg["text"].strip() for seg in filtered_segments)
    
    # Perform speaker diarization if enabled
    speaker_segments = None
    if enable_diarization:
        speaker_segments = perform_speaker_diarization(audio_path, num_speakers)
        if speaker_segments is None:
            logger.warning(
                "Speaker diarization failed or is not available. All segments will be labeled as 'UNKNOWN'. "
                "To enable: pip install pyannote.audio, Hugging Face token, huggingface-cli login, "
                "accept terms at https://huggingface.co/pyannote/speaker-diarization-3.1. "
                "Or run with --no-diarization to disable."
            )
    
    # Match speakers to transcription segments
    segments = result.get("segments", [])
    if speaker_segments and segments:
        segments = match_speakers_to_segments(segments, speaker_segments)
        result["segments"] = segments
    elif enable_diarization and segments:
        # Diarization was enabled but failed - add UNKNOWN speaker labels
        for segment in segments:
            segment["speaker"] = "UNKNOWN"

    # Overlap detection: flag segments where two or more speakers talk at once
    overlap_regions: List[Tuple[float, float]] = []
    if enable_overlap_detection:
        overlap_regions = perform_overlap_detection(audio_path)
        # Also derive overlap from diarization (catches 3-way overlap when
        # pyannote outputs overlapping speaker segments)
        if speaker_segments:
            diar_overlaps = compute_overlap_regions_from_diarization(speaker_segments)
            overlap_regions = _merge_overlap_regions(overlap_regions + diar_overlaps)
    # Small padding to account for timing drift between Whisper and pyannote
    OVERLAP_MATCH_PADDING_S = 0.15
    for segment in segments:
        seg_overlap = False
        seg_start, seg_end = segment["start"], segment["end"]
        for ov_start, ov_end in overlap_regions:
            # Expand overlap region slightly to catch boundary misalignments
            pad_start = max(0, ov_start - OVERLAP_MATCH_PADDING_S)
            pad_end = ov_end + OVERLAP_MATCH_PADDING_S
            if _segments_overlap(seg_start, seg_end, pad_start, pad_end):
                seg_overlap = True
                break
        segment["overlap"] = seg_overlap
    
    # Generate annotated text
    if segments and speaker_segments:
        # Build text with speaker annotations
        annotated_lines = []
        current_speaker = None
        
        for segment in segments:
            speaker = segment.get("speaker", "UNKNOWN")
            text = segment["text"].strip()
            if segment.get("overlap", False):
                text += " [!]"
            
            if speaker != current_speaker:
                if annotated_lines:  # Add blank line between speakers
                    annotated_lines.append("")
                annotated_lines.append(f"[{speaker}]:")
                current_speaker = speaker
            
            annotated_lines.append(text)
        
        annotated_text = "\n".join(annotated_lines)
    elif segments:
        # No diarization: still include overlap markers
        parts = []
        for seg in segments:
            t = seg["text"].strip()
            if seg.get("overlap", False):
                t += " [!]"
            parts.append(t)
        annotated_text = " ".join(parts)
    else:
        annotated_text = result["text"]
    
    # Save annotated text to data/transcripts/ (speaker labels, [!] overlap markers)
    audio_file = Path(audio_path)
    TRANSCRIPTS_FOLDER.mkdir(parents=True, exist_ok=True)
    output_path = TRANSCRIPTS_FOLDER / f"{audio_file.stem}_{model_label}.txt"
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(annotated_text)
    
    logger.info("Transcription complete! Text saved to: %s", output_path)

    if speaker_segments:
        num_unique_speakers = len(set(seg.get("speaker", "UNKNOWN") for seg in segments))
        logger.info("Identified %d speaker(s)", num_unique_speakers)

    logger.info("Transcribed text (%d characters):\n%s\n%s", len(annotated_text), "-" * 80, annotated_text)
    logger.info("-" * 80)

    # Also log segments with speaker labels if available
    if segments:
        logger.info("Segments with timestamps and speakers:")
        for segment in segments:
            start = segment["start"]
            end = segment["end"]
            speaker = segment.get("speaker", "UNKNOWN")
            text = segment["text"]
            logger.info("[%.1fs - %.1fs] [%s]: %s", start, end, speaker, text)

    # Clean up memory
    logger.info("Cleaning up memory...")
    if 'model' in locals():
        del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result["text"]


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # Check if pyannote.audio is available (for better error messages)
    pyannote_available = False
    try:
        import pyannote.audio
        pyannote_available = True
    except ImportError:
        pass
    
    # Default to test_debate.mp3 if no argument provided
    audio_file = sys.argv[1] if len(sys.argv) > 1 else "test_debate.mp3"
    
    # Optional: specify model (tiny, base, small, medium, large, or Hugging Face path like inesc-id/WhisperLv3-EP-X)
    model_size = sys.argv[2] if len(sys.argv) > 2 else "base"
    
    # Optional: specify number of speakers (if known)
    num_speakers = None
    if len(sys.argv) > 3:
        try:
            num_speakers = int(sys.argv[3])
        except ValueError:
            logger.warning("Invalid number of speakers '%s', using auto-detection", sys.argv[3])
    
    # Optional: disable diarization with --no-diarization flag
    enable_diarization = "--no-diarization" not in sys.argv
    
    # Optional: disable VAD with --no-vad flag
    enable_vad = "--no-vad" not in sys.argv

    # Optional: disable overlap detection
    enable_overlap_detection = "--no-overlap-detection" not in sys.argv

    # Optional: Whisper anti-repetition (defaults reduce hallucinations)
    condition_on_previous_text = False
    compression_ratio_threshold = 2.0
    chunk_length_s = None
    if "--condition-on-previous-text" in sys.argv:
        idx = sys.argv.index("--condition-on-previous-text")
        if idx + 1 < len(sys.argv):
            condition_on_previous_text = sys.argv[idx + 1].lower() in ("1", "true", "yes", "y", "on")
    if "--compression-ratio-threshold" in sys.argv:
        idx = sys.argv.index("--compression-ratio-threshold")
        if idx + 1 < len(sys.argv):
            compression_ratio_threshold = float(sys.argv[idx + 1])
    if "--chunk-length-s" in sys.argv:
        idx = sys.argv.index("--chunk-length-s")
        if idx + 1 < len(sys.argv):
            chunk_length_s = int(sys.argv[idx + 1])
    
    # Optional: custom initial prompt
    initial_prompt = None
    if "--prompt" in sys.argv:
        try:
            prompt_idx = sys.argv.index("--prompt")
            if prompt_idx + 1 < len(sys.argv):
                initial_prompt = sys.argv[prompt_idx + 1]
        except (ValueError, IndexError):
            logger.warning("--prompt flag provided but no prompt text found")
    
    # Warn if diarization is enabled but pyannote is not available
    if enable_diarization and not pyannote_available:
        uv_lock_exists = Path("uv.lock").exists()
        logger.warning(
            "pyannote.audio is not available. Speaker diarization will be disabled. "
            "Run with: %s",
            "uv run python transcribe_audio.py <audio_file>" if uv_lock_exists else "pip install pyannote.audio",
        )

    logger.info("=" * 80)
    logger.info("Whisper Audio Transcription with Speaker Diarization")
    logger.info("=" * 80)
    logger.info("Audio file: %s", audio_file)
    logger.info("Model: %s", model_size)
    logger.info("Language: Portuguese (pt)")
    logger.info("Speaker diarization: %s", "Enabled" if enable_diarization else "Disabled")
    logger.info("VAD (Voice Activity Detection): %s", "Enabled" if enable_vad else "Disabled")
    logger.info("Overlap detection: %s", "Enabled" if enable_overlap_detection else "Disabled")
    if initial_prompt:
        logger.info("Initial prompt: %s...", initial_prompt[:80])
    if num_speakers:
        logger.info("Number of speakers: %s", num_speakers)
    logger.info("=" * 80)
    
    transcribe_audio(
        audio_file,
        language="pt",
        model_size=model_size,
        enable_diarization=enable_diarization,
        num_speakers=num_speakers,
        enable_vad=enable_vad,
        initial_prompt=initial_prompt,
        condition_on_previous_text=condition_on_previous_text,
        compression_ratio_threshold=compression_ratio_threshold,
        chunk_length_s=chunk_length_s,
        enable_overlap_detection=enable_overlap_detection,
    )
