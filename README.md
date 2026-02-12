# Debates

Extract, transcribe, and benchmark political debate audio from CNN Portugal videos.

## Setup with uv

```powershell
uv venv
.venv\Scripts\activate
uv pip install -e .
playwright install chromium
```

## Usage

**Download and transcribe all debates** (from `data/links/debates_unified.csv`):
```powershell
uv run python process_debates.py
```

**Download only:**
```powershell
uv run python download_all_debates.py
```

## Audio Transcription with Speaker Diarization

The `transcribe_audio.py` script transcribes audio files using OpenAI Whisper and automatically identifies speakers.

### Setup for Speaker Diarization

1. **Install dependencies:**
   ```powershell
   uv pip install -e .
   ```
   This will install all dependencies including `pyannote.audio`.

2. **Get a Hugging Face token:**
   - Go to https://huggingface.co/settings/tokens
   - Create a new token (read access is sufficient)

3. **Accept the model terms:**
   - Visit https://huggingface.co/pyannote/speaker-diarization-3.1
   - Accept the model terms and conditions

4. **Authenticate:**
   ```powershell
   huggingface-cli login
   ```
   Enter your token when prompted.

### Usage

**Basic transcription with speaker diarization:**
```powershell
python transcribe_audio.py audio_file.mp3
```

**Specify Whisper model size:**
```powershell
python transcribe_audio.py audio_file.mp3 base
```

**Specify number of speakers (if known):**
```powershell
python transcribe_audio.py audio_file.mp3 base 2
```

**Disable speaker diarization:**
```powershell
python transcribe_audio.py audio_file.mp3 base --no-diarization
```

**Disable overlap detection** (faster; segments where two speakers talk at once won't be flagged):
```powershell
python transcribe_audio.py audio_file.mp3 --no-overlap-detection
```

**Whisper anti-repetition options** (reduce repetitive hallucinations):
```powershell
python transcribe_audio.py audio_file.mp3 --condition-on-previous-text false --compression-ratio-threshold 2.0
```

The output will be saved to `audio_file.txt` with speaker annotations in the format:
```
[SPEAKER_00]:
Text spoken by speaker 00...

[SPEAKER_01]:
Text spoken by speaker 01...
```

## Scripts

| Script | Purpose |
|--------|---------|
| `scripts/test_first_debate.py` | Download + transcribe first debate (end-to-end test) |
| `scripts/benchmark_transcription.py` | Compare transcription to reference, compute WER/CER |
| `scripts/cut_audio.py` | Cut MP3 by start/end or duration |
| `scripts/migrate_links_to_unified.py` | Regenerate `data/links/debates_unified.csv` from per-election CSVs |

**Test first debate** (download + transcribe):
```powershell
uv run python scripts/test_first_debate.py
```

**Benchmark** (add refs to `data/benchmark/refs/`):
```powershell
uv run python scripts/benchmark_transcription.py
```

**Regenerate unified links:**
```powershell
uv run python scripts/migrate_links_to_unified.py
```

## Dependencies

- `playwright` - For browser automation to intercept network requests
- `yt-dlp` - For downloading and converting video streams to audio
- `openai-whisper` - For audio transcription
- `pyannote.audio` - For speaker diarization and overlapped speech detection
- `librosa` - For audio processing
- `tqdm` - For progress bars
- `jiwer` - For WER/CER benchmarking


