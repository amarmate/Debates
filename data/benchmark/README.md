# Benchmark Layout

- **refs/** — Reference transcripts (manually corrected). Filename stem must match the audio file (e.g. `2025_04_07_AD_CDU.txt` for `2025_04_07_AD_CDU.mp3` in `data/debates/`).
- **results/** — Benchmark score history (JSON files, one per run).

Run the benchmark:
```bash
uv run python scripts/benchmark_transcription.py
```
