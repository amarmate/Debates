#!/usr/bin/env python3
"""
Benchmark transcription quality by comparing model output to reference transcripts.
Computes WER (Word Error Rate) and CER (Character Error Rate) for evaluation.
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

import jiwer

BENCHMARK_REFS = Path("data/benchmark/refs")
BENCHMARK_RESULTS = Path("data/benchmark/results")
AUDIO_DIR = Path("data/debates")


def normalize_text(text: str) -> str:
    """Light normalization for fair WER/CER comparison."""
    return " ".join(text.lower().split()).strip()


def discover_benchmark_pairs(audio_dir: Path, ref_dir: Path) -> list[tuple[Path, Path]]:
    """Find (audio, reference) pairs by matching filenames (stem)."""
    pairs = []
    if not ref_dir.exists():
        return pairs
    for ref_file in ref_dir.glob("*.txt"):
        stem = ref_file.stem
        for ext in (".mp3", ".wav", ".m4a", ".ogg"):
            audio_file = audio_dir / f"{stem}{ext}"
            if audio_file.exists():
                pairs.append((audio_file, ref_file))
                break
    return pairs


def run_benchmark(
    audio_path: Path,
    ref_path: Path,
    model_size: str = "base",
    enable_diarization: bool = False,
    enable_vad: bool = True,
) -> tuple[str, str, float, float]:
    """Transcribe audio, compare to reference, return (reference, hypothesis, wer, cer)."""
    # Add project root to path so transcribe_audio can be imported
    project_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(project_root))
    from transcribe_audio import transcribe_audio

    ref_text = ref_path.read_text(encoding="utf-8")
    hypothesis = transcribe_audio(
        str(audio_path),
        language="pt",
        model_size=model_size,
        enable_diarization=enable_diarization,
        enable_vad=enable_vad,
        condition_on_previous_text=False,
        compression_ratio_threshold=2.0,
    )
    if hypothesis is None:
        hypothesis = ""

    ref_norm = normalize_text(ref_text)
    hyp_norm = normalize_text(hypothesis)

    w = jiwer.wer(ref_norm, hyp_norm)
    c = jiwer.cer(ref_norm, hyp_norm)
    return ref_text, hypothesis, w, c


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark transcription quality (WER/CER) against reference transcripts."
    )
    parser.add_argument("--audio-dir", type=Path, default=AUDIO_DIR, help="Directory containing audio files")
    parser.add_argument("--ref-dir", type=Path, default=BENCHMARK_REFS, help="Directory containing reference .txt transcripts")
    parser.add_argument("--model", default="base", help="Whisper model size (tiny, base, small, medium, large)")
    parser.add_argument("--no-diarization", action="store_true", help="Disable speaker diarization (faster for benchmarking)")
    parser.add_argument("--no-vad", action="store_true", help="Disable VAD")
    parser.add_argument("--output", type=Path, help="Write results JSON path")
    parser.add_argument("--audio", type=Path, help="Single audio file (requires --ref)")
    parser.add_argument("--ref", type=Path, help="Reference transcript (with --audio)")
    args = parser.parse_args()

    if args.audio and args.ref:
        pairs = [(args.audio, args.ref)]
        if not args.audio.exists():
            print(f"Error: Audio file not found: {args.audio}", file=sys.stderr)
            return 1
        if not args.ref.exists():
            print(f"Error: Reference file not found: {args.ref}", file=sys.stderr)
            return 1
    else:
        pairs = discover_benchmark_pairs(args.audio_dir, args.ref_dir)
        if not pairs:
            print("No benchmark pairs found.")
            print(f"  Add reference transcripts to {args.ref_dir}")
            print(f"  (e.g. 2025_04_07_AD_CDU.txt for 2025_04_07_AD_CDU.mp3 in {args.audio_dir})")
            return 0

    print(f"Found {len(pairs)} benchmark pair(s)")
    print(f"Model: {args.model}\n")

    results = []
    total_wer = total_cer = 0.0
    for i, (audio_path, ref_path) in enumerate(pairs):
        print(f"[{i + 1}/{len(pairs)}] {audio_path.name} ...")
        _, _, w, c = run_benchmark(
            audio_path, ref_path,
            model_size=args.model,
            enable_diarization=not args.no_diarization,
            enable_vad=not args.no_vad,
        )
        total_wer += w
        total_cer += c
        results.append({"audio": str(audio_path), "reference": str(ref_path), "wer": w, "cer": c})
        print(f"  WER: {w:.4f}  CER: {c:.4f}")

    n = len(results)
    avg_wer = total_wer / n if n else 0.0
    avg_cer = total_cer / n if n else 0.0

    print("\n" + "=" * 60)
    print(f"SUMMARY (n={n})")
    print(f"  Average WER: {avg_wer:.4f}")
    print(f"  Average CER: {avg_cer:.4f}")
    print("=" * 60)

    out_path = args.output or BENCHMARK_RESULTS / f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {"timestamp": datetime.now().isoformat(), "model": args.model, "n": n, "avg_wer": avg_wer, "avg_cer": avg_cer, "results": results},
            f, indent=2, ensure_ascii=False,
        )
    print(f"Results saved to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
