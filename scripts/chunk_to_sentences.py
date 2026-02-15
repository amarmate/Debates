#!/usr/bin/env python3
"""
Parse transcription chunks, reconstruct full text,
segment into sentences, and export to CSV for fact-checking model training.

Supports JSONL (debug_frame format) or plain text (one chunk per line).
Uses shared sentence segmentation logic from pipeline.sentence_buffer.

  uv run python scripts/chunk_to_sentences.py -i data/debug_chunks.jsonl -o data/sentences.csv
  uv run python scripts/chunk_to_sentences.py -i chunks.txt --format lines
"""
import argparse
import csv
import json
import logging
import sys
from pathlib import Path

# Add project root for pipeline imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.sentence_buffer import segment_sentences

logger = logging.getLogger(__name__)


def parse_chunks(path: Path, fmt: str) -> list[str]:
    """
    Parse chunks from file. Returns list of chunk strings in order.
    """
    lines = path.read_text(encoding="utf-8").splitlines()
    chunks: list[str] = []

    if fmt == "auto":
        first_line = next((ln.strip() for ln in lines if ln.strip()), "")
        fmt = "jsonl" if first_line.startswith("{") else "lines"
        logger.info("Auto-detected format: %s", fmt)

    if fmt == "jsonl":
        for i, line in enumerate(lines, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                merge_info = obj.get("merge_info") if isinstance(obj.get("merge_info"), dict) else {}
                text = merge_info.get("new_content") or obj.get("text") or obj.get("raw")
                if text is not None:
                    chunks.append(str(text).strip())
                else:
                    logger.debug("Skipping line %d: no raw/text field", i)
            except json.JSONDecodeError as e:
                logger.warning("Invalid JSON at line %d: %s", i, e)
        return chunks

    # fmt == "lines"
    for line in lines:
        line = line.strip()
        if line:
            chunks.append(line)
    return chunks


def reconstruct_full_text(chunks: list[str]) -> str:
    """
    Reconstruct full text by joining ordered chunk content.
    """
    return " ".join(chunk for chunk in chunks if chunk).strip()


def export_csv(sentences: list[str], path: Path) -> None:
    """
    Export sentences to CSV with columns text, is_fact.
    is_fact is left empty for manual annotation.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["text", "is_fact"])
        for s in sentences:
            writer.writerow([s, ""])
    logger.info("Exported %d sentences to %s", len(sentences), path)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Parse overlapping chunks, deduplicate, segment into sentences, export CSV.",
    )
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        required=True,
        help="Path to log file (JSONL or plain text)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output CSV path (default: {input_stem}_sentences.csv)",
    )
    parser.add_argument(
        "--format",
        choices=["auto", "jsonl", "lines"],
        default="auto",
        help="Input format; auto detects from first line",
    )
    parser.add_argument(
        "--language",
        default="portuguese",
        help="Language for sentence tokenizer (e.g. portuguese, english)",
    )
    args = parser.parse_args()

    if not args.input.exists():
        logger.error("Input file not found: %s", args.input)
        return 1

    if args.output is None:
        args.output = args.input.with_stem(args.input.stem + "_sentences").with_suffix(".csv")

    chunks = parse_chunks(args.input, args.format)
    logger.info("Parsed %d chunks", len(chunks))

    if not chunks:
        logger.error("No chunks found in input")
        return 1

    full_text = reconstruct_full_text(chunks)
    logger.info("Reconstructed full text: %d characters", len(full_text))

    sentences = segment_sentences(full_text, args.language)
    seen: set[str] = set()
    sentences = [s for s in sentences if s not in seen and not seen.add(s)]
    logger.info("Segmented into %d sentences (after dedup)", len(sentences))

    export_csv(sentences, args.output)
    return 0


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    sys.exit(main())
