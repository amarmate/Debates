"""
Lightweight debate metadata lookup for pipeline transcription priming.
No torch/whisper dependencies.
"""
import csv
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = PROJECT_ROOT / "data" / "links" / "debates_unified.csv"


def lookup_debate_metadata(audio_path: str | Path) -> Optional[dict[str, str]]:
    """
    Look up debate metadata from the unified CSV based on audio filename.

    Args:
        audio_path: Path to the audio file (or filename)

    Returns:
        Dictionary with metadata (candidate1, candidate2, date, channel) or None
    """
    audio_file = Path(audio_path)
    filename = audio_file.stem.lower()

    if not CSV_PATH.exists():
        return None

    def normalize_name(name: str) -> str:
        name = unicodedata.normalize("NFD", name.lower())
        name = "".join(c for c in name if unicodedata.category(c) != "Mn")
        return name.replace(" ", "").replace("-", "")

    try:
        with open(CSV_PATH, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            matches: list[tuple[int, dict[str, str]]] = []

            for row in reader:
                date_str = (row.get("date") or "").strip()
                party1 = (row.get("party1") or "").strip()
                party2 = (row.get("party2") or "").strip()
                candidate1 = (row.get("candidate1") or "").strip()
                candidate2 = (row.get("candidate2") or "").strip()
                channel = (row.get("channel") or "").strip()

                name1 = candidate1 or party1
                name2 = candidate2 or party2

                if not date_str or len(date_str) < 10:
                    continue
                date_prefix = date_str[:10].lower()
                date_prefix_alt = date_str[:10].replace("-", "_").lower()
                if not filename.startswith(date_prefix) and not filename.startswith(date_prefix_alt):
                    continue

                metadata: dict[str, str] = {
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


def build_initial_prompt(metadata: Optional[dict[str, Any]] = None) -> str:
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

    parts = [base_prompt]

    candidate1 = (metadata.get("candidate1") or "").strip()
    candidate2 = (metadata.get("candidate2") or "").strip()
    if candidate1 and candidate2:
        parts.append(f"entre {candidate1} e {candidate2}")
    elif candidate1:
        parts.append(f"com {candidate1}")
    elif candidate2:
        parts.append(f"com {candidate2}")

    date_str = (metadata.get("date") or "").strip()
    if date_str:
        try:
            date_obj = datetime.strptime(date_str, "%Y-%m-%d")
            months_pt = [
                "janeiro", "fevereiro", "março", "abril", "maio", "junho",
                "julho", "agosto", "setembro", "outubro", "novembro", "dezembro",
            ]
            formatted_date = f"{date_obj.day} de {months_pt[date_obj.month - 1]} de {date_obj.year}"
            parts.append(f"realizado em {formatted_date}")
        except ValueError:
            if date_str:
                parts.append(f"realizado em {date_str}")

    channel = (metadata.get("channel") or "").strip()
    if channel:
        parts.append(f"transmitido pela {channel}")

    return ", ".join(parts) + "."
