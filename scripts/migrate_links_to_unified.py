#!/usr/bin/env python3
"""
Regenerate data/links/debates_unified.csv from the per-election CSVs.
"""

import csv
import logging
import uuid
from pathlib import Path

logger = logging.getLogger(__name__)

SOURCE_FILES = [
    ("legislativas", Path("data/links/legislativas_debates_2025.csv")),
    ("presidenciais", Path("data/links/presidencial_debates_2026.csv")),
]
UNIFIED_PATH = Path("data/links/debates_unified.csv")


def _uuid5(*parts: str) -> str:
    ns = uuid.UUID("3d2d9b6a-9d6c-4e44-8d52-81f76f26d0c2")
    return str(uuid.uuid5(ns, "|".join(p.strip() for p in parts if p)))


def main() -> int:
    rows = []
    for typ, path in SOURCE_FILES:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                date = (row.get("date") or "").strip()
                url = (row.get("url") or "").strip()
                if not url.startswith("http") or len(date) < 10:
                    continue
                c1 = (row.get("candidate1") or "").strip()
                c2 = (row.get("candidate2") or "").strip()
                party1, party2 = (c1, c2) if typ == "legislativas" else ("", "")
                cand1, cand2 = ("", "") if typ == "legislativas" else (c1, c2)
                rows.append({
                    "uuid": _uuid5(typ, date, party1, party2, cand1, cand2, url),
                    "year": date[:4],
                    "date": date,
                    "party1": party1,
                    "party2": party2,
                    "candidate1": cand1,
                    "candidate2": cand2,
                    "type": typ,
                    "channel": (row.get("channel") or "").strip(),
                    "url": url,
                })
    if not rows:
        logger.info("No rows to write.")
        return 0
    UNIFIED_PATH.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["uuid", "year", "date", "party1", "party2", "candidate1", "candidate2", "type", "channel", "url"]
    with UNIFIED_PATH.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    logger.info("Wrote %d rows to %s", len(rows), UNIFIED_PATH)
    return 0


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    raise SystemExit(main())
