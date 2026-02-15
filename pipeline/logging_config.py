"""
Logging configuration for the Speech-to-Fact pipeline server.

- File: Full logs to logs/YYYY-MM-DD_HH-MM-SS.log (timestamped per server start)
- Console: Minimal output (server management, init, errors only)
"""
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


def _console_filter(record: logging.LogRecord) -> bool:
    """
    Allow on console: WARNING+ only. Block all INFO.
    Terminal stays minimal; full logs go to file.
    """
    return record.levelno >= logging.WARNING


def setup_server_logging(
    log_dir: Optional[Path] = None,
    project_root: Optional[Path] = None,
) -> Path:
    """
    Configure logging for the pipeline server.

    - File handler: Full logs to logs/YYYY-MM-DD_HH-MM-SS.log
    - Console handler: Minimal (server management, init, errors)

    Returns:
        Path to the log file.
    """
    if project_root is None:
        project_root = Path(__file__).resolve().parent.parent
    if log_dir is None:
        log_dir = project_root / "logs"

    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_path = log_dir / f"{timestamp}.log"

    file_fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(file_fmt))

    console_fmt = "%(message)s"
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter(console_fmt))
    console_handler.addFilter(_console_filter)

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    for h in root.handlers[:]:
        root.removeHandler(h)
    root.addHandler(file_handler)
    root.addHandler(console_handler)

    print(f"Logging to {log_path}", file=sys.stdout)
    return log_path
