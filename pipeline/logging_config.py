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
    Allow on console: WARNING+, and INFO only for server/init messages.
    Block verbose INFO (Merge, Prompt, etc.) to keep terminal clean.
    """
    if record.levelno >= logging.WARNING:
        return True
    if record.levelno != logging.INFO:
        return True
    # INFO: allow only server management and init-style messages
    msg = (record.getMessage() or "").lower()
    name = (record.name or "").lower()
    allow_substrings = (
        "server", "uvicorn", "start", "port", "listen",
        "model loaded", "transcription", "websocket", "disconnect",
        "ready", "loaded", "logging",
    )
    if any(s in msg or s in name for s in allow_substrings):
        return True
    # Block verbose pipeline logs (Merge, Prompt, etc.)
    if "pipeline.sentence_buffer" in record.name:
        return False
    if "pipeline.src.transcriber" in record.name and "prompt" in msg:
        return False
    if "pipeline.punctuation_restore" in record.name:
        return False
    return True


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

    logging.getLogger(__name__).info("Logging to %s", log_path)
    return log_path
