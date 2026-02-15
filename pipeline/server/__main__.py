"""Run the web server: python -m pipeline.server [--port PORT]"""
import argparse
import logging
import os
import sys
from pathlib import Path

from pipeline.logging_config import setup_server_logging
from pipeline.server.app import run_server

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent.parent
    setup_server_logging(project_root=project_root)

    parser = argparse.ArgumentParser(description="Speech-to-Fact web server")
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("PORT", "8000")),
        help="Port to bind (default: 8000 or PORT env var)",
    )
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind")
    args = parser.parse_args()
    run_server(host=args.host, port=args.port)
