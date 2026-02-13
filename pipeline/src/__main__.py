"""Entry point for `python -m pipeline.src`."""
from pipeline.src.main import run_pipeline

if __name__ == "__main__":
    import logging
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    run_pipeline()
