"""
Punctuation restoration for transcribed text.
Tries kredor/punctuate-all (transformers, torch 2.x compatible) first, then respunct.
Lazy-loads the model. If both fail, returns text unchanged (no-op).
"""
import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)

_model: Optional[object] = None
_model_type: Optional[str] = None  # "punctuate_all" | "respunct"
_available: Optional[bool] = None  # None=not tried, True=ok, False=failed


def _get_model() -> Optional[object]:
    """Lazy-load punctuation model. Prefer punctuate-all (torch 2.x), fallback to respunct."""
    global _model, _model_type, _available
    if _available is False:
        return None
    if _model is not None:
        return _model

    # Try kredor/punctuate-all first (works with torch 2.x, supports Portuguese)
    try:
        from transformers import pipeline
        pipe = pipeline(
            "token-classification",
            model="kredor/punctuate-all",
            aggregation_strategy="first",
        )
        _model = pipe
        _model_type = "punctuate_all"
        _available = True
        logger.info("Loaded punctuation restoration model (kredor/punctuate-all)")
        return _model
    except Exception as e:
        logger.debug("punctuate-all not available: %s", e)

    # Fallback to respunct (may conflict with torch 2.x)
    try:
        from respunct import RestorePuncts
        _model = RestorePuncts()
        _model_type = "respunct"
        _available = True
        logger.info("Loaded punctuation restoration model (respunct)")
        return _model
    except ImportError:
        logger.warning(
            "No punctuation restoration available. "
            "Install: pip install transformers (for kredor/punctuate-all) or pip install respunct"
        )
        _available = False
        return None
    except Exception:
        logger.exception("Failed to load punctuation restoration model")
        _available = False
        return None


def _restore_punctuate_all(text: str, model) -> str:
    """Restore punctuation using kredor/punctuate-all token classification output."""
    entities = model(text)
    if not entities:
        return text
    parts = []
    for item in entities:
        word = item.get("word", "").strip()
        entity = item.get("entity_group", item.get("entity", "0"))
        if not word:
            continue
        punct = entity if entity and entity != "0" else ""
        parts.append(word + punct)
    return " ".join(parts) if parts else text


def restore_punctuation(text: str) -> str:
    """
    Restore punctuation and capitalization in transcribed text.

    Args:
        text: Raw transcribed text without or with poor punctuation

    Returns:
        Text with restored punctuation, or unchanged if no model available
    """
    if not text or len(text.strip()) < 5:
        return text
    model = _get_model()
    if model is None:
        return text
    try:
        if _model_type == "punctuate_all":
            return _restore_punctuate_all(text, model)
        if _model_type == "respunct":
            return model.restore_puncts(text).strip() or text
        return text
    except Exception:
        logger.exception("Punctuation restoration failed")
        return text


def cleanup_chunk_artifacts(text: str) -> str:
    """
    Fix common chunk-boundary artifacts: ellipsis runs, period-space-period.
    """
    if not text or len(text.strip()) < 3:
        return text
    # Replace "... ..." or multiple dots with single ellipsis or period
    text = re.sub(r"\.\s*\.\s*\.(\s*\.\s*\.\s*\.)*", "...", text)
    text = re.sub(r"\.\s+\.", ".", text)
    # Collapse multiple spaces
    text = re.sub(r"  +", " ", text)
    return text.strip()
