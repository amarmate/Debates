"""
Punctuation restoration for transcribed text using respunct (Portuguese BERT-based model).
Lazy-loads the model to avoid overhead when disabled.
If respunct is not installed or fails to load, returns text unchanged (no-op).
"""
import logging
from typing import Optional

logger = logging.getLogger(__name__)

_model: Optional[object] = None
_available: Optional[bool] = None  # None=not tried, True=ok, False=failed


def _get_model() -> Optional[object]:
    """Lazy-load the RestorePuncts model. Returns None if unavailable."""
    global _model, _available
    if _available is False:
        return None
    if _model is not None:
        return _model
    try:
        from respunct import RestorePuncts
        _model = RestorePuncts()
        _available = True
        logger.info("Loaded punctuation restoration model (respunct)")
        return _model
    except ImportError:
        logger.warning(
            "respunct not available for punctuation restoration. "
            "Install with: pip install respunct (may conflict with torch 2.x)"
        )
        _available = False
        return None
    except Exception:
        logger.exception("Failed to load punctuation restoration model")
        _available = False
        return None


def restore_punctuation(text: str) -> str:
    """
    Restore punctuation and capitalization in transcribed text.

    Args:
        text: Raw transcribed text without or with poor punctuation

    Returns:
        Text with restored punctuation and capitalization, or unchanged if respunct unavailable
    """
    if not text or len(text.strip()) < 5:
        return text
    model = _get_model()
    if model is None:
        return text
    try:
        result = model.restore_puncts(text)
        return result.strip() if result else text
    except Exception:
        logger.exception("Punctuation restoration failed")
        return text
