"""Pipeline utilities."""


def resolve_compute_type(compute_type: str, device: str) -> str:
    """Resolve compute type: if 'auto', use float16 for GPU (cuda/mps) and int8 for CPU."""
    if compute_type == "auto":
        return "float16" if device in ("cuda", "mps") else "int8"
    return compute_type


def resolve_device(device: str) -> str:
    """Resolve device: cuda if available, else mps, else cpu."""
    if device == "auto":
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass
        return "cpu"
    if device == "cuda":
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
        except ImportError:
            pass
        return "cpu"
    if device == "mps":
        try:
            import torch
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass
        return "cpu"
    return "cpu"
