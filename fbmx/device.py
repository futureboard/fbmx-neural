"""Device selection.

Everything in the stack must run unchanged on a laptop CPU, on CUDA, and in a
Colab runtime.  We therefore never hard-code a device anywhere else; call
:func:`auto_device` once at the top of a script and thread it through.
"""

from __future__ import annotations

import torch

__all__ = ["auto_device", "describe_device", "amp_supported"]


def auto_device(requested: str | None = None) -> torch.device:
    """Resolve a device string.

    ``None`` or ``"auto"`` picks CUDA if it is usable, then Apple MPS, then CPU.
    Anything else is passed straight to :class:`torch.device` so a config can
    pin ``"cpu"`` for reproducibility runs.
    """
    if requested and requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def describe_device(device: torch.device) -> str:
    """Human readable one-liner for logs and checkpoint metadata."""
    device = torch.device(device)
    if device.type == "cuda":
        index = device.index if device.index is not None else torch.cuda.current_device()
        name = torch.cuda.get_device_name(index)
        cap = ".".join(str(c) for c in torch.cuda.get_device_capability(index))
        return f"cuda:{index} ({name}, sm_{cap.replace('.', '')})"
    if device.type == "mps":
        return "mps (Apple Metal)"
    return f"cpu (torch {torch.__version__})"


def amp_supported(device: torch.device) -> bool:
    """AMP is optional everywhere.  FP32 correctness comes first."""
    return torch.device(device).type == "cuda"
