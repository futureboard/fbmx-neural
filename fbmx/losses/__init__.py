"""Loss registry and composition.

A training config names losses and weights::

    loss:
      - {name: mae, weight: 1.0}
      - {name: mrstft, weight: 0.5, max_length: 4096}

:class:`CompositeLoss` returns the weighted total *and* the individual terms,
so the log shows which one is actually moving.
"""

from __future__ import annotations

from typing import Any, Callable, Iterable, Mapping

import torch
import torch.nn as nn

from fbmx.losses.auxiliary import AuxTraceLoss, EnvelopeLoss, TransientLoss
from fbmx.losses.spectral import MultiResolutionSTFTLoss, STFTLoss
from fbmx.losses.waveform import (
    DCLoss,
    ESRLoss,
    FBMXLoss,
    MAELoss,
    MSELoss,
    NormalizedMAELoss,
    PreEmphasis,
    SilenceDCLoss,
)

__all__ = [
    "FBMXLoss",
    "MAELoss",
    "MSELoss",
    "NormalizedMAELoss",
    "ESRLoss",
    "DCLoss",
    "SilenceDCLoss",
    "PreEmphasis",
    "STFTLoss",
    "MultiResolutionSTFTLoss",
    "EnvelopeLoss",
    "TransientLoss",
    "AuxTraceLoss",
    "CompositeLoss",
    "LOSS_REGISTRY",
    "register_loss",
    "build_loss",
]

LOSS_REGISTRY: dict[str, Callable[..., FBMXLoss]] = {
    "mae": MAELoss,
    "l1": MAELoss,
    "norm_mae": NormalizedMAELoss,
    "mse": MSELoss,
    "esr": ESRLoss,
    "dc": DCLoss,
    "silence_dc": SilenceDCLoss,
    "stft": STFTLoss,
    "mrstft": MultiResolutionSTFTLoss,
    "envelope": EnvelopeLoss,
    "transient": TransientLoss,
    "aux_trace": AuxTraceLoss,
}


def register_loss(name: str) -> Callable[[Callable[..., FBMXLoss]], Callable[..., FBMXLoss]]:
    def decorate(factory: Callable[..., FBMXLoss]) -> Callable[..., FBMXLoss]:
        if name in LOSS_REGISTRY:
            raise ValueError(f"loss {name!r} already registered")
        LOSS_REGISTRY[name] = factory
        return factory

    return decorate


class CompositeLoss(nn.Module):
    """Weighted sum of named losses, reporting every term."""

    def __init__(self, terms: Mapping[str, tuple[float, FBMXLoss]]) -> None:
        super().__init__()
        self.weights = {name: float(w) for name, (w, _) in terms.items()}
        self.terms = nn.ModuleDict({name: fn for name, (_, fn) in terms.items()})

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        aux: dict[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        total = pred.new_zeros(())
        parts: dict[str, float] = {}
        for name, fn in self.terms.items():
            value = fn(pred, target, aux)
            parts[name] = float(value.detach())
            total = total + self.weights[name] * value
        parts["total"] = float(total.detach())
        return total, parts

    def describe(self) -> str:
        return " + ".join(f"{w:g}*{n}" for n, w in self.weights.items())


def build_loss(spec: Iterable[Mapping[str, Any]] | None) -> CompositeLoss:
    """Build a :class:`CompositeLoss` from a config list.

    Defaults to plain MAE if nothing is specified, which is the least
    surprising thing a config-less call can do.
    """
    entries = list(spec or [{"name": "mae", "weight": 1.0}])
    terms: dict[str, tuple[float, FBMXLoss]] = {}
    for entry in entries:
        entry = dict(entry)
        name = entry.pop("name", None)
        if name is None:
            raise ValueError("each loss entry needs a 'name'")
        if name not in LOSS_REGISTRY:
            raise KeyError(f"unknown loss {name!r}; known: {sorted(LOSS_REGISTRY)}")
        weight = float(entry.pop("weight", 1.0))
        key = entry.pop("as", name)
        if key in terms:
            raise ValueError(f"duplicate loss key {key!r}; use 'as' to rename")
        terms[key] = (weight, LOSS_REGISTRY[name](**entry))
    return CompositeLoss(terms)
