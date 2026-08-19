"""Reporting metrics.

Kept separate from ``fbmx.losses``: what a model is optimised for and what it
is judged by should be allowed to differ, and mixing the two makes it easy to
accidentally report the training objective as if it were an evaluation.

ESR in dB is the number to quote when comparing against the literature; MAE and
peak error are what catch a model that is fine on average and wrong on
transients.
"""

from __future__ import annotations

import math

import torch

__all__ = [
    "esr",
    "esr_db",
    "mae",
    "rmse",
    "peak_error",
    "dc_offset",
    "waveform_metrics",
    "assert_finite",
]


def esr(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> float:
    num = torch.sum((target - pred) ** 2)
    den = torch.sum(target**2) + eps
    return float(num / den)


def esr_db(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> float:
    value = esr(pred, target, eps)
    return 10.0 * math.log10(max(value, 1e-30))


def mae(pred: torch.Tensor, target: torch.Tensor) -> float:
    return float(torch.mean(torch.abs(pred - target)))


def rmse(pred: torch.Tensor, target: torch.Tensor) -> float:
    return float(torch.sqrt(torch.mean((pred - target) ** 2)))


def peak_error(pred: torch.Tensor, target: torch.Tensor) -> float:
    return float(torch.max(torch.abs(pred - target)))


def dc_offset(pred: torch.Tensor, target: torch.Tensor) -> float:
    return float(torch.mean(pred) - torch.mean(target))


def waveform_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    return {
        "esr": esr(pred, target),
        "esr_db": esr_db(pred, target),
        "mae": mae(pred, target),
        "rmse": rmse(pred, target),
        "peak_error": peak_error(pred, target),
        "dc_offset": dc_offset(pred, target),
    }


def assert_finite(tensor: torch.Tensor, what: str = "tensor") -> None:
    """Fail fast on NaN/Inf.

    Called at every training step.  A recurrent model that goes non-finite
    keeps producing plausible-looking loss numbers for a while afterwards
    because the state is poisoned, so detecting it late wastes a whole run.
    """
    if not torch.isfinite(tensor).all():
        n_nan = int(torch.isnan(tensor).sum())
        n_inf = int(torch.isinf(tensor).sum())
        raise FloatingPointError(f"{what} is not finite: {n_nan} NaN, {n_inf} Inf")
