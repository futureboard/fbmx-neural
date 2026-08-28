"""Give the analyser's output the range of the thing it is estimating.

A minimum-error predictor of a noisy target is always **under-dispersed**. It
has to be: where the evidence is weak the safest guess is the mean, so the
predictions bunch. Measured here, the fitted rule's prominence has a standard
deviation of about 0.05 against a target standard deviation of 0.21 — it is
right about which notes are the prominent ones and says so in a fifth of the
range available.

That is fine for a number in a report and useless for a control. An Accent lane
whose bars only ever move between 0.45 and 0.55 cannot express what it is
measuring, cannot be read at a glance, and cannot drive an audible difference
downstream without a hidden gain somewhere else — which is the worse option,
because a hidden gain is a fudge nobody can see.

So the spread is matched explicitly, as one documented affine transform per
head:

    calibrated = mean_target + gain * (raw - mean_raw)
    gain       = std_target / std_raw

This is **monotone**, so it changes no ordering: every rank correlation is
identical before and after, and "which note of this phrase is the most
prominent" — the question section 47 says matters most — has exactly the same
answer. What it does change is MAE, which gets worse, and both numbers are
reported. That trade is the honest one to make here: the ranking is the signal,
the absolute scalar is the presentation, and a presentation that cannot show
the signal is not worth a better MAE.

The gain is never allowed below 1.0 (shrinking a prediction only makes it less
usable) and never above `MAX_GAIN` (past which it is amplifying the model's
noise rather than its signal, and the right answer is a better model).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from .features import ACCENT_TARGETS

#: Ceiling on the spread gain. Four is roughly the point at which a prediction
#: correlating 0.3 with its target would be stretched to the target's full
#: range, which is as far as this transform can go while still being a
#: presentation choice rather than an invention.
MAX_GAIN = 4.0


@dataclass
class SpreadCalibration:
    """One affine transform per head, fitted on the training pool."""

    gains: dict[str, float] = field(default_factory=dict)
    raw_means: dict[str, float] = field(default_factory=dict)
    target_means: dict[str, float] = field(default_factory=dict)

    def apply(self, predictions: np.ndarray) -> np.ndarray:
        """`(notes, targets)` in, the same shape out, clipped to ``0..1``."""

        out = np.empty_like(predictions)
        for index, name in enumerate(ACCENT_TARGETS):
            gain = self.gains.get(name, 1.0)
            raw_mean = self.raw_means.get(name, 0.5)
            target_mean = self.target_means.get(name, 0.5)
            out[:, index] = np.clip(
                target_mean + gain * (predictions[:, index] - raw_mean), 0.0, 1.0
            )
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "gains": {k: round(v, 6) for k, v in self.gains.items()},
            "raw_means": {k: round(v, 6) for k, v in self.raw_means.items()},
            "target_means": {k: round(v, 6) for k, v in self.target_means.items()},
            "max_gain": MAX_GAIN,
            "note": (
                "monotone affine spread match; preserves every ranking, "
                "worsens MAE, fitted on the training pool"
            ),
        }

    @staticmethod
    def from_dict(payload: dict[str, Any]) -> "SpreadCalibration":
        return SpreadCalibration(
            gains={k: float(v) for k, v in payload.get("gains", {}).items()},
            raw_means={k: float(v) for k, v in payload.get("raw_means", {}).items()},
            target_means={k: float(v) for k, v in payload.get("target_means", {}).items()},
        )


def fit_calibration(
    predictions: np.ndarray, targets: np.ndarray, mask: np.ndarray
) -> SpreadCalibration:
    """Match each head's predicted spread to its target's, on training data."""

    gains: dict[str, float] = {}
    raw_means: dict[str, float] = {}
    target_means: dict[str, float] = {}
    for index, name in enumerate(ACCENT_TARGETS):
        keep = mask[:, index] > 0
        if keep.sum() < 8:
            gains[name], raw_means[name], target_means[name] = 1.0, 0.5, 0.5
            continue
        raw = predictions[keep, index].astype(np.float64)
        actual = targets[keep, index].astype(np.float64)
        raw_std = float(raw.std())
        gains[name] = (
            float(np.clip(actual.std() / raw_std, 1.0, MAX_GAIN)) if raw_std > 1e-6 else 1.0
        )
        raw_means[name] = float(raw.mean())
        target_means[name] = float(actual.mean())
    return SpreadCalibration(gains=gains, raw_means=raw_means, target_means=target_means)


def stack(sequences: Sequence) -> tuple[np.ndarray, np.ndarray]:
    """`(targets, mask)` concatenated over sequences."""

    return (
        np.concatenate([sequence.targets for sequence in sequences], axis=0),
        np.concatenate([sequence.mask for sequence in sequences], axis=0),
    )
