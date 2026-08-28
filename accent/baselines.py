"""What the Accent Analyzer has to beat, and the rule analyser it ships with.

Four baselines, in increasing order of how much they know:

``uniform``
    Predict the training mean for every note. A model that cannot beat this has
    learned nothing at all. It is a stronger baseline than it looks, because
    the targets are locally normalised and therefore centred by construction.

``velocity``
    A least-squares fit from score velocity alone. This is the baseline that
    matters most for the brief's central prohibition: if the neural analyser
    does not clearly beat "accent is velocity", then whatever it has learned is
    not worth the model. The analyser never sees velocity, so this is a genuine
    outside comparison rather than an ablation of itself.

``meter``
    A least-squares fit from metrical strength alone — "strong beat = accent",
    the second thing the brief forbids implementing.

``rule``
    A ridge regression over nine interpretable score features. This is the
    deterministic analyser the DAW falls back to when no model is loaded, so it
    has to be a real answer and not a straw man.

The rule's coefficients are **fitted, not chosen**. Section 16 asks for a
deterministic baseline whose constants are documented rather than arbitrary,
and the most defensible way to document a constant is to say which data
produced it. They are refitted whenever the analyser is retrained, written into
the training report, and copied into the Rust fallback by
`scripts/export_accent.py --rule`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from .features import ACCENT_INPUT_FEATURES, ACCENT_TARGETS, AccentSequence

#: Score features the rule analyser is allowed to use, by name.
#:
#: Nine, chosen for interpretability rather than accuracy: every one of them is
#: something a musician would name out loud as a reason a note is emphasised,
#: so the fitted coefficient on each is a claim that can be argued with. The
#: neural analyser gets all thirty-three.
RULE_FEATURES: tuple[str, ...] = (
    "metrical_strength",
    "syncopation",
    "duration_vs_local",
    "starts_after_rest",
    "is_window_peak",
    "pitch_vs_local",
    "leap_into",
    "is_phrase_start",
    "long_after_short",
)

#: Ridge penalty for the rule fit. Small: with nine features and ~5000 notes
#: there is no collinearity crisis to regularise away, and this is here to keep
#: a coefficient from running away on a feature that fires 3% of the time.
RULE_RIDGE = 1.0


@dataclass
class RuleAnalyzer:
    """A linear accent rule over interpretable score features."""

    #: `target -> {feature: coefficient}`.
    coefficients: dict[str, dict[str, float]] = field(default_factory=dict)
    intercepts: dict[str, float] = field(default_factory=dict)
    #: What the rule states about its own certainty. It has no uncertainty
    #: estimate — a least-squares fit reports one number per note and no spread
    #: — so it declares a fixed middling confidence rather than claiming the
    #: neural model's calibrated one. Stated here rather than buried at a call
    #: site so nothing can mistake it for a measurement.
    confidence: float = 0.5

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        """`(notes, features)` in feature order, `(notes, targets)` out."""

        columns = [ACCENT_INPUT_FEATURES.index(name) for name in RULE_FEATURES]
        design = inputs[:, columns]
        out = np.empty((inputs.shape[0], len(ACCENT_TARGETS)), dtype=np.float32)
        for index, target in enumerate(ACCENT_TARGETS):
            weights = np.asarray(
                [self.coefficients[target][name] for name in RULE_FEATURES], dtype=np.float64
            )
            out[:, index] = np.clip(design @ weights + self.intercepts[target], 0.0, 1.0)
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "features": list(RULE_FEATURES),
            "ridge": RULE_RIDGE,
            "confidence": self.confidence,
            "coefficients": {
                target: {name: round(value, 6) for name, value in weights.items()}
                for target, weights in self.coefficients.items()
            },
            "intercepts": {k: round(v, 6) for k, v in self.intercepts.items()},
        }

    @staticmethod
    def from_dict(payload: dict[str, Any]) -> "RuleAnalyzer":
        return RuleAnalyzer(
            coefficients={
                target: {k: float(v) for k, v in weights.items()}
                for target, weights in payload["coefficients"].items()
            },
            intercepts={k: float(v) for k, v in payload["intercepts"].items()},
            confidence=float(payload.get("confidence", 0.5)),
        )


def _ridge(design: np.ndarray, target: np.ndarray, penalty: float) -> tuple[np.ndarray, float]:
    """Ridge regression with an unpenalised intercept."""

    mean_x = design.mean(axis=0)
    mean_y = float(target.mean())
    centred = design - mean_x
    gram = centred.T @ centred + penalty * np.eye(design.shape[1])
    weights = np.linalg.solve(gram, centred.T @ (target - mean_y))
    return weights, float(mean_y - mean_x @ weights)


def fit_rule(sequences: Sequence[AccentSequence]) -> RuleAnalyzer:
    """Fit the rule analyser on a split (in practice, training only)."""

    inputs = np.concatenate([sequence.inputs for sequence in sequences], axis=0).astype(np.float64)
    targets = np.concatenate([sequence.targets for sequence in sequences], axis=0).astype(
        np.float64
    )
    mask = np.concatenate([sequence.mask for sequence in sequences], axis=0)
    columns = [ACCENT_INPUT_FEATURES.index(name) for name in RULE_FEATURES]
    design = inputs[:, columns]

    coefficients: dict[str, dict[str, float]] = {}
    intercepts: dict[str, float] = {}
    for index, target in enumerate(ACCENT_TARGETS):
        keep = mask[:, index] > 0
        weights, intercept = _ridge(design[keep], targets[keep, index], RULE_RIDGE)
        coefficients[target] = {
            name: float(weights[position]) for position, name in enumerate(RULE_FEATURES)
        }
        intercepts[target] = intercept
    return RuleAnalyzer(coefficients=coefficients, intercepts=intercepts)


@dataclass
class LinearBaseline:
    """One-feature least-squares predictor, per target."""

    name: str
    feature: str | None
    slopes: dict[str, float] = field(default_factory=dict)
    intercepts: dict[str, float] = field(default_factory=dict)

    def predict(self, inputs: np.ndarray, extra: np.ndarray | None = None) -> np.ndarray:
        if self.feature is None:
            column = np.zeros(inputs.shape[0]) if extra is None else extra
        elif self.feature == "__extra__":
            column = extra if extra is not None else np.zeros(inputs.shape[0])
        else:
            column = inputs[:, ACCENT_INPUT_FEATURES.index(self.feature)]
        out = np.empty((inputs.shape[0], len(ACCENT_TARGETS)), dtype=np.float32)
        for index, target in enumerate(ACCENT_TARGETS):
            out[:, index] = np.clip(
                self.slopes[target] * column + self.intercepts[target], 0.0, 1.0
            )
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "feature": self.feature,
            "slopes": {k: round(v, 6) for k, v in self.slopes.items()},
            "intercepts": {k: round(v, 6) for k, v in self.intercepts.items()},
        }


def attach_rule_base(rule: RuleAnalyzer, sequences: Sequence[AccentSequence]) -> None:
    """Fill each sequence's `base` with the rule's prediction, in place.

    Must be called with a rule fitted on *training* sequences only, including
    when filling the validation and test sequences: a rule refitted per split
    would give the network a different floor on each one and make the
    neural-versus-rule comparison meaningless.
    """

    for sequence in sequences:
        sequence.base = rule.predict(sequence.inputs).astype(np.float32)


def fit_linear_baseline(
    sequences: Sequence[AccentSequence],
    *,
    name: str,
    feature: str | None,
    extra: np.ndarray | None = None,
) -> LinearBaseline:
    """Least squares from one column (or from a constant, for `uniform`)."""

    inputs = np.concatenate([sequence.inputs for sequence in sequences], axis=0).astype(np.float64)
    targets = np.concatenate([sequence.targets for sequence in sequences], axis=0).astype(
        np.float64
    )
    mask = np.concatenate([sequence.mask for sequence in sequences], axis=0)

    if feature is None:
        column = np.zeros(inputs.shape[0])
    elif feature == "__extra__":
        column = np.asarray(extra, dtype=np.float64)
    else:
        column = inputs[:, ACCENT_INPUT_FEATURES.index(feature)]

    slopes: dict[str, float] = {}
    intercepts: dict[str, float] = {}
    for index, target in enumerate(ACCENT_TARGETS):
        keep = mask[:, index] > 0
        x = column[keep]
        y = targets[keep, index]
        if feature is None or x.std() < 1e-9:
            slopes[target] = 0.0
            intercepts[target] = float(y.mean())
            continue
        design = np.column_stack([x, np.ones_like(x)])
        solution, *_ = np.linalg.lstsq(design, y, rcond=None)
        slopes[target] = float(solution[0])
        intercepts[target] = float(solution[1])
    return LinearBaseline(name=name, feature=feature, slopes=slopes, intercepts=intercepts)


def score_velocity_column(sequences: Sequence[AccentSequence]) -> np.ndarray:
    """Score velocity per note, normalised to ``0..1``.

    Read straight off the records rather than from the feature matrix, because
    velocity is deliberately absent from the feature matrix. Building the
    velocity baseline is the only place in this package that touches it.
    """

    values: list[float] = []
    for sequence in sequences:
        for record in sequence.records:
            values.append(float(record.get("score_velocity") or 64) / 127.0)
    return np.asarray(values, dtype=np.float64)
