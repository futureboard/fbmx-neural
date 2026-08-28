"""The Accent Analyzer: a score in, a reading of which notes matter out.

This model makes no audio, no curves, and no timing. It answers one question
per note — how strongly should this be emphasised, and in which of the four
ways — and everything downstream is free to disagree with it, because its
output lands in the project as ordinary editable data.

Bidirectional by construction. Whether a note is a phrase's peak, whether it is
about to be followed by a rest, whether the long note it leans into is still to
come — none of that is knowable causally, and a causal accent analyser would be
answering a different and much harder question than the one a DAW asks. The
Performer keeps a `live` variant for a future keyboard path; this does not,
because "analyse the clip I already have" is the only use it has.

## Why this is a separate model from the Performer

The brief asks for the two to share a phrase encoder if that can be done
cleanly. It cannot, and the reason is not efficiency but *editability*.

`AccentState` has to be real project data a user can drag. That means the
Performer must **read** accent as an input — if it did not, editing a note's
accent would change a number in the project and change nothing about the sound,
which is the definition of a disconnected control. A shared trunk with an
accent head and performance heads gives the opposite: accent and performance
would both be read off the same hidden state, the accent value would be a
*report* of what the trunk decided rather than a cause of it, and a user
lowering it would be overruled silently on the next regeneration.

So the pipeline is Accent → Performer, two encoders, and the Performer's input
grows by the four accent components. The cost of the second encoder is one more
GRU pass over a few hundred notes — tens of microseconds — against which the
alternative is a control that does not control anything.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from .features import (
    ACCENT_FEATURE_SCHEMA_VERSION,
    ACCENT_INPUT_FEATURES,
    ACCENT_OUTPUT_SIZE,
    ACCENT_TARGETS,
    CONFIDENCE_INDEX,
)

ACCENT_MODEL_SCHEMA_VERSION = 1

#: Clamp on the log-variance head, in the natural log of the target's own
#: units. `exp(-6)` is a standard deviation of 0.05 on a 0..1 scale and
#: `exp(2)` is 2.7; outside that range the Gaussian term either divides by
#: nothing or stops caring about the residual at all.
LOG_VARIANCE_RANGE = (-6.0, 2.0)


@dataclass
class AccentConfig:
    input_size: int = len(ACCENT_INPUT_FEATURES)
    hidden_size: int = 16
    num_layers: int = 1
    output_size: int = ACCENT_OUTPUT_SIZE
    dropout: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": ACCENT_MODEL_SCHEMA_VERSION,
            "feature_schema_version": ACCENT_FEATURE_SCHEMA_VERSION,
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "output_size": self.output_size,
            "bidirectional": True,
            "dropout": self.dropout,
            "input_features": list(ACCENT_INPUT_FEATURES),
            "targets": list(ACCENT_TARGETS),
        }

    @staticmethod
    def from_dict(payload: dict[str, Any]) -> "AccentConfig":
        return AccentConfig(
            input_size=int(payload["input_size"]),
            hidden_size=int(payload["hidden_size"]),
            num_layers=int(payload.get("num_layers", 1)),
            output_size=int(payload["output_size"]),
            dropout=float(payload.get("dropout", 0.0)),
        )


class AccentAnalyzer(nn.Module):
    """BiGRU over a note sequence, one linear head per accent component.

    The same shape as the Performer, deliberately: it reuses the Performer's
    exported tensor layout and therefore the Rust GRU that already reads it, so
    a second sequence runtime does not have to exist. What differs is the
    feature vocabulary, the targets, and the extra head.

    Separate heads rather than one four-wide projection for the same reason the
    Performer has them: the components must stay independently meaningful, and
    a user who edits only `agogic` is entitled to have only timing change.
    """

    def __init__(self, config: AccentConfig | None = None) -> None:
        super().__init__()
        self.config = config or AccentConfig()
        self.rnn = nn.GRU(
            input_size=self.config.input_size,
            hidden_size=self.config.hidden_size,
            num_layers=self.config.num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=self.config.dropout if self.config.num_layers > 1 else 0.0,
        )
        trunk = self.config.hidden_size * 2
        self.heads = nn.ModuleList(
            [nn.Linear(trunk, 1) for _ in range(self.config.output_size)]
        )
        # Start at *no correction*. The network's output is added to the fitted
        # rule's answer, so a zero-initialised head means the untrained model is
        # exactly the rule — which is the floor the whole residual formulation
        # exists to guarantee. Initialising these at 0.5, as an absolute
        # predictor would want, would start training half a range away from a
        # perfectly good answer.
        with torch.no_grad():
            for index in range(len(ACCENT_TARGETS)):
                self.heads[index].weight.zero_()
                self.heads[index].bias.zero_()
            self.heads[CONFIDENCE_INDEX].bias.fill_(-2.0)

        self.register_buffer("input_mean", torch.zeros(self.config.input_size))
        self.register_buffer("input_std", torch.ones(self.config.input_size))

    def set_normalization(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        self.input_mean.copy_(mean.to(self.input_mean.dtype))
        self.input_std.copy_(std.to(self.input_std.dtype))

    def forward(self, notes: torch.Tensor) -> torch.Tensor:
        """``(batch, notes, features)`` in, ``(batch, notes, outputs)`` out.

        Outputs are raw: the four accent components are *not* squashed here.
        A sigmoid would make the head's gradient vanish exactly where the
        interesting notes are — the strongly and weakly accented ones at the
        ends of the range — and the targets are already bounded, so clamping at
        decode time costs nothing and keeps the loss well conditioned.
        """

        normalized = (notes - self.input_mean) / self.input_std
        encoded, _ = self.rnn(normalized)
        return torch.cat([head(encoded) for head in self.heads], dim=-1)

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def export_spec(self) -> dict[str, Any]:
        """What the ``.fbmx`` writer needs about this architecture."""

        return {
            "model_type": "accent-gru",
            "sample_rate": 0,
            "channels": 0,
            "causal": False,
            "recurrent": True,
            "receptive_field": 0,
            "parameter_count": self.parameter_count(),
            "hparams": {
                "hidden_size": self.config.hidden_size,
                "num_layers": self.config.num_layers,
                "bidirectional": True,
                "mode": "studio",
                "input_size": self.config.input_size,
                "output_size": self.config.output_size,
                "rnn": "gru",
            },
            "input_features": {
                "layout": "BNF",
                "order": list(ACCENT_INPUT_FEATURES),
                "size": self.config.input_size,
                "normalization": "per-feature mean/std stored as input_mean and input_std",
            },
            "output_spec": {
                "layout": "BNO",
                "regression": list(ACCENT_TARGETS),
                "scales": {name: 1.0 for name in ACCENT_TARGETS},
                "classification": [],
                "uncertainty": ["prominence_log_variance"],
                "size": self.config.output_size,
            },
            "conditioning": {"continuous": [], "categorical": []},
            "output_heads": ["accent"],
        }


def confidence_from_log_variance(log_variance: torch.Tensor) -> torch.Tensor:
    """Turn the uncertainty head into the ``0..1`` confidence the UI shows.

    The head predicts the log variance of `prominence`. A standard deviation of
    zero would be certainty and one of 0.25 — a quarter of the whole range — is
    no useful opinion at all, so the mapping is linear in standard deviation
    between those, clamped.

    This is the only honest thing `confidence` can be. It is not a measurement
    of how confident the *player* was, which nothing observed; it is how much
    the analyser's own training data agreed about notes like this one.
    """

    clamped = log_variance.clamp(*LOG_VARIANCE_RANGE)
    deviation = torch.exp(0.5 * clamped)
    return (1.0 - deviation / 0.25).clamp(0.0, 1.0)
