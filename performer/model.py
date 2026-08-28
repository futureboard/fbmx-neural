"""The FBMX Performer: a score in, a way of playing it out.

This model does not make audio and has no path to making audio. It reads a
sequence of written notes and predicts, for each one, how a violinist would
place it, shape it, tune it, and colour it. The engine already knows what a
violin sounds like; this is the part that knows what a player does.

Two variants over the same weights-shaped body:

``studio``
    Bidirectional. Sees the whole phrase before deciding anything, which is what
    lets it shape a ritardando into a phrase ending or start a crescendo before
    the note that needs it. This is the one a DAW uses, because a DAW already
    has the notes.

``live``
    Causal. Same feature vector, same heads, but only past and present. Kept
    small and separate so a future MIDI-keyboard path has somewhere to go
    without redesigning the outputs.

The size is deliberate. A note sequence is a few hundred steps long, not a few
hundred thousand samples, so the whole phrase costs less than a millisecond and
there is no reason to reach for a Transformer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn

from .features import INPUT_FEATURES, OUTPUT_SIZE, TARGETS

MODEL_SCHEMA_VERSION = 1


@dataclass
class PerformerConfig:
    input_size: int = len(INPUT_FEATURES)
    hidden_size: int = 64
    num_layers: int = 1
    output_size: int = OUTPUT_SIZE
    #: ``studio`` reads the phrase in both directions; ``live`` only forwards.
    mode: str = "studio"
    dropout: float = 0.0

    @property
    def bidirectional(self) -> bool:
        return self.mode == "studio"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": MODEL_SCHEMA_VERSION,
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "output_size": self.output_size,
            "mode": self.mode,
            "bidirectional": self.bidirectional,
            "dropout": self.dropout,
            "input_features": list(INPUT_FEATURES),
            "targets": list(TARGETS),
        }

    @staticmethod
    def from_dict(payload: dict[str, Any]) -> "PerformerConfig":
        return PerformerConfig(
            input_size=int(payload["input_size"]),
            hidden_size=int(payload["hidden_size"]),
            num_layers=int(payload.get("num_layers", 1)),
            output_size=int(payload["output_size"]),
            mode=str(payload.get("mode", "studio")),
            dropout=float(payload.get("dropout", 0.0)),
        )


class Performer(nn.Module):
    """GRU over a note sequence with one linear head per performance dimension.

    A GRU rather than an LSTM: with a few thousand training notes the extra gate
    is parameters spent on capacity the data cannot fill, and the exported
    weight layout stays simpler for the runtime to read.

    One shared trunk and separate heads, rather than one head emitting all nine
    numbers, because the dimensions have to stay independently regenerable —
    a "retake the vibrato, keep the timing" request has to be expressible, and
    that is much harder if every output shares a final projection.
    """

    def __init__(self, config: PerformerConfig | None = None) -> None:
        super().__init__()
        self.config = config or PerformerConfig()
        self.rnn = nn.GRU(
            input_size=self.config.input_size,
            hidden_size=self.config.hidden_size,
            num_layers=self.config.num_layers,
            batch_first=True,
            bidirectional=self.config.bidirectional,
            dropout=self.config.dropout if self.config.num_layers > 1 else 0.0,
        )
        trunk = self.config.hidden_size * (2 if self.config.bidirectional else 1)
        self.heads = nn.ModuleList(
            [nn.Linear(trunk, 1) for _ in range(self.config.output_size)]
        )
        # Normalisation is carried inside the module so a checkpoint is
        # self-sufficient: loading it cannot silently pick up statistics from a
        # different dataset.
        self.register_buffer("input_mean", torch.zeros(self.config.input_size))
        self.register_buffer("input_std", torch.ones(self.config.input_size))

    def set_normalization(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        self.input_mean.copy_(mean.to(self.input_mean.dtype))
        self.input_std.copy_(std.to(self.input_std.dtype))

    def forward(self, notes: torch.Tensor) -> torch.Tensor:
        """``(batch, notes, features)`` in, ``(batch, notes, outputs)`` out."""

        normalized = (notes - self.input_mean) / self.input_std
        encoded, _ = self.rnn(normalized)
        return torch.cat([head(encoded) for head in self.heads], dim=-1)

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


@dataclass
class BaselineStats:
    """What the dataset's own averages predict, per target.

    Section 27's requirement: before believing a network, check it beats the
    obvious answer. For most of these targets the obvious answer is "do what a
    violinist does on average", and a model that cannot beat that has learned
    nothing about the specific note it was given.
    """

    mean: list[float] = field(default_factory=list)
    vibrato_rate: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {"mean": self.mean, "vibrato_rate": self.vibrato_rate}
