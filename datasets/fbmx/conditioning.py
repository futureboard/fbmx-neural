"""Generic parameter-conditioning API.

The FBMX model API knows nothing about any particular effect.  A model is
handed a :class:`ConditioningSchema` describing the knobs it must respond to,
and it builds whatever projections / embedding tables that implies.  The same
schema object is serialised into the checkpoint and into the ``.fbmx`` file, so
a runtime can discover the control surface without out-of-band knowledge.

Two kinds of parameter exist and they are *not* interchangeable:

``ContinuousParam``
    A real-valued knob with a range and a unit -- Input, Attack, Release,
    Drive, Mix.  Normalised to ``[-1, 1]`` before it reaches the network.

``CategoricalParam``
    A discrete state with no meaningful ordering or distance -- a Revision
    switch, or the "all buttons in" mode of an FET limiter.  These get a
    learned embedding.  Encoding them as a number on a continuous axis tells
    the network that "revision D is halfway between C and E", which is false,
    and it costs accuracy at exactly the settings people care about.

A ratio *switch* with positions 4:1 / 8:1 / 12:1 / 20:1 is genuinely
ambiguous: it is categorical hardware but has a monotone acoustic meaning.
Declare it whichever way the capture supports -- both are first-class here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import torch

__all__ = [
    "ContinuousParam",
    "CategoricalParam",
    "ConditioningSchema",
    "ParamBatch",
]


@dataclass(frozen=True)
class ContinuousParam:
    """A real-valued control, normalised to ``[-1, 1]`` for the network."""

    name: str
    minimum: float = 0.0
    maximum: float = 1.0
    default: float = 0.0
    unit: str = ""
    description: str = ""

    def __post_init__(self) -> None:
        if self.maximum <= self.minimum:
            raise ValueError(f"{self.name}: maximum must be > minimum")
        if not (self.minimum <= self.default <= self.maximum):
            raise ValueError(f"{self.name}: default {self.default} outside range")

    def normalize(self, value: float) -> float:
        span = self.maximum - self.minimum
        clamped = min(max(float(value), self.minimum), self.maximum)
        return 2.0 * (clamped - self.minimum) / span - 1.0

    def denormalize(self, value: float) -> float:
        span = self.maximum - self.minimum
        return self.minimum + (min(max(float(value), -1.0), 1.0) + 1.0) * 0.5 * span

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "continuous",
            "name": self.name,
            "minimum": self.minimum,
            "maximum": self.maximum,
            "default": self.default,
            "unit": self.unit,
            "description": self.description,
        }

    @staticmethod
    def from_dict(d: Mapping[str, Any]) -> "ContinuousParam":
        return ContinuousParam(
            name=d["name"],
            minimum=float(d.get("minimum", 0.0)),
            maximum=float(d.get("maximum", 1.0)),
            default=float(d.get("default", 0.0)),
            unit=d.get("unit", ""),
            description=d.get("description", ""),
        )


@dataclass(frozen=True)
class CategoricalParam:
    """A discrete control backed by a learned embedding table."""

    name: str
    categories: tuple[str, ...] = ()
    default: str = ""
    embedding_dim: int = 4
    description: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "categories", tuple(str(c) for c in self.categories))
        if not self.categories:
            raise ValueError(f"{self.name}: needs at least one category")
        if len(set(self.categories)) != len(self.categories):
            raise ValueError(f"{self.name}: duplicate categories")
        if not self.default:
            object.__setattr__(self, "default", self.categories[0])
        if self.default not in self.categories:
            raise ValueError(f"{self.name}: default {self.default!r} not in categories")
        if self.embedding_dim < 1:
            raise ValueError(f"{self.name}: embedding_dim must be >= 1")

    @property
    def num_categories(self) -> int:
        return len(self.categories)

    def index(self, value: Any) -> int:
        if isinstance(value, bool):
            value = int(value)
        if isinstance(value, int):
            if not 0 <= value < len(self.categories):
                raise ValueError(f"{self.name}: index {value} out of range")
            return value
        value = str(value)
        if value not in self.categories:
            raise ValueError(
                f"{self.name}: {value!r} not one of {list(self.categories)}"
            )
        return self.categories.index(value)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "categorical",
            "name": self.name,
            "categories": list(self.categories),
            "default": self.default,
            "embedding_dim": self.embedding_dim,
            "description": self.description,
        }

    @staticmethod
    def from_dict(d: Mapping[str, Any]) -> "CategoricalParam":
        return CategoricalParam(
            name=d["name"],
            categories=tuple(d["categories"]),
            default=d.get("default", ""),
            embedding_dim=int(d.get("embedding_dim", 4)),
            description=d.get("description", ""),
        )


@dataclass
class ParamBatch:
    """A batch of encoded parameter values.

    ``continuous`` is ``[B, n_continuous]``, already normalised to ``[-1, 1]``.
    ``categorical`` is ``[B, n_categorical]`` of int64 category indices.
    Either may have a zero-width second dimension; an unconditioned model has
    both empty, and every model must accept that case.
    """

    continuous: torch.Tensor
    categorical: torch.Tensor

    def __post_init__(self) -> None:
        if self.continuous.dim() != 2 or self.categorical.dim() != 2:
            raise ValueError("ParamBatch tensors must be 2-D [batch, n_params]")
        if self.continuous.shape[0] != self.categorical.shape[0]:
            raise ValueError("ParamBatch: batch size mismatch")
        if self.categorical.dtype != torch.long:
            self.categorical = self.categorical.long()

    @property
    def batch_size(self) -> int:
        return int(self.continuous.shape[0])

    @property
    def is_empty(self) -> bool:
        return self.continuous.shape[1] == 0 and self.categorical.shape[1] == 0

    def to(self, device: torch.device | str) -> "ParamBatch":
        return ParamBatch(self.continuous.to(device), self.categorical.to(device))

    def expand_to(self, batch_size: int) -> "ParamBatch":
        """Broadcast a single-row batch up to ``batch_size`` rows."""
        if self.batch_size == batch_size:
            return self
        if self.batch_size != 1:
            raise ValueError(
                f"cannot expand ParamBatch of {self.batch_size} rows to {batch_size}"
            )
        return ParamBatch(
            self.continuous.expand(batch_size, -1).contiguous(),
            self.categorical.expand(batch_size, -1).contiguous(),
        )

    @staticmethod
    def collate(items: Sequence["ParamBatch"]) -> "ParamBatch":
        return ParamBatch(
            torch.cat([i.continuous for i in items], dim=0),
            torch.cat([i.categorical for i in items], dim=0),
        )


@dataclass(frozen=True)
class ConditioningSchema:
    """An ordered description of a model's control surface."""

    continuous: tuple[ContinuousParam, ...] = field(default_factory=tuple)
    categorical: tuple[CategoricalParam, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        object.__setattr__(self, "continuous", tuple(self.continuous))
        object.__setattr__(self, "categorical", tuple(self.categorical))
        names = [p.name for p in self.continuous] + [p.name for p in self.categorical]
        if len(set(names)) != len(names):
            raise ValueError("ConditioningSchema: duplicate parameter names")

    # -- shape helpers -------------------------------------------------
    @property
    def n_continuous(self) -> int:
        return len(self.continuous)

    @property
    def n_categorical(self) -> int:
        return len(self.categorical)

    @property
    def embedding_dim(self) -> int:
        return sum(p.embedding_dim for p in self.categorical)

    @property
    def cond_dim(self) -> int:
        """Width of the vector a model sees after encoding + embedding."""
        return self.n_continuous + self.embedding_dim

    @property
    def is_empty(self) -> bool:
        return self.n_continuous == 0 and self.n_categorical == 0

    @property
    def names(self) -> list[str]:
        return [p.name for p in self.continuous] + [p.name for p in self.categorical]

    def get(self, name: str) -> ContinuousParam | CategoricalParam:
        for p in self.continuous:
            if p.name == name:
                return p
        for p in self.categorical:
            if p.name == name:
                return p
        raise KeyError(name)

    # -- encoding ------------------------------------------------------
    def encode(
        self,
        values: Mapping[str, Any] | None = None,
        *,
        device: torch.device | str = "cpu",
        strict: bool = True,
    ) -> ParamBatch:
        """Encode one setting of the controls into a single-row ParamBatch.

        Missing parameters fall back to their declared defaults.  ``strict``
        rejects unknown names, which catches config typos early instead of
        silently training a model that ignores a knob.
        """
        values = dict(values or {})
        if strict:
            unknown = set(values) - set(self.names)
            if unknown:
                raise ValueError(f"unknown parameters: {sorted(unknown)}")
        cont = [p.normalize(values.get(p.name, p.default)) for p in self.continuous]
        cat = [p.index(values.get(p.name, p.default)) for p in self.categorical]
        return ParamBatch(
            torch.tensor(cont, dtype=torch.float32, device=device).reshape(1, len(cont)),
            torch.tensor(cat, dtype=torch.long, device=device).reshape(1, len(cat)),
        )

    def empty_batch(
        self, batch_size: int = 1, device: torch.device | str = "cpu"
    ) -> ParamBatch:
        """Defaults for every parameter, repeated ``batch_size`` times."""
        return self.encode({}, device=device).expand_to(batch_size)

    def decode(self, batch: ParamBatch, row: int = 0) -> dict[str, Any]:
        """Inverse of :meth:`encode`, for logging and inspection."""
        out: dict[str, Any] = {}
        for i, p in enumerate(self.continuous):
            out[p.name] = p.denormalize(float(batch.continuous[row, i]))
        for i, p in enumerate(self.categorical):
            out[p.name] = p.categories[int(batch.categorical[row, i])]
        return out

    # -- serialisation -------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "continuous": [p.to_dict() for p in self.continuous],
            "categorical": [p.to_dict() for p in self.categorical],
        }

    @staticmethod
    def from_dict(d: Mapping[str, Any] | None) -> "ConditioningSchema":
        if not d:
            return ConditioningSchema()
        return ConditioningSchema(
            continuous=tuple(
                ContinuousParam.from_dict(x) for x in d.get("continuous", [])
            ),
            categorical=tuple(
                CategoricalParam.from_dict(x) for x in d.get("categorical", [])
            ),
        )

    @staticmethod
    def from_config(spec: Iterable[Mapping[str, Any]] | None) -> "ConditioningSchema":
        """Build from a flat YAML list where each entry carries a ``kind``."""
        cont: list[ContinuousParam] = []
        cat: list[CategoricalParam] = []
        for entry in spec or []:
            kind = entry.get("kind", "continuous")
            if kind == "continuous":
                cont.append(ContinuousParam.from_dict(entry))
            elif kind == "categorical":
                cat.append(CategoricalParam.from_dict(entry))
            else:
                raise ValueError(f"unknown parameter kind {kind!r}")
        return ConditioningSchema(tuple(cont), tuple(cat))

    def __str__(self) -> str:  # pragma: no cover - display only
        if self.is_empty:
            return "ConditioningSchema(unconditioned)"
        parts = [f"{p.name}[{p.minimum},{p.maximum}]{p.unit}" for p in self.continuous]
        parts += [p.name + "{" + "|".join(p.categories) + "}" for p in self.categorical]
        return "ConditioningSchema(" + ", ".join(parts) + f" -> {self.cond_dim}d)"
