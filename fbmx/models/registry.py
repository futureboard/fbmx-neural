"""Tiny model registry.

Keeps ``build_model`` out of every call site so a config string is all that is
needed to pick an architecture -- including one added later (S4, WaveNet-style
GCN, gray-box hybrid) without touching the trainer or the exporter.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping

from fbmx.conditioning import ConditioningSchema
from fbmx.models.base import StreamingModel

__all__ = ["register_model", "get_model_class", "build_model", "MODEL_REGISTRY"]

MODEL_REGISTRY: dict[str, type[StreamingModel]] = {}


def register_model(name: str) -> Callable[[type[StreamingModel]], type[StreamingModel]]:
    def decorate(cls: type[StreamingModel]) -> type[StreamingModel]:
        if name in MODEL_REGISTRY:
            raise ValueError(f"model {name!r} already registered")
        cls.model_type = name
        MODEL_REGISTRY[name] = cls
        return cls

    return decorate


def get_model_class(name: str) -> type[StreamingModel]:
    try:
        return MODEL_REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"unknown model type {name!r}; registered: {sorted(MODEL_REGISTRY)}"
        ) from None


def build_model(
    cfg: Mapping[str, Any], schema: ConditioningSchema | None = None
) -> StreamingModel:
    """Build from ``{"type": "lstm", "hidden_size": 32, ...}``."""
    cfg = dict(cfg)
    name = cfg.pop("type", None)
    if name is None:
        raise ValueError("model config needs a 'type' key")
    return get_model_class(name).from_config(cfg, schema)
