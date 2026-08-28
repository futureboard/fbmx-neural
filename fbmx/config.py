"""YAML experiment configuration.

A config is a plain nested dict with four top-level blocks -- ``model``,
``data``, ``loss``, ``train`` -- plus optional ``run`` metadata.  There is no
schema class and no framework: the blocks are handed to the registries
(:func:`fbmx.models.build_model`, :func:`fbmx.datasets.build_dataset`,
:func:`fbmx.losses.build_loss`) which each validate their own keys.

``extends:`` inherits from another config file, so ``lstm32.yaml`` can be the
smoke config with the knobs that matter changed and nothing else duplicated.

The whole resolved dict is snapshotted into every checkpoint, so a run can be
reproduced from its artifacts even if the file on disk has moved on.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml

__all__ = [
    "load_config",
    "deep_merge",
    "apply_overrides",
    "build_experiment",
    "config_digest",
]


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Recursive dict merge; lists and scalars are replaced, not concatenated."""
    out = copy.deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), Mapping):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_config(path: str | Path, _seen: set[Path] | None = None) -> dict[str, Any]:
    path = Path(path).resolve()
    _seen = _seen or set()
    if path in _seen:
        raise ValueError(f"circular 'extends' chain at {path}")
    _seen.add(path)

    cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"{path}: config must be a mapping at the top level")

    parent_ref = cfg.pop("extends", None)
    if parent_ref:
        parent = load_config((path.parent / parent_ref).resolve(), _seen)
        cfg = deep_merge(parent, cfg)
    cfg.setdefault("run", {}).setdefault("config_path", str(path))
    return cfg


def _coerce(text: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def apply_overrides(cfg: Mapping[str, Any], overrides: Iterable[str] | None) -> dict[str, Any]:
    """Apply ``--set train.lr=1e-4`` style overrides.

    Values are parsed as JSON when possible, so ``true``, ``null``, ``3e-4``
    and ``[1,2]`` all do the obvious thing, and anything else stays a string.
    """
    out = copy.deepcopy(dict(cfg))
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"override {item!r} is not of the form key.path=value")
        key, _, raw = item.partition("=")
        node: Any = out
        parts = key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = _coerce(raw)
    return out


def config_digest(cfg: Mapping[str, Any]) -> str:
    """Stable short hash of a config, for run naming."""
    import hashlib

    blob = json.dumps(cfg, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:12]


def build_experiment(cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Instantiate datasets, model, loss and trainer config from a config dict.

    Returns the pieces rather than a Trainer so that ``validate.py`` and
    ``export_fbmx.py`` can reuse the same construction path without starting a
    training run.
    """
    from fbmx.datasets import build_dataset
    from fbmx.losses import build_loss
    from fbmx.models import build_model
    from fbmx.training.trainer import TrainerConfig

    data_cfg = dict(cfg.get("data", {}))
    train_split = data_cfg.pop("train_split", "train")
    val_split = data_cfg.pop("val_split", "val")

    # ``None`` is an intentional split selector for datasets that should use
    # every manifest entry.  ``build_dataset`` cannot distinguish an omitted
    # override from an explicit ``None`` because its own default is ``train``,
    # so carry the explicit value through the dataset config.
    if train_split is None:
        train_data_cfg = dict(data_cfg)
        train_data_cfg["split"] = None
        train_dataset = build_dataset(train_data_cfg, split=None)
    else:
        train_dataset = build_dataset(data_cfg, split=train_split)
    val_dataset = build_dataset(data_cfg, split=val_split) if val_split else None

    trainer_cfg = TrainerConfig.from_dict(cfg.get("train", {}))

    # The dataset owns the conditioning schema: the model must respond to the
    # controls the data actually varies, not to a list retyped in the config.
    model = build_model(cfg.get("model", {"type": "lstm"}), train_dataset.schema)

    loss_spec = cfg.get("loss")
    if loss_spec:
        loss_spec = [
            dict(entry, max_length=entry.get("max_length", trainer_cfg.chunk_size))
            if entry.get("name") == "mrstft"
            else entry
            for entry in loss_spec
        ]
    loss_fn = build_loss(loss_spec)

    return {
        "model": model,
        "train_dataset": train_dataset,
        "val_dataset": val_dataset,
        "loss_fn": loss_fn,
        "trainer_cfg": trainer_cfg,
        "config": dict(cfg),
    }
