"""Checkpointing: survive a disconnected Colab runtime.

A checkpoint here is a *complete* description of the run at a point in time --
weights, optimiser and scheduler state, epoch and step counters, the best
metric so far, every RNG stream, and a snapshot of the config that produced it.
Resuming reconstructs the model from the checkpoint itself, never from the
config file, so a config edited after the fact cannot silently change the
architecture underneath a resume.

These are ``torch.save`` files: development artifacts, tied to this PyTorch
version and this repository.  They are *not* the distribution format -- that is
``.fbmx``, which is self-describing, independently readable, and contains no
executable object graph.

RNG state is stored as plain ints/lists so that the file loads under
``torch.load(weights_only=True)``.
"""

from __future__ import annotations

import copy
import platform
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from fbmx import FBMX_FORMAT_VERSION, __version__
from fbmx.conditioning import ConditioningSchema
from fbmx.models import build_model
from fbmx.models.base import StreamingModel

__all__ = [
    "CHECKPOINT_FORMAT",
    "capture_rng_state",
    "restore_rng_state",
    "set_seed",
    "save_checkpoint",
    "load_checkpoint",
    "model_from_checkpoint",
    "CheckpointManager",
]

CHECKPOINT_FORMAT = 1


# ---------------------------------------------------------------------------
# randomness
# ---------------------------------------------------------------------------
def set_seed(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        # cuDNN's autotuner picks different algorithms run to run; pinning it
        # costs throughput and is only worth it when chasing a discrepancy.
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def capture_rng_state() -> dict[str, Any]:
    py_state = random.getstate()
    np_state = np.random.get_state()
    return {
        "python": [py_state[0], list(py_state[1]), py_state[2]],
        "numpy": [np_state[0], [int(v) for v in np_state[1]], int(np_state[2]),
                  int(np_state[3]), float(np_state[4])],
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng_state(state: Mapping[str, Any] | None) -> None:
    if not state:
        return
    py = state.get("python")
    if py:
        random.setstate((py[0], tuple(int(v) for v in py[1]), py[2]))
    npy = state.get("numpy")
    if npy:
        np.random.set_state(
            (npy[0], np.array(npy[1], dtype=np.uint32), int(npy[2]), int(npy[3]), float(npy[4]))
        )
    if state.get("torch") is not None:
        torch.set_rng_state(state["torch"].to(torch.uint8).cpu())
    cuda = state.get("cuda")
    if cuda is not None and len(cuda) and torch.cuda.is_available():
        try:
            torch.cuda.set_rng_state_all([s.to(torch.uint8).cpu() for s in cuda])
        except Exception:  # different GPU count on resume; not fatal
            pass


# ---------------------------------------------------------------------------
# save / load
# ---------------------------------------------------------------------------
def save_checkpoint(
    path: str | Path,
    model: StreamingModel,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    epoch: int = 0,
    global_step: int = 0,
    best_metric: float | None = None,
    monitor: str = "val_loss",
    config: Mapping[str, Any] | None = None,
    metrics: Mapping[str, float] | None = None,
    history: list[dict[str, Any]] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model_config = dict(model.hparams())
    model_config["type"] = model.model_type
    payload: dict[str, Any] = {
        "checkpoint_format": CHECKPOINT_FORMAT,
        "fbmx_version": __version__,
        "fbmx_container_version": FBMX_FORMAT_VERSION,
        # str(): torch.__version__ is a TorchVersion object, and storing one
        # would force weights_only=False on every load
        "torch_version": str(torch.__version__),
        "python": platform.python_version(),
        "model_type": model.model_type,
        "model_config": model_config,
        "conditioning": model.schema.to_dict(),
        "export_spec": model.export_spec(),
        "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "epoch": int(epoch),
        "global_step": int(global_step),
        "best_metric": best_metric,
        "monitor": monitor,
        "metrics": dict(metrics or {}),
        "history": list(history or []),
        "config": copy.deepcopy(dict(config or {})),
        "rng": capture_rng_state(),
        "extra": dict(extra or {}),
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)  # atomic-ish: a killed runtime never leaves a torn file
    return path


def load_checkpoint(
    path: str | Path, map_location: torch.device | str = "cpu"
) -> dict[str, Any]:
    ckpt = torch.load(str(path), map_location=map_location, weights_only=True)
    version = ckpt.get("checkpoint_format")
    if version != CHECKPOINT_FORMAT:
        raise ValueError(
            f"{path}: checkpoint format {version} but this build reads {CHECKPOINT_FORMAT}"
        )
    return ckpt


def model_from_checkpoint(
    ckpt: Mapping[str, Any], device: torch.device | str = "cpu"
) -> StreamingModel:
    """Rebuild the exact architecture the checkpoint was written from."""
    schema = ConditioningSchema.from_dict(ckpt.get("conditioning"))
    model = build_model(ckpt["model_config"], schema)
    model.load_state_dict(ckpt["model_state"])
    return model.to(device)


class CheckpointManager:
    """Writes ``last.pt`` every save and ``best.pt`` when the monitor improves."""

    def __init__(
        self,
        directory: str | Path,
        monitor: str = "val_loss",
        mode: str = "min",
        keep_epochs: bool = False,
    ) -> None:
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.monitor = monitor
        if mode not in ("min", "max"):
            raise ValueError("mode must be 'min' or 'max'")
        self.mode = mode
        self.keep_epochs = keep_epochs
        self.best_metric: float | None = None

    @property
    def last_path(self) -> Path:
        return self.dir / "last.pt"

    @property
    def best_path(self) -> Path:
        return self.dir / "best.pt"

    def is_improvement(self, value: float) -> bool:
        if value is None:
            return False
        if self.best_metric is None:
            return True
        return value < self.best_metric if self.mode == "min" else value > self.best_metric

    def save(self, model: StreamingModel, metrics: Mapping[str, float], **kwargs: Any) -> dict[str, Path]:
        value = metrics.get(self.monitor)
        written: dict[str, Path] = {}
        improved = value is not None and self.is_improvement(float(value))
        if improved:
            self.best_metric = float(value)
        kwargs.setdefault("monitor", self.monitor)
        written["last"] = save_checkpoint(
            self.last_path, model, metrics=metrics, best_metric=self.best_metric, **kwargs
        )
        if improved:
            written["best"] = save_checkpoint(
                self.best_path, model, metrics=metrics, best_metric=self.best_metric, **kwargs
            )
        if self.keep_epochs:
            epoch = int(kwargs.get("epoch", 0))
            written["epoch"] = save_checkpoint(
                self.dir / f"epoch{epoch:04d}.pt",
                model,
                metrics=metrics,
                best_metric=self.best_metric,
                **kwargs,
            )
        return written
