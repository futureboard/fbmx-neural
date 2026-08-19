"""Truncated-BPTT trainer.

The whole design turns on one thing: **the recurrent state is carried across
chunk boundaries, and only the autograd graph is cut.**  That is what lets a
32-unit LSTM learn an 80 ms release from 4096-sample chunks -- the state has
seen the whole sequence even though gradients only reach back
``chunk_size * tbptt_chunks`` samples.

Per sequence in the batch::

    state = zeros
    for chunk in sequence:                     # contiguous, in order
        y, state = model(chunk, params, state)
        window_loss += loss(y, wet_chunk)
        if window is full:
            window_loss.backward()             # graph spans the window
            clip; optimiser.step(); zero_grad
            state = detach(state)              # <-- state survives, graph does not

The first ``warmup_samples`` of every sequence are excluded from the loss: the
model starts from a zero state that the target does not share, and scoring that
transient teaches it to guess rather than to converge.

Everything else here is deliberately ordinary -- Adam, optional cosine decay,
gradient clipping, optional AMP.  FP32 is the reference; AMP is opt-in and the
streaming-equality tests always run in FP32.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

import torch
from torch.utils.data import DataLoader

from fbmx.conditioning import ParamBatch
from fbmx.datasets.base import PairedSequenceDataset, collate_pairs
from fbmx.device import amp_supported, auto_device, describe_device
from fbmx.losses import CompositeLoss, build_loss
from fbmx.models.base import StreamingModel, detach_state
from fbmx.training.checkpoint import (
    CheckpointManager,
    load_checkpoint,
    restore_rng_state,
    set_seed,
)
from fbmx.training.metrics import assert_finite, waveform_metrics

__all__ = ["TrainerConfig", "Trainer"]


@dataclass
class TrainerConfig:
    # -- schedule
    epochs: int = 10
    max_steps: int | None = None
    batch_size: int = 4
    num_workers: int = 0

    # -- TBPTT
    chunk_size: int = 4096
    tbptt_chunks: int = 1  # optimiser steps every N chunks
    warmup_samples: int = 1024  # excluded from the loss at sequence start

    # -- optimisation
    lr: float = 3e-3
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    scheduler: str | None = None  # None | "cosine" | "plateau"
    min_lr: float = 1e-5

    # -- runtime
    device: str = "auto"
    amp: bool = False
    seed: int = 0
    deterministic: bool = False
    dtype: str = "float32"
    #: CPU threads for this process. Set it when running two experiments side
    #: by side, so they do not fight over the same cores and make each other's
    #: timings meaningless.
    threads: int | None = None

    # -- bookkeeping
    log_every: int = 10
    validate_every: int = 1  # epochs
    monitor: str = "val_loss"
    monitor_mode: str = "min"
    #: Stop when the monitored metric has not improved by `min_delta` for this
    #: many validations. 0 disables it. Long-horizon runs on a small corpus
    #: start overfitting quietly, and the best checkpoint is kept anyway, so
    #: there is nothing to gain from grinding past the turn.
    early_stopping_patience: int = 0
    early_stopping_min_delta: float = 0.0
    checkpoint_dir: str = "checkpoints"
    keep_epoch_checkpoints: bool = False
    run_name: str = "run"

    extra: dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def from_dict(d: Mapping[str, Any] | None) -> "TrainerConfig":
        d = dict(d or {})
        known = set(TrainerConfig.__dataclass_fields__) - {"extra"}
        extra = {k: v for k, v in d.items() if k not in known}
        cfg = TrainerConfig(**{k: v for k, v in d.items() if k in known})
        cfg.extra = extra
        return cfg

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class Trainer:
    def __init__(
        self,
        model: StreamingModel,
        train_dataset: PairedSequenceDataset,
        val_dataset: PairedSequenceDataset | None = None,
        cfg: TrainerConfig | Mapping[str, Any] | None = None,
        loss_fn: CompositeLoss | None = None,
        config_snapshot: Mapping[str, Any] | None = None,
    ) -> None:
        self.cfg = cfg if isinstance(cfg, TrainerConfig) else TrainerConfig.from_dict(cfg)
        if self.cfg.threads:
            torch.set_num_threads(int(self.cfg.threads))
        self.device = auto_device(self.cfg.device)
        set_seed(self.cfg.seed, self.cfg.deterministic)

        self.model = model.to(self.device)
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.loss_fn = (loss_fn or build_loss(None)).to(self.device)
        self.config_snapshot = dict(config_snapshot or {})

        self.train_loader = self._make_loader(train_dataset, shuffle=True)
        self.val_loader = (
            self._make_loader(val_dataset, shuffle=False) if val_dataset is not None else None
        )

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay
        )
        self.scheduler = self._make_scheduler()
        self.use_amp = bool(self.cfg.amp) and amp_supported(self.device)
        if self.cfg.amp and not self.use_amp:
            print("[trainer] AMP requested but unavailable on this device; using FP32")
        self.scaler = torch.amp.GradScaler(self.device.type, enabled=self.use_amp)

        self.ckpt = CheckpointManager(
            self.cfg.checkpoint_dir,
            monitor=self.cfg.monitor,
            mode=self.cfg.monitor_mode,
            keep_epochs=self.cfg.keep_epoch_checkpoints,
        )
        self.epoch = 0
        self.global_step = 0
        self.epochs_without_improvement = 0
        self.stopped_early = False
        self.history: list[dict[str, Any]] = []
        self.log_path = Path(self.cfg.checkpoint_dir) / "training_log.jsonl"

    # -- setup helpers ---------------------------------------------------
    def _make_loader(self, dataset: PairedSequenceDataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=min(self.cfg.batch_size, len(dataset)),
            shuffle=shuffle,
            num_workers=self.cfg.num_workers,
            collate_fn=collate_pairs,
            drop_last=False,
        )

    def _make_scheduler(self):
        if self.cfg.scheduler in (None, "none"):
            return None
        if self.cfg.scheduler == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=max(self.cfg.epochs, 1), eta_min=self.cfg.min_lr
            )
        if self.cfg.scheduler == "plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode=self.cfg.monitor_mode, factor=0.5, patience=3,
                min_lr=self.cfg.min_lr,
            )
        raise ValueError(f"unknown scheduler {self.cfg.scheduler!r}")

    # -- core loop -------------------------------------------------------
    def _chunks(self, n_samples: int) -> list[tuple[int, int]]:
        chunk = self.cfg.chunk_size
        bounds = [(s, min(s + chunk, n_samples)) for s in range(0, n_samples, chunk)]
        return [b for b in bounds if b[1] > b[0]]

    def _loss_on_chunk(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        start: int,
        aux: dict[str, torch.Tensor] | None,
    ) -> tuple[torch.Tensor | None, dict[str, float]]:
        """Loss for one chunk, with the sequence-start warmup masked out."""
        skip = max(0, min(self.cfg.warmup_samples - start, pred.shape[-1]))
        if skip >= pred.shape[-1]:
            return None, {}
        if skip:
            pred, target = pred[..., skip:], target[..., skip:]
            aux = {k: v[..., skip:] for k, v in aux.items()} if aux else aux
        return self.loss_fn(pred, target, aux)

    def train_epoch(self) -> dict[str, float]:
        self.model.train()
        totals: dict[str, float] = {}
        n_scored_chunks = 0
        n_windows = 0
        for batch in self.train_loader:
            dry = batch["dry"].to(self.device)
            wet = batch["wet"].to(self.device)
            params: ParamBatch = batch["params"].to(self.device)
            aux_full = {k: v.to(self.device) for k, v in batch.get("aux", {}).items()}
            batch_size, _, n_samples = dry.shape

            state = self.model.init_state(batch_size, device=self.device)
            self.optimizer.zero_grad(set_to_none=True)
            window_loss = None
            window_count = 0

            bounds = self._chunks(n_samples)
            for i, (start, end) in enumerate(bounds):
                with torch.amp.autocast(self.device.type, enabled=self.use_amp):
                    pred, model_aux, state = self.model.forward_aux(
                        dry[..., start:end], params, state
                    )
                    # Dataset traces and the model's predictions of them go into
                    # the same dict; AuxTraceLoss pairs `gain` with `pred_gain`.
                    aux = {k: v[..., start:end] for k, v in aux_full.items()}
                    aux.update({k: v.float() for k, v in model_aux.items()})
                    loss, parts = self._loss_on_chunk(
                        pred.float(), wet[..., start:end], start, aux or None
                    )
                if loss is not None:
                    window_loss = loss if window_loss is None else window_loss + loss
                    window_count += 1
                    n_scored_chunks += 1
                    for k, v in parts.items():
                        totals[k] = totals.get(k, 0.0) + v

                is_boundary = (i + 1) % self.cfg.tbptt_chunks == 0 or i == len(bounds) - 1
                if not is_boundary:
                    continue

                if window_loss is not None:
                    scaled = window_loss / max(window_count, 1)
                    assert_finite(scaled.detach(), "training loss")
                    self.scaler.scale(scaled).backward()
                    if self.cfg.grad_clip:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.cfg.grad_clip
                        )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)
                    self.global_step += 1
                    n_windows += 1
                    if self.cfg.log_every and self.global_step % self.cfg.log_every == 0:
                        print(
                            f"  step {self.global_step:6d} "
                            f"loss {float(scaled.detach()):.6f}",
                            flush=True,
                        )
                # State survives the boundary; only the graph is cut.
                state = detach_state(state)
                window_loss = None
                window_count = 0

                if self.cfg.max_steps and self.global_step >= self.cfg.max_steps:
                    break
            if self.cfg.max_steps and self.global_step >= self.cfg.max_steps:
                break

        count = max(n_scored_chunks, 1)
        out = {f"train_{k}": v / count for k, v in totals.items()}
        out["train_windows"] = float(n_windows)
        return out

    @torch.no_grad()
    def validate(self) -> dict[str, float]:
        """Chunked, stateful validation -- the same path inference will take."""
        if self.val_loader is None:
            return {}
        self.model.eval()
        loss_total = 0.0
        n_batches = 0
        metric_totals: dict[str, float] = {}
        for batch in self.val_loader:
            dry = batch["dry"].to(self.device)
            wet = batch["wet"].to(self.device)
            params: ParamBatch = batch["params"].to(self.device)
            batch_size, _, n_samples = dry.shape

            aux_full = {k: v.to(self.device) for k, v in batch.get("aux", {}).items()}

            state = self.model.init_state(batch_size, device=self.device)
            preds = []
            aux_preds: dict[str, list[torch.Tensor]] = {}
            for start, end in self._chunks(n_samples):
                pred, model_aux, state = self.model.forward_aux(
                    dry[..., start:end], params, state
                )
                preds.append(pred)
                for k, v in model_aux.items():
                    aux_preds.setdefault(k, []).append(v)
            pred = torch.cat(preds, dim=-1)
            assert_finite(pred, "validation prediction")

            skip = min(self.cfg.warmup_samples, n_samples - 1)
            aux = {k: v[..., skip:] for k, v in aux_full.items()}
            aux.update({k: torch.cat(v, dim=-1)[..., skip:] for k, v in aux_preds.items()})
            loss, _ = self.loss_fn(pred[..., skip:], wet[..., skip:], aux or None)
            loss_total += float(loss)
            for k, v in waveform_metrics(pred[..., skip:], wet[..., skip:]).items():
                metric_totals[k] = metric_totals.get(k, 0.0) + v
            # Auxiliary traces are reported separately: they are what says
            # whether the model learned the *dynamics* or only the waveform.
            for name in self.model.aux_heads:
                key, pred_key = name, f"pred_{name}"
                if key in aux and pred_key in aux:
                    # Undo the loss's normalisation so the number is in the
                    # trace's own units (dB for gain reduction), which is what
                    # anyone reading the log wants to compare against the
                    # teacher.
                    scale = self._aux_scale(name)
                    err = float(torch.mean(torch.abs(aux[pred_key] / scale - aux[key])))
                    metric_totals[f"{name}_mae"] = metric_totals.get(f"{name}_mae", 0.0) + err
            n_batches += 1

        out = {"val_loss": loss_total / max(n_batches, 1)}
        out.update({f"val_{k}": v / max(n_batches, 1) for k, v in metric_totals.items()})
        return out

    def _aux_scale(self, name: str) -> float:
        """The scale an :class:`AuxTraceLoss` applies to this trace, if any."""
        from fbmx.losses.auxiliary import AuxTraceLoss

        for term in self.loss_fn.terms.values():
            if isinstance(term, AuxTraceLoss) and term.key == name and term.scale != 0.0:
                return term.scale
        return 1.0

    # -- driver ----------------------------------------------------------
    def fit(self) -> dict[str, Any]:
        print(f"[trainer] device: {describe_device(self.device)}")
        print(f"[trainer] model:  {self.model.model_type} "
              f"({self.model.num_parameters():,} trainable parameters)")
        print(f"[trainer] loss:   {self.loss_fn.describe()}")
        print(f"[trainer] data:   {self.train_dataset.describe()}")
        if self.val_dataset is not None:
            print(f"[trainer] val:    {self.val_dataset.describe()}")

        start_epoch = self.epoch
        for epoch in range(start_epoch, self.cfg.epochs):
            self.epoch = epoch
            t0 = time.time()
            metrics = self.train_epoch()
            if self.cfg.validate_every and (epoch + 1) % self.cfg.validate_every == 0:
                metrics.update(self.validate())
            metrics["epoch"] = epoch
            metrics["lr"] = self.optimizer.param_groups[0]["lr"]
            metrics["seconds"] = time.time() - t0
            self._step_scheduler(metrics)
            self.history.append(metrics)
            self._log(metrics)

            self.ckpt.save(
                self.model,
                metrics=metrics,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                epoch=epoch + 1,
                global_step=self.global_step,
                config=self.config_snapshot,
                history=self.history,
                extra={
                    "dataset": self.train_dataset.provenance(),
                    "trainer": self.cfg.to_dict(),
                    "device": describe_device(self.device),
                },
            )
            if self.cfg.max_steps and self.global_step >= self.cfg.max_steps:
                print(f"[trainer] stopping: max_steps={self.cfg.max_steps} reached")
                break
            if self._should_stop_early(metrics):
                self.stopped_early = True
                print(
                    f"[trainer] early stop: {self.cfg.monitor} has not improved by "
                    f"{self.cfg.early_stopping_min_delta} in "
                    f"{self.cfg.early_stopping_patience} validations "
                    f"(best {self.ckpt.best_metric:.6g})"
                )
                break
        self.epoch = min(self.epoch + 1, self.cfg.epochs)
        return {
            "epochs_completed": self.epoch,
            "global_step": self.global_step,
            "stopped_early": self.stopped_early,
            "best_metric": self.ckpt.best_metric,
            "history": self.history,
            "checkpoint_dir": str(self.ckpt.dir),
        }

    def _realign_schedule(self) -> None:
        """Make a restored schedule agree with *this* run's epoch count.

        ``CosineAnnealingLR.state_dict()`` carries ``T_max``, so resuming a run
        that was configured for 12 epochs under a config that asks for 30
        restores the 12-epoch period -- the learning rate reaches its minimum
        at epoch 12 and then *climbs back up* the other side of the cosine.
        That happened, quietly, and cost a comparison run: the model was still
        improving at epoch 13 and had been driven back to a worse place by
        epoch 19.

        A restored schedule describes where the run is, not how long it is.
        """
        wanted = max(int(self.cfg.epochs), 1)
        current = getattr(self.scheduler, "T_max", None)
        if current is not None and current != wanted:
            print(
                f"[trainer] schedule restored with T_max={current} but this run has "
                f"{wanted} epochs; realigning so the cosine anneals to the end"
            )
            self.scheduler.T_max = wanted
        print(
            f"[trainer] learning rate resumes at {self.optimizer.param_groups[0]['lr']:.3g}"
        )

    def _should_stop_early(self, metrics: Mapping[str, float]) -> bool:
        """Patience against the *best* value, not the previous epoch's.

        Comparing against the previous epoch would stop on any single noisy
        validation; comparing against the best is what "no longer improving"
        actually means.
        """
        patience = int(self.cfg.early_stopping_patience or 0)
        if patience <= 0:
            return False
        value = metrics.get(self.cfg.monitor)
        if value is None:
            return False
        best = self.ckpt.best_metric
        delta = float(self.cfg.early_stopping_min_delta)
        improved = (
            best is None
            or (self.cfg.monitor_mode == "min" and value < best - delta)
            or (self.cfg.monitor_mode == "max" and value > best + delta)
        )
        # `best` has already been updated by the checkpoint manager for this
        # epoch, so an improvement shows up as equality with it.
        if improved or (best is not None and abs(value - best) < 1e-12):
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1
        return self.epochs_without_improvement >= patience

    def _step_scheduler(self, metrics: Mapping[str, float]) -> None:
        if self.scheduler is None:
            return
        if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            value = metrics.get(self.cfg.monitor)
            if value is not None:
                self.scheduler.step(value)
        else:
            self.scheduler.step()

    def _log(self, metrics: Mapping[str, Any]) -> None:
        line = " | ".join(
            f"{k}={v:.6g}" if isinstance(v, float) else f"{k}={v}"
            for k, v in metrics.items()
        )
        print(f"[epoch {metrics.get('epoch')}] {line}", flush=True)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.log_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(dict(metrics)) + "\n")

    # -- resume ----------------------------------------------------------
    def resume(self, path: str | Path) -> None:
        """Restore weights, optimiser, scheduler, counters and RNG streams.

        The model architecture is *not* rebuilt here -- the caller is expected
        to have constructed it from the same checkpoint (see
        ``fbmx.training.checkpoint.model_from_checkpoint``), which is what
        ``scripts/train.py --resume`` does.
        """
        ckpt = load_checkpoint(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.to(self.device)
        if ckpt.get("optimizer_state"):
            self.optimizer.load_state_dict(ckpt["optimizer_state"])
        if self.scheduler is not None and ckpt.get("scheduler_state"):
            self.scheduler.load_state_dict(ckpt["scheduler_state"])
            self._realign_schedule()
        self.epoch = int(ckpt.get("epoch", 0))
        self.global_step = int(ckpt.get("global_step", 0))
        self.history = list(ckpt.get("history", []))
        self.ckpt.best_metric = ckpt.get("best_metric")
        restore_rng_state(ckpt.get("rng"))
        print(
            f"[trainer] resumed from {path}: epoch {self.epoch}, step {self.global_step}, "
            f"best {self.ckpt.monitor}={self.ckpt.best_metric}"
        )
