"""Early stopping and thread control.

Long-horizon runs on a small corpus turn over quietly: the best checkpoint is
kept regardless, so grinding past the turn costs hours and buys nothing.
"""

from __future__ import annotations

import pytest
import torch

from fbmx.datasets import SYNTHETIC_SCHEMA, SyntheticSmokeDataset
from fbmx.losses import build_loss
from fbmx.models import build_model
from fbmx.training.trainer import Trainer, TrainerConfig


def make_trainer(tmp_path, **overrides):
    train = SyntheticSmokeDataset(num_sequences=4, sequence_length=8192, seed=0, split="train")
    val = SyntheticSmokeDataset(num_sequences=2, sequence_length=8192, seed=0, split="val")
    cfg = TrainerConfig.from_dict(
        {
            "epochs": 8,
            "batch_size": 2,
            "chunk_size": 2048,
            "warmup_samples": 512,
            "lr": 0.01,
            "device": "cpu",
            "log_every": 0,
            "checkpoint_dir": str(tmp_path / "ckpt"),
            **overrides,
        }
    )
    model = build_model({"type": "lstm", "hidden_size": 8}, SYNTHETIC_SCHEMA)
    return Trainer(model, train, val, cfg, loss_fn=build_loss([{"name": "mae"}]))


def test_disabled_by_default(tmp_path):
    trainer = make_trainer(tmp_path, epochs=3)
    result = trainer.fit()
    assert result["epochs_completed"] == 3
    assert result["stopped_early"] is False


def test_stops_when_the_monitor_stalls(tmp_path):
    """A patience of 1 against an impossible improvement target must stop."""
    trainer = make_trainer(
        tmp_path,
        epochs=20,
        early_stopping_patience=1,
        early_stopping_min_delta=1e6,  # nothing can ever count as an improvement
    )
    result = trainer.fit()
    assert result["stopped_early"] is True
    assert result["epochs_completed"] < 20


def test_the_best_checkpoint_survives_an_early_stop(tmp_path):
    from fbmx.training.checkpoint import load_checkpoint

    trainer = make_trainer(
        tmp_path, epochs=20, early_stopping_patience=1, early_stopping_min_delta=1e6
    )
    trainer.fit()
    best = load_checkpoint(trainer.ckpt.best_path)
    assert best["best_metric"] == trainer.ckpt.best_metric
    assert best["metrics"]["val_loss"] == pytest.approx(trainer.ckpt.best_metric)


def test_patience_counts_against_the_best_not_the_previous_epoch(tmp_path):
    """A single noisy validation must not end the run."""
    trainer = make_trainer(tmp_path, epochs=6, early_stopping_patience=3)
    trainer.ckpt.best_metric = 0.0  # nothing will beat this
    trainer.epochs_without_improvement = 0
    for _ in range(2):
        assert trainer._should_stop_early({"val_loss": 1.0}) is False
    assert trainer._should_stop_early({"val_loss": 1.0}) is True


def test_thread_count_is_applied(tmp_path):
    before = torch.get_num_threads()
    try:
        make_trainer(tmp_path, threads=2)
        assert torch.get_num_threads() == 2
    finally:
        torch.set_num_threads(before)
