"""Training loop, TBPTT, checkpointing and resume."""

from __future__ import annotations

import pytest
import torch

from fbmx.datasets import SYNTHETIC_SCHEMA, SyntheticSmokeDataset
from fbmx.losses import build_loss
from fbmx.models import build_model
from fbmx.training.checkpoint import (
    CheckpointManager,
    load_checkpoint,
    model_from_checkpoint,
    save_checkpoint,
    set_seed,
)
from fbmx.training.metrics import assert_finite, esr, esr_db, waveform_metrics
from fbmx.training.trainer import Trainer, TrainerConfig


def make_trainer(tmp_path, **overrides):
    set_seed(0)
    train = SyntheticSmokeDataset(num_sequences=4, sequence_length=8192, seed=0, split="train")
    val = SyntheticSmokeDataset(num_sequences=2, sequence_length=8192, seed=0, split="val")
    cfg = TrainerConfig.from_dict({
        "epochs": 2,
        "batch_size": 2,
        "chunk_size": 2048,
        "tbptt_chunks": 1,
        "warmup_samples": 512,
        "lr": 0.01,
        "device": "cpu",
        "log_every": 0,
        "checkpoint_dir": str(tmp_path / "ckpt"),
        **overrides,
    })
    model = build_model({"type": "lstm", "hidden_size": 16}, SYNTHETIC_SCHEMA)
    return Trainer(model, train, val, cfg, loss_fn=build_loss([{"name": "mae"}]))


# -- metrics -----------------------------------------------------------------
def test_metrics_on_perfect_prediction():
    x = torch.randn(1, 1, 1000)
    m = waveform_metrics(x, x)
    assert m["esr"] == pytest.approx(0.0, abs=1e-12)
    assert m["mae"] == pytest.approx(0.0, abs=1e-12)
    assert esr_db(x, x) < -100


def test_esr_is_scale_relative():
    target = torch.randn(1, 1, 1000)
    assert esr(target * 0.5, target) == pytest.approx(esr(target * 5, target * 10), rel=1e-5)


def test_assert_finite_catches_nan():
    with pytest.raises(FloatingPointError):
        assert_finite(torch.tensor([1.0, float("nan")]), "probe")


# -- training ----------------------------------------------------------------
def test_loss_decreases(tmp_path):
    trainer = make_trainer(tmp_path, epochs=4)
    result = trainer.fit()
    history = result["history"]
    assert len(history) == 4
    assert history[-1]["train_total"] < history[0]["train_total"]
    assert history[-1]["val_loss"] < history[0]["val_loss"]


def test_training_produces_finite_weights(tmp_path):
    trainer = make_trainer(tmp_path)
    trainer.fit()
    for name, param in trainer.model.named_parameters():
        assert torch.isfinite(param).all(), name


def test_tbptt_windows_group_chunks(tmp_path):
    """tbptt_chunks=2 must take half as many optimiser steps as tbptt_chunks=1."""
    one = make_trainer(tmp_path / "a", epochs=1, tbptt_chunks=1)
    two = make_trainer(tmp_path / "b", epochs=1, tbptt_chunks=2)
    one.fit()
    two.fit()
    assert one.global_step > two.global_step


def test_max_steps_stops_early(tmp_path):
    trainer = make_trainer(tmp_path, epochs=50, max_steps=3)
    trainer.fit()
    assert trainer.global_step == 3


def test_warmup_is_excluded_from_the_loss(tmp_path):
    """A warmup longer than the whole sequence leaves nothing to train on."""
    trainer = make_trainer(tmp_path, epochs=1, warmup_samples=10**9)
    trainer.fit()
    assert trainer.global_step == 0


# -- checkpoints -------------------------------------------------------------
def test_checkpoint_round_trip(tmp_path):
    set_seed(3)
    model = build_model({"type": "lstm", "hidden_size": 16}, SYNTHETIC_SCHEMA).eval()
    path = save_checkpoint(tmp_path / "c.pt", model, epoch=2, global_step=42, config={"a": 1})
    ckpt = load_checkpoint(path)
    restored = model_from_checkpoint(ckpt).eval()

    assert ckpt["epoch"] == 2 and ckpt["global_step"] == 42
    assert ckpt["config"] == {"a": 1}
    assert restored.hidden_size == model.hidden_size
    assert restored.schema == model.schema
    x = torch.randn(1, 1, 2048) * 0.3
    params = SYNTHETIC_SCHEMA.empty_batch(1)
    with torch.no_grad():
        assert torch.equal(model(x, params, None)[0], restored(x, params, None)[0])


def test_checkpoint_manager_tracks_best(tmp_path):
    model = build_model({"type": "lstm", "hidden_size": 8}, SYNTHETIC_SCHEMA)
    manager = CheckpointManager(tmp_path, monitor="val_loss", mode="min")
    manager.save(model, {"val_loss": 1.0}, epoch=1)
    assert manager.best_path.exists() and manager.best_metric == 1.0
    manager.save(model, {"val_loss": 2.0}, epoch=2)
    assert manager.best_metric == 1.0
    manager.save(model, {"val_loss": 0.5}, epoch=3)
    assert manager.best_metric == 0.5
    assert load_checkpoint(manager.last_path)["epoch"] == 3
    assert load_checkpoint(manager.best_path)["epoch"] == 3


def test_resume_restores_optimizer_and_counters(tmp_path):
    trainer = make_trainer(tmp_path, epochs=2)
    trainer.fit()
    step_after_first = trainer.global_step
    momentum = trainer.optimizer.state_dict()["state"]

    resumed = make_trainer(tmp_path / "resumed", epochs=4)
    resumed.resume(trainer.ckpt.last_path)
    assert resumed.epoch == 2
    assert resumed.global_step == step_after_first
    assert len(resumed.optimizer.state_dict()["state"]) == len(momentum)
    for name, param in resumed.model.named_parameters():
        assert torch.equal(param, dict(trainer.model.named_parameters())[name])


def test_resume_continues_rather_than_restarting(tmp_path):
    """Two epochs then resume for two more == the interrupted run continued."""
    trainer = make_trainer(tmp_path, epochs=2)
    trainer.fit()
    loss_at_2 = trainer.history[-1]["val_loss"]

    resumed = make_trainer(tmp_path / "resumed", epochs=4)
    resumed.resume(trainer.ckpt.last_path)
    result = resumed.fit()
    assert result["epochs_completed"] == 4
    assert len(result["history"]) == 4  # the first two came from the checkpoint
    assert result["history"][-1]["val_loss"] < loss_at_2


def test_checkpoint_loads_without_arbitrary_code(tmp_path):
    """load_checkpoint uses weights_only=True; keep it that way."""
    model = build_model({"type": "lstm", "hidden_size": 8}, SYNTHETIC_SCHEMA)
    path = save_checkpoint(tmp_path / "c.pt", model)
    loaded = torch.load(path, map_location="cpu", weights_only=True)
    assert loaded["model_type"] == "lstm"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_trains_on_cuda(tmp_path):
    trainer = make_trainer(tmp_path, epochs=1, device="cuda")
    trainer.fit()
    assert trainer.history[-1]["train_total"] > 0
    assert next(trainer.model.parameters()).device.type == "cuda"


def test_resume_realigns_a_restored_cosine_schedule(tmp_path):
    """A restored schedule says where the run is, not how long it is.

    `CosineAnnealingLR.state_dict()` carries `T_max`. Resuming a 12-epoch run
    under a 30-epoch config used to restore the 12-epoch period, so the
    learning rate bottomed out at epoch 12 and climbed back up the far side of
    the cosine for the rest of the run.
    """
    short = make_trainer(tmp_path / "short", epochs=12, scheduler="cosine")
    short.fit()
    assert short.scheduler.T_max == 12

    long = make_trainer(tmp_path / "long", epochs=30, scheduler="cosine")
    long.resume(short.ckpt.last_path)
    assert long.scheduler.T_max == 30, "the schedule must follow this run's epoch count"

    # ...and the rate must keep falling from here rather than turning around.
    rates = []
    for _ in range(6):
        long.scheduler.step()
        rates.append(long.optimizer.param_groups[0]["lr"])
    assert rates == sorted(rates, reverse=True), rates
