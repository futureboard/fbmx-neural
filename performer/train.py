"""Train the FBMX Performer and measure it against the baselines it must beat.

Reproducible by construction: the seed, the split report, the normalisation
statistics, and the config all travel with the checkpoint, and nothing here
reads an absolute path that is not passed in.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from accent.features import fit_prominence_basis

from .features import (
    FEATURE_SCHEMA_VERSION,
    INPUT_FEATURES,
    TARGET_SCALES,
    TARGETS,
    VIBRATO_PRESENT_INDEX,
    Sequence_,
    input_normalization,
    load_sequences,
)
from .model import Performer, PerformerConfig

#: Weight per regression target in the loss. Timing and pitch are what the ear
#: notices first, so they are not allowed to be drowned out by the targets with
#: the most notes behind them.
TARGET_WEIGHTS: dict[str, float] = {
    "onset_deviation": 1.5,
    "log_duration_ratio": 1.0,
    "pitch_offset": 1.5,
    "entry_offset": 0.75,
    "intensity": 1.0,
    "vibrato_rate": 1.0,
    "vibrato_depth": 1.0,
    "vibrato_delay": 0.5,
}

VIBRATO_PRESENCE_WEIGHT = 1.0


@dataclass
class TrainConfig:
    dataset_dir: str
    checkpoint_dir: str
    mode: str = "studio"
    #: Notes per training example, and how far the window slides between
    #: examples. A part is one long sequence but it is also one gradient step,
    #: and 21 steps an epoch is not enough to fit anything: windowing the same
    #: notes into overlapping phrase-length examples turns the corpus into
    #: hundreds of steps without inventing data. Evaluation still runs whole
    #: parts, so the reported numbers are for the sequences the model will
    #: actually be given.
    window_notes: int = 48
    window_stride: int = 12
    hidden_size: int = 64
    num_layers: int = 1
    dropout: float = 0.0
    epochs: int = 120
    learning_rate: float = 3e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    seed: int = 20260828
    #: Huber transition point, in normalised target units. Below this the loss
    #: is quadratic; above it linear, so one badly aligned note cannot dominate
    #: a batch the way a squared error would.
    huber_beta: float = 1.0
    patience: int = 25


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def window_sequences(
    sequences: list[Sequence_], *, window: int, stride: int
) -> list[Sequence_]:
    """Slice each part into overlapping windows of notes.

    Windows shorter than the full length are kept only when a part is shorter
    than one window, so every example carries the phrase context the
    bidirectional pass needs.
    """

    out: list[Sequence_] = []
    for sequence in sequences:
        count = sequence.inputs.shape[0]
        if count <= window:
            out.append(sequence)
            continue
        for start in range(0, count - window + 1, stride):
            stop = start + window
            out.append(
                Sequence_(
                    part_id=f"{sequence.part_id}[{start}:{stop}]",
                    piece_id=sequence.piece_id,
                    inputs=sequence.inputs[start:stop],
                    targets=sequence.targets[start:stop],
                    mask=sequence.mask[start:stop],
                    vibrato_present=sequence.vibrato_present[start:stop],
                    records=sequence.records[start:stop],
                )
            )
    return out


def _batch(sequence: Sequence_, device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "inputs": torch.from_numpy(sequence.inputs).unsqueeze(0).to(device),
        "targets": torch.from_numpy(sequence.targets).unsqueeze(0).to(device),
        "mask": torch.from_numpy(sequence.mask).unsqueeze(0).to(device),
        "present": torch.from_numpy(sequence.vibrato_present).unsqueeze(0).to(device),
    }


def compute_loss(
    prediction: torch.Tensor,
    batch: dict[str, torch.Tensor],
    *,
    huber_beta: float,
    weights: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    regression = prediction[..., : len(TARGETS)]
    presence_logit = prediction[..., VIBRATO_PRESENT_INDEX]

    error = regression - batch["targets"]
    absolute = error.abs()
    huber = torch.where(
        absolute <= huber_beta,
        0.5 * error**2 / huber_beta,
        absolute - 0.5 * huber_beta,
    )
    masked = huber * batch["mask"] * weights
    denominator = (batch["mask"] * weights).sum().clamp_min(1e-6)
    regression_loss = masked.sum() / denominator

    presence_loss = nn.functional.binary_cross_entropy_with_logits(
        presence_logit, batch["present"]
    )
    total = regression_loss + VIBRATO_PRESENCE_WEIGHT * presence_loss
    return total, {
        "regression": float(regression_loss.detach()),
        "presence": float(presence_loss.detach()),
    }


def evaluate(
    model: Performer, sequences: list[Sequence_], device: torch.device
) -> dict[str, Any]:
    """Per-target metrics in the units a musician would state them in."""

    model.eval()
    sums: dict[str, float] = {name: 0.0 for name in TARGETS}
    counts: dict[str, float] = {name: 0.0 for name in TARGETS}
    predicted_all: dict[str, list[float]] = {name: [] for name in TARGETS}
    actual_all: dict[str, list[float]] = {name: [] for name in TARGETS}
    presence_correct = 0.0
    presence_total = 0.0
    true_positive = false_positive = false_negative = 0.0

    with torch.no_grad():
        for sequence in sequences:
            batch = _batch(sequence, device)
            prediction = model(batch["inputs"])
            regression = prediction[..., : len(TARGETS)]
            presence = torch.sigmoid(prediction[..., VIBRATO_PRESENT_INDEX])

            error = (regression - batch["targets"]).abs() * batch["mask"]
            for index, name in enumerate(TARGETS):
                sums[name] += float(error[..., index].sum()) * TARGET_SCALES[name]
                counts[name] += float(batch["mask"][..., index].sum())
                keep = batch["mask"][..., index] > 0
                predicted_all[name].extend(
                    (regression[..., index][keep] * TARGET_SCALES[name]).cpu().tolist()
                )
                actual_all[name].extend(
                    (batch["targets"][..., index][keep] * TARGET_SCALES[name]).cpu().tolist()
                )

            hard = (presence >= 0.5).float()
            presence_correct += float((hard == batch["present"]).sum())
            presence_total += float(batch["present"].numel())
            true_positive += float(((hard == 1) & (batch["present"] == 1)).sum())
            false_positive += float(((hard == 1) & (batch["present"] == 0)).sum())
            false_negative += float(((hard == 0) & (batch["present"] == 1)).sum())

    metrics: dict[str, Any] = {}
    for name in TARGETS:
        mae = sums[name] / counts[name] if counts[name] else float("nan")
        predicted = np.asarray(predicted_all[name])
        actual = np.asarray(actual_all[name])
        if predicted.size > 2 and predicted.std() > 1e-9 and actual.std() > 1e-9:
            correlation = float(np.corrcoef(predicted, actual)[0, 1])
        else:
            correlation = float("nan")
        metrics[name] = {
            "mae": mae,
            "n": int(counts[name]),
            "correlation": correlation,
            "predicted_std": float(predicted.std()) if predicted.size else float("nan"),
            "actual_std": float(actual.std()) if actual.size else float("nan"),
        }
    precision = true_positive / max(true_positive + false_positive, 1e-9)
    recall = true_positive / max(true_positive + false_negative, 1e-9)
    metrics["vibrato_present"] = {
        "accuracy": presence_correct / max(presence_total, 1e-9),
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / max(precision + recall, 1e-9),
        "n": int(presence_total),
    }
    return metrics


def baseline_metrics(
    train: list[Sequence_], evaluation: list[Sequence_]
) -> dict[str, Any]:
    """What predicting the training mean achieves on the evaluation split.

    This is the number the network has to beat to have learned anything. For
    timing especially it is a strong baseline: human microtiming is close to
    zero-mean, so "play it exactly as written" is already a decent guess and a
    model that merely reproduces it has added nothing.
    """

    train_targets = np.concatenate([sequence.targets for sequence in train], axis=0)
    train_mask = np.concatenate([sequence.mask for sequence in train], axis=0)
    means = np.zeros(len(TARGETS), dtype=np.float64)
    for index in range(len(TARGETS)):
        keep = train_mask[:, index] > 0
        means[index] = float(train_targets[keep, index].mean()) if keep.any() else 0.0

    present_rate = float(
        np.concatenate([sequence.vibrato_present for sequence in train]).mean()
    )

    targets = np.concatenate([sequence.targets for sequence in evaluation], axis=0)
    mask = np.concatenate([sequence.mask for sequence in evaluation], axis=0)
    present = np.concatenate([sequence.vibrato_present for sequence in evaluation])

    metrics: dict[str, Any] = {}
    for index, name in enumerate(TARGETS):
        keep = mask[:, index] > 0
        if not keep.any():
            metrics[name] = {"mae": float("nan"), "n": 0}
            continue
        error = np.abs(targets[keep, index] - means[index]) * TARGET_SCALES[name]
        metrics[name] = {"mae": float(error.mean()), "n": int(keep.sum())}

    # Majority-class guess for presence.
    guess = 1.0 if present_rate >= 0.5 else 0.0
    metrics["vibrato_present"] = {
        "accuracy": float((present == guess).mean()),
        "n": int(present.size),
        "strategy": f"always {'present' if guess else 'absent'}",
    }
    metrics["_train_means_normalised"] = means.tolist()
    return metrics


def train(config: TrainConfig) -> dict[str, Any]:
    seed_everything(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset_dir = Path(config.dataset_dir)
    # The accent inputs are the measured accent, and the recipe that turns four
    # evidence columns into a prominence is fitted on *training* records only.
    # Fitting it over the corpus would let the test split's covariance into the
    # definition of a feature the model is scored with.
    train_records = [
        json.loads(line)
        for line in (dataset_dir / "notes.train.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    accent_basis = fit_prominence_basis(train_records)

    train_sequences = load_sequences(dataset_dir / "notes.train.jsonl", accent_basis)
    validation_sequences = load_sequences(dataset_dir / "notes.validation.jsonl", accent_basis)
    test_sequences = load_sequences(dataset_dir / "notes.test.jsonl", accent_basis)
    if not train_sequences:
        raise RuntimeError("no training sequences; run the dataset build and split first")

    mean, std = input_normalization(train_sequences)
    train_windows = window_sequences(
        train_sequences, window=config.window_notes, stride=config.window_stride
    )
    model_config = PerformerConfig(
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        dropout=config.dropout,
        mode=config.mode,
    )
    model = Performer(model_config).to(device)
    model.set_normalization(torch.from_numpy(mean), torch.from_numpy(std))

    weights = torch.tensor(
        [TARGET_WEIGHTS[name] for name in TARGETS], dtype=torch.float32, device=device
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)

    checkpoint_dir = Path(config.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_path = checkpoint_dir / "training_log.jsonl"
    log_path.write_text("", encoding="utf-8")

    order = list(range(len(train_windows)))
    best_score = float("inf")
    best_epoch = -1
    since_improvement = 0

    for epoch in range(config.epochs):
        model.train()
        random.shuffle(order)
        epoch_loss = 0.0
        parts: dict[str, float] = {}
        for index in order:
            batch = _batch(train_windows[index], device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(batch["inputs"])
            loss, detail = compute_loss(
                prediction, batch, huber_beta=config.huber_beta, weights=weights
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            epoch_loss += float(loss.detach())
            for key, value in detail.items():
                parts[key] = parts.get(key, 0.0) + value
        scheduler.step()

        validation = evaluate(model, validation_sequences, device)
        # One number to select on: the weighted sum of normalised MAEs, so no
        # single target can be traded away for another that happens to be
        # measured in smaller units.
        score = float(
            sum(
                validation[name]["mae"] / TARGET_SCALES[name] * TARGET_WEIGHTS[name]
                for name in TARGETS
                if validation[name]["n"] > 0 and np.isfinite(validation[name]["mae"])
            )
        )
        entry = {
            "epoch": epoch,
            "train_loss": epoch_loss / max(len(order), 1),
            "train_regression": parts.get("regression", 0.0) / max(len(order), 1),
            "train_presence": parts.get("presence", 0.0) / max(len(order), 1),
            "validation_score": score,
            "lr": scheduler.get_last_lr()[0],
            "onset_mae_ms": validation["onset_deviation"]["mae"] * 1000.0,
            "pitch_mae_cents": validation["pitch_offset"]["mae"],
            "vibrato_f1": validation["vibrato_present"]["f1"],
        }
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry) + "\n")

        if score < best_score - 1e-5:
            best_score, best_epoch, since_improvement = score, epoch, 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "config": model_config.to_dict(),
                    "train_config": asdict(config),
                    "input_mean": mean,
                    "input_std": std,
                    "accent_basis": accent_basis.to_dict(),
                    "epoch": epoch,
                    "validation_score": score,
                    "feature_schema_version": FEATURE_SCHEMA_VERSION,
                },
                checkpoint_dir / "best.pt",
            )
        else:
            since_improvement += 1
            if since_improvement >= config.patience:
                break

    state = torch.load(checkpoint_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(state["model"])

    report = {
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config": asdict(config),
        "model": model_config.to_dict(),
        "parameter_count": model.parameter_count(),
        "best_epoch": best_epoch,
        "best_validation_score": best_score,
        "sequences": {
            "train_parts": len(train_sequences),
            "train_windows": len(train_windows),
            "train": len(train_sequences),
            "validation": len(validation_sequences),
            "test": len(test_sequences),
        },
        "accent_basis": accent_basis.to_dict(),
        "notes": {
            "train": int(sum(s.inputs.shape[0] for s in train_sequences)),
            "validation": int(sum(s.inputs.shape[0] for s in validation_sequences)),
            "test": int(sum(s.inputs.shape[0] for s in test_sequences)),
        },
        "validation": evaluate(model, validation_sequences, device),
        "test": evaluate(model, test_sequences, device),
        "baseline_validation": baseline_metrics(train_sequences, validation_sequences),
        "baseline_test": baseline_metrics(train_sequences, test_sequences),
    }
    (checkpoint_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, default=float) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the FBMX Performer")
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--mode", default="studio", choices=("studio", "live"))
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--window-notes", type=int, default=48)
    parser.add_argument("--window-stride", type=int, default=12)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--seed", type=int, default=20260828)
    arguments = parser.parse_args()

    report = train(
        TrainConfig(
            dataset_dir=arguments.dataset_dir,
            checkpoint_dir=arguments.checkpoint_dir,
            mode=arguments.mode,
            hidden_size=arguments.hidden_size,
            num_layers=arguments.num_layers,
            dropout=arguments.dropout,
            epochs=arguments.epochs,
            learning_rate=arguments.learning_rate,
            weight_decay=arguments.weight_decay,
            window_notes=arguments.window_notes,
            window_stride=arguments.window_stride,
            patience=arguments.patience,
            seed=arguments.seed,
        )
    )
    print(f"parameters      {report['parameter_count']}")
    print(f"best epoch      {report['best_epoch']}")
    for split in ("validation", "test"):
        print(f"\n== {split} ==")
        print(f"{'target':22s} {'model MAE':>12s} {'baseline MAE':>13s} {'corr':>7s}")
        baseline = report[f"baseline_{split}"]
        for name in TARGETS:
            model_mae = report[split][name]["mae"]
            base_mae = baseline[name]["mae"]
            corr = report[split][name]["correlation"]
            print(f"{name:22s} {model_mae:12.4f} {base_mae:13.4f} {corr:7.3f}")
        print(
            f"{'vibrato_present f1':22s} {report[split]['vibrato_present']['f1']:12.4f} "
            f"{baseline['vibrato_present']['accuracy']:13.4f} (baseline is accuracy)"
        )


if __name__ == "__main__":
    main()
