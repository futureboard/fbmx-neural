"""Grouped cross-validation over URMP sessions.

A single 15% validation split of a corpus this small measures the split as much
as the model: the first Performer runs disagreed with themselves depending on
whether `44_K515` — one part whose score and performance differ in tempo by a
factor of two — landed in validation or test. With fourteen session clusters
there is no split that is simultaneously large enough to trust and small enough
to leave a useful training set.

So every cluster takes a turn being held out. Each target is then reported as a
mean and spread across folds, against the same mean-prediction baseline computed
per fold from that fold's own training data. That is the number that answers
section 27's question — did the network learn anything — rather than one draw
from a noisy distribution.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..datasets.urmp.split import session_clusters
from .features import TARGET_SCALES, TARGETS, Sequence_, input_normalization, load_sequences
from .model import Performer, PerformerConfig
from .train import (
    TARGET_WEIGHTS,
    TrainConfig,
    baseline_metrics,
    compute_loss,
    evaluate,
    seed_everything,
    window_sequences,
    _batch,
)


def _cluster_of_part(dataset_dir: Path) -> dict[str, str]:
    records = [
        json.loads(line)
        for line in (dataset_dir / "notes.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    clusters = session_clusters(records)
    return {record["part_id"]: clusters[record["group_id"]] for record in records}


def _train_fold(
    train_sequences: list[Sequence_],
    validation_sequences: list[Sequence_],
    config: TrainConfig,
    device: torch.device,
) -> tuple[Performer, dict[str, Any]]:
    mean, std = input_normalization(train_sequences)
    model_config = PerformerConfig(
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        dropout=config.dropout,
        mode=config.mode,
    )
    model = Performer(model_config).to(device)
    model.set_normalization(torch.from_numpy(mean), torch.from_numpy(std))

    windows = window_sequences(
        train_sequences, window=config.window_notes, stride=config.window_stride
    )
    weights = torch.tensor(
        [TARGET_WEIGHTS[name] for name in TARGETS], dtype=torch.float32, device=device
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)

    import random

    order = list(range(len(windows)))
    best_score, best_state, since = float("inf"), None, 0
    for _ in range(config.epochs):
        model.train()
        random.shuffle(order)
        for index in order:
            batch = _batch(windows[index], device)
            optimizer.zero_grad(set_to_none=True)
            loss, _ = compute_loss(
                model(batch["inputs"]), batch, huber_beta=config.huber_beta, weights=weights
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
        scheduler.step()

        metrics = evaluate(model, validation_sequences, device)
        score = float(
            sum(
                metrics[name]["mae"] / TARGET_SCALES[name] * TARGET_WEIGHTS[name]
                for name in TARGETS
                if metrics[name]["n"] > 0 and np.isfinite(metrics[name]["mae"])
            )
        )
        if score < best_score - 1e-5:
            best_score, since = score, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            since += 1
            if since >= config.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, evaluate(model, validation_sequences, device)


def crossval(
    dataset_dir: str,
    output: str,
    *,
    hidden_size: int = 8,
    dropout: float = 0.3,
    weight_decay: float = 3e-2,
    epochs: int = 80,
    seed: int = 20260828,
) -> dict[str, Any]:
    directory = Path(dataset_dir)
    seed_everything(seed)
    device = torch.device("cpu")

    part_cluster = _cluster_of_part(directory)
    everything: list[Sequence_] = []
    for name in ("train", "validation", "test"):
        path = directory / f"notes.{name}.jsonl"
        if path.exists():
            everything.extend(load_sequences(path))

    clusters = sorted({part_cluster[sequence.part_id] for sequence in everything})
    folds: dict[str, list[Sequence_]] = {cluster: [] for cluster in clusters}
    for sequence in everything:
        folds[part_cluster[sequence.part_id]].append(sequence)

    per_fold: list[dict[str, Any]] = []
    for held_out in clusters:
        validation = folds[held_out]
        train = [s for cluster, group in folds.items() if cluster != held_out for s in group]
        if not validation or not train:
            continue
        config = TrainConfig(
            dataset_dir=dataset_dir,
            checkpoint_dir="",
            hidden_size=hidden_size,
            dropout=dropout,
            weight_decay=weight_decay,
            epochs=epochs,
            learning_rate=2e-3,
            patience=20,
        )
        _, metrics = _train_fold(train, validation, config, device)
        baseline = baseline_metrics(train, validation)
        row = {
            "fold": held_out,
            "validation_notes": int(sum(s.inputs.shape[0] for s in validation)),
            "parts": [s.part_id for s in validation],
            "model": {name: metrics[name]["mae"] for name in TARGETS},
            "baseline": {name: baseline[name]["mae"] for name in TARGETS},
            "correlation": {name: metrics[name]["correlation"] for name in TARGETS},
            "vibrato_f1": metrics["vibrato_present"]["f1"],
            "vibrato_accuracy": metrics["vibrato_present"]["accuracy"],
            "vibrato_baseline_accuracy": baseline["vibrato_present"]["accuracy"],
        }
        per_fold.append(row)
        wins = sum(
            1
            for name in TARGETS
            if np.isfinite(row["model"][name])
            and np.isfinite(row["baseline"][name])
            and row["model"][name] < row["baseline"][name]
        )
        print(f"fold {held_out:11s} notes {row['validation_notes']:5d}  beats {wins}/{len(TARGETS)}")

    summary: dict[str, Any] = {"folds": len(per_fold), "targets": {}}
    for name in TARGETS:
        model_values = np.asarray(
            [row["model"][name] for row in per_fold if np.isfinite(row["model"][name])]
        )
        base_values = np.asarray(
            [row["baseline"][name] for row in per_fold if np.isfinite(row["baseline"][name])]
        )
        correlations = np.asarray(
            [
                row["correlation"][name]
                for row in per_fold
                if np.isfinite(row["correlation"][name])
            ]
        )
        # How often the model beat the baseline, across folds. A model that is
        # better on average because of one lucky fold is not better.
        wins = sum(
            1
            for row in per_fold
            if np.isfinite(row["model"][name])
            and np.isfinite(row["baseline"][name])
            and row["model"][name] < row["baseline"][name]
        )
        summary["targets"][name] = {
            "model_mae_mean": float(model_values.mean()) if model_values.size else float("nan"),
            "model_mae_sd": float(model_values.std()) if model_values.size else float("nan"),
            "baseline_mae_mean": float(base_values.mean()) if base_values.size else float("nan"),
            "improvement_percent": (
                float((1.0 - model_values.mean() / base_values.mean()) * 100.0)
                if model_values.size and base_values.size and base_values.mean() > 0
                else float("nan")
            ),
            "folds_won": wins,
            "folds": len(per_fold),
            "correlation_mean": float(correlations.mean()) if correlations.size else float("nan"),
        }

    f1 = np.asarray([row["vibrato_f1"] for row in per_fold])
    accuracy = np.asarray([row["vibrato_accuracy"] for row in per_fold])
    baseline_accuracy = np.asarray([row["vibrato_baseline_accuracy"] for row in per_fold])
    summary["vibrato_present"] = {
        "f1_mean": float(f1.mean()),
        "accuracy_mean": float(accuracy.mean()),
        "baseline_accuracy_mean": float(baseline_accuracy.mean()),
        "folds_won": int((accuracy > baseline_accuracy).sum()),
        "folds": len(per_fold),
    }
    summary["config"] = {
        "hidden_size": hidden_size,
        "dropout": dropout,
        "weight_decay": weight_decay,
        "epochs": epochs,
        "seed": seed,
    }

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(
        json.dumps({"summary": summary, "per_fold": per_fold}, indent=2, default=float) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Leave-one-session-out cross-validation")
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--hidden-size", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--weight-decay", type=float, default=3e-2)
    parser.add_argument("--epochs", type=int, default=80)
    arguments = parser.parse_args()

    summary = crossval(
        arguments.dataset_dir,
        arguments.output,
        hidden_size=arguments.hidden_size,
        dropout=arguments.dropout,
        weight_decay=arguments.weight_decay,
        epochs=arguments.epochs,
    )
    print(f"\n{summary['folds']} folds")
    print(f"{'target':22s} {'model':>9s} {'baseline':>9s} {'improve':>9s} {'won':>7s} {'corr':>6s}")
    for name, row in summary["targets"].items():
        print(
            f"{name:22s} {row['model_mae_mean']:9.4f} {row['baseline_mae_mean']:9.4f} "
            f"{row['improvement_percent']:8.1f}% {row['folds_won']:3d}/{row['folds']:<3d} "
            f"{row['correlation_mean']:6.3f}"
        )
    vibrato = summary["vibrato_present"]
    print(
        f"{'vibrato_present':22s} acc {vibrato['accuracy_mean']:.4f} "
        f"baseline {vibrato['baseline_accuracy_mean']:.4f} "
        f"won {vibrato['folds_won']}/{vibrato['folds']} f1 {vibrato['f1_mean']:.4f}"
    )


if __name__ == "__main__":
    main()
