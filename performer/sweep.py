"""Find the Performer capacity this corpus can actually support.

URMP's violin content is 60 minutes and about five thousand aligned notes from
a dozen recording sessions. That is a lot of *audio* and very little *evidence
about playing decisions*, and the first Performer trained on it memorised the
training windows within four epochs. Rather than guess at a smaller model, this
sweeps capacity and regularisation and reports, per target, whether the network
beat the only baseline that matters: predicting the training mean.

A target the network cannot beat is reported as such. Some of them are not
failures of the model — a violinist's vibrato rate barely moves, so the mean is
very hard to improve on and there may be nothing in the score that predicts the
remaining variation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .features import TARGETS
from .train import TrainConfig, train


def sweep(dataset_dir: str, output_dir: str, *, epochs: int = 120) -> dict[str, Any]:
    grid = [
        {"hidden_size": 8, "dropout": 0.2, "weight_decay": 1e-2},
        {"hidden_size": 8, "dropout": 0.3, "weight_decay": 3e-2},
        {"hidden_size": 16, "dropout": 0.2, "weight_decay": 1e-2},
        {"hidden_size": 16, "dropout": 0.3, "weight_decay": 3e-2},
        {"hidden_size": 24, "dropout": 0.3, "weight_decay": 3e-2},
        {"hidden_size": 32, "dropout": 0.3, "weight_decay": 1e-1},
    ]

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []

    for index, setting in enumerate(grid):
        name = f"h{setting['hidden_size']}_d{setting['dropout']}_wd{setting['weight_decay']}"
        report = train(
            TrainConfig(
                dataset_dir=dataset_dir,
                checkpoint_dir=str(root / name),
                hidden_size=setting["hidden_size"],
                dropout=setting["dropout"],
                weight_decay=setting["weight_decay"],
                learning_rate=2e-3,
                epochs=epochs,
                patience=30,
            )
        )
        wins = [
            name_
            for name_ in TARGETS
            if report["validation"][name_]["mae"] < report["baseline_validation"][name_]["mae"]
        ]
        results.append(
            {
                "name": name,
                **setting,
                "parameters": report["parameter_count"],
                "best_epoch": report["best_epoch"],
                "validation_score": report["best_validation_score"],
                "beats_baseline": wins,
                "beats_baseline_count": len(wins),
                "validation": {
                    key: report["validation"][key]["mae"] for key in TARGETS
                },
                "baseline": {
                    key: report["baseline_validation"][key]["mae"] for key in TARGETS
                },
                "vibrato_f1": report["validation"]["vibrato_present"]["f1"],
            }
        )
        print(
            f"[{index + 1}/{len(grid)}] {name:28s} params {report['parameter_count']:6d} "
            f"epoch {report['best_epoch']:3d} beats {len(wins)}/{len(TARGETS)}"
        )

    results.sort(key=lambda row: (-row["beats_baseline_count"], row["validation_score"]))
    (root / "sweep.json").write_text(
        json.dumps(results, indent=2, sort_keys=True, default=float) + "\n", encoding="utf-8"
    )
    return {"results": results, "best": results[0] if results else None}


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep Performer capacity")
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=120)
    arguments = parser.parse_args()
    outcome = sweep(arguments.dataset_dir, arguments.output_dir, epochs=arguments.epochs)
    best = outcome["best"]
    if best is None:
        print("no results")
        return
    print(f"\nbest: {best['name']} beats baseline on {best['beats_baseline']}")


if __name__ == "__main__":
    main()
