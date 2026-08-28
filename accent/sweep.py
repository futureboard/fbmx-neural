"""Search the Accent Analyzer's capacity and learning rate.

The first run of `accent.train` selected its epoch-0 checkpoint: validation got
worse from the first epoch onward, and the nine-feature linear rule beat the
network on every head. That is the signature of a model with more capacity and
more steps than 4958 notes can support, not of a task with no signal in it —
the same 4958 notes let a ridge regression reach a test correlation of 0.34 on
prominence.

This sweeps the two knobs that decide it and reports validation only. The test
split is not touched here; picking a configuration on test is how a held-out
split stops being held out.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from .features import ACCENT_TARGETS
from .train import AccentTrainConfig, TARGET_WEIGHTS, train


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--out", required=True, help="directory for per-run checkpoints")
    parser.add_argument("--hidden", type=int, nargs="+", default=[4, 8, 16, 32])
    parser.add_argument("--lr", type=float, nargs="+", default=[3e-4, 1e-3, 3e-3])
    parser.add_argument("--weight-decay", type=float, nargs="+", default=[1e-3, 1e-2])
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--window-stride", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--patience", type=int, default=25)
    arguments = parser.parse_args()

    out = Path(arguments.out)
    out.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []

    for hidden in arguments.hidden:
        for learning_rate in arguments.lr:
            for decay in arguments.weight_decay:
                tag = f"h{hidden}_lr{learning_rate:g}_wd{decay:g}"
                report = train(
                    AccentTrainConfig(
                        dataset_dir=arguments.dataset_dir,
                        checkpoint_dir=str(out / tag),
                        hidden_size=hidden,
                        learning_rate=learning_rate,
                        weight_decay=decay,
                        dropout=arguments.dropout,
                        window_stride=arguments.window_stride,
                        epochs=arguments.epochs,
                        patience=arguments.patience,
                    )
                )
                validation = report["validation"]
                rule = validation["rule"]
                neural = validation["neural"]
                row = {
                    "tag": tag,
                    "hidden": hidden,
                    "lr": learning_rate,
                    "weight_decay": decay,
                    "parameters": report["parameter_count"],
                    "best_epoch": report["best_epoch"],
                    "score": report["best_validation_score"],
                    "beats_rule": sum(
                        1
                        for name in ACCENT_TARGETS
                        if neural[name].get("n") and neural[name]["mae"] < rule[name]["mae"]
                    ),
                    **{
                        f"{name}_mae": neural[name].get("mae", float("nan"))
                        for name in ACCENT_TARGETS
                    },
                    **{
                        f"{name}_rho": neural[name].get("spearman", float("nan"))
                        for name in ACCENT_TARGETS
                    },
                }
                rows.append(row)
                print(
                    f"{tag:24s} params {row['parameters']:5d} epoch {row['best_epoch']:3d} "
                    f"score {row['score']:.4f} beats-rule {row['beats_rule']}/4 "
                    f"prom rho {row['prominence_rho']:+.3f}"
                )

    rows.sort(key=lambda row: row["score"])
    (out / "sweep.json").write_text(
        json.dumps(rows, indent=2, default=float) + "\n", encoding="utf-8"
    )
    print("\nbest by validation score:")
    for row in rows[:5]:
        print(f"  {row['tag']:24s} {row['score']:.4f}  beats-rule {row['beats_rule']}/4")


if __name__ == "__main__":
    main()
