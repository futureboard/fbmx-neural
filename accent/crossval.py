"""Leave-one-session-out cross-validation for the Accent Analyzer.

A single held-out split cannot answer "did this learn anything" on a corpus of
twenty-three performance groups. It was tried: on the fixed validation split
*no* method beat predicting the training mean — the fitted rule scored a
prominence MAE of 0.1810 against uniform's 0.1809 — while on the test split the
same rule reached a rank correlation of 0.33. Those two numbers describe the
splits, not the methods.

So every session cluster takes a turn being held out, and the verdict is how
many folds each method wins. This is the same methodology the Performer's
`crossval.py` established for the same corpus and the same reason.

What is compared, per fold and per head:

    uniform     the fold's own training mean
    velocity    least squares from score velocity
    meter       least squares from metrical strength
    rule        the nine-feature ridge, fitted on the fold's training data
    neural      that rule plus the fold's learned correction

Every one of them is refitted inside the fold. A rule fitted once on everything
and reused across folds would have seen each fold's held-out session.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from datasets.urmp.split import session_clusters
from .baselines import attach_rule_base, fit_linear_baseline, fit_rule, score_velocity_column
from .features import (
    ACCENT_TARGETS,
    AccentSequence,
    fit_prominence_basis,
    input_normalization,
    load_accent_sequences,
)
from .model import AccentAnalyzer, AccentConfig
from .train import (
    TARGET_WEIGHTS,
    AccentTrainConfig,
    _batch,
    compute_loss,
    model_predictions,
    score_predictions,
    seed_everything,
    window_sequences,
)

METHODS = ("neural", "rule", "meter", "velocity", "uniform")


def _cluster_of_part(dataset_dir: Path) -> tuple[dict[str, str], list[dict[str, Any]]]:
    records = [
        json.loads(line)
        for line in (dataset_dir / "notes.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    clusters = session_clusters(records)
    return {record["part_id"]: clusters[record["group_id"]] for record in records}, records


def _train_fold(
    train_sequences: Sequence[AccentSequence],
    config: AccentTrainConfig,
    device: torch.device,
    *,
    inner_validation: Sequence[AccentSequence] | None = None,
) -> AccentAnalyzer:
    """Train one fold, stopping when an inner split stops improving.

    The epoch count must not be a chosen number. A fixed budget was tried and
    the choice mattered enormously: at six epochs the learned correction beat
    the rule on six of eight head-fold comparisons, and at thirty it was worse
    than the rule on the first two folds it reached. Picking whichever of those
    looked better and then reporting cross-validation on the same folds would be
    selecting the hyperparameter on the test data.

    So each fold holds out one further session cluster from its own training
    set, trains until that stops improving, and restores the best state. The
    outer fold is never looked at. Where there is nothing to learn the early
    stop fires almost immediately and the residual formulation leaves the model
    at — or close to — the rule, which is the right answer rather than a
    fallback.
    """

    mean, std = input_normalization(train_sequences)
    model = AccentAnalyzer(
        AccentConfig(
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            dropout=config.dropout,
        )
    ).to(device)
    model.set_normalization(torch.from_numpy(mean), torch.from_numpy(std))

    windows = window_sequences(
        list(train_sequences), window=config.window_notes, stride=config.window_stride
    )
    weights = torch.tensor(
        [TARGET_WEIGHTS[name] for name in ACCENT_TARGETS], dtype=torch.float32, device=device
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)

    order = list(range(len(windows)))
    best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    best_score = float("inf")
    if inner_validation:
        # Epoch zero is the rule, exactly. Scoring it first means the early stop
        # has something to beat and a model that never improves on the rule is
        # returned *as* the rule rather than as a slightly-worse-than-rule
        # model that happened to be the least bad epoch.
        best_score = _inner_score(model, inner_validation, device)
    since_improvement = 0

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

        if not inner_validation:
            continue
        score = _inner_score(model, inner_validation, device)
        if score < best_score - 1e-6:
            best_score = score
            best_state = {
                key: value.detach().clone() for key, value in model.state_dict().items()
            }
            since_improvement = 0
        else:
            since_improvement += 1
            if since_improvement >= INNER_PATIENCE:
                break

    if inner_validation:
        model.load_state_dict(best_state)
    return model


#: Epochs without an inner improvement before a fold stops.
INNER_PATIENCE = 4


def _inner_score(
    model: AccentAnalyzer, sequences: Sequence[AccentSequence], device: torch.device
) -> float:
    """Weighted MAE on the inner split, in the units the lane shows."""

    predicted, actual, mask, _ = model_predictions(model, sequences, device)
    scored = score_predictions(predicted, actual, mask)
    return float(
        sum(
            scored[name]["mae"] * TARGET_WEIGHTS[name]
            for name in ACCENT_TARGETS
            if scored[name].get("n") and np.isfinite(scored[name]["mae"])
        )
    )


def crossval(config: AccentTrainConfig, *, folds: int | None = None) -> dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_dir = Path(config.dataset_dir)
    cluster_of_part, records = _cluster_of_part(dataset_dir)

    # The whole corpus, as one pool. Splitting is by fold from here on.
    basis = fit_prominence_basis(records)
    everything = load_accent_sequences(dataset_dir / "notes.jsonl", basis)
    clusters = sorted({cluster_of_part[sequence.part_id] for sequence in everything})
    if folds is not None:
        clusters = clusters[:folds]

    fold_reports: list[dict[str, Any]] = []
    for fold, held_out in enumerate(clusters):
        seed_everything(config.seed + fold)
        test = [s for s in everything if cluster_of_part[s.part_id] == held_out]
        train = [s for s in everything if cluster_of_part[s.part_id] != held_out]
        if not test or not train:
            continue

        rule = fit_rule(train)
        attach_rule_base(rule, train)
        attach_rule_base(rule, test)
        uniform = fit_linear_baseline(train, name="uniform", feature=None)
        velocity = fit_linear_baseline(
            train, name="velocity", feature="__extra__", extra=score_velocity_column(train)
        )
        meter_only = fit_linear_baseline(train, name="meter", feature="metrical_strength")

        # One further cluster out of the fold's own training set, for the early
        # stop. Rotated by fold index so no single session is always the inner
        # validation and therefore never trained on.
        inner_clusters = sorted({cluster_of_part[s.part_id] for s in train})
        inner_held = inner_clusters[fold % len(inner_clusters)]
        inner = [s for s in train if cluster_of_part[s.part_id] == inner_held]
        outer_train = [s for s in train if cluster_of_part[s.part_id] != inner_held]

        model = _train_fold(outer_train, config, device, inner_validation=inner)
        predicted, actual, mask, _ = model_predictions(model, test, device)

        inputs = np.concatenate([s.inputs for s in test], axis=0)
        entry: dict[str, Any] = {
            "fold": fold,
            "held_out": held_out,
            "parts": sorted(s.part_id for s in test),
            "notes": int(sum(s.inputs.shape[0] for s in test)),
            "neural": score_predictions(predicted, actual, mask),
            "rule": score_predictions(rule.predict(inputs), actual, mask),
            "uniform": score_predictions(uniform.predict(inputs), actual, mask),
            "velocity": score_predictions(
                velocity.predict(inputs, extra=score_velocity_column(test)), actual, mask
            ),
            "meter": score_predictions(meter_only.predict(inputs), actual, mask),
        }
        fold_reports.append(entry)
        print(
            f"fold {fold:2d}  {held_out:11s} {entry['notes']:4d} notes  "
            + "  ".join(
                f"{name[:4]} {entry[name]['prominence']['mae']:.4f}" for name in METHODS
            )
        )

    summary: dict[str, Any] = {
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config": asdict(config),
        "prominence_basis": basis.to_dict(),
        "folds": len(fold_reports),
        "per_fold": fold_reports,
        "verdict": {},
    }

    for head in ACCENT_TARGETS:
        row: dict[str, Any] = {}
        for method in METHODS:
            maes = [
                report[method][head]["mae"]
                for report in fold_reports
                if report[method][head].get("n")
            ]
            rhos = [
                report[method][head].get("spearman", float("nan"))
                for report in fold_reports
                if report[method][head].get("n")
            ]
            finite = [value for value in rhos if np.isfinite(value)]
            row[method] = {
                "mae_mean": float(np.mean(maes)) if maes else float("nan"),
                "mae_std": float(np.std(maes)) if maes else float("nan"),
                "spearman_mean": float(np.mean(finite)) if finite else float("nan"),
            }
        # The verdict the Performer's report used: not "is the average better"
        # — one bad fold moves an average — but "on how many of the folds does
        # this win". A method that wins nine of fourteen has found something; a
        # method that wins seven has found a coin.
        for method in ("neural", "rule"):
            row[f"{method}_beats_uniform"] = sum(
                1
                for report in fold_reports
                if report[method][head].get("n")
                and report[method][head]["mae"] < report["uniform"][head]["mae"]
            )
        row["neural_beats_rule"] = sum(
            1
            for report in fold_reports
            if report["neural"][head].get("n")
            and report["neural"][head]["mae"] < report["rule"][head]["mae"]
        )
        summary["verdict"][head] = row

    return summary


def print_verdict(summary: dict[str, Any]) -> None:
    folds = summary["folds"]
    print(f"\n== leave-one-session-out, {folds} folds ==")
    print(
        f"{'head':12s} {'method':10s} {'MAE mean':>9s} {'+/-':>7s} {'rho':>7s} "
        f"{'beats uniform':>14s} {'beats rule':>11s}"
    )
    for head, row in summary["verdict"].items():
        for method in METHODS:
            detail = row[method]
            beats_uniform = (
                f"{row[method + '_beats_uniform']}/{folds}"
                if method in ("neural", "rule")
                else ""
            )
            beats_rule = f"{row['neural_beats_rule']}/{folds}" if method == "neural" else ""
            print(
                f"{head if method == 'neural' else '':12s} {method:10s} "
                f"{detail['mae_mean']:9.4f} {detail['mae_std']:7.4f} "
                f"{detail['spearman_mean']:7.3f} {beats_uniform:>14s} {beats_rule:>11s}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--hidden-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--window-stride", type=int, default=16)
    parser.add_argument("--folds", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20260828)
    arguments = parser.parse_args()

    summary = crossval(
        AccentTrainConfig(
            dataset_dir=arguments.dataset_dir,
            checkpoint_dir=arguments.out,
            hidden_size=arguments.hidden_size,
            epochs=arguments.epochs,
            learning_rate=arguments.learning_rate,
            weight_decay=arguments.weight_decay,
            window_stride=arguments.window_stride,
            seed=arguments.seed,
        ),
        folds=arguments.folds,
    )
    out = Path(arguments.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "crossval.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=float) + "\n", encoding="utf-8"
    )
    print_verdict(summary)


if __name__ == "__main__":
    main()
