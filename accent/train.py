"""Train the Accent Analyzer and measure it against every baseline it must beat.

Metrics are reported three ways per head, because they answer different
questions and a model can win one while losing another:

``mae``
    How far off, in the 0..1 units a user sees in the Accent lane.

``pearson``
    Whether the model's ups and downs are the performance's ups and downs.

``spearman``
    Whether it gets the *order* right — which note of a phrase is the most
    prominent, regardless of by how much. For accent this is arguably the real
    question, and it is the one an MAE-only report hides: a model that squashes
    every prediction toward the mean scores a good MAE and a terrible rank
    correlation, which is exactly the failure section 49 asks to be caught.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch import nn

from .baselines import (
    LinearBaseline,
    RuleAnalyzer,
    attach_rule_base,
    fit_linear_baseline,
    fit_rule,
    score_velocity_column,
)
from .features import (
    ACCENT_FEATURE_SCHEMA_VERSION,
    ACCENT_INPUT_FEATURES,
    ACCENT_TARGETS,
    CONFIDENCE_INDEX,
    AccentSequence,
    ProminenceBasis,
    fit_prominence_basis,
    input_normalization,
    load_accent_sequences,
)
from .model import LOG_VARIANCE_RANGE, AccentAnalyzer, AccentConfig

#: Weight per head in the loss.
#:
#: `prominence` is what the user sees and what conditions everything
#: downstream, so it carries the most. The other three are still trained on
#: their own measurements rather than derived from prominence, because a note
#: can be agogically emphasised without being loud and the whole point of
#: keeping four components is that they can disagree.
TARGET_WEIGHTS: dict[str, float] = {
    "prominence": 2.0,
    "attack": 1.0,
    "agogic": 1.0,
    "timbre": 0.75,
}

#: Weight of the Gaussian negative log-likelihood that trains the uncertainty
#: head. Low: it is a side channel, and letting it compete with the regression
#: would trade prediction accuracy for well-calibrated wrongness.
UNCERTAINTY_WEIGHT = 0.2


@dataclass
class AccentTrainConfig:
    dataset_dir: str
    checkpoint_dir: str
    window_notes: int = 32
    window_stride: int = 8
    hidden_size: int = 16
    num_layers: int = 1
    dropout: float = 0.0
    epochs: int = 200
    learning_rate: float = 3e-3
    weight_decay: float = 1e-3
    grad_clip: float = 1.0
    seed: int = 20260828
    huber_beta: float = 0.15
    patience: int = 30


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def window_sequences(
    sequences: Sequence[AccentSequence], *, window: int, stride: int
) -> list[AccentSequence]:
    """Slice parts into overlapping windows so an epoch is more than 21 steps.

    Shorter windows than the Performer uses (32 notes against 48) because the
    grouping this model reads is shorter: the median group is 11 notes, so a
    48-note window is mostly other phrases.
    """

    out: list[AccentSequence] = []
    for sequence in sequences:
        count = sequence.inputs.shape[0]
        if count <= window:
            out.append(sequence)
            continue
        for start in range(0, count - window + 1, stride):
            stop = start + window
            out.append(
                AccentSequence(
                    part_id=f"{sequence.part_id}[{start}:{stop}]",
                    piece_id=sequence.piece_id,
                    inputs=sequence.inputs[start:stop],
                    targets=sequence.targets[start:stop],
                    mask=sequence.mask[start:stop],
                    records=sequence.records[start:stop],
                    base=None if sequence.base is None else sequence.base[start:stop],
                )
            )
    return out


def _batch(sequence: AccentSequence, device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "inputs": torch.from_numpy(sequence.inputs).unsqueeze(0).to(device),
        # The network predicts the rule's *error*, so this is what it is scored
        # against during training. The reported metrics use the finished
        # prediction (rule + correction); see `model_predictions`.
        "targets": torch.from_numpy(sequence.residual_targets()).unsqueeze(0).to(device),
        "base": torch.from_numpy(
            sequence.base
            if sequence.base is not None
            else np.zeros_like(sequence.targets)
        )
        .unsqueeze(0)
        .to(device),
        "mask": torch.from_numpy(sequence.mask).unsqueeze(0).to(device),
    }


def compute_loss(
    prediction: torch.Tensor,
    batch: dict[str, torch.Tensor],
    *,
    huber_beta: float,
    weights: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    regression = prediction[..., : len(ACCENT_TARGETS)]
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

    # Gaussian NLL for the uncertainty head, against a *detached* residual. The
    # head learns how wrong the prediction is; the prediction must not learn to
    # be conveniently wrong where the head has already said it would be.
    log_variance = prediction[..., CONFIDENCE_INDEX].clamp(*LOG_VARIANCE_RANGE)
    residual = (regression[..., 0] - batch["targets"][..., 0]).detach() ** 2
    keep = batch["mask"][..., 0]
    nll = 0.5 * (log_variance + residual / torch.exp(log_variance))
    uncertainty_loss = (nll * keep).sum() / keep.sum().clamp_min(1e-6)

    total = regression_loss + UNCERTAINTY_WEIGHT * uncertainty_loss
    return total, {
        "regression": float(regression_loss.detach()),
        "uncertainty": float(uncertainty_loss.detach()),
    }


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Rank correlation, computed here rather than pulled in from scipy.

    The repository has no scipy dependency and adding one for a rank transform
    is not a concrete need. Ties get average ranks, which is what
    `scipy.stats.spearmanr` does and what makes the coefficient correct on the
    binary-ish features this is applied to.
    """

    if a.size < 3 or b.size < 3:
        return float("nan")

    def rank(values: np.ndarray) -> np.ndarray:
        order = np.argsort(values, kind="mergesort")
        ranks = np.empty(values.size, dtype=np.float64)
        ranks[order] = np.arange(values.size, dtype=np.float64)
        # Average the ranks inside each run of equal values.
        sorted_values = values[order]
        start = 0
        for index in range(1, values.size + 1):
            if index == values.size or sorted_values[index] != sorted_values[start]:
                if index - start > 1:
                    ranks[order[start:index]] = ranks[order[start:index]].mean()
                start = index
        return ranks

    ranked_a, ranked_b = rank(a), rank(b)
    if ranked_a.std() < 1e-12 or ranked_b.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(ranked_a, ranked_b)[0, 1])


def score_predictions(
    predicted: np.ndarray, actual: np.ndarray, mask: np.ndarray
) -> dict[str, Any]:
    """Per-head MAE, Pearson, Spearman, and the predicted spread."""

    metrics: dict[str, Any] = {}
    for index, name in enumerate(ACCENT_TARGETS):
        keep = mask[:, index] > 0
        if not keep.any():
            metrics[name] = {"n": 0, "mae": float("nan")}
            continue
        p = predicted[keep, index].astype(np.float64)
        a = actual[keep, index].astype(np.float64)
        pearson = (
            float(np.corrcoef(p, a)[0, 1]) if p.std() > 1e-9 and a.std() > 1e-9 else float("nan")
        )
        metrics[name] = {
            "n": int(keep.sum()),
            "mae": float(np.abs(p - a).mean()),
            "pearson": pearson,
            "spearman": _spearman(p, a),
            "predicted_std": float(p.std()),
            "actual_std": float(a.std()),
            "predicted_mean": float(p.mean()),
        }
    return metrics


def model_predictions(
    model: AccentAnalyzer, sequences: Sequence[AccentSequence], device: torch.device
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """`(predicted, actual, mask, log_variance)` over whole parts."""

    model.eval()
    predicted_rows, actual_rows, mask_rows, variance_rows = [], [], [], []
    with torch.no_grad():
        for sequence in sequences:
            batch = _batch(sequence, device)
            out = model(batch["inputs"]).squeeze(0).cpu().numpy()
            base = (
                sequence.base
                if sequence.base is not None
                else np.zeros_like(sequence.targets)
            )
            # The finished prediction, which is what every reported number is
            # about: the rule's answer plus the correction the network learned.
            predicted_rows.append(
                np.clip(out[:, : len(ACCENT_TARGETS)] + base, 0.0, 1.0)
            )
            variance_rows.append(out[:, CONFIDENCE_INDEX])
            actual_rows.append(sequence.targets)
            mask_rows.append(sequence.mask)
    return (
        np.concatenate(predicted_rows),
        np.concatenate(actual_rows),
        np.concatenate(mask_rows),
        np.concatenate(variance_rows),
    )


def baseline_report(
    predictor, sequences: Sequence[AccentSequence], *, extra: np.ndarray | None = None
) -> dict[str, Any]:
    inputs = np.concatenate([sequence.inputs for sequence in sequences], axis=0)
    actual = np.concatenate([sequence.targets for sequence in sequences], axis=0)
    mask = np.concatenate([sequence.mask for sequence in sequences], axis=0)
    if isinstance(predictor, LinearBaseline):
        predicted = predictor.predict(inputs, extra=extra)
    else:
        predicted = predictor.predict(inputs)
    return score_predictions(predicted, actual, mask)


def uncertainty_calibration(
    predicted: np.ndarray, actual: np.ndarray, mask: np.ndarray, log_variance: np.ndarray
) -> dict[str, Any]:
    """Does the confidence head actually predict where the model is wrong?

    Reported as the correlation between the head's predicted standard deviation
    and the realised absolute error on `prominence`. A confidence output that
    does not correlate with error is decoration, and saying so is cheaper than
    letting a UI show it as if it meant something.
    """

    keep = mask[:, 0] > 0
    if keep.sum() < 8:
        return {"n": 0}
    deviation = np.exp(0.5 * np.clip(log_variance[keep], *LOG_VARIANCE_RANGE))
    error = np.abs(predicted[keep, 0] - actual[keep, 0])
    if deviation.std() < 1e-9:
        return {"n": int(keep.sum()), "pearson": float("nan"), "note": "head is constant"}
    return {
        "n": int(keep.sum()),
        "pearson": float(np.corrcoef(deviation, error)[0, 1]),
        "spearman": _spearman(deviation, error),
        "mean_predicted_sigma": float(deviation.mean()),
        "mean_absolute_error": float(error.mean()),
    }


def train(config: AccentTrainConfig) -> dict[str, Any]:
    seed_everything(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_dir = Path(config.dataset_dir)

    train_records = [
        json.loads(line)
        for line in (dataset_dir / "notes.train.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    # Fitted on training only. A basis fitted over the whole corpus would let
    # the test split's covariance structure into the definition of the target
    # the model is scored against.
    basis = fit_prominence_basis(train_records)

    train_sequences = load_accent_sequences(dataset_dir / "notes.train.jsonl", basis)
    validation_sequences = load_accent_sequences(dataset_dir / "notes.validation.jsonl", basis)
    test_sequences = load_accent_sequences(dataset_dir / "notes.test.jsonl", basis)
    if not train_sequences:
        raise RuntimeError("no training sequences; run scripts/build_urmp.py first")

    # The rule is fitted first and on training only: it is the floor the
    # network learns a correction to, so it has to be the *same* floor on every
    # split or "neural beats rule" would be comparing two different rules.
    rule = fit_rule(train_sequences)
    for group in (train_sequences, validation_sequences, test_sequences):
        attach_rule_base(rule, group)

    mean, std = input_normalization(train_sequences)
    windows = window_sequences(
        train_sequences, window=config.window_notes, stride=config.window_stride
    )

    model_config = AccentConfig(
        hidden_size=config.hidden_size, num_layers=config.num_layers, dropout=config.dropout
    )
    model = AccentAnalyzer(model_config).to(device)
    model.set_normalization(torch.from_numpy(mean), torch.from_numpy(std))

    weights = torch.tensor(
        [TARGET_WEIGHTS[name] for name in ACCENT_TARGETS], dtype=torch.float32, device=device
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)

    checkpoint_dir = Path(config.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_path = checkpoint_dir / "training_log.jsonl"
    log_path.write_text("", encoding="utf-8")

    order = list(range(len(windows)))
    best_score, best_epoch, since_improvement = float("inf"), -1, 0

    for epoch in range(config.epochs):
        model.train()
        random.shuffle(order)
        epoch_loss = 0.0
        for index in order:
            batch = _batch(windows[index], device)
            optimizer.zero_grad(set_to_none=True)
            loss, _ = compute_loss(
                model(batch["inputs"]), batch, huber_beta=config.huber_beta, weights=weights
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            epoch_loss += float(loss.detach())
        scheduler.step()

        predicted, actual, mask, _ = model_predictions(model, validation_sequences, device)
        validation = score_predictions(predicted, actual, mask)
        # Selected on weighted MAE. Not on correlation, which is scale-free and
        # therefore indifferent to a model that predicts the right shape at a
        # tenth of the right amplitude — and amplitude is what reaches the ear.
        score = float(
            sum(
                validation[name]["mae"] * TARGET_WEIGHTS[name]
                for name in ACCENT_TARGETS
                if validation[name]["n"] > 0 and np.isfinite(validation[name]["mae"])
            )
        )
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "epoch": epoch,
                        "train_loss": epoch_loss / max(len(order), 1),
                        "validation_score": score,
                        "prominence_mae": validation["prominence"]["mae"],
                        "prominence_spearman": validation["prominence"]["spearman"],
                        "lr": scheduler.get_last_lr()[0],
                    }
                )
                + "\n"
            )

        if score < best_score - 1e-6:
            best_score, best_epoch, since_improvement = score, epoch, 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "config": model_config.to_dict(),
                    "train_config": asdict(config),
                    "input_mean": mean,
                    "input_std": std,
                    "prominence_basis": basis.to_dict(),
                    "rule": rule.to_dict(),
                    "epoch": epoch,
                    "validation_score": score,
                    "feature_schema_version": ACCENT_FEATURE_SCHEMA_VERSION,
                },
                checkpoint_dir / "best.pt",
            )
        else:
            since_improvement += 1
            if since_improvement >= config.patience:
                break

    state = torch.load(checkpoint_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(state["model"])

    uniform = fit_linear_baseline(train_sequences, name="uniform", feature=None)
    velocity = fit_linear_baseline(
        train_sequences,
        name="velocity",
        feature="__extra__",
        extra=score_velocity_column(train_sequences),
    )
    meter_only = fit_linear_baseline(train_sequences, name="meter", feature="metrical_strength")

    report: dict[str, Any] = {
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config": asdict(config),
        "model": model_config.to_dict(),
        "parameter_count": model.parameter_count(),
        "best_epoch": best_epoch,
        "best_validation_score": best_score,
        "prominence_basis": basis.to_dict(),
        "rule": rule.to_dict(),
        "baselines": {
            name: predictor.to_dict()
            for name, predictor in (
                ("uniform", uniform),
                ("velocity", velocity),
                ("meter", meter_only),
            )
        },
        "sequences": {
            "train_parts": len(train_sequences),
            "train_windows": len(windows),
            "validation_parts": len(validation_sequences),
            "test_parts": len(test_sequences),
        },
        "notes": {
            "train": int(sum(s.inputs.shape[0] for s in train_sequences)),
            "validation": int(sum(s.inputs.shape[0] for s in validation_sequences)),
            "test": int(sum(s.inputs.shape[0] for s in test_sequences)),
        },
    }

    for split, sequences in (
        ("validation", validation_sequences),
        ("test", test_sequences),
    ):
        predicted, actual, mask, log_variance = model_predictions(model, sequences, device)
        report[split] = {
            "neural": score_predictions(predicted, actual, mask),
            "rule": baseline_report(rule, sequences),
            "uniform": baseline_report(uniform, sequences),
            "velocity": baseline_report(
                velocity, sequences, extra=score_velocity_column(sequences)
            ),
            "meter": baseline_report(meter_only, sequences),
            "uncertainty": uncertainty_calibration(predicted, actual, mask, log_variance),
            "distribution": {
                name: {
                    "predicted_histogram": np.histogram(
                        predicted[mask[:, index] > 0, index], bins=10, range=(0.0, 1.0)
                    )[0].tolist(),
                    "actual_histogram": np.histogram(
                        actual[mask[:, index] > 0, index], bins=10, range=(0.0, 1.0)
                    )[0].tolist(),
                }
                for index, name in enumerate(ACCENT_TARGETS)
            },
        }

    (checkpoint_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, default=float) + "\n", encoding="utf-8"
    )
    return report


def print_report(report: dict[str, Any]) -> None:
    print(f"parameters      {report['parameter_count']}")
    print(f"best epoch      {report['best_epoch']}")
    basis = report["prominence_basis"]
    print(
        "prominence      "
        + "  ".join(f"{k}={v:+.3f}" for k, v in basis["loadings"].items())
        + f"   (PC1, {basis['explained_variance']:.1%} of variance)"
    )
    for split in ("validation", "test"):
        print(f"\n== {split} ==")
        header = f"{'head':12s} {'method':10s} {'MAE':>8s} {'pearson':>9s} {'spearman':>9s} {'pred std':>9s}"
        print(header)
        for name in ACCENT_TARGETS:
            for method in ("neural", "rule", "meter", "velocity", "uniform"):
                row = report[split][method][name]
                if not row.get("n"):
                    continue
                print(
                    f"{name if method == 'neural' else '':12s} {method:10s} "
                    f"{row['mae']:8.4f} {row.get('pearson', float('nan')):9.3f} "
                    f"{row.get('spearman', float('nan')):9.3f} {row.get('predicted_std', float('nan')):9.3f}"
                )
        calibration = report[split]["uncertainty"]
        if calibration.get("n"):
            print(
                f"{'confidence':12s} {'':10s} {'':8s} "
                f"{calibration.get('pearson', float('nan')):9.3f} "
                f"{calibration.get('spearman', float('nan')):9.3f}"
                "   (predicted sigma vs realised |error|)"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the Accent Analyzer")
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--hidden-size", type=int, default=16)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--window-notes", type=int, default=32)
    parser.add_argument("--window-stride", type=int, default=8)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260828)
    arguments = parser.parse_args()

    print_report(
        train(
            AccentTrainConfig(
                dataset_dir=arguments.dataset_dir,
                checkpoint_dir=arguments.checkpoint_dir,
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
    )


if __name__ == "__main__":
    main()
