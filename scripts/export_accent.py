"""Fit, export, and ship the Accent Analyzer.

One command produces everything the DAW needs:

* ``SoloViolinAccent.fbmx`` — the trained correction, in the existing container.
* ``rule_coefficients.json`` — the fitted linear rule and its spread
  calibration, copied into the Futureboard source tree so the runtime's
  fallback is the same rule the model was trained as a correction to.
* ``accent_parity.json`` — feature vectors and rule outputs on real phrases,
  which the Rust implementation is checked against.

## Each split has exactly one job

======================  ==============================================
``train``               fits the rule, and trains the correction
``validation``          stops the training, and fits the spread calibration
``test``                is reported, and is touched by nothing else
======================  ==============================================

The calibration is fitted on validation rather than on train because the
network has already been fitted on train: its predictions there are
optimistically tight, and a spread matched against them would come out too
narrow. Validation is out of sample for the network — the early stop sees only
a scalar score from it — so the spread measured there is the spread the model
really has.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

import _bootstrap  # noqa: F401

from accent.baselines import (
    attach_rule_base,
    fit_linear_baseline,
    fit_rule,
    score_velocity_column,
)
from accent.calibration import fit_calibration, stack
from accent.crossval import _train_fold
from accent.features import (
    ACCENT_FEATURE_SCHEMA_VERSION,
    ACCENT_INPUT_FEATURES,
    ACCENT_TARGETS,
    contexts_from_records,
    fit_prominence_basis,
    input_normalization,
    load_accent_sequences,
    phrase_feature_matrix,
)
from accent.model import AccentAnalyzer, AccentConfig
from accent.train import (
    AccentTrainConfig,
    model_predictions,
    score_predictions,
    uncertainty_calibration,
)
from fbmx.export.fbmx import FBMXMetadata, Normalization, read_fbmx, write_fbmx


def export(
    dataset_dir: str | Path,
    output: str | Path,
    *,
    rule_out: str | Path | None,
    parity_out: str | Path | None,
    config: AccentTrainConfig,
) -> dict[str, Any]:
    dataset_dir = Path(dataset_dir)
    device = torch.device("cpu")

    train_records = [
        json.loads(line)
        for line in (dataset_dir / "notes.train.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    basis = fit_prominence_basis(train_records)

    train = load_accent_sequences(dataset_dir / "notes.train.jsonl", basis)
    validation = load_accent_sequences(dataset_dir / "notes.validation.jsonl", basis)
    test = load_accent_sequences(dataset_dir / "notes.test.jsonl", basis)

    rule = fit_rule(train)
    for group in (train, validation, test):
        attach_rule_base(rule, group)

    model = _train_fold(train, config, device, inner_validation=validation)
    model.eval()

    # Spread match, measured where the network has not been fitted.
    validation_predicted, validation_actual, validation_mask, _ = model_predictions(
        model, validation, device
    )
    calibration = fit_calibration(validation_predicted, validation_actual, validation_mask)

    uniform = fit_linear_baseline(train, name="uniform", feature=None)
    velocity = fit_linear_baseline(
        train, name="velocity", feature="__extra__", extra=score_velocity_column(train)
    )
    meter_only = fit_linear_baseline(train, name="meter", feature="metrical_strength")

    predicted, actual, mask, log_variance = model_predictions(model, test, device)
    calibrated = calibration.apply(predicted)
    inputs = np.concatenate([sequence.inputs for sequence in test], axis=0)
    rule_raw = rule.predict(inputs)

    report: dict[str, Any] = {
        "fitted_on": "urmp-violin train; early-stopped and calibrated on validation",
        "notes": {
            "train": int(sum(s.inputs.shape[0] for s in train)),
            "validation": int(sum(s.inputs.shape[0] for s in validation)),
            "test": int(sum(s.inputs.shape[0] for s in test)),
        },
        "parameter_count": model.parameter_count(),
        "prominence_basis": basis.to_dict(),
        "rule": rule.to_dict(),
        "calibration": calibration.to_dict(),
        "test": {
            # The shipped predictor, spread-matched: what the Accent lane shows.
            "neural_calibrated": score_predictions(calibrated, actual, mask),
            # The same predictor before the presentation transform: what a
            # minimum-error comparison against the baselines should use, since
            # none of them is spread-matched either.
            "neural": score_predictions(predicted, actual, mask),
            "rule": score_predictions(rule_raw, actual, mask),
            "rule_calibrated": score_predictions(
                calibration.apply(rule_raw), actual, mask
            ),
            "meter": score_predictions(meter_only.predict(inputs), actual, mask),
            "velocity": score_predictions(
                velocity.predict(inputs, extra=score_velocity_column(test)), actual, mask
            ),
            "uniform": score_predictions(uniform.predict(inputs), actual, mask),
            "uncertainty": uncertainty_calibration(predicted, actual, mask, log_variance),
            "distribution": {
                name: {
                    "predicted": np.histogram(
                        calibrated[mask[:, index] > 0, index], bins=10, range=(0.0, 1.0)
                    )[0].tolist(),
                    "actual": np.histogram(
                        actual[mask[:, index] > 0, index], bins=10, range=(0.0, 1.0)
                    )[0].tolist(),
                    "predicted_mean": float(calibrated[mask[:, index] > 0, index].mean()),
                    "predicted_std": float(calibrated[mask[:, index] > 0, index].std()),
                    "actual_mean": float(actual[mask[:, index] > 0, index].mean()),
                    "actual_std": float(actual[mask[:, index] > 0, index].std()),
                }
                for index, name in enumerate(ACCENT_TARGETS)
            },
        },
    }

    # How far the correction actually moved from the rule. A correction of zero
    # is a real and reportable outcome — it means the early stop found nothing
    # worth keeping — and it must not be hidden behind a model file that exists.
    correction = predicted - rule_raw
    report["correction_magnitude"] = {
        name: {
            "mean_abs": float(np.abs(correction[mask[:, index] > 0, index]).mean()),
            "max_abs": float(np.abs(correction[mask[:, index] > 0, index]).max()),
        }
        for index, name in enumerate(ACCENT_TARGETS)
    }

    metadata = FBMXMetadata.from_dict(
        {
            "name": "Solo Violin Accent Analyzer",
            "description": (
                "Estimates how strongly each note of a score should be "
                "emphasised, and by which means: prominence, attack, agogic "
                "and timbral weight, plus the model's own confidence. Produces "
                "no audio. Its output is a *correction* to the fitted linear "
                "rule shipped alongside it, not a standalone prediction."
            ),
            "attribution": (
                "Trained on the University of Rochester Multi-Modal Music "
                "Performance (URMP) dataset."
            ),
            "tags": ["violin", "accent", "analysis", "expression"],
            "dataset": {
                "name": "urmp-accent",
                "source": "URMP isolated violin stems, notes, and F0 annotations",
                "source_type": "derived",
                "license": "CC BY-NC-SA 4.0",
                "version": "urmp-violin-2",
            },
            "training": {
                "epochs": config.epochs,
                "monitor": "inner-split early stop; leave-one-session-out cross-validated",
                "torch_version": torch.__version__,
            },
            "notes": (
                "Accent analysis model. Reads a sequence of score feature "
                "vectors and returns a per-note correction to the linear rule. "
                "The host must add the rule's prediction and then apply the "
                "spread calibration; running this model alone yields "
                "corrections centred on zero, not accents."
            ),
        }
    )

    path = write_fbmx(
        output,
        model,
        metadata,
        normalization=Normalization(),
        extra={
            "accent": {
                "input_features": list(ACCENT_INPUT_FEATURES),
                "targets": list(ACCENT_TARGETS),
                "feature_schema_version": ACCENT_FEATURE_SCHEMA_VERSION,
                "residual_of": "linear rule; see rule_coefficients.json",
                "prominence_basis": basis.to_dict(),
                "rule": rule.to_dict(),
                "calibration": calibration.to_dict(),
            }
        },
    )
    container = read_fbmx(path)
    report["export"] = {
        "path": str(path),
        "bytes": Path(path).stat().st_size,
        "model_type": container.header["model"]["type"],
        "parameter_count": container.header["model"]["parameter_count"],
    }

    if rule_out:
        payload = rule.to_dict()
        payload["fitted_on"] = report["fitted_on"]
        payload["feature_schema_version"] = ACCENT_FEATURE_SCHEMA_VERSION
        payload["calibration"] = calibration.to_dict()
        Path(rule_out).parent.mkdir(parents=True, exist_ok=True)
        Path(rule_out).write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        report["rule_written_to"] = str(rule_out)

    if parity_out:
        report["parity_written_to"] = str(write_parity_fixture(parity_out, rule, test))

    return report


def write_parity_fixture(path, rule, sequences) -> Path:
    """A fixture the Rust implementation is checked against.

    Real phrases from the held-out split, with their score facts, the feature
    vector this side computed, and the rule components that follow. A parity
    test over random inputs proves the arithmetic agrees; one over real notes
    also proves the *feature extraction* agrees, which is where two
    independently written implementations actually drift.

    Features are recomputed from the truncated excerpt rather than sliced out of
    the whole part's matrix. Several of them compare a note with a window of its
    neighbours, and the Rust side only ever sees the excerpt.
    """

    cases: list[dict[str, Any]] = []
    for sequence in sequences[:3]:
        rows = sequence.records[:24]
        if len(rows) < 8:
            continue
        beat_seconds = 60.0 / max(float(rows[0].get("tempo_bpm") or 120.0), 1e-6)
        inputs = phrase_feature_matrix(contexts_from_records(rows))
        components = rule.predict(inputs)
        cases.append(
            {
                "part_id": sequence.part_id,
                "tempo_bpm": float(rows[0].get("tempo_bpm") or 120.0),
                "time_signature": list(rows[0]["time_signature"]),
                "notes": [
                    {
                        "pitch": int(record["score_pitch"]),
                        "onset_beats": round(
                            float(record["score_onset_seconds"]) / beat_seconds, 6
                        ),
                        "duration_beats": round(
                            float(record["score_duration_seconds"]) / beat_seconds, 6
                        ),
                    }
                    for record in rows
                ],
                "features": [[round(float(v), 6) for v in row] for row in inputs],
                # Uncalibrated: the runtime applies the spread match after the
                # learned correction, so the fixture pins the raw rule.
                "rule_components": [[round(float(v), 6) for v in row] for row in components],
            }
        )

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "feature_schema_version": ACCENT_FEATURE_SCHEMA_VERSION,
                "input_features": list(ACCENT_INPUT_FEATURES),
                "targets": list(ACCENT_TARGETS),
                "cases": cases,
            },
            indent=1,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--output", required=True, help="destination .fbmx")
    parser.add_argument("--rule-out", help="destination rule_coefficients.json")
    parser.add_argument("--parity-out", help="destination accent_parity.json")
    parser.add_argument("--hidden-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--window-stride", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260828)
    arguments = parser.parse_args()

    report = export(
        arguments.dataset_dir,
        arguments.output,
        rule_out=arguments.rule_out,
        parity_out=arguments.parity_out,
        config=AccentTrainConfig(
            dataset_dir=arguments.dataset_dir,
            checkpoint_dir=str(Path(arguments.output).parent),
            hidden_size=arguments.hidden_size,
            epochs=arguments.epochs,
            learning_rate=arguments.learning_rate,
            weight_decay=arguments.weight_decay,
            window_stride=arguments.window_stride,
            seed=arguments.seed,
        ),
    )
    Path(arguments.output).with_suffix(".report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, default=float) + "\n", encoding="utf-8"
    )
    print(f"{report['export']['path']}  {report['export']['bytes']} bytes")
    print(f"parameters {report['parameter_count']}")
    print(f"\n{'head':12s} {'method':18s} {'MAE':>8s} {'pearson':>9s} {'spearman':>9s} {'std':>8s}")
    for head in ACCENT_TARGETS:
        for method in (
            "neural_calibrated",
            "neural",
            "rule",
            "meter",
            "velocity",
            "uniform",
        ):
            row = report["test"][method][head]
            if not row.get("n"):
                continue
            print(
                f"{head if method == 'neural_calibrated' else '':12s} {method:18s} "
                f"{row['mae']:8.4f} {row.get('pearson', float('nan')):9.3f} "
                f"{row.get('spearman', float('nan')):9.3f} "
                f"{row.get('predicted_std', float('nan')):8.4f}"
            )
        magnitude = report["correction_magnitude"][head]
        print(
            f"{'':12s} {'correction':18s} {magnitude['mean_abs']:8.4f} "
            f"{'':9s} {'':9s} {magnitude['max_abs']:8.4f}   (mean / max |neural - rule|)"
        )


if __name__ == "__main__":
    main()
