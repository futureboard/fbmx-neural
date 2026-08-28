"""Ship just the rule half of the Accent Analyzer.

The rule and the parity fixture are what the Futureboard build embeds, and
neither depends on the trained correction: the rule is fitted by least squares
in a second, and the fixture is a record of what the *feature extraction*
produced. Keeping this separate from `export_accent.py` means the DAW can be
built and its parity test run without a training run first, which matters
because the parity test is the thing that catches the two feature
implementations drifting apart.

    python scripts/export_accent_rule.py \
        --dataset-dir datasets/urmp-violin \
        --rule-out  <futureboard>/src/solfege/accent/rule_coefficients.json \
        --parity-out <futureboard>/src/solfege/accent/accent_parity.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import _bootstrap  # noqa: F401

from accent.baselines import attach_rule_base, fit_rule
from accent.calibration import fit_calibration, stack
from accent.features import (
    ACCENT_FEATURE_SCHEMA_VERSION,
    ACCENT_INPUT_FEATURES,
    ACCENT_TARGETS,
    contexts_from_records,
    phrase_feature_matrix,
    fit_prominence_basis,
    load_accent_sequences,
)
from accent.train import score_predictions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--rule-out", required=True)
    parser.add_argument("--parity-out", required=True)
    parser.add_argument("--report-out")
    arguments = parser.parse_args()

    dataset_dir = Path(arguments.dataset_dir)
    records = []
    for name in ("train", "validation"):
        records.extend(
            json.loads(line)
            for line in (dataset_dir / f"notes.{name}.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        )
    basis = fit_prominence_basis(records)

    pool = []
    for name in ("train", "validation"):
        pool.extend(load_accent_sequences(dataset_dir / f"notes.{name}.jsonl", basis))
    test = load_accent_sequences(dataset_dir / "notes.test.jsonl", basis)

    rule = fit_rule(pool)
    attach_rule_base(rule, pool)
    attach_rule_base(rule, test)

    pool_inputs = np.concatenate([s.inputs for s in pool], axis=0)
    pool_targets, pool_mask = stack(pool)
    raw = rule.predict(pool_inputs)
    calibration = fit_calibration(raw, pool_targets, pool_mask)

    test_inputs = np.concatenate([s.inputs for s in test], axis=0)
    test_targets, test_mask = stack(test)
    test_raw = rule.predict(test_inputs)
    test_calibrated = calibration.apply(test_raw)

    payload = rule.to_dict()
    payload["fitted_on"] = "urmp-violin train+validation"
    payload["feature_schema_version"] = ACCENT_FEATURE_SCHEMA_VERSION
    payload["calibration"] = calibration.to_dict()
    Path(arguments.rule_out).parent.mkdir(parents=True, exist_ok=True)
    Path(arguments.rule_out).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    # ── parity fixture ───────────────────────────────────────────────────
    cases = []
    for sequence in test[:3]:
        rows = sequence.records[:24]
        if len(rows) < 8:
            continue
        beat_seconds = 60.0 / max(float(rows[0].get("tempo_bpm") or 120.0), 1e-6)
        # Recomputed from the truncated note list, not sliced out of the whole
        # part's matrix. Several features compare a note with a window of its
        # neighbours, so a note six from the end of the excerpt has a different
        # neighbourhood in the excerpt than it does in the part -- and the Rust
        # side only ever sees the excerpt. Slicing produced a fixture the
        # runtime could not reproduce, which the parity test caught.
        inputs = phrase_feature_matrix(contexts_from_records(rows))
        components = np.stack(
            [
                np.asarray(
                    [
                        sum(
                            rule.coefficients[target][name]
                            * inputs[row, ACCENT_INPUT_FEATURES.index(name)]
                            for name in rule.coefficients[target]
                        )
                        + rule.intercepts[target]
                        for target in ACCENT_TARGETS
                    ]
                )
                for row in range(len(rows))
            ]
        )
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
    Path(arguments.parity_out).parent.mkdir(parents=True, exist_ok=True)
    Path(arguments.parity_out).write_text(
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

    report = {
        "fitted_on": payload["fitted_on"],
        "calibration": calibration.to_dict(),
        "test_raw": score_predictions(test_raw, test_targets, test_mask),
        "test_calibrated": score_predictions(test_calibrated, test_targets, test_mask),
        "parity_cases": len(cases),
        "parity_notes": sum(len(case["notes"]) for case in cases),
    }
    if arguments.report_out:
        Path(arguments.report_out).write_text(
            json.dumps(report, indent=2, sort_keys=True, default=float) + "\n", encoding="utf-8"
        )

    print(f"rule    -> {arguments.rule_out}")
    print(f"parity  -> {arguments.parity_out}  ({report['parity_notes']} notes)")
    print(f"\n{'head':12s} {'gain':>6s} {'MAE raw':>9s} {'MAE cal':>9s} "
          f"{'rho raw':>9s} {'rho cal':>9s} {'std raw':>9s} {'std cal':>9s}")
    for head in ACCENT_TARGETS:
        raw_row = report["test_raw"][head]
        cal_row = report["test_calibrated"][head]
        print(
            f"{head:12s} {calibration.gains[head]:6.2f} "
            f"{raw_row['mae']:9.4f} {cal_row['mae']:9.4f} "
            f"{raw_row['spearman']:9.3f} {cal_row['spearman']:9.3f} "
            f"{raw_row['predicted_std']:9.4f} {cal_row['predicted_std']:9.4f}"
        )


if __name__ == "__main__":
    main()
