"""Does accent actually condition the Performer?

The question acceptance criterion 12 asks, and the one a diagram of the pipeline
cannot answer. A model can be *given* an input and ignore it; if this Performer
ignores its accent columns then the Accent lane is a control that changes a
number in the project and nothing else, however many arrows point from it.

Two measurements, and they answer different halves:

**Sensitivity.** Run the trained model over the held-out split twice — once with
the accent it was given, once with every accent neutral — and report how far
each performance target moves. This is the direct question: turning the control
must turn something.

**Worth.** Train the same architecture on the same split with the accent columns
held permanently neutral, and compare. Sensitivity says the model reacts;
this says the reaction is *informed* rather than the model having learned to
route noise. A model can be sensitive to a useless input.

And a caveat the second measurement alone would hide. In **training** the accent
input is measured from the same performance whose timing is the target, so some
of any improvement is the target reaching the input by the back door. At
**runtime** accent comes from the analyser, which is a much weaker signal. So the
comparison is run a third way — with the analyser's own predictions substituted
for the measured accent — and it is that number, not the optimistic one, that
says what a user gets.

Per-target, all of them are reported against the scale the target is measured
in, so a 2 ms onset move is not confused with a 2-cent pitch move.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

import _bootstrap  # noqa: F401

from accent.features import contexts_from_records, fit_prominence_basis
from accent.predict import ShippedAnalyzer
from performer.features import (
    ACCENT_INPUTS,
    INPUT_FEATURES,
    NEUTRAL_ACCENT,
    TARGETS,
    TARGET_SCALES,
    VIBRATO_PRESENT_INDEX,
    load_sequences,
)
from performer.model import Performer, PerformerConfig
from performer.train import TrainConfig, evaluate, train


def _neutralised(sequence) -> np.ndarray:
    """The same feature matrix with every accent column at neutral."""

    inputs = sequence.inputs.copy()
    for name in ACCENT_INPUTS:
        inputs[:, INPUT_FEATURES.index(name)] = NEUTRAL_ACCENT
    return inputs


def _with_predicted_accent(sequences, analyzer: ShippedAnalyzer) -> None:
    """Replace each sequence's measured accent columns with the analyser's.

    In place, on a copy of the feature matrix, so the caller can evaluate the
    same model against the input it will really be given.
    """

    columns = [INPUT_FEATURES.index(name) for name in ACCENT_INPUTS]
    for sequence in sequences:
        predicted = analyzer.analyze(contexts_from_records(sequence.records))
        inputs = sequence.inputs.copy()
        for row, accent in enumerate(predicted):
            for column, name in zip(columns, ACCENT_INPUTS):
                inputs[row, column] = accent[name.removeprefix("accent_")]
        sequence.inputs = inputs


def sensitivity(model: Performer, sequences) -> dict[str, Any]:
    model.eval()
    with_accent: list[np.ndarray] = []
    without_accent: list[np.ndarray] = []
    with torch.no_grad():
        for sequence in sequences:
            given = torch.from_numpy(sequence.inputs).unsqueeze(0)
            flat = torch.from_numpy(_neutralised(sequence)).unsqueeze(0)
            with_accent.append(model(given).squeeze(0).numpy())
            without_accent.append(model(flat).squeeze(0).numpy())
    given = np.concatenate(with_accent)
    flat = np.concatenate(without_accent)
    delta = given - flat

    out: dict[str, Any] = {}
    for index, name in enumerate(TARGETS):
        scale = TARGET_SCALES[name]
        out[name] = {
            "mean_abs_delta": float(np.abs(delta[:, index]).mean() * scale),
            "max_abs_delta": float(np.abs(delta[:, index]).max() * scale),
            # As a fraction of how much the target itself varies across the
            # split: a 1 ms move on a target whose spread is 60 ms is not a
            # conditioning path anyone would hear.
            "relative_to_spread": float(
                np.abs(delta[:, index]).mean() / max(given[:, index].std(), 1e-9)
            ),
            "unit_scale": scale,
        }
    presence = 1.0 / (1.0 + np.exp(-given[:, VIBRATO_PRESENT_INDEX]))
    presence_flat = 1.0 / (1.0 + np.exp(-flat[:, VIBRATO_PRESENT_INDEX]))
    out["vibrato_present"] = {
        "mean_abs_delta": float(np.abs(presence - presence_flat).mean()),
        "decisions_flipped": int(((presence >= 0.5) != (presence_flat >= 0.5)).sum()),
        "notes": int(presence.size),
    }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--checkpoint", required=True, help="the accent-conditioned Performer")
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--train-control",
        action="store_true",
        help="also train a Performer with accent held neutral, for the worth half",
    )
    parser.add_argument("--hidden-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument(
        "--accent-rule",
        help="rule_coefficients.json; evaluates the model against the accent it "
        "will really be given at runtime",
    )
    parser.add_argument("--accent-checkpoint")
    arguments = parser.parse_args()

    dataset_dir = Path(arguments.dataset_dir)
    train_records = [
        json.loads(line)
        for line in (dataset_dir / "notes.train.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    basis = fit_prominence_basis(train_records)
    test = load_sequences(dataset_dir / "notes.test.jsonl", basis)
    validation = load_sequences(dataset_dir / "notes.validation.jsonl", basis)

    state = torch.load(arguments.checkpoint, map_location="cpu", weights_only=False)
    model = Performer(PerformerConfig.from_dict(state["config"]))
    model.load_state_dict(state["model"])

    report: dict[str, Any] = {
        "checkpoint": str(arguments.checkpoint),
        "input_size": model.config.input_size,
        "accent_inputs": list(ACCENT_INPUTS),
        "sensitivity": {
            "test": sensitivity(model, test),
            "validation": sensitivity(model, validation),
        },
        "conditioned": {
            "test": evaluate(model, test, torch.device("cpu")),
        },
    }

    if arguments.accent_rule:
        # The same trained model, evaluated against the analyser's predictions
        # rather than the measured accent it trained on. This is the runtime
        # number.
        analyzer = ShippedAnalyzer.load(
            arguments.accent_rule, arguments.accent_checkpoint
        )
        predicted_test = load_sequences(dataset_dir / "notes.test.jsonl", basis)
        _with_predicted_accent(predicted_test, analyzer)
        report["predicted_accent"] = {
            "test": evaluate(model, predicted_test, torch.device("cpu")),
            "note": "accent from the shipped analyser, as at runtime",
        }

    if arguments.train_control:
        # The same architecture, the same split, the same seed — trained on a
        # dataset whose accent columns are all neutral. Anything the
        # accent-conditioned model does better than this, it does because of
        # accent.
        control_dir = Path(arguments.out) / "control"
        control_dir.mkdir(parents=True, exist_ok=True)
        import performer.features as performer_features

        # Neutralise the accent columns for the whole control run by making the
        # dataset loader attach a neutral accent to every record. Same
        # architecture, same split, same seed, same everything else.
        original = performer_features.attach_measured_accent

        def neutral(records, basis):  # noqa: ARG001
            for record in records:
                record['accent'] = {
                    name: performer_features.NEUTRAL_ACCENT
                    for name in ('prominence', 'attack', 'agogic', 'timbre')
                }

        performer_features.attach_measured_accent = neutral
        try:
            control = train(
                TrainConfig(
                    dataset_dir=str(dataset_dir),
                    checkpoint_dir=str(control_dir),
                    hidden_size=arguments.hidden_size,
                    epochs=arguments.epochs,
                )
            )
        finally:
            performer_features.attach_measured_accent = original
        report["control"] = {"test": control["test"], "note": "accent columns neutral"}

    out = Path(arguments.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "accent-conditioning.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, default=float) + "\n", encoding="utf-8"
    )

    print(f"Performer input size {report['input_size']} "
          f"({len(ACCENT_INPUTS)} of them accent)\n")
    print(f"{'target':22s} {'unit':>12s} {'mean |delta|':>14s} {'max |delta|':>13s} "
          f"{'vs spread':>10s}")
    for name in TARGETS:
        row = report["sensitivity"]["test"][name]
        print(
            f"{name:22s} {row['unit_scale']:12.3f} {row['mean_abs_delta']:14.5f} "
            f"{row['max_abs_delta']:13.5f} {row['relative_to_spread']:10.3f}"
        )
    presence = report["sensitivity"]["test"]["vibrato_present"]
    print(
        f"{'vibrato_present':22s} {'probability':>12s} {presence['mean_abs_delta']:14.5f} "
        f"{'':13s} {presence['decisions_flipped']:6d} flipped of {presence['notes']}"
    )


if __name__ == "__main__":
    main()
