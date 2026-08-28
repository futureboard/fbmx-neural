"""Does the model beat passing the signal through unchanged?

The only question that decides whether a correction model may ship. A residual
model always *can* be trained, and a training loss always *can* come down; both
are true of a model that makes the sound worse. The bypass baseline is the
comparison that cannot be argued with: on data the model never saw, is
``model(dry)`` closer to ``wet`` than ``dry`` already was?

Reported per metric, and per pair kind, because the two kinds ask different
things of the model:

``identity``
    the note played from its own recorded entry. The right answer is "change
    almost nothing". A model that improves the interpolation cases by damaging
    these is not an improvement.
``interp``
    the same recording read at a shifted pitch by the realtime interpolator,
    against a 64-tap float64 reference read. This is where the correction has
    work to do.

A model that does not beat bypass on the aggregate, or that makes ``identity``
materially worse, must not be shipped — say so and keep the correction off,
rather than shipping a regression and calling the mix control a fix.

Usage:
    python scripts/bypass_baseline.py --checkpoint checkpoints/RUN/best.pt --split test
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

import _bootstrap  # noqa: F401  (sys.path)

from fbmx.config import build_experiment, load_config
from fbmx.datasets.base import collate_pairs
from fbmx.training.checkpoint import load_checkpoint, model_from_checkpoint


def metrics(target: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    error = target - candidate
    energy = float(np.sum(target.astype(np.float64) ** 2))
    error_energy = float(np.sum(error.astype(np.float64) ** 2))
    esr = error_energy / energy if energy > 1e-20 else float("nan")
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error.astype(np.float64) ** 2))),
        "esr": esr,
        "peak_error": float(np.max(np.abs(error))) if error.size else 0.0,
        "dc_error": float(np.mean(error)),
        # Kept raw so the aggregate can pool energy across sequences instead of
        # averaging ratios — see `aggregate`.
        "_error_energy": error_energy,
        "_target_energy": energy,
        "_samples": float(target.size),
    }


def aggregate(rows: list[dict[str, float]]) -> dict[str, float]:
    """Pool the error energy; do not average per-sequence ratios.

    ESR is a ratio, and this material has near-silent sequences — the tail after
    a pizzicato, a short spiccato padded to the cap. Averaging per-sequence ESR
    lets one quiet sequence with a tiny denominator outweigh every loud one, so
    the headline number would be decided by the silence between notes rather
    than by the notes. Summing error energy and target energy separately and
    dividing once is the definition that matches what a listener integrates.

    The per-sequence median is still reported, because a pooled figure can hide
    a model that is excellent on loud material and destructive on quiet.
    """
    if not rows:
        return {}
    out: dict[str, float] = {"sequences": len(rows)}
    error_energy = sum(r["_error_energy"] for r in rows)
    target_energy = sum(r["_target_energy"] for r in rows)
    samples = sum(r["_samples"] for r in rows)
    out["esr"] = error_energy / target_energy if target_energy > 1e-20 else float("nan")
    if out["esr"] == out["esr"] and out["esr"] > 0:
        out["esr_db"] = 10.0 * math.log10(out["esr"])
    out["rmse"] = math.sqrt(error_energy / samples) if samples else float("nan")
    for key in ("mae", "esr", "peak_error"):
        values = np.array([r[key] for r in rows if r[key] == r[key]])
        if values.size:
            out[f"{key}_median"] = float(np.median(values))
    out["mae_mean"] = float(np.mean([r["mae"] for r in rows]))
    out["peak_error_max"] = float(np.max([r["peak_error"] for r in rows]))
    out["abs_dc_error_mean"] = float(np.mean([abs(r["dc_error"]) for r in rows]))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--chunk-size", type=int, default=4096)
    parser.add_argument("--max-sequences", type=int, default=0, help="0 = all")
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    payload = load_checkpoint(args.checkpoint, map_location="cpu")
    model = model_from_checkpoint(payload).eval()
    # `--config` matters for more than convenience: two runs are only
    # comparable if they are scored on the *same* sequences. A run trained on a
    # rebalanced manifest carries that manifest in its checkpoint, and its test
    # split is rebalanced too — so evaluating from the checkpoint's own config
    # would move the goalposts between runs. Point every comparison at one
    # config and the split is fixed.
    config = load_config(args.config) if args.config else payload["config"]
    # Whole sequences, not fixed segments: the acceptance question is about the
    # rendered note, and a segment boundary is an artefact of training.
    config = dict(config)
    config["data"] = {**config["data"], "segment_length": None, "train_split": args.split}
    # `build_experiment` returns a dict of pieces, and `train_split` is what
    # selects which split `train_dataset` holds — set above to the split under
    # test, so this evaluates the held-out data by construction.
    experiment = build_experiment(config)
    dataset = experiment["train_dataset"]

    by_kind: dict[str, dict[str, list[dict[str, float]]]] = {}
    with torch.no_grad():
        for index in range(len(dataset)):
            if args.max_sequences and index >= args.max_sequences:
                break
            batch = collate_pairs([dataset[index]])
            dry = batch["dry"]
            wet = batch["wet"].numpy()[0, 0]
            params = batch["params"]

            # Chunked and stateful — the path inference actually takes.
            state = model.init_state(1)
            pieces = []
            for start in range(0, dry.shape[-1], args.chunk_size):
                piece, state = model(dry[..., start : start + args.chunk_size], params, state)
                pieces.append(piece)
            wet_hat = torch.cat(pieces, dim=-1).numpy()[0, 0]
            dry_np = dry.numpy()[0, 0]

            key = dataset.entries[dataset._index[index][0]].key
            kind = "interp" if "__shift" in key else "identity"
            slot = by_kind.setdefault(kind, {"bypass": [], "model": []})
            slot["bypass"].append(metrics(wet, dry_np))
            slot["model"].append(metrics(wet, wet_hat))

    report: dict[str, object] = {
        "format": "fbmx-bypass-baseline-v1",
        "checkpoint": str(args.checkpoint),
        "split": args.split,
        "by_kind": {},
        "all": {},
    }
    every: dict[str, list[dict[str, float]]] = {"bypass": [], "model": []}
    for kind, slot in sorted(by_kind.items()):
        report["by_kind"][kind] = {
            "bypass": aggregate(slot["bypass"]),
            "model": aggregate(slot["model"]),
        }
        every["bypass"].extend(slot["bypass"])
        every["model"].extend(slot["model"])
    report["all"] = {"bypass": aggregate(every["bypass"]), "model": aggregate(every["model"])}

    print(
        f"{'':<12}{'':<9}{'ESR pooled':>12}{'ESR dB':>9}{'ESR med':>12}"
        f"{'MAE mean':>12}{'|DC| mean':>12}{'peak err':>11}"
    )
    for scope in list(report["by_kind"]) + ["all"]:
        block = report["by_kind"].get(scope, report["all"])
        for which in ("bypass", "model"):
            stats = block[which]
            if not stats:
                continue
            print(
                f"{scope:<12}{which:<9}{stats.get('esr', float('nan')):>12.6f}"
                f"{stats.get('esr_db', float('nan')):>9.2f}"
                f"{stats.get('esr_median', float('nan')):>12.6f}"
                f"{stats.get('mae_mean', float('nan')):>12.6f}"
                f"{stats.get('abs_dc_error_mean', float('nan')):>12.2e}"
                f"{stats.get('peak_error_max', float('nan')):>11.5f}"
            )
        bypass = block["bypass"].get("esr")
        model_esr = block["model"].get("esr")
        if bypass and model_esr:
            change = 10.0 * math.log10(model_esr / bypass)
            verdict = "BETTER" if model_esr < bypass else "WORSE"
            print(f"{'':<12}{'-> ' + verdict:<9}{change:>+11.2f} dB vs bypass\n")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    overall = report["all"]
    identity = report["by_kind"].get("identity", {})
    beats_overall = overall["model"].get("esr", float("inf")) < overall["bypass"].get(
        "esr", float("inf")
    )
    # A model that wins on average by wrecking the notes it should not touch is
    # not an improvement. Allow the identity case to get a little worse — it
    # starts near zero, so any change is a large ratio — but not by an order of
    # magnitude.
    identity_ok = True
    if identity:
        before = identity["bypass"].get("esr", 0.0)
        after = identity["model"].get("esr", float("inf"))
        identity_ok = after <= max(before * 10.0, 1e-4)
        if not identity_ok:
            print(
                f"identity case degraded {10.0 * math.log10(after / max(before, 1e-20)):+.1f} dB "
                "— the model is colouring notes that were already correct"
            )
    ships = beats_overall and identity_ok
    print("VERDICT:", "model beats bypass" if ships else "model is WORSE than bypass — do not ship")
    return 0 if ships else 1


if __name__ == "__main__":
    raise SystemExit(main())
