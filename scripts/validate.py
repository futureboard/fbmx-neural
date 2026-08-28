"""Evaluate a checkpoint.

    python scripts/validate.py --checkpoint checkpoints/smoke/best.pt

Reports the standard metrics (ESR, MAE, RMSE, peak error, DC offset) on the
validation split, and -- because it is the property that decides whether the
model can ship at all -- the offline/blocked streaming difference at a range of
block sizes.

The dataset comes from the config snapshot stored inside the checkpoint, so
this works on a checkpoint from another machine with no config file to hand.
Pass ``--config`` to evaluate against a different dataset instead.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

import _bootstrap  # noqa: F401  (sys.path)

from fbmx.config import apply_overrides, build_experiment, load_config
from fbmx.datasets.base import collate_pairs
from fbmx.device import auto_device, describe_device
from fbmx.streaming.inference import streaming_equivalence
from fbmx.training.checkpoint import load_checkpoint, model_from_checkpoint
from fbmx.training.metrics import assert_finite, waveform_metrics

BLOCK_SIZES = (16, 32, 64, 128, 256, 512, 1024)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate an FBMX checkpoint")
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--config", type=Path, default=None,
                   help="evaluate against this config's data instead of the stored one")
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    p.add_argument("--split", default="val")
    p.add_argument("--device", default="auto")
    p.add_argument("--chunk-size", type=int, default=4096)
    p.add_argument("--warmup", type=int, default=1024)
    p.add_argument("--max-sequences", type=int, default=8)
    p.add_argument("--skip-streaming", action="store_true")
    p.add_argument("--json", type=Path, default=None, help="also write metrics here")
    return p.parse_args()


@torch.no_grad()
def main() -> int:
    args = parse_args()
    device = auto_device(args.device)
    ckpt = load_checkpoint(args.checkpoint, map_location="cpu")
    model = model_from_checkpoint(ckpt, device=device).eval()

    cfg = load_config(args.config) if args.config else dict(ckpt.get("config") or {})
    if not cfg:
        raise SystemExit(
            f"{args.checkpoint} carries no config snapshot; pass --config explicitly"
        )
    cfg = apply_overrides(cfg, args.set)
    cfg.setdefault("data", {})["val_split"] = args.split
    exp = build_experiment(cfg)
    dataset = exp["val_dataset"] or exp["train_dataset"]
    loss_fn = exp["loss_fn"].to(device)

    print(f"[validate] device      {describe_device(device)}")
    print(f"[validate] checkpoint  {args.checkpoint} "
          f"(epoch {ckpt.get('epoch')}, step {ckpt.get('global_step')})")
    print(f"[validate] model       {model.model_type}, {model.num_parameters():,} parameters")
    print(f"[validate] data        {dataset.describe()}")

    n = min(args.max_sequences, len(dataset))
    items = [dataset[i] for i in range(n)]
    reports = []
    first_probe = None
    for item in items:
        # VSCO clips are intentionally variable-length. Evaluate one item at a
        # time instead of padding/collating them into a fake batch; training's
        # DataLoader already uses batch_size=1 for the same reason.
        batch = collate_pairs([item])
        dry = batch["dry"].to(device)
        wet = batch["wet"].to(device)
        params = batch["params"].to(device)

        # Chunked and stateful: the same path inference takes.
        state = model.init_state(1, device=device)
        preds = []
        for start in range(0, dry.shape[-1], args.chunk_size):
            pred, state = model(dry[..., start : start + args.chunk_size], params, state)
            preds.append(pred)
        pred = torch.cat(preds, dim=-1)
        assert_finite(pred, "prediction")

        skip = min(args.warmup, pred.shape[-1] - 1)
        loss, parts = loss_fn(pred[..., skip:], wet[..., skip:])
        metrics = waveform_metrics(pred[..., skip:], wet[..., skip:])
        metrics["loss"] = float(loss)
        reports.append((metrics, parts))
        if first_probe is None:
            first_probe = (dry, params)

    metrics = {
        key: sum(report[0].get(key, 0.0) for report in reports) / max(len(reports), 1)
        for key in {key for report in reports for key in report[0]}
    }
    parts = {
        key: sum(report[1].get(key, 0.0) for report in reports) / max(len(reports), 1)
        for key in {key for report in reports for key in report[1]}
    }

    print("\n[validate] metrics")
    for key, value in {**metrics, **{f"loss_{k}": v for k, v in parts.items()}}.items():
        print(f"  {key:<12} {value: .6g}")

    report = {
        "checkpoint": str(args.checkpoint),
        "split": args.split,
        "sequences": n,
        "metrics": metrics,
        "loss_terms": parts,
        "parameters": model.num_parameters(),
    }

    if not args.skip_streaming:
        probe_dry, probe_params = first_probe
        probe = probe_dry[:1, :, : min(probe_dry.shape[-1], 16384)]
        diffs = streaming_equivalence(model, probe, BLOCK_SIZES, probe_params)
        print("\n[validate] streaming equivalence (max |offline - blocked|)")
        for block, diff in diffs.items():
            print(f"  block {block:<5} {diff:.3e}")
        report["streaming_max_abs_diff"] = diffs

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\n[validate] wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
