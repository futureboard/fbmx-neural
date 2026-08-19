"""Train a model from a YAML config.

    python scripts/train.py --config configs/smoke.yaml
    python scripts/train.py --config configs/smoke.yaml --resume checkpoints/smoke/last.pt
    python scripts/train.py --config configs/smoke.yaml --set train.lr=0.001 train.epochs=20

Resuming rebuilds the architecture from the checkpoint, not from the config, so
editing the config between runs cannot silently change the model under a
resume.  Everything else -- optimiser, scheduler, epoch/step counters, RNG
streams -- is restored too: an interrupted Colab session continues where it
stopped rather than starting over.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401  (sys.path)

from fbmx.config import apply_overrides, build_experiment, config_digest, load_config
from fbmx.training.checkpoint import load_checkpoint, model_from_checkpoint
from fbmx.training.trainer import Trainer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train an FBMX model")
    p.add_argument("--config", required=True, type=Path, help="YAML experiment config")
    p.add_argument("--resume", type=Path, default=None, help="checkpoint to continue from")
    p.add_argument("--init-weights", type=Path, default=None,
                   help="start from this checkpoint's weights but with a fresh optimiser, "
                        "schedule and step count (fine-tuning, or moving to another dataset)")
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                   help="dotted config overrides, e.g. train.lr=1e-3")
    p.add_argument("--device", default=None, help="override train.device")
    p.add_argument("--epochs", type=int, default=None, help="override train.epochs")
    p.add_argument("--max-steps", type=int, default=None, help="stop after N optimiser steps")
    p.add_argument("--checkpoint-dir", default=None, help="override train.checkpoint_dir")
    p.add_argument("--seed", type=int, default=None, help="override train.seed")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = apply_overrides(load_config(args.config), args.set)
    train_cfg = cfg.setdefault("train", {})
    for key, value in (
        ("device", args.device),
        ("epochs", args.epochs),
        ("max_steps", args.max_steps),
        ("checkpoint_dir", args.checkpoint_dir),
        ("seed", args.seed),
    ):
        if value is not None:
            train_cfg[key] = value

    print(f"[train] config {args.config} (digest {config_digest(cfg)})")
    exp = build_experiment(cfg)

    if args.resume and args.init_weights:
        raise SystemExit("--resume continues a run; --init-weights starts a new one. Pick one.")
    if args.resume:
        ckpt = load_checkpoint(args.resume, map_location="cpu")
        exp["model"] = model_from_checkpoint(ckpt, device="cpu")
        print(f"[train] architecture rebuilt from {args.resume}")
    elif args.init_weights:
        # Weights only: no optimiser state, no epoch counter, no RNG. This is a
        # new experiment that happens to start somewhere useful, and its
        # checkpoints must not claim to be a continuation of the old run.
        ckpt = load_checkpoint(args.init_weights, map_location="cpu")
        exp["model"] = model_from_checkpoint(ckpt, device="cpu")
        print(f"[train] initialised weights from {args.init_weights} "
              f"(epoch {ckpt.get('epoch')}); optimiser and schedule start fresh")

    trainer = Trainer(
        model=exp["model"],
        train_dataset=exp["train_dataset"],
        val_dataset=exp["val_dataset"],
        cfg=exp["trainer_cfg"],
        loss_fn=exp["loss_fn"],
        config_snapshot=cfg,
    )
    if args.resume:
        trainer.resume(args.resume)

    result = trainer.fit()

    summary = {
        "run": cfg.get("run", {}).get("name", "run"),
        "parameters": exp["model"].num_parameters(),
        "epochs_completed": result["epochs_completed"],
        "global_step": result["global_step"],
        "best_metric": result["best_metric"],
        "monitor": exp["trainer_cfg"].monitor,
        "checkpoints": result["checkpoint_dir"],
    }
    print("\n[train] done")
    print(json.dumps(summary, indent=2, default=str))
    print(f"[train] best checkpoint: {Path(result['checkpoint_dir']) / 'best.pt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
