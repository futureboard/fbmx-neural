"""Export a checkpoint to the ``.fbmx`` container.

    python scripts/export_fbmx.py --checkpoint checkpoints/smoke/best.pt \
        --output models/smoke.fbmx

Licence, dataset provenance and ``model_source_type`` default to whatever the
training run recorded; the flags below can add to that but the intent is that
you cannot get a cleaner-looking provenance by exporting differently.

``--validated`` is a claim about a human having measured or listened to this
model against its reference.  It defaults to false and nothing sets it
automatically.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

import _bootstrap  # noqa: F401  (sys.path)

from fbmx.datasets.base import SOURCE_TYPES
from fbmx.export.fbmx import export_from_checkpoint, read_fbmx
from fbmx.streaming.inference import process_blocked, process_offline


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export a checkpoint to .fbmx")
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--name", default=None)
    p.add_argument("--description", default=None)
    p.add_argument("--author", default=None)
    p.add_argument("--license", default=None, help="SPDX id where possible")
    p.add_argument("--license-url", default=None)
    p.add_argument("--attribution", default=None)
    p.add_argument("--source-type", default=None, choices=SOURCE_TYPES)
    p.add_argument("--tags", nargs="*", default=None)
    p.add_argument("--notes", default=None)
    p.add_argument("--validated", action="store_true",
                   help="assert this model has been validated against its reference")
    p.add_argument("--skip-verify", action="store_true",
                   help="skip the reload + numerical equivalence check")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    metadata = {
        k: v
        for k, v in {
            "name": args.name or args.output.stem,
            "description": args.description,
            "author": args.author,
            "license": args.license,
            "license_url": args.license_url,
            "attribution": args.attribution,
            "model_source_type": args.source_type,
            "tags": args.tags,
            "notes": args.notes,
            "validated": args.validated or None,
        }.items()
        if v is not None
    }

    path, container = export_from_checkpoint(args.checkpoint, args.output, metadata=metadata)
    print(f"[export] wrote {path} ({path.stat().st_size:,} bytes)")
    print()
    print(container.summary())

    if not args.skip_verify:
        # Reload from the file alone and check it computes the same thing as
        # the checkpoint's model. An exporter that drops a tensor still writes
        # a valid-looking file, so this check is not optional in practice.
        from fbmx.training.checkpoint import load_checkpoint, model_from_checkpoint

        reference = model_from_checkpoint(load_checkpoint(args.checkpoint), "cpu").eval()
        restored = read_fbmx(path).build_model("cpu")
        torch.manual_seed(0)
        probe = torch.randn(1, 1, 4096) * 0.3
        params = restored.schema.empty_batch(1)
        with torch.no_grad():
            a = process_offline(reference, probe, params)
            b = process_offline(restored, probe, params)
            c = process_blocked(restored, probe, 128, params)
        print()
        print(f"[export] reload max |checkpoint - fbmx|   {float((a - b).abs().max()):.3e}")
        print(f"[export] reload max |offline  - blocked|  {float((b - c).abs().max()):.3e}")
        if not torch.equal(a, b):
            print("[export] note: differences are float32 kernel noise, not a weight mismatch"
                  if float((a - b).abs().max()) < 1e-5 else
                  "[export] WARNING: reloaded model does not match the checkpoint")

    print()
    print(json.dumps({"uuid": container.model_uuid, "path": str(path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
