"""Inspect a ``.fbmx`` file.

    python scripts/inspect_fbmx.py models/smoke.fbmx
    python scripts/inspect_fbmx.py models/smoke.fbmx --tensors --json

Reads the container without building a model, verifies both checksums, and
prints the header.  This is also the reference for what a Rust reader has to
parse, so it deliberately touches every field the format defines.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401  (sys.path)

from fbmx.export.fbmx import read_fbmx


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Inspect a .fbmx model container")
    p.add_argument("path", type=Path)
    p.add_argument("--tensors", action="store_true", help="list every tensor")
    p.add_argument("--json", action="store_true", help="dump the raw header as JSON")
    p.add_argument("--no-verify", action="store_true", help="skip checksum verification")
    p.add_argument("--load", action="store_true",
                   help="also rebuild the model and run one block through it")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    container = read_fbmx(args.path, verify=not args.no_verify)

    if args.json:
        print(json.dumps(container.header, indent=2))
        return 0

    size = args.path.stat().st_size
    print(f"file           {args.path}  ({size:,} bytes)")
    print(f"checksums      {'verified' if not args.no_verify else 'NOT CHECKED'}")
    print(container.summary())

    meta = container.metadata
    print()
    print("metadata")
    print(f"  name         {meta.name}")
    print(f"  description  {meta.description or '-'}")
    print(f"  author       {meta.author or '-'}")
    print(f"  attribution  {meta.attribution or '-'}")
    print(f"  tags         {', '.join(meta.tags) if meta.tags else '-'}")
    if meta.dataset:
        print("  dataset")
        for key in ("name", "source", "source_type", "license", "version", "checksum"):
            if meta.dataset.get(key):
                print(f"    {key:<10} {meta.dataset[key]}")
    if meta.training:
        print("  training")
        for key, value in meta.training.items():
            if key == "trainer":
                continue
            print(f"    {key:<10} {value}")
    if meta.notes:
        print(f"  notes        {meta.notes}")

    if args.tensors:
        print()
        print("tensors")
        total = 0
        for entry in container.header["tensors"]:
            total += entry["nbytes"]
            shape = "x".join(str(s) for s in entry["shape"]) or "scalar"
            print(f"  {entry['name']:<28} {entry['dtype']:<4} {shape:<16} "
                  f"{entry['nbytes']:>9,} B @ {entry['offset']:,}")
        print(f"  {'total':<28} {'':<4} {'':<16} {total:>9,} B")

    if args.load:
        import torch

        model = container.build_model("cpu")
        params = container.schema.empty_batch(1)
        with torch.no_grad():
            y, _ = model(torch.zeros(1, 1, 128), params, None)
        print()
        print(f"load check     rebuilt {model.model_type} "
              f"({model.num_parameters():,} parameters), "
              f"128-sample block -> {tuple(y.shape)}, finite={bool(torch.isfinite(y).all())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
