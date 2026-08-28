"""Create Solfege bowed-string -> VSCO Solo Violin FBMX pairs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401

from datasets.vsco.fbmx_pairs import MAX_PAIR_SECONDS, build_vsco_fbmx_pairs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("../solfage-datasets/vsco2-ce"),
        help="prepared VSCO workspace root",
    )
    parser.add_argument(
        "--sfm",
        type=Path,
        default=Path("../artifacts/solfage/SoloViolin/SoloViolin.sfm"),
        help="compiled self-contained voicebank SFM",
    )
    parser.add_argument("--renderer", type=Path, help="solfage-model executable")
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=MAX_PAIR_SECONDS,
        help=(
            "maximum reference duration per training pair; the cut is faded, "
            f"default {MAX_PAIR_SECONDS} s"
        ),
    )
    parser.add_argument("--force", action="store_true", help="rerender existing pair files")
    args = parser.parse_args()
    result = build_vsco_fbmx_pairs(
        args.dataset_root,
        sfm=args.sfm,
        renderer=args.renderer,
        force=args.force,
        max_seconds=args.max_seconds,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
