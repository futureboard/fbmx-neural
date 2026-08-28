"""Build and split the URMP violin performance dataset.

Previously done by hand at a REPL, which meant the holdout choice lived only in
the report it produced. It is here now so a rebuild reproduces the *same* split
rather than a new one: the Performer already trained under this assignment, and
a dataset rebuilt with different pieces in test makes every number in its report
incomparable.

    python scripts/build_urmp.py --root "<URMP root>" --out datasets/urmp-violin
"""

from __future__ import annotations

import argparse
import json

import _bootstrap  # noqa: F401

from datasets.urmp.build import build
from datasets.urmp.split import write_split

#: Pieces forced into the test split, fixed since the first build. Both are
#: pieces whose violin parts appear nowhere else in the corpus, so holding them
#: out costs no training material that a reused take would have supplied anyway.
HOLDOUT_PIECES = {"13_Hark_vn_vn_va", "32_Fugue_vn_vn_va_vc"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="URMP dataset root")
    parser.add_argument("--out", required=True, help="output dataset directory")
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="re-split an existing notes.jsonl without re-reading the audio",
    )
    arguments = parser.parse_args()

    if not arguments.skip_build:
        summary = build(arguments.root, arguments.out)
        print(
            f"built {summary['total_notes']} notes from {summary['parts_kept']} parts "
            f"({summary['performance_groups']} performance groups)"
        )

    report = write_split(arguments.out, holdout_pieces=HOLDOUT_PIECES)
    for name in ("train", "validation", "test"):
        detail = report["splits"][name]
        print(f"{name:11s} {detail['notes']:5d} notes  {len(detail['pieces']):2d} pieces")
    print(json.dumps(report["group_assignment"], indent=1))


if __name__ == "__main__":
    main()
