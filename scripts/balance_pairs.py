"""Write a manifest that weights the two pair kinds evenly.

The v3 pair set is 320 identity pairs against 1287 interpolation pairs, and the
identity residual is ~15x smaller, so the objective barely notices an identity
error. The first v3 run showed exactly the consequence: interpolation improved
1.95 dB against bypass while the identity case — notes the model should leave
alone — degraded 19 dB. A model that wins on average by colouring correct notes
is not an improvement.

Repeating the identity entries until the two kinds carry comparable weight is
the standard remedy and the honest one: it changes what the objective *asks
for*, and it changes nothing about what the data *is*. Repeated entries point at
the same files under suffixed keys, so nothing is copied and the checksums still
verify.

Usage:
    python scripts/balance_pairs.py --manifest ../solfage-datasets/vsco2-ce/manifests/violin-fbmx.json \\
        --out ../solfage-datasets/vsco2-ce/manifests/violin-fbmx-balanced.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def kind_of(key: str) -> str:
    return "interp" if "__shift" in key else "identity"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--target-ratio",
        type=float,
        default=1.0,
        help="identity weight relative to interp (1.0 = equal)",
    )
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    entries = manifest["entries"]
    identity = [e for e in entries if kind_of(e["key"]) == "identity"]
    interp = [e for e in entries if kind_of(e["key"]) == "interp"]
    if not identity or not interp:
        raise SystemExit(f"expected both kinds; got {len(identity)} identity, {len(interp)} interp")

    repeats = max(1, round(len(interp) * args.target_ratio / len(identity)))
    balanced = list(interp)
    for index in range(repeats):
        for entry in identity:
            copy = dict(entry)
            if index:
                # A distinct key per repeat; `dry`/`wet` still point at the one
                # pair of files on disk, so nothing is duplicated in bytes.
                copy["key"] = f"{entry['key']}~r{index}"
            balanced.append(copy)

    manifest["entries"] = balanced
    info = dict(manifest.get("info", {}))
    info["version"] = f"{info.get('version', 'v3')}-balanced-x{repeats}"
    extra = dict(info.get("extra", {}))
    extra["balance"] = {
        "identity_pairs": len(identity),
        "interp_pairs": len(interp),
        "identity_repeats": repeats,
        "target_ratio": args.target_ratio,
    }
    info["extra"] = extra
    manifest["info"] = info

    args.out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(
        f"{len(identity)} identity x{repeats} + {len(interp)} interp "
        f"= {len(balanced)} entries -> {args.out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
