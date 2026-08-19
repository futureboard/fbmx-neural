"""Emit golden vectors for the Python <-> Rust parity test.

    python scripts/make_golden.py --fbmx models/smoke.fbmx \
        --out tests/golden/smoke_lstm32 --set drive=0.7 mix=0.9 mode=hard

Writes ``<out>.json`` next to a copy of the model at ``<out>.fbmx``, so the
Rust test has one self-contained directory to point at and neither side can
drift onto a different model without the other noticing (the UUID is recorded
and checked).

The reference values come from the *reloaded* container, not from the training
checkpoint: what Rust must match is what the file says, and any exporter bug
between the two is exactly what this test exists to catch.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import torch

import _bootstrap  # noqa: F401  (sys.path)

from fbmx.export.fbmx import read_fbmx


def probe_signal(n: int, sample_rate: int, seed: int = 12345) -> np.ndarray:
    """A deterministic signal that exercises the parts that break.

    Silence (state decay), an impulse and a hard step (attack), a low-frequency
    sine (intra-cycle behaviour) and noise (broadband). Nothing here is
    musical; it is chosen so that a state bug cannot hide in a quiet passage.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n) / sample_rate
    x = np.zeros(n, dtype=np.float64)

    quarter = n // 4
    x[:quarter] = 0.3 * np.sin(2 * np.pi * 50.0 * t[:quarter])
    x[quarter : 2 * quarter] = rng.standard_normal(quarter) * 0.25
    x[2 * quarter : 3 * quarter] = 0.0
    x[3 * quarter :] = 0.8 * np.sin(2 * np.pi * 1000.0 * t[3 * quarter :])
    x[quarter // 2] = 0.95  # impulse
    x[2 * quarter : 2 * quarter + 64] = 0.9  # step into silence and back
    return x.astype(np.float32)


def parse_set(pairs: list[str]) -> dict[str, object]:
    out: dict[str, object] = {}
    for item in pairs:
        if "=" not in item:
            raise SystemExit(f"--set expects name=value, got {item!r}")
        key, _, raw = item.partition("=")
        try:
            out[key] = float(raw)
        except ValueError:
            out[key] = raw
    return out


def main() -> int:
    p = argparse.ArgumentParser(description="Export Python->Rust golden vectors")
    p.add_argument("--fbmx", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path, help="output path without extension")
    p.add_argument("--set", nargs="*", default=[], metavar="NAME=VALUE",
                   help="parameter values; defaults are used for the rest")
    p.add_argument("--samples", type=int, default=2048)
    p.add_argument("--seed", type=int, default=12345)
    args = p.parse_args()

    container = read_fbmx(args.fbmx)
    model = container.build_model("cpu").eval()
    schema = container.schema

    values = parse_set(args.set)
    params = schema.encode(values)
    x = probe_signal(args.samples, container.sample_rate, args.seed)

    with torch.no_grad():
        tensor = torch.from_numpy(x).reshape(1, 1, -1)
        y, state = model(tensor, params, None)
    h, c = (state, torch.zeros_like(state)) if torch.is_tensor(state) else state

    if not torch.isfinite(y).all():
        raise SystemExit("reference output is not finite; refusing to write a golden vector")

    out_json = args.out.with_suffix(".json")
    out_fbmx = args.out.with_suffix(".fbmx")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    if Path(args.fbmx).resolve() != out_fbmx.resolve():
        shutil.copyfile(args.fbmx, out_fbmx)

    payload = {
        "description": (
            "Golden vectors for the FBMX Rust runtime. Produced by "
            "neural/scripts/make_golden.py from the .fbmx next to this file."
        ),
        "model_file": out_fbmx.name,
        "model_uuid": container.model_uuid,
        "model_type": container.model_type,
        "sample_rate": container.sample_rate,
        "hidden_size": container.hparams.get("hidden_size"),
        "conditioning": container.header["conditioning"],
        "parameters": schema.decode(params),
        "initial_state": "zeros",
        "torch_version": str(torch.__version__),
        "input": [float(v) for v in x],
        "output": [float(v) for v in y.reshape(-1)],
        "final_h": [float(v) for v in h.reshape(-1)],
        "final_c": [float(v) for v in c.reshape(-1)],
    }
    out_json.write_text(json.dumps(payload), encoding="utf-8")

    print(f"[golden] model      {args.fbmx} (uuid {container.model_uuid})")
    print(f"[golden] parameters {payload['parameters']}")
    print(f"[golden] samples    {len(payload['input'])}")
    print(f"[golden] output rms {float(np.sqrt(np.mean(np.square(payload['output'])))):.6f}")
    print(f"[golden] wrote      {out_json} and {out_fbmx}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
