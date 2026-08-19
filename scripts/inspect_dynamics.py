"""Diagnose how much memory a trained recurrent model actually has.

    python scripts/inspect_dynamics.py --fbmx models/fa76-revd-v2.fbmx

A compressor's release is a decaying exponential, and an LSTM can only hold one
by keeping its forget gate close to 1: a time constant of ``tau`` seconds at
``fs`` needs ``f = exp(-1 / (tau * fs))``. At 48 kHz a 500 ms release needs
``f = 0.99996``. The gradient that would push a gate that far is proportionally
tiny, which is the standard argument for state-space parametrisations — and it
is testable rather than merely arguable.

This drives the model with a step down (loud tone to quiet) and, for every
hidden unit, fits the decay of its cell state to an exponential. The resulting
histogram of time constants says directly whether the model *can* hold the
release it is being asked to reproduce, or whether it has settled for the
fastest thing its gates allow.

Reports the achieved forget-gate statistics too, straight from the recurrence.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import _bootstrap  # noqa: F401  (sys.path)

from fbmx.export.fbmx import read_fbmx


def step_signal(sample_rate: int, hz: float, loud_s: float, quiet_s: float,
                loud_dbfs: float, quiet_dbfs: float) -> torch.Tensor:
    n_loud = int(loud_s * sample_rate)
    n_quiet = int(quiet_s * sample_rate)
    t = np.arange(n_loud + n_quiet) / sample_rate
    amp = np.concatenate([
        np.full(n_loud, 10 ** (loud_dbfs / 20)),
        np.full(n_quiet, 10 ** (quiet_dbfs / 20)),
    ])
    return torch.from_numpy((amp * np.sin(2 * np.pi * hz * t)).astype(np.float32))


def time_constants_from_decay(cell: np.ndarray, sample_rate: int) -> tuple[np.ndarray, int]:
    """Per-unit exponential time constant of |c| after the step, in ms.

    Returns `(tau_ms, n_static)`.

    Only units that *actually release* are fitted: a unit whose |c| never falls
    below 60 % of its post-step value inside the window is holding a bias, not
    carrying a decay, and fitting a line to its noise yields a meaningless
    multi-second "time constant". Those are counted separately, because
    "25 units with tau > 6 s" would otherwise read as evidence of long memory
    when it is mostly evidence of constants.
    """
    n_units = cell.shape[1]
    taus = np.full(n_units, np.nan)
    static = 0
    for u in range(n_units):
        trace = np.abs(cell[:, u])
        start = int(trace.argmax())
        if start > len(trace) - 200:
            static += 1
            continue
        segment = trace[start:]
        if segment.min() > 0.6 * segment[0]:
            static += 1
            continue
        floor = max(segment.max() * 1e-3, 1e-6)
        usable = segment > floor
        if usable.sum() < 100:
            static += 1
            continue
        y = np.log(segment[usable])
        x = np.arange(len(segment))[usable] / sample_rate
        slope, _ = np.polyfit(x, y, 1)
        if slope >= -1e-6:
            static += 1
            continue
        taus[u] = -1.0 / slope * 1000.0
    return taus, static


def _t63_ms(trace: np.ndarray, start: float, end: float, rate: float) -> float:
    """Time to cross 63 % of the way from `start` to `end`, in ms."""
    target = start + 0.63 * (end - start)
    reached = trace >= target if end > start else trace <= target
    if not reached.any():
        return float("nan")
    return float(np.argmax(reached)) / rate * 1000.0


def main() -> int:
    p = argparse.ArgumentParser(description="Measure a model's usable memory")
    p.add_argument("--fbmx", required=True, type=Path)
    p.add_argument("--set", nargs="*", default=[], metavar="NAME=VALUE")
    p.add_argument("--hz", type=float, default=1000.0)
    p.add_argument("--json", type=Path, default=None)
    args = p.parse_args()

    container = read_fbmx(args.fbmx)
    model = container.build_model("cpu").eval()
    schema = container.schema

    values: dict[str, object] = {}
    for item in args.set:
        key, _, raw = item.partition("=")
        try:
            values[key] = float(raw)
        except ValueError:
            values[key] = raw
    params = schema.encode(values)

    sample_rate = container.sample_rate
    x = step_signal(sample_rate, args.hz, 1.0, 1.5, -3.0, -40.0).reshape(1, 1, -1)

    # Run in one pass, capturing the cell state at every sample.
    cells = []
    state = None
    with torch.no_grad():
        for i in range(x.shape[-1]):
            _, state = model(x[..., i : i + 1], params, state)
            cells.append(state[1].reshape(-1).clone())
    cell = torch.stack(cells).numpy()

    n_loud = int(1.0 * sample_rate)
    taus, static = time_constants_from_decay(cell[n_loud:], sample_rate)
    finite = taus[np.isfinite(taus)]

    # The forget gate is what sets those constants. Recover the implied gate
    # value for each measured time constant.
    implied_f = np.exp(-1.0 / (finite / 1000.0 * sample_rate)) if finite.size else np.array([])

    print(f"model            {args.fbmx}")
    print(f"                 {container.model_type}, hidden {container.hparams.get('hidden_size')},"
          f" {sample_rate} Hz")
    print(f"parameters       {schema.decode(params)}")
    print()
    print(f"units that release             {finite.size} of {taus.size}")
    print(f"units holding a steady bias    {static}")
    if finite.size:
        for q in (50, 75, 90, 100):
            print(f"  {q:>3}th percentile tau        {np.percentile(finite, q):8.1f} ms")
        print(f"  slowest unit's forget gate   {implied_f.max():.6f}")
        print()
        print("  a release of ...  needs a forget gate of")
        for tau_ms in (50.0, 100.0, 250.0, 500.0):
            need = np.exp(-1.0 / (tau_ms / 1000.0 * sample_rate))
            reached = (finite >= tau_ms).sum()
            print(f"  {tau_ms:>6.0f} ms          {need:.6f}   units reaching it: {reached}")
    else:
        print("  no unit showed a clean exponential decay")

    # -- release sweep --------------------------------------------------
    # The state can hold a long decay; the question is whether the model
    # *uses* it, and whether the Release dial moves it. Both the auxiliary
    # gain head and the audio envelope are measured, because they can
    # disagree: the head reads the state directly, the audio has to express it.
    print()
    print("release t63 against the Release dial")
    print(f"  {'dial':>6}  {'GR head':>10}  {'audio env':>10}   (circuit, for reference: "
          "547 ms at 1, 255 ms at 4, 242 ms at 7)")
    for dial in (1.0, 4.0, 7.0):
        swept = dict(values)
        swept["Release"] = dial
        p_sweep = schema.encode(swept)
        with torch.no_grad():
            y, aux, _ = model.forward_aux(x, p_sweep, None)
        gr_scale = 0.04  # matches the aux_trace scale in configs/fa76_revd_v2.yaml
        head = aux.get("pred_gain")
        head_ms = float("nan")
        if head is not None:
            trace = head.reshape(-1).numpy()[n_loud:] / gr_scale
            head_ms = _t63_ms(trace, trace[0], trace[-2000:].mean(), sample_rate)
        audio = y.reshape(-1).numpy()[n_loud:]
        half = max(int(sample_rate / args.hz / 2), 1)
        env = np.array([np.abs(audio[i : i + half]).max() for i in range(0, len(audio), half)])
        env_db = 20 * np.log10(np.maximum(env, 1e-12))
        audio_ms = _t63_ms(env_db, env_db[0], env_db[-40:].mean(), sample_rate / half)
        print(f"  {dial:>6.0f}  {head_ms:>9.1f}ms  {audio_ms:>9.1f}ms")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(
                {
                    "model": str(args.fbmx),
                    "uuid": container.model_uuid,
                    "sample_rate": sample_rate,
                    "tau_ms": [None if not np.isfinite(v) else float(v) for v in taus],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
