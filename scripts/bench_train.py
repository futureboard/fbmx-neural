"""Is this model faster to train on the GPU? Measure, do not assume.

    python scripts/bench_train.py --config configs/fa76_revd_v2.yaml
    python scripts/bench_train.py --config configs/fa76_revd_v2.yaml --batches 4 16 64

A 32-unit LSTM is a bad fit for a GPU on paper: the per-timestep work is a
4x8 and a 4x32 matrix-vector product, the timesteps are strictly sequential,
and a 1060 has 1280 cores waiting on 128 units of parallel work. Whether cuDNN's
fused kernel wins anyway is an empirical question, and it depends on the batch
size — which is the parameter a GPU actually rewards.

This times the *training* inner loop, not inference: `tbptt_chunks` forward
passes accumulating one window, one backward, one optimiser step. That is the
thing a training run is made of, so its cost is the thing worth comparing.

Projected epoch times assume the corpus in the config; they are what to decide
on, since a 3x faster step at a quarter of the step count is not a win.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

import _bootstrap  # noqa: F401  (sys.path)

from fbmx.config import load_config
from fbmx.datasets import SYNTHETIC_SCHEMA
from fbmx.losses import build_loss
from fbmx.models import build_model
from fbmx.models.base import detach_state


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare training throughput across devices")
    p.add_argument("--config", type=Path, default=Path("configs/fa76_revd_v2.yaml"))
    p.add_argument("--devices", nargs="*", default=None, help="default: cpu, plus cuda if present")
    p.add_argument("--batches", nargs="*", type=int, default=None, help="default: the config's")
    p.add_argument("--windows", type=int, default=6, help="timed TBPTT windows per measurement")
    p.add_argument("--threads", type=int, default=6, help="CPU threads (6 physical cores here)")
    return p.parse_args()


def describe(device: str) -> str:
    if device == "cpu":
        return f"cpu ({torch.get_num_threads()} threads)"
    name = torch.cuda.get_device_name(0)
    cap = ".".join(str(c) for c in torch.cuda.get_device_capability(0))
    arches = ", ".join(torch.cuda.get_arch_list())
    return f"{name} (sm_{cap.replace('.', '')}; wheel built for: {arches})"


def time_windows(model, loss_fn, device, batch, chunk, tbptt, windows):
    """One measurement: `windows` TBPTT windows of forward + backward + step."""
    model = model.to(device).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    schema = model.schema
    params = schema.empty_batch(batch, device=device)

    torch.manual_seed(0)
    window_len = chunk * tbptt
    x = torch.randn(batch, 1, window_len, device=device) * 0.3
    target = torch.tanh(x * 2.0)
    # Stand-ins for the teacher's traces: the auxiliary heads have to be
    # evaluated and back-propagated for the timing to mean anything, but what
    # they are compared against does not affect the cost.
    dataset_aux = {k: torch.zeros_like(x) for k in model.aux_heads}

    def one_window():
        state = model.init_state(batch, device=device)
        optimizer.zero_grad(set_to_none=True)
        total = None
        for c in range(tbptt):
            lo, hi = c * chunk, (c + 1) * chunk
            pred, model_aux, state = model.forward_aux(x[..., lo:hi], params, state)
            merged = {k: v[..., lo:hi] for k, v in dataset_aux.items()}
            merged.update(model_aux)
            value, _ = loss_fn(pred, target[..., lo:hi], merged or None)
            total = value if total is None else total + value
        (total / tbptt).backward()
        optimizer.step()
        detach_state(state)

    for _ in range(2):  # warm up allocators, cuDNN algorithm choice, autotune
        one_window()
    if device != "cpu":
        torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(windows):
        one_window()
    if device != "cpu":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    per_window = elapsed / windows
    samples_per_window = batch * window_len
    return per_window, samples_per_window / per_window


def main() -> int:
    args = parse_args()
    torch.set_num_threads(args.threads)

    cfg = load_config(args.config)
    train_cfg = cfg.get("train", {})
    chunk = int(train_cfg.get("chunk_size", 4096))
    tbptt = int(train_cfg.get("tbptt_chunks", 8))
    config_batch = int(train_cfg.get("batch_size", 4))
    sequence_len = 120_000  # 2.5 s at 48 kHz, the phase-3 corpus

    # The schema has to match the config's dataset; use the FA76 one when the
    # manifest is present, else the synthetic one, so this runs anywhere.
    manifest = Path(cfg.get("data", {}).get("manifest", ""))
    manifest = (args.config.parent / manifest).resolve() if str(manifest) else None
    if manifest and manifest.exists():
        from fbmx.datasets.manifest import DatasetManifest

        schema = DatasetManifest.load(manifest).schema
        corpus_sequences = 419
    else:
        schema = SYNTHETIC_SCHEMA
        corpus_sequences = 419

    model_cfg = dict(cfg.get("model", {}))
    devices = args.devices or (["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
    batches = args.batches or [config_batch]

    print(f"config        {args.config}")
    print(f"model         {model_cfg.get('type')} hidden {model_cfg.get('hidden_size')}, "
          f"aux heads {model_cfg.get('aux_heads', [])}")
    print(f"window        {tbptt} x {chunk} = {chunk * tbptt} samples")
    print(f"corpus        {corpus_sequences} sequences x {sequence_len} samples\n")

    if "cuda" in devices and not torch.cuda.is_available():
        print("cuda requested but torch.cuda.is_available() is False")
        devices = [d for d in devices if d != "cuda"]

    results: dict[tuple[str, int], tuple[float, float]] = {}
    for device in devices:
        print(f"--- {describe(device)}")
        print(f"{'batch':>6}  {'ms/window':>11}  {'ksamples/s':>12}  {'steps/epoch':>12}  {'est s/epoch':>12}")
        for batch in batches:
            model = build_model(dict(model_cfg), schema)
            loss_fn = build_loss(cfg.get("loss")).to(device)
            try:
                per_window, throughput = time_windows(
                    model, loss_fn, device, batch, chunk, tbptt, args.windows
                )
            except RuntimeError as e:
                print(f"{batch:>6}  failed: {str(e).splitlines()[0]}")
                continue
            results[(device, batch)] = (per_window, throughput)
            windows_per_sequence = max(sequence_len // (chunk * tbptt), 1)
            steps = (corpus_sequences / batch) * windows_per_sequence
            epoch_s = steps * per_window
            print(
                f"{batch:>6}  {per_window * 1000:>11.1f}  {throughput / 1000:>12.1f}  "
                f"{steps:>12.0f}  {epoch_s:>12.0f}"
            )
        print()

    if len(devices) > 1 and results:
        base = results.get(("cpu", batches[0]))
        if base:
            print("summary")
            for (device, batch), (per_window, _) in sorted(results.items()):
                speedup = base[0] / per_window
                print(f"  {device:<5} batch {batch:<4} {speedup:>5.2f}x the cpu step at batch {batches[0]}")
            print(
                "\nA faster step at a larger batch is only a win if the step count still "
                "trains the model; compare est s/epoch, not ms/window."
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
