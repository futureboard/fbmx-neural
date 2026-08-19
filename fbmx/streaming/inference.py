"""Block-wise inference -- the shape the Rust runtime will have.

A host hands the plugin whatever block size it feels like, and it may change it
between calls.  The only thing that must not change is the output: processing
100k samples in one call and processing them as 782 blocks of 128 must give the
same signal, because a listener will hear the difference as a click or a
pumping artefact at every buffer boundary.

That property is not automatic.  It holds only if *every* piece of temporal
state -- recurrent hidden/cell state, convolution input caches, and later any
SSM state -- is carried explicitly across the boundary.  It is verified in
``tests/test_streaming_equivalence.py`` at 16/32/64/128/256/512/1024 samples,
and :func:`streaming_equivalence` is the function that test calls.

Latency is zero by construction: the models are causal and there is no
lookahead buffer anywhere in this file.
"""

from __future__ import annotations

import time
from typing import Iterable, Sequence

import torch

from fbmx.conditioning import ParamBatch
from fbmx.models.base import StreamingModel

__all__ = [
    "StreamingProcessor",
    "process_offline",
    "process_blocked",
    "streaming_equivalence",
    "realtime_factor",
]


class StreamingProcessor:
    """Stateful wrapper around a model, for block-at-a-time processing.

    Not thread-safe and not intended to be: one processor instance corresponds
    to one voice / one audio callback.
    """

    def __init__(
        self,
        model: StreamingModel,
        params: ParamBatch | None = None,
        *,
        batch_size: int = 1,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.device = torch.device(device) if device is not None else next(model.parameters()).device
        self.model = model.to(self.device).eval()
        self.dtype = dtype
        self.batch_size = batch_size
        self.params = params.to(self.device).expand_to(batch_size) if params is not None else None
        self.state = None
        self.reset()

    @property
    def latency_samples(self) -> int:
        return 0

    def reset(self) -> None:
        """Return to the "silence forever" state.  Call on transport stop."""
        self.state = self.model.init_state(self.batch_size, device=self.device, dtype=self.dtype)

    def set_params(self, params: ParamBatch | None) -> None:
        """Change the control setting without disturbing the audio state.

        Parameters are read once per block, so an automation move lands on a
        block boundary.  Per-sample smoothing of the conditioning vector is a
        runtime concern and is deliberately not simulated here.
        """
        self.params = params.to(self.device).expand_to(self.batch_size) if params is not None else None

    @torch.no_grad()
    def process(self, block: torch.Tensor) -> torch.Tensor:
        """Process one block.  Accepts ``[T]``, ``[1, T]`` or ``[B, 1, T]``."""
        squeeze_to = block.dim()
        x = block
        if x.dim() == 1:
            x = x.reshape(1, 1, -1)
        elif x.dim() == 2:
            x = x.unsqueeze(0) if x.shape[0] == 1 else x.unsqueeze(1)
        x = x.to(device=self.device, dtype=self.dtype)
        y, self.state = self.model(x, self.params, self.state)
        if squeeze_to == 1:
            return y.reshape(-1)
        if squeeze_to == 2:
            return y.reshape(y.shape[0] * y.shape[1], -1)
        return y


@torch.no_grad()
def process_offline(
    model: StreamingModel,
    x: torch.Tensor,
    params: ParamBatch | None = None,
    state=None,
) -> torch.Tensor:
    """Process a whole signal in one call.  ``x`` is ``[B, 1, T]``."""
    model.eval()
    y, _ = model(x, params, state)
    return y


@torch.no_grad()
def process_blocked(
    model: StreamingModel,
    x: torch.Tensor,
    block_size: int,
    params: ParamBatch | None = None,
) -> torch.Tensor:
    """Process ``[B, 1, T]`` as consecutive blocks, carrying state."""
    model.eval()
    processor = StreamingProcessor(
        model, params, batch_size=int(x.shape[0]), device=x.device, dtype=x.dtype
    )
    out = [processor.process(x[..., s : s + block_size]) for s in range(0, x.shape[-1], block_size)]
    return torch.cat(out, dim=-1)


@torch.no_grad()
def streaming_equivalence(
    model: StreamingModel,
    x: torch.Tensor,
    block_sizes: Iterable[int] = (16, 32, 64, 128, 256, 512, 1024),
    params: ParamBatch | None = None,
) -> dict[int, float]:
    """Max absolute difference between offline and blocked processing.

    Values are not expected to be exactly zero: cuDNN/oneDNN pick different
    kernels for different sequence lengths, so the difference is float32
    accumulation noise (order 1e-7 relative).  A block-size-dependent *state*
    bug is orders of magnitude larger and immediately obvious.
    """
    reference = process_offline(model, x, params)
    results: dict[int, float] = {}
    for block in block_sizes:
        blocked = process_blocked(model, x, int(block), params)
        results[int(block)] = float(torch.max(torch.abs(reference - blocked)))
    return results


@torch.no_grad()
def realtime_factor(
    model: StreamingModel,
    block_size: int = 128,
    sample_rate: int | None = None,
    n_blocks: int = 200,
    params: ParamBatch | None = None,
    device: torch.device | str = "cpu",
) -> dict[str, float]:
    """Crude throughput probe: audio-seconds processed per wall-clock second.

    A sanity check, not a benchmark.  The real number is whatever the Rust
    runtime achieves; PyTorch's per-call overhead dominates at these block
    sizes and will flatter or slander the model depending on the day.
    """
    sample_rate = sample_rate or model.sample_rate
    processor = StreamingProcessor(model, params, device=device)
    block = torch.zeros(1, 1, block_size, device=processor.device)
    processor.process(block)  # warm up allocators / lazy init
    start = time.perf_counter()
    for _ in range(n_blocks):
        processor.process(block)
    elapsed = time.perf_counter() - start
    audio_seconds = n_blocks * block_size / sample_rate
    return {
        "block_size": float(block_size),
        "wall_seconds": elapsed,
        "audio_seconds": audio_seconds,
        "realtime_factor": audio_seconds / elapsed if elapsed > 0 else float("inf"),
    }


def block_schedule(total: int, block_sizes: Sequence[int]) -> list[int]:
    """Cycle through ``block_sizes`` until ``total`` samples are covered.

    Hosts change their buffer size mid-session (freeze, bounce, device switch).
    Tests use this to check that a *varying* block size is also exact, which is
    a strictly stronger property than any single fixed size.
    """
    out: list[int] = []
    remaining = total
    i = 0
    while remaining > 0:
        n = min(block_sizes[i % len(block_sizes)], remaining)
        out.append(n)
        remaining -= n
        i += 1
    return out
