"""Block-size independence.

The single most important property in this repository.  A model that fails
these tests cannot be used in a plugin at any quality, because the artefact it
produces is tied to the host's buffer size and will change when the user
changes their audio device.

Tolerance: recurrent models over the same weights in FP32 accumulate in a
different order for different sequence lengths, so exact equality is not
required.  1e-5 absolute on signals of order 0.3 is ~90 dB below the signal,
several orders of magnitude tighter than any state-carrying bug.
"""

from __future__ import annotations

import pytest
import torch

from fbmx.conditioning import ParamBatch
from fbmx.streaming.inference import (
    StreamingProcessor,
    block_schedule,
    process_blocked,
    process_offline,
    streaming_equivalence,
)

TOL = 1e-5
BLOCK_SIZES = (16, 32, 64, 128, 256, 512, 1024)


@pytest.mark.parametrize("block_size", BLOCK_SIZES)
def test_blocked_matches_offline(model, schema, probe_signal, block_size):
    params = schema.empty_batch(1)
    reference = process_offline(model, probe_signal, params)
    blocked = process_blocked(model, probe_signal, block_size, params)
    assert blocked.shape == reference.shape
    assert float((reference - blocked).abs().max()) < TOL


def test_all_block_sizes_at_once(lstm32, schema, probe_signal):
    diffs = streaming_equivalence(lstm32, probe_signal, BLOCK_SIZES, schema.empty_batch(1))
    assert set(diffs) == set(BLOCK_SIZES)
    assert max(diffs.values()) < TOL


def test_lstm_block_processing_is_bit_exact(lstm32, schema, probe_signal):
    """For the baseline specifically, we expect exactness, not just closeness.

    A single-layer LSTM in FP32 evaluates the same arithmetic per sample
    regardless of how the samples were grouped, so any nonzero difference here
    means something is being recomputed differently -- worth knowing about.
    """
    params = schema.empty_batch(1)
    reference = process_offline(lstm32, probe_signal, params)
    for block in (32, 128, 1024):
        assert torch.equal(reference, process_blocked(lstm32, probe_signal, block, params))


def test_varying_block_size_is_also_exact(model, schema, probe_signal):
    """Hosts change buffer size mid-session; a fixed size passing is not enough."""
    params = schema.empty_batch(1)
    reference = process_offline(model, probe_signal, params)
    processor = StreamingProcessor(model, params)
    out, pos = [], 0
    for n in block_schedule(probe_signal.shape[-1], [64, 1, 333, 128, 7, 512]):
        out.append(processor.process(probe_signal[..., pos : pos + n]))
        pos += n
    blocked = torch.cat(out, dim=-1)
    assert blocked.shape == reference.shape
    assert float((reference - blocked).abs().max()) < TOL


def test_reset_returns_to_initial_conditions(model, schema, probe_signal):
    params = schema.empty_batch(1)
    processor = StreamingProcessor(model, params)
    first = processor.process(probe_signal)
    processor.process(torch.randn_like(probe_signal))  # dirty the state
    processor.reset()
    second = processor.process(probe_signal)
    assert torch.equal(first, second)


def test_state_is_not_hidden_inside_the_module(lstm32, schema, probe_signal):
    """Two processors must not interfere -- no module-level scratch buffers."""
    params = schema.empty_batch(1)
    a = StreamingProcessor(lstm32, params)
    b = StreamingProcessor(lstm32, params)
    a.process(torch.ones(1, 1, 4096) * 0.9)  # drive a's state somewhere else
    solo = StreamingProcessor(lstm32, params).process(probe_signal)
    assert torch.equal(b.process(probe_signal), solo)


def test_streaming_output_is_finite(model, schema):
    """Long run, including silence, must not drift to NaN/Inf."""
    x = torch.cat(
        [torch.randn(1, 1, 8192) * 0.5, torch.zeros(1, 1, 8192), torch.ones(1, 1, 4096)],
        dim=-1,
    )
    y = process_blocked(model, x, 128, schema.empty_batch(1))
    assert torch.isfinite(y).all()


def test_batched_streaming_matches_single(lstm32, schema, probe_signal):
    """Batch dimension must not couple sequences together."""
    params = ParamBatch.collate([schema.encode({"drive": 0.2}), schema.encode({"drive": 0.9})])
    batched_input = torch.cat([probe_signal, probe_signal * 0.5], dim=0)
    batched = process_blocked(lstm32, batched_input, 256, params)
    for row in range(2):
        single = process_blocked(
            lstm32,
            batched_input[row : row + 1],
            256,
            ParamBatch(params.continuous[row : row + 1], params.categorical[row : row + 1]),
        )
        assert float((batched[row : row + 1] - single).abs().max()) < TOL


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_streaming_equivalence_on_cuda(schema, probe_signal):
    from fbmx.models import build_model

    model = build_model({"type": "lstm", "hidden_size": 32}, schema).cuda().eval()
    x = probe_signal.cuda()
    diffs = streaming_equivalence(model, x, (64, 256, 1024), schema.empty_batch(1, "cuda"))
    assert max(diffs.values()) < TOL
