"""Shapes, determinism, causality, state handling, device coverage."""

from __future__ import annotations

import pytest
import torch

from fbmx.models import build_model, detach_state
from fbmx.models.base import StreamingModel


def test_output_shape_matches_input(model, schema):
    x = torch.randn(2, 1, 1024) * 0.2
    params = schema.empty_batch(2)
    y, state = model(x, params, None)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()
    assert state is not None


def test_accepts_two_dimensional_input(lstm32, schema):
    y, _ = lstm32(torch.zeros(3, 512), schema.empty_batch(3), None)
    assert y.shape == (3, 1, 512)


def test_rejects_stereo(lstm32, schema):
    with pytest.raises(ValueError):
        lstm32(torch.zeros(1, 2, 128), schema.empty_batch(1), None)


def test_inference_is_deterministic(model, schema):
    x = torch.randn(1, 1, 2048) * 0.3
    params = schema.empty_batch(1)
    with torch.no_grad():
        a, _ = model(x, params, None)
        b, _ = model(x, params, None)
    assert torch.equal(a, b)


def test_zero_state_is_reproducible(model, schema):
    """A fresh state must be identical every time, or nothing else is testable."""
    s1 = model.init_state(2)
    s2 = model.init_state(2)
    flat1 = s1 if isinstance(s1, torch.Tensor) else list(s1)
    flat2 = s2 if isinstance(s2, torch.Tensor) else list(s2)
    if isinstance(flat1, torch.Tensor):
        assert torch.equal(flat1, flat2) and float(flat1.abs().max()) == 0.0
    else:
        for a, b in zip(flat1, flat2):
            assert torch.equal(a, b) and float(a.abs().max()) == 0.0


def test_model_is_causal(model, schema):
    """Changing the future must not change the past.

    This is the property that makes a realtime model possible at all, and it is
    easy to lose accidentally (a centred convolution, a reversed scan, a global
    normalisation).  Test it directly rather than trusting the constructor.
    """
    torch.manual_seed(0)
    x = torch.randn(1, 1, 1024) * 0.3
    params = schema.empty_batch(1)
    edited = x.clone()
    edited[..., 512:] = torch.randn(1, 1, 512) * 0.9
    with torch.no_grad():
        a, _ = model(x, params, None)
        b, _ = model(edited, params, None)
    assert torch.allclose(a[..., :512], b[..., :512], atol=1e-6)


def test_state_carries_information(lstm32, schema):
    """Feeding a chunk with a used state must differ from a fresh state."""
    torch.manual_seed(0)
    x = torch.randn(1, 1, 512) * 0.5
    params = schema.empty_batch(1)
    with torch.no_grad():
        _, state = lstm32(torch.ones(1, 1, 2048) * 0.8, params, None)
        warm, _ = lstm32(x, params, state)
        cold, _ = lstm32(x, params, None)
    assert not torch.allclose(warm, cold, atol=1e-6)


def test_lstm_state_shapes(lstm32):
    h, c = lstm32.init_state(4)
    assert h.shape == (lstm32.num_layers, 4, lstm32.hidden_size)
    assert c.shape == h.shape


def test_detach_state_cuts_the_graph(lstm32, schema):
    x = torch.randn(1, 1, 256, requires_grad=False) * 0.2
    y, state = lstm32(x, schema.empty_batch(1), None)
    assert state[0].requires_grad
    detached = detach_state(state)
    assert not detached[0].requires_grad
    assert torch.equal(detached[0], state[0].detach())


def test_conditioning_changes_the_output(schema):
    """A conditioned model that ignores its parameters is a silent failure."""
    torch.manual_seed(0)
    model = build_model({"type": "lstm", "hidden_size": 16}, schema).eval()
    # Untrained weights make the effect tiny but non-zero; scale the projection
    # so the test measures wiring, not training.
    with torch.no_grad():
        model.rnn.weight_ih_l0.mul_(3.0)
    x = torch.randn(1, 1, 1024) * 0.4
    a, _ = model(x, schema.encode({"drive": 0.0, "mix": 0.0, "mode": "soft"}), None)
    b, _ = model(x, schema.encode({"drive": 1.0, "mix": 1.0, "mode": "hard"}), None)
    assert not torch.allclose(a, b, atol=1e-6)


def test_unconditioned_model_still_builds():
    from fbmx.conditioning import ConditioningSchema

    model = build_model({"type": "lstm", "hidden_size": 8}, ConditioningSchema())
    y, _ = model(torch.zeros(1, 1, 64), None, None)
    assert y.shape == (1, 1, 64)


def test_lstm32_parameter_count_is_small(schema):
    """The baseline has to fit in an audio callback; keep an eye on the size."""
    model = build_model({"type": "lstm", "hidden_size": 32}, schema)
    assert model.num_parameters() < 10_000


def test_export_spec_is_complete(model):
    spec = model.export_spec()
    for key in (
        "model_type",
        "sample_rate",
        "channels",
        "causal",
        "recurrent",
        "receptive_field",
        "parameter_count",
        "hparams",
        "conditioning",
    ):
        assert key in spec, key
    assert spec["causal"] is True


def test_no_bidirectional_layers(model):
    for module in model.modules():
        if isinstance(module, torch.nn.RNNBase):
            assert module.bidirectional is False


def test_model_is_a_streaming_model(model):
    assert isinstance(model, StreamingModel)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_runs_on_device(device, schema):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available on this machine")
    model = build_model({"type": "lstm", "hidden_size": 32}, schema).to(device).eval()
    x = torch.randn(2, 1, 1024, device=device) * 0.3
    params = schema.empty_batch(2, device=device)
    y, state = model(x, params, None)
    assert y.device.type == device
    assert torch.isfinite(y).all()
    assert state[0].device.type == device
