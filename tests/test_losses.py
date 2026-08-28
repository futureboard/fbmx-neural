"""Losses: identity gives zero, everything stays finite, composition works."""

from __future__ import annotations

import math

import pytest
import torch

from fbmx.losses import (
    CompositeLoss,
    DCLoss,
    SilenceDCLoss,
    ESRLoss,
    EnvelopeLoss,
    MAELoss,
    MultiResolutionSTFTLoss,
    TransientLoss,
    build_loss,
)
from fbmx.losses.auxiliary import AuxTraceLoss


@pytest.fixture
def signals():
    torch.manual_seed(0)
    target = torch.randn(2, 1, 4096) * 0.3
    pred = target + torch.randn_like(target) * 0.01
    return pred, target


@pytest.mark.parametrize(
    "loss",
    [MAELoss(), ESRLoss(), DCLoss(), SilenceDCLoss(), EnvelopeLoss(), TransientLoss(),
     MultiResolutionSTFTLoss(fft_sizes=(256, 512), hop_sizes=(64, 128), win_lengths=(256, 512))],
)
def test_identity_is_near_zero(loss, signals):
    _, target = signals
    value = loss(target, target)
    assert torch.isfinite(value)
    assert float(value) < 1e-4


@pytest.mark.parametrize(
    "loss",
    [MAELoss(), ESRLoss(), EnvelopeLoss(), TransientLoss(),
     MultiResolutionSTFTLoss(fft_sizes=(256,), hop_sizes=(64,), win_lengths=(256,))],
)
def test_worse_prediction_scores_worse(loss, signals):
    pred, target = signals
    worse = target + torch.randn_like(target) * 0.2
    assert float(loss(pred, target)) < float(loss(worse, target))


def test_stft_loss_is_finite_on_silence():
    """log|X| on a digital-silence block is the classic NaN source."""
    silence = torch.zeros(1, 1, 2048)
    loss = MultiResolutionSTFTLoss(fft_sizes=(256,), hop_sizes=(64,), win_lengths=(256,))
    value = loss(silence, silence)
    assert torch.isfinite(value)


def test_stft_loss_backward_is_finite():
    pred = torch.zeros(1, 1, 2048, requires_grad=True)
    target = torch.randn(1, 1, 2048) * 0.2
    loss = MultiResolutionSTFTLoss(fft_sizes=(256,), hop_sizes=(64,), win_lengths=(256,))
    loss(pred, target).backward()
    assert torch.isfinite(pred.grad).all()


def test_stft_loss_handles_a_short_final_training_chunk():
    pred = torch.zeros(1, 1, 469, requires_grad=True)
    target = torch.randn(1, 1, 469) * 0.2
    loss = MultiResolutionSTFTLoss(
        fft_sizes=(512, 1024, 2048),
        hop_sizes=(128, 256, 512),
        win_lengths=(512, 1024, 2048),
    )
    value = loss(pred, target)
    value.backward()
    assert torch.isfinite(value)
    assert torch.isfinite(pred.grad).all()


def test_resolutions_longer_than_the_chunk_are_dropped():
    with pytest.warns(UserWarning, match="skipping STFT resolution"):
        loss = MultiResolutionSTFTLoss(
            fft_sizes=(256, 4096), hop_sizes=(64, 1024), win_lengths=(256, 4096),
            max_length=1024,
        )
    assert len(loss.losses) == 1


def test_every_resolution_dropped_is_an_error():
    with pytest.raises(ValueError, match="every STFT resolution"):
        with pytest.warns(UserWarning):
            MultiResolutionSTFTLoss(
                fft_sizes=(4096,), hop_sizes=(1024,), win_lengths=(4096,), max_length=512
            )


def test_build_loss_composes_and_reports_terms(signals):
    pred, target = signals
    loss_fn = build_loss([
        {"name": "mae", "weight": 1.0},
        {"name": "mrstft", "weight": 0.5, "fft_sizes": [256], "hop_sizes": [64],
         "win_lengths": [256]},
    ])
    assert isinstance(loss_fn, CompositeLoss)
    total, parts = loss_fn(pred, target)
    assert set(parts) == {"mae", "mrstft", "total"}
    assert float(total) == pytest.approx(parts["mae"] + 0.5 * parts["mrstft"], rel=1e-5)


def test_build_loss_defaults_to_mae():
    assert set(build_loss(None).weights) == {"mae"}


def test_build_loss_rejects_unknown_and_duplicate():
    with pytest.raises(KeyError):
        build_loss([{"name": "not_a_loss"}])
    with pytest.raises(ValueError, match="duplicate loss key"):
        build_loss([{"name": "mae"}, {"name": "mae"}])


def test_aux_trace_loss_fails_loudly_without_its_target(signals):
    pred, target = signals
    with pytest.raises(KeyError, match="AuxTraceLoss"):
        AuxTraceLoss("gain")(pred, target, {"gain": torch.zeros_like(pred)})


def test_aux_trace_loss_scores_when_both_sides_present(signals):
    pred, target = signals
    aux = {"gain": torch.ones_like(pred), "pred_gain": torch.ones_like(pred) * 0.5}
    assert float(AuxTraceLoss("gain")(pred, target, aux)) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# level-normalised waveform loss
# ---------------------------------------------------------------------------
def test_normalized_mae_is_zero_on_a_perfect_match(signals):
    _, target = signals
    from fbmx.losses import NormalizedMAELoss

    assert float(NormalizedMAELoss()(target, target)) == pytest.approx(0.0, abs=1e-9)


def test_normalized_mae_weights_quiet_passages_as_heavily_as_loud_ones():
    """The property the phase-4 reweighting run exists to get.

    Two identical *relative* errors, one under a loud passage and one under a
    quiet one. Plain MAE scores the quiet one at a hundredth of the loud one;
    the normalised version scores them the same.
    """
    from fbmx.losses import MAELoss, NormalizedMAELoss

    n = 24000
    loud = torch.ones(1, 1, n) * 0.5
    quiet = torch.ones(1, 1, n) * 0.005  # 40 dB down
    target = torch.cat([loud, quiet], dim=-1)

    err_loud = target.clone()
    err_loud[..., :n] *= 1.1  # 10 % error under the loud half
    err_quiet = target.clone()
    err_quiet[..., n:] *= 1.1  # the same 10 % error under the quiet half

    plain = MAELoss()
    normed = NormalizedMAELoss(floor_dbfs=-60.0)
    ratio_plain = float(plain(err_loud, target)) / float(plain(err_quiet, target))
    ratio_normed = float(normed(err_loud, target)) / float(normed(err_quiet, target))

    assert ratio_plain > 50, f"plain MAE should be dominated by the loud half ({ratio_plain:.1f})"
    assert 0.5 < ratio_normed < 2.0, f"normalised should treat them alike ({ratio_normed:.2f})"


def test_normalized_mae_floor_stops_silence_from_dominating():
    from fbmx.losses import NormalizedMAELoss

    n = 12000
    target = torch.cat([torch.ones(1, 1, n) * 0.5, torch.zeros(1, 1, n)], dim=-1)
    pred = target.clone()
    pred[..., n:] += 1e-4  # a tiny error in digital silence
    quiet_only = float(NormalizedMAELoss(floor_dbfs=-40.0)(pred, target))
    assert quiet_only < 1e-3, "silence was amplified into the objective"


def test_normalized_mae_keeps_the_magnitude_of_a_plain_mae():
    """So the other lambdas in a config do not have to be retuned."""
    from fbmx.losses import MAELoss, NormalizedMAELoss

    torch.manual_seed(0)
    target = torch.randn(2, 1, 8192) * 0.3
    pred = target + torch.randn_like(target) * 0.01
    plain = float(MAELoss()(pred, target))
    normed = float(NormalizedMAELoss()(pred, target))
    assert 0.2 * plain < normed < 5.0 * plain, f"{normed:.5f} vs {plain:.5f}"


def test_normalized_mae_is_selectable_from_a_config():
    loss_fn = build_loss([{"name": "norm_mae", "weight": 1.0, "floor_dbfs": -45.0}])
    assert set(loss_fn.weights) == {"norm_mae"}
    assert loss_fn.terms["norm_mae"].floor == pytest.approx(10 ** (-45.0 / 20.0))


def test_normalized_mae_exponent_bounds_the_weight_range():
    """The knob that made the loss trainable.

    A 40 dB level range spans 100:1 of weight at exponent 1 and 10:1 at 0.5.
    The first diverged in training; the second is what the run uses.
    """
    from fbmx.losses import NormalizedMAELoss

    n = 24000
    target = torch.cat(
        [torch.ones(1, 1, n) * 0.5, torch.ones(1, 1, n) * 0.005], dim=-1
    )
    for exponent, expected in ((1.0, 100.0), (0.5, 10.0), (0.0, 1.0)):
        loss = NormalizedMAELoss(floor_dbfs=-60.0, exponent=exponent)
        with torch.no_grad():
            from fbmx.losses.waveform import one_pole_envelope

            env = one_pole_envelope(target, loss.coeff).clamp_min(loss.floor)
            w = env.pow(-loss.exponent)
        span = float(w[..., -100:].mean() / w[..., n // 2 - 100 : n // 2].mean())
        assert span == pytest.approx(expected, rel=0.2), f"exponent {exponent}: {span:.1f}"


def test_normalized_mae_exponent_zero_is_plain_mae():
    from fbmx.losses import MAELoss, NormalizedMAELoss

    torch.manual_seed(3)
    target = torch.randn(1, 1, 4096) * 0.3
    pred = target + torch.randn_like(target) * 0.02
    assert float(NormalizedMAELoss(exponent=0.0)(pred, target)) == pytest.approx(
        float(MAELoss()(pred, target)), rel=1e-5
    )


# --------------------------------------------------------------------------
# SilenceDCLoss
# --------------------------------------------------------------------------
# The defect it exists for: with silence in, the trained models put out a
# constant of up to -28 dBFS where the circuit puts out zero. See the class
# docstring and configs/fa76_revd_v3_dcfix.yaml.


def _signal(seed: int = 0) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.randn(2, 1, 16384) * 0.1


def _silence(n: int = 16384) -> torch.Tensor:
    return torch.zeros(1, 1, n)


def _tone(hz: float, n: int = 48_000, amplitude: float = 0.5) -> torch.Tensor:
    t = torch.arange(n, dtype=torch.float32) / 48_000.0
    return (amplitude * torch.sin(2 * math.pi * hz * t)).reshape(1, 1, -1)


def test_a_clean_prediction_costs_nothing():
    silence = _silence()
    assert float(SilenceDCLoss()(silence, silence)) == pytest.approx(0.0, abs=1e-9)


def test_the_tolerance_is_where_the_loss_reads_one():
    """`target_dbfs` is calibrated, not decorative: an offset at the tolerance
    costs about 1.0, so a weight in a config means what it looks like."""
    silence = _silence()
    offset = 10.0 ** (-90.0 / 20.0)
    value = float(SilenceDCLoss(target_dbfs=-90.0)(silence + offset, silence))
    # Just under 1.0 by the lowpass's start-up over the window, not by much.
    assert 0.85 < value < 1.0


def test_the_measured_offset_costs_about_what_it_is_worth():
    """-50 dBFS is 40 dB above the tolerance, so it should cost ~100x."""
    silence = _silence()
    at_tolerance = float(SilenceDCLoss()(silence + 10.0 ** (-90.0 / 20.0), silence))
    measured = float(SilenceDCLoss()(silence + 10.0 ** (-50.0 / 20.0), silence))
    assert measured / at_tolerance == pytest.approx(100.0, rel=0.05)


def test_it_charges_nothing_for_a_low_bass_note():
    """The whole reason the term is gated on silence rather than filtered.

    Telling a -90 dBFS constant from a -6 dBFS tone at 40 Hz needs ~84 dB of
    rejection at 40 Hz, which no filter cheap enough to run in a loss provides;
    a term that charged for low-frequency signal would buy the offset back as a
    bass rolloff. Where the teacher is silent the question does not arise.
    """
    tone = _tone(40.0)
    assert float(SilenceDCLoss()(tone, tone)) == pytest.approx(0.0, abs=1e-9)


def test_it_ignores_signal_error_that_carries_no_dc():
    """A wrong waveform with the right mean is some other term's business."""
    tone = _tone(200.0)
    assert float(SilenceDCLoss()(-tone, tone)) == pytest.approx(0.0, abs=1e-9)


def test_the_gate_opens_after_a_passage_ends_not_during_its_decay():
    """A release tail is the waveform term's business, and the lowpass needs the
    signal before it to have washed out before its output means anything."""
    loud = _tone(200.0, n=24_000)
    quiet = torch.cat([loud, _silence(24_000)], dim=-1)
    offset = torch.zeros_like(quiet)
    offset[..., 24_000:] = 10.0 ** (-50.0 / 20.0)
    scored = float(SilenceDCLoss()(quiet + offset, quiet))
    # The offset is only in the silent half, and is scored there.
    assert scored > 50.0
    # But an identical offset placed only under the signal is not scored.
    hidden = torch.zeros_like(quiet)
    hidden[..., :24_000] = 10.0 ** (-50.0 / 20.0)
    assert float(SilenceDCLoss()(quiet + hidden, quiet)) < 5.0


def test_it_survives_a_wholly_silent_window():
    """The reason this is not `DCLoss`, which divides by the target's power."""
    silence = _silence(4096)
    value = float(SilenceDCLoss()(silence + 0.003, silence))
    assert math.isfinite(value) and value > 1.0


def test_the_same_offset_scores_the_same_whatever_else_is_in_the_sequence():
    """Contrast with `DCLoss`, whose normaliser is the target's power, so the
    same three millivolts is priced across orders of magnitude depending on how
    loud the rest of the sequence happens to be — and in a silent window is
    priced by `eps`, which is a number with no physical meaning."""
    offset = 0.003
    quiet = torch.cat([_tone(200.0, n=24_000, amplitude=0.02), _silence(24_000)], dim=-1)
    loud = torch.cat([_tone(200.0, n=24_000, amplitude=0.8), _silence(24_000)], dim=-1)
    mask = torch.zeros_like(quiet)
    mask[..., 24_000:] = offset

    ours = SilenceDCLoss()
    a, b = float(ours(quiet + mask, quiet)), float(ours(loud + mask, loud))
    assert a == pytest.approx(b, rel=0.05)

    theirs = DCLoss()
    c, d = float(theirs(quiet + mask, quiet)), float(theirs(loud + mask, loud))
    assert c / d > 100.0


def test_a_batch_with_no_silence_at_all_is_finite_and_free():
    """`gate.sum()` is zero there; the clamp is what keeps it from being NaN."""
    tone = _tone(1000.0)
    value = float(SilenceDCLoss()(tone + 0.01, tone))
    assert math.isfinite(value)
    assert value == pytest.approx(0.0, abs=1e-9)
