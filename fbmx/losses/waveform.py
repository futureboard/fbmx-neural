"""Time-domain losses.

MAE on the waveform is the workhorse: it is what most black-box effect
modelling work uses as the sample-accurate term, it is cheap, and unlike MSE it
does not let a handful of transient samples dominate the gradient.

ESR (error-to-signal ratio) is reported as a metric everywhere in this
literature, so it is implemented here too and is usable as a loss.  It is
scale-invariant per batch, which is a virtue for reporting and a hazard for
training on material with long silences -- hence the epsilon and the note.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "FBMXLoss",
    "MAELoss",
    "MSELoss",
    "ESRLoss",
    "DCLoss",
    "PreEmphasis",
    "NormalizedMAELoss",
    "one_pole_envelope",
    "one_pole_lowpass",
    "SilenceDCLoss",
]


def one_pole_envelope(x: torch.Tensor, coeff: float, max_taps: int = 4096) -> torch.Tensor:
    """Smoothed |x| via a causal one-pole, as a fixed-length convolution.

    ``env[n] = (1-a) * sum_k a^(n-k) |x[k]|``, truncated at ``max_taps``. The
    recursion written out in Python would dominate the step time; the
    truncation is far below the level anything here is sensitive to.
    """
    kernel_len = min(x.shape[-1], max_taps)
    taps = coeff ** torch.arange(kernel_len, device=x.device, dtype=x.dtype)
    taps = (taps * (1.0 - coeff)).flip(0).reshape(1, 1, -1)
    rectified = torch.abs(x).reshape(-1, 1, x.shape[-1])
    padded = F.pad(rectified, (kernel_len - 1, 0))
    return F.conv1d(padded, taps).reshape(x.shape)


def one_pole_lowpass(x: torch.Tensor, coeff: float, max_taps: int = 8192) -> torch.Tensor:
    """Causal one-pole lowpass, **signed** — :func:`one_pole_envelope` without
    the rectifier, so the result keeps DC instead of measuring amplitude.

    Same fixed-length-convolution trick and the same truncation caveat.
    """
    kernel_len = min(x.shape[-1], max_taps)
    taps = coeff ** torch.arange(kernel_len, device=x.device, dtype=x.dtype)
    taps = (taps * (1.0 - coeff)).flip(0).reshape(1, 1, -1)
    flat = x.reshape(-1, 1, x.shape[-1])
    padded = F.pad(flat, (kernel_len - 1, 0))
    return F.conv1d(padded, taps).reshape(x.shape)


def _coeff_for(sample_rate: int, time_ms: float) -> float:
    return float(torch.exp(torch.tensor(-1.0 / (sample_rate * time_ms * 1e-3))))


class FBMXLoss(nn.Module):
    """Common signature for every loss in this package.

    ``pred`` and ``target`` are ``[B, 1, T]``.  ``aux`` carries whatever extra
    targets a dataset happened to provide (a gain trace, a control voltage, an
    envelope); a loss that does not need it ignores it.  Keeping the signature
    uniform is what lets :class:`fbmx.losses.CompositeLoss` mix them freely.
    """

    def forward(  # pragma: no cover - abstract
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        aux: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        raise NotImplementedError


class MAELoss(FBMXLoss):
    def forward(self, pred, target, aux=None):
        return torch.mean(torch.abs(pred - target))


class MSELoss(FBMXLoss):
    def forward(self, pred, target, aux=None):
        return torch.mean((pred - target) ** 2)


class ESRLoss(FBMXLoss):
    """``sum((y - y_hat)^2) / sum(y^2)``, the standard metric in this field."""

    def __init__(self, eps: float = 1e-8) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, pred, target, aux=None):
        num = torch.sum((target - pred) ** 2, dim=-1)
        den = torch.sum(target**2, dim=-1) + self.eps
        return torch.mean(num / den)


class DCLoss(FBMXLoss):
    """Penalise a DC offset between prediction and target.

    Cheap insurance: a residual model can drift a few millivolts away and the
    waveform loss barely notices, but the offset is audible as a click when the
    plugin is bypassed.
    """

    def __init__(self, eps: float = 1e-8) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, pred, target, aux=None):
        diff = torch.mean(pred, dim=-1) - torch.mean(target, dim=-1)
        den = torch.mean(target**2, dim=-1) + self.eps
        return torch.mean(diff**2 / den)


class NormalizedMAELoss(FBMXLoss):
    """Waveform L1 divided by a slow envelope of the target.

    Plain MAE is an *absolute* error, so a passage 40 dB below the loud part of
    a sequence contributes 1/100th as much — and the tail of a compressor's
    release is exactly such a passage. Phase 3 measured the consequence: the
    model reproduced a release of ~150 ms whatever the Release dial said, while
    the circuit ranged over 242-547 ms, and nothing in the objective objected.
    The state was not the limit (its measured time constants run to seconds);
    the loss was.

    Dividing by the target's own envelope turns the term into a *relative*
    error, so a 3 dB gain mistake costs the same whether it happens under a
    loud note or 400 ms into the decay after it.

    ``floor_dbfs`` bounds how far down that reweighting goes. Without it,
    digital silence would be amplified by whatever the epsilon happens to be,
    and the model would be trained mostly on the noise between notes.

    ``exponent`` interpolates between the two extremes: 0 is a plain absolute
    L1, 1 is a fully relative error, and anything between is a compromise. It
    is not a free knob — it was added because the fully relative version
    *diverged*. With a -40 dBFS floor, `exponent = 1` spans a 100:1 weight
    range within a single chunk, which makes the effective learning rate in
    near-silence a hundred times the nominal one; the spectral term spiked at
    epoch 4 of the first attempt and the NaN guard stopped the run at epoch 7.
    At 0.5 the same 40 dB level range becomes a 10:1 weight range, which is
    still an order of magnitude more attention than plain L1 gives the tail of
    a release, and it is stable.

    The weight is derived from the target only, and detached: it changes what
    the loss cares about, never which direction it pushes.
    """

    def __init__(
        self,
        sample_rate: int = 48000,
        time_ms: float = 25.0,
        floor_dbfs: float = -40.0,
        exponent: float = 1.0,
    ) -> None:
        super().__init__()
        self.coeff = _coeff_for(sample_rate, time_ms)
        self.floor = 10.0 ** (floor_dbfs / 20.0)
        self.exponent = float(exponent)

    def forward(self, pred, target, aux=None):
        with torch.no_grad():
            envelope = one_pole_envelope(target, self.coeff).clamp_min(self.floor)
            weight = envelope.pow(-self.exponent)
            # Renormalise so the term keeps the magnitude of a plain MAE and the
            # other loss weights in the config do not have to be retuned.
            weight = weight / weight.mean()
        return torch.mean(weight * torch.abs(pred - target))


class PreEmphasis(nn.Module):
    """First-order high-pass, ``y[n] = x[n] - a * x[n-1]``.

    Wrapping a waveform loss in this weights the error towards high frequencies,
    which is a common trick when a model gets lazy about the top octave.  Kept
    separate so it can be composed rather than baked into a loss.
    """

    def __init__(self, coeff: float = 0.85) -> None:
        super().__init__()
        self.coeff = coeff

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        prev = torch.nn.functional.pad(x, (1, 0))[..., :-1]
        return x - self.coeff * prev


class SilenceDCLoss(FBMXLoss):
    """Penalise output that is not silent when the teacher is.

    The circuit puts out exactly zero for zero in. A residual RNN does not:
    with silence at the input its state settles onto some fixed point, and
    whatever the readout makes of that fixed point is a constant at the output.
    Measured on the phase-3 models by `fa76-neural-lab compare` it reaches
    **-28 dBFS** at INPUT 2 — four percent of full scale, with nothing playing.

    Nothing in the waveform objective objects. Three millivolts of L1 is
    nothing beside the error in the parts of a sequence that have signal in
    them, and dividing by an envelope (:class:`NormalizedMAELoss`) does not
    help either, because a silent passage sits on the envelope floor and its
    weight saturates. Both trained models carry the offset at the same size,
    which is what a term that cannot see the defect looks like.

    A level probe cannot see it and an FFT analyser cannot miss it: a constant
    across the capture window is energy at DC, spread by the window into the
    lowest bins, which is why the defect draws a deep notch and a peak below
    ~60 Hz in a plugin analyser while a stepped sine sweep of the same model
    reads flat to within a decibel.

    **Why it is gated on silence rather than simply lowpassing the residual.**
    Separating a -90 dBFS constant from a -6 dBFS tone at 40 Hz needs some 84 dB
    of rejection at 40 Hz; a one-pole at 5 Hz gives 18. A filter steep enough to
    tell them apart would take a second to settle, and a term that charges for
    low-frequency *signal* is an EQ by another name — it would trade the offset
    for a bass rolloff and call it progress. Where the teacher is silent there
    is no low-frequency signal to confuse the measurement with and the right
    answer is exactly zero, which is both the defect and the whole of it.

    The gate follows a deliberately slow (100 ms) envelope of the target, so it
    opens well after a passage ends rather than during its decay — by then the
    lowpass no longer holds anything from the signal that preceded it, and the
    release tail stays the waveform term's business.

    It also requires a short running *peak* of the target to be below the
    threshold, which is not redundant. A smoothed envelope starts each chunk at
    zero and takes its time constant to catch up, so the slow gate alone
    declares the first tens of milliseconds of every chunk silent and scores
    whatever signal is there — at the start of a 200 Hz tone, that is the whole
    tone. A running peak is exactly zero in silence, has no start-up ramp, and
    unlike a fast smoothed envelope does not dip back through the threshold at
    every zero crossing of a low note. The peak closes the gate the moment
    anything arrives; the slow envelope is what keeps it closed through a
    decay.

    The score is divided by ``target_dbfs``, so a loss of 1.0 *is* the
    tolerance. That makes the scale absolute rather than relative, which is
    right here — a DC offset is not more acceptable in a loud sequence — and it
    makes a weight of 1.0 mean "an offset at the tolerance costs about as much
    as a unit of waveform error".

    Distinct from :class:`DCLoss`, which normalises by the target's power and
    therefore diverges in exactly the silent passages this exists for.
    """

    def __init__(
        self,
        sample_rate: int = 48_000,
        corner_hz: float = 5.0,
        gate_time_ms: float = 100.0,
        gate_peak_ms: float = 5.0,
        gate_dbfs: float = -60.0,
        target_dbfs: float = -90.0,
    ) -> None:
        super().__init__()
        # tau = 1/(2*pi*f): 32 ms at 5 Hz, which settles well inside a chunk.
        tau_ms = 1000.0 / (2.0 * math.pi * corner_hz)
        self.coeff = _coeff_for(sample_rate, tau_ms)
        self.gate_coeff = _coeff_for(sample_rate, gate_time_ms)
        # Odd, so the window is centred: deciding whether a passage is silent
        # may look slightly ahead — it is a mask, not part of the signal path.
        self.gate_peak_taps = int(sample_rate * gate_peak_ms * 1e-3) | 1
        self.gate_level = 10.0 ** (gate_dbfs / 20.0)
        self.scale = 10.0 ** (target_dbfs / 20.0)

    def forward(self, pred, target, aux=None):
        with torch.no_grad():
            slow = one_pole_envelope(target, self.gate_coeff)
            peak = F.max_pool1d(
                torch.abs(target).reshape(-1, 1, target.shape[-1]),
                kernel_size=self.gate_peak_taps,
                stride=1,
                padding=self.gate_peak_taps // 2,
            ).reshape(target.shape)
            quiet = torch.maximum(slow, peak) < self.gate_level
            gate = quiet.to(pred.dtype)
        dc = one_pole_lowpass(pred - target, self.coeff)
        # Mean over the gated samples only, so the score does not depend on how
        # much of the batch happened to be silent.
        return (gate * torch.abs(dc)).sum() / gate.sum().clamp_min(1.0) / self.scale
