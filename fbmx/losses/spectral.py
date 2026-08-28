"""Multi-resolution STFT loss.

A waveform loss alone tends to produce models that are phase-correct and
timbrally dull, because a small broadband spectral error is cheap in the time
domain.  The standard remedy is a sum of STFT losses at several window sizes,
each combining spectral convergence and log-magnitude error; it is what the
efficient-DRC-modelling work (arXiv:2102.06200) uses alongside L1 and what the
NablAFx framework exposes as its default composite.

Two practical notes:

* Resolutions whose window is longer than the training chunk are skipped, with
  a warning at construction time.  A 2048-point window on a 1024-sample chunk
  is not a loss, it is zero-padding noise.
* The STFT here is only ever applied to *offline* training chunks.  It never
  runs in the audio path, so its latency is irrelevant.
"""

from __future__ import annotations

import warnings

import torch
import torch.nn.functional as F

from fbmx.losses.waveform import FBMXLoss

__all__ = ["STFTLoss", "MultiResolutionSTFTLoss"]


def _stft_magnitude(
    x: torch.Tensor, n_fft: int, hop: int, win_length: int, window: torch.Tensor
) -> torch.Tensor:
    spec = torch.stft(
        x,
        n_fft=n_fft,
        hop_length=hop,
        win_length=win_length,
        window=window,
        center=True,
        return_complex=True,
    )
    # clamp before sqrt: |X|^2 can round to exactly 0 and the gradient of
    # sqrt at 0 is inf, which shows up as a NaN a few hundred steps in
    return torch.sqrt(torch.clamp(spec.real**2 + spec.imag**2, min=1e-10))


class STFTLoss(FBMXLoss):
    """Spectral convergence + log-magnitude error at one resolution."""

    def __init__(
        self,
        n_fft: int = 1024,
        hop_length: int = 256,
        win_length: int = 1024,
        sc_weight: float = 1.0,
        mag_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.sc_weight = sc_weight
        self.mag_weight = mag_weight
        self.register_buffer("window", torch.hann_window(win_length), persistent=False)

    def forward(self, pred, target, aux=None):
        pred = pred.reshape(-1, pred.shape[-1])
        target = target.reshape(-1, target.shape[-1])
        # The trainer keeps the final partial chunk of each variable-length
        # sequence.  ``torch.stft(..., center=True, pad_mode="reflect")``
        # cannot reflect-pad a chunk shorter than half its window.  Pad only
        # this loss input; waveform loss and recurrent state still see the
        # original unpadded chunk.
        if pred.shape[-1] < self.win_length:
            pad = self.win_length - pred.shape[-1]
            pred = F.pad(pred, (0, pad))
            target = F.pad(target, (0, pad))
        window = self.window.to(device=pred.device, dtype=pred.dtype)
        pred_mag = _stft_magnitude(pred, self.n_fft, self.hop_length, self.win_length, window)
        target_mag = _stft_magnitude(
            target, self.n_fft, self.hop_length, self.win_length, window
        )
        sc = torch.norm(target_mag - pred_mag, p="fro") / (
            torch.norm(target_mag, p="fro") + 1e-8
        )
        mag = torch.mean(torch.abs(torch.log(pred_mag) - torch.log(target_mag)))
        return self.sc_weight * sc + self.mag_weight * mag


class MultiResolutionSTFTLoss(FBMXLoss):
    def __init__(
        self,
        fft_sizes: tuple[int, ...] = (512, 1024, 2048),
        hop_sizes: tuple[int, ...] = (128, 256, 512),
        win_lengths: tuple[int, ...] = (512, 1024, 2048),
        sc_weight: float = 1.0,
        mag_weight: float = 1.0,
        max_length: int | None = None,
    ) -> None:
        super().__init__()
        if not (len(fft_sizes) == len(hop_sizes) == len(win_lengths)):
            raise ValueError("fft_sizes, hop_sizes and win_lengths must be the same length")
        losses = []
        for n_fft, hop, win in zip(fft_sizes, hop_sizes, win_lengths):
            if max_length is not None and win > max_length:
                warnings.warn(
                    f"skipping STFT resolution n_fft={n_fft} win={win}: longer than "
                    f"the {max_length}-sample training chunk",
                    stacklevel=2,
                )
                continue
            losses.append(STFTLoss(n_fft, hop, win, sc_weight, mag_weight))
        if not losses:
            raise ValueError(
                "every STFT resolution was skipped; lower fft_sizes or raise the chunk size"
            )
        self.losses = torch.nn.ModuleList(losses)

    def forward(self, pred, target, aux=None):
        total = pred.new_zeros(())
        for loss in self.losses:
            total = total + loss(pred, target)
        return total / len(self.losses)
