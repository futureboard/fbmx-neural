"""Auxiliary-target losses: the extension points that matter for dynamics.

Named ``auxiliary`` and not ``aux`` because ``AUX`` is a reserved DOS device
name. Windows itself no longer minds -- a file called ``aux.py`` can be created
and read here -- but git does: ``core.protectNTFS`` is on by default on
Windows and refuses to index such a path, reporting the file as missing even
while it sits there in the listing::

    error: open("fbmx/losses/aux.py"): No such file or directory
    error: unable to index file 'fbmx/losses/aux.py'

The guard is right to exist and turning it off would only move the failure to
whoever next checks the repository out on Windows, so the module is named
around the problem instead.

A compressor is badly served by waveform + spectral terms alone.  What a
listener reacts to is the *gain trajectory* -- how fast it grabs, how it lets
go, what it does on the second hit -- and that trajectory is a small fraction
of the signal energy, so it is nearly free to get wrong under an L1 loss.

Three shapes of extra supervision are anticipated, in increasing order of how
much the data has to give you:

1. **Envelope loss** -- needs nothing extra.  Compare envelope followers run
   over prediction and target.  Available today.
2. **Transient loss** -- needs nothing extra.  Weight the error by the local
   rate of change of the target, so attacks count for more.  Available today.
3. **Gain-reduction / control-voltage loss** -- needs the teacher or the rig to
   have exported the trace (``aux["gain"]``, ``aux["cv"]``).  A hardware
   capture usually cannot provide it; a circuit-model teacher can, and a
   gray-box model can be asked to predict it as a second output head.

(3) is implemented but inert unless the dataset supplies the key, and it
raises rather than silently scoring zero -- a loss that quietly does nothing is
worse than one that is absent.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from fbmx.losses.waveform import FBMXLoss, one_pole_envelope

__all__ = ["EnvelopeLoss", "TransientLoss", "AuxTraceLoss"]


class EnvelopeLoss(FBMXLoss):
    """L1 between the smoothed envelopes of prediction and target."""

    def __init__(self, sample_rate: int = 48000, time_ms: float = 20.0) -> None:
        super().__init__()
        self.coeff = float(torch.exp(torch.tensor(-1.0 / (sample_rate * time_ms * 1e-3))))

    def forward(self, pred, target, aux=None):
        return torch.mean(
            torch.abs(one_pole_envelope(pred, self.coeff) - one_pole_envelope(target, self.coeff))
        )


class TransientLoss(FBMXLoss):
    """Waveform L1 weighted by the target's local rate of change.

    Emphasises attacks without needing any extra target.  ``floor`` keeps
    steady-state material from being ignored entirely.
    """

    def __init__(self, floor: float = 0.1) -> None:
        super().__init__()
        self.floor = floor

    def forward(self, pred, target, aux=None):
        delta = torch.abs(F.pad(target[..., 1:] - target[..., :-1], (1, 0)))
        weight = self.floor + delta / (delta.amax(dim=-1, keepdim=True) + 1e-8)
        return torch.mean(weight * torch.abs(pred - target))


class AuxTraceLoss(FBMXLoss):
    """Supervise an auxiliary trace the model predicts alongside audio.

    ``key`` names both the dataset target (``aux[key]``) and the model's
    prediction (``aux["pred_" + key]``).

    ``scale`` multiplies the *target* before the comparison, so a trace in
    physical units can be brought to the order of magnitude the head can reach
    from its initialisation.  A gain-reduction trace runs 0 to -25 dB; asking a
    freshly initialised linear head on a ``tanh``-bounded state to output -25
    is asking for a long, slow fit that contributes nothing for the first
    several epochs.  ``scale: 0.04`` makes the head predict dB/25 instead.  The
    factor is part of the training configuration, not of the model: the head is
    ignored at inference and never reaches the runtime.
    """

    def __init__(self, key: str = "gain", norm: str = "l1", scale: float = 1.0) -> None:
        super().__init__()
        self.key = key
        self.norm = norm
        self.scale = float(scale)

    def forward(self, pred, target, aux=None):
        aux = aux or {}
        pred_key = f"pred_{self.key}"
        if self.key not in aux or pred_key not in aux:
            raise KeyError(
                f"AuxTraceLoss({self.key!r}) needs aux[{self.key!r}] from the dataset and "
                f"aux[{pred_key!r}] from the model; got keys {sorted(aux)}"
            )
        diff = aux[pred_key] - aux[self.key] * self.scale
        return torch.mean(torch.abs(diff)) if self.norm == "l1" else torch.mean(diff**2)
