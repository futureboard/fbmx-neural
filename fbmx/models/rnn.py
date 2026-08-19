"""Shared scaffolding for the recurrent baselines (LSTM, GRU).

Both architectures differ only in the cell and in the shape of the carried
state, so the conditioning, the output head and the residual connection live
here once.

Design notes / deltas from the references:

* Conditioning by **concatenation** to the per-sample input is the default,
  following the parametric RNN amp/compressor line of work (and PANAMA's
  parametric GRU).  It is the cheapest thing that works and costs the runtime
  one extra input row per sample.
* Conditioning by **FiLM** on the recurrent output is also available.  The
  conditioning-mechanism survey (arXiv:2408.04829) finds FiLM-family
  mechanisms generally stronger than plain concatenation for effects with
  strong parameter dependence, so ``conditioning: both`` is a supported
  experiment rather than a claim.
* The output is **residual**: ``y = x + head(h)``.  The network learns the
  difference between dry and wet, which is much easier to fit at low capacity
  than the signal itself, and it makes a fresh model close to a bypass rather
  than to noise.
"""

from __future__ import annotations

from typing import Any, Sequence

import torch
import torch.nn as nn

from fbmx.conditioning import ConditioningSchema, ParamBatch
from fbmx.models.base import ConditioningEncoder, FiLM, StreamingModel

__all__ = ["RecurrentModel"]

CONDITIONING_MODES = ("none", "concat", "film", "both")


class RecurrentModel(StreamingModel):
    """A single-direction RNN with parameter conditioning and a residual head."""

    #: subclasses set these
    rnn_class: type[nn.RNNBase] = nn.LSTM
    has_cell_state: bool = True

    def __init__(
        self,
        *,
        hidden_size: int = 32,
        num_layers: int = 1,
        sample_rate: int = 48000,
        schema: ConditioningSchema | None = None,
        conditioning: str = "concat",
        cond_proj_dim: int | None = None,
        head_hidden: int = 0,
        residual: bool = True,
        dropout: float = 0.0,
        aux_heads: Sequence[str] = (),
    ) -> None:
        super().__init__(sample_rate=sample_rate, schema=schema)
        if conditioning not in CONDITIONING_MODES:
            raise ValueError(
                f"conditioning must be one of {CONDITIONING_MODES}, got {conditioning!r}"
            )
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)
        self.conditioning = conditioning
        self.residual = bool(residual)
        self.head_hidden = int(head_hidden)
        self.dropout = float(dropout)

        self.cond_encoder = ConditioningEncoder(self.schema, proj_dim=cond_proj_dim)
        cond_dim = self.cond_encoder.out_dim
        self.cond_dim = cond_dim

        use_concat = conditioning in ("concat", "both")
        self.input_size = 1 + (cond_dim if use_concat else 0)
        self.rnn = self.rnn_class(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=False,  # never: the runtime has no future
            dropout=self.dropout if self.num_layers > 1 else 0.0,
        )
        self.film = (
            FiLM(cond_dim, self.hidden_size)
            if conditioning in ("film", "both")
            else None
        )
        if self.head_hidden > 0:
            self.head: nn.Module = nn.Sequential(
                nn.Linear(self.hidden_size, self.head_hidden),
                nn.Tanh(),
                nn.Linear(self.head_hidden, 1),
            )
        else:
            self.head = nn.Linear(self.hidden_size, 1)

        # Optional auxiliary heads. They read the same recurrent state as the
        # audio head, so asking the model to predict the gain trajectory is a
        # statement that the state must *contain* that trajectory -- which is
        # the point: an LSTM can fit a compressor's waveform while representing
        # its dynamics badly, and this is the cheapest pressure against that.
        # They cost the audio path nothing at inference and the Rust runtime
        # ignores their weights.
        self._aux_heads = tuple(aux_heads)
        self.aux = nn.ModuleDict(
            {name: nn.Linear(self.hidden_size, 1) for name in self._aux_heads}
        )

    # -- streaming contract --------------------------------------------
    def init_state(
        self,
        batch_size: int = 1,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ):
        device = device or next(self.parameters()).device
        shape = (self.num_layers, batch_size, self.hidden_size)
        h = torch.zeros(shape, device=device, dtype=dtype)
        if self.has_cell_state:
            return (h, torch.zeros(shape, device=device, dtype=dtype))
        return h

    @property
    def aux_heads(self) -> tuple[str, ...]:
        return self._aux_heads

    def forward(
        self,
        x: torch.Tensor,
        params: ParamBatch | None = None,
        state=None,
    ) -> tuple[torch.Tensor, Any]:
        y, _, new_state = self.forward_aux(x, params, state)
        return y, new_state

    def forward_aux(
        self,
        x: torch.Tensor,
        params: ParamBatch | None = None,
        state=None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], Any]:
        x = self._check_input(x)
        batch, _, n_samples = x.shape
        if state is None:
            state = self.init_state(batch, device=x.device, dtype=x.dtype)

        seq = x.transpose(1, 2)  # [B, T, 1]
        cond = self.cond_encoder(params, batch)  # [B, D]
        if self.conditioning in ("concat", "both") and cond.shape[-1] > 0:
            seq = torch.cat([seq, cond.unsqueeze(1).expand(-1, n_samples, -1)], dim=-1)

        h_seq, new_state = self.rnn(seq, state)  # [B, T, H]
        if self.film is not None:
            h_seq = self.film(h_seq.transpose(1, 2), cond).transpose(1, 2)

        y = self.head(h_seq).transpose(1, 2)  # [B, 1, T]
        if self.residual:
            y = y + x

        aux = {
            f"pred_{name}": head(h_seq).transpose(1, 2)
            for name, head in self.aux.items()
        }
        return y, aux, new_state

    # -- description ----------------------------------------------------
    def hparams(self) -> dict[str, Any]:
        return {
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "sample_rate": self.sample_rate,
            "conditioning": self.conditioning,
            "cond_proj_dim": (
                self.cond_encoder.out_dim if self.cond_encoder.proj is not None else None
            ),
            "head_hidden": self.head_hidden,
            "residual": self.residual,
            "dropout": self.dropout,
            "aux_heads": list(self._aux_heads),
        }

    def export_spec(self) -> dict[str, Any]:
        spec = super().export_spec()
        spec["input_features"] = {
            "layout": "BCT",
            "channels": 1,
            "rnn_input_size": self.input_size,
            "order": ["audio"]
            + (
                ["conditioning"]
                if self.conditioning in ("concat", "both") and self.cond_dim
                else []
            ),
        }
        spec["state_spec"] = {
            "kind": "lstm" if self.has_cell_state else "gru",
            "tensors": ["h", "c"] if self.has_cell_state else ["h"],
            "shape": [self.num_layers, "batch", self.hidden_size],
        }
        return spec
