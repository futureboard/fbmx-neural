"""Causal temporal convolutional network -- experimental, untuned.

Structurally this follows the efficient-DRC-modelling line of work
(arXiv:2102.06200): a stack of dilated causal 1-D convolutions with FiLM
conditioning per block and a residual path, ending in a 1x1 projection back to
one channel.

Two deliberate differences from that paper:

* Their model is evaluated offline on whole segments; ours must be exact under
  block processing, so every block owns an explicit input cache of
  ``(kernel_size - 1) * dilation`` samples and the state carries those caches.
  Streaming equality is a test, not an assumption.
* No batch/global normalisation anywhere -- a statistic computed over a whole
  file is not available to an audio callback.

Not the baseline.  Provided so the receptive-field / cost trade-off against
LSTM-32 can be measured later without rewriting the trainer.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from fbmx.conditioning import ConditioningSchema, ParamBatch
from fbmx.models.base import ConditioningEncoder, FiLM, StreamingModel
from fbmx.models.registry import register_model

__all__ = ["TCNModel", "CausalTCNBlock"]


class CausalTCNBlock(nn.Module):
    """One dilated causal conv + FiLM + activation, with a 1x1 residual."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        cond_dim: int,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation)
        self.film = FiLM(cond_dim, out_channels)
        self.act = nn.PReLU(out_channels)
        self.residual = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv1d(in_channels, out_channels, 1, bias=False)
        )

    def init_cache(self, batch_size: int, device, dtype) -> torch.Tensor:
        return torch.zeros(
            (batch_size, self.in_channels, self.pad), device=device, dtype=dtype
        )

    def forward(
        self, x: torch.Tensor, cond: torch.Tensor, cache: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        padded = torch.cat([cache, x], dim=-1)
        y = self.conv(padded)
        y = self.act(self.film(y, cond))
        new_cache = padded[..., padded.shape[-1] - self.pad :] if self.pad else cache
        return y + self.residual(x), new_cache


@register_model("tcn")
class TCNModel(StreamingModel):
    def __init__(
        self,
        *,
        channels: int = 16,
        num_blocks: int = 6,
        kernel_size: int = 5,
        dilation_growth: int = 3,
        sample_rate: int = 48000,
        schema: ConditioningSchema | None = None,
        cond_proj_dim: int | None = 16,
        residual: bool = True,
    ) -> None:
        super().__init__(sample_rate=sample_rate, schema=schema, channels=1)
        self.hidden_channels = int(channels)
        self.num_blocks = int(num_blocks)
        self.kernel_size = int(kernel_size)
        self.dilation_growth = int(dilation_growth)
        self.residual = bool(residual)

        self.cond_encoder = ConditioningEncoder(self.schema, proj_dim=cond_proj_dim)
        cond_dim = self.cond_encoder.out_dim
        self.cond_dim = cond_dim

        blocks = []
        in_ch = 1
        for i in range(self.num_blocks):
            blocks.append(
                CausalTCNBlock(
                    in_ch,
                    self.hidden_channels,
                    self.kernel_size,
                    self.dilation_growth**i,
                    cond_dim,
                )
            )
            in_ch = self.hidden_channels
        self.blocks = nn.ModuleList(blocks)
        self.out_proj = nn.Conv1d(self.hidden_channels, 1, 1)
        nn.init.zeros_(self.out_proj.bias)

    # -- streaming contract --------------------------------------------
    def init_state(
        self,
        batch_size: int = 1,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> list[torch.Tensor]:
        device = device or next(self.parameters()).device
        return [b.init_cache(batch_size, device, dtype) for b in self.blocks]

    def forward(
        self,
        x: torch.Tensor,
        params: ParamBatch | None = None,
        state=None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        x = self._check_input(x)
        batch = x.shape[0]
        if state is None:
            state = self.init_state(batch, device=x.device, dtype=x.dtype)
        cond = self.cond_encoder(params, batch)

        h = x
        new_state: list[torch.Tensor] = []
        for block, cache in zip(self.blocks, state):
            h, cache = block(h, cond, cache)
            new_state.append(cache)
        y = self.out_proj(h)
        if self.residual:
            y = y + x
        return y, new_state

    @property
    def receptive_field(self) -> int:
        return 1 + sum(b.pad for b in self.blocks)

    @property
    def is_recurrent(self) -> bool:
        return False

    # -- description ----------------------------------------------------
    def hparams(self) -> dict[str, Any]:
        return {
            "channels": self.hidden_channels,
            "num_blocks": self.num_blocks,
            "kernel_size": self.kernel_size,
            "dilation_growth": self.dilation_growth,
            "sample_rate": self.sample_rate,
            "cond_proj_dim": (
                self.cond_encoder.out_dim if self.cond_encoder.proj is not None else None
            ),
            "residual": self.residual,
        }

    def export_spec(self) -> dict[str, Any]:
        spec = super().export_spec()
        spec["input_features"] = {"layout": "BCT", "channels": 1, "order": ["audio"]}
        spec["state_spec"] = {
            "kind": "conv_cache",
            "tensors": [f"block{i}_cache" for i in range(self.num_blocks)],
            "cache_lengths": [b.pad for b in self.blocks],
        }
        return spec
