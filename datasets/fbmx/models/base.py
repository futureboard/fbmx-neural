"""Model contract shared by every FBMX architecture.

The contract is deliberately narrow, because the eventual consumer is a Rust
audio callback and not a Python script:

* **Causal.**  Sample ``n`` of the output may depend on inputs ``<= n`` and on
  the carried state.  No lookahead, no bidirectionality, no global
  normalisation over a whole file.
* **Stateful.**  Everything the model needs to continue from where it stopped
  lives in an explicit ``state`` object.  There is no hidden buffer inside the
  module, so two calls of 512 samples must equal one call of 1024.
* **Shape-stable.**  Audio is ``[batch, channels, time]`` everywhere, mono for
  now (``channels == 1``).  Parameters arrive as a
  :class:`~fbmx.conditioning.ParamBatch`, never as positional floats.
* **Self-describing.**  :meth:`StreamingModel.export_spec` returns everything
  the ``.fbmx`` writer needs, so adding an architecture never means editing the
  exporter.

A future S4 / state-space backbone slots in by implementing this same
interface: its recurrent state is just another tensor tuple, and the trainer,
the streaming runner and the exporter do not change.
"""

from __future__ import annotations

import abc
from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn

from fbmx.conditioning import ConditioningSchema, ParamBatch

__all__ = [
    "StreamingModel",
    "ConditioningEncoder",
    "FiLM",
    "detach_state",
    "state_to_device",
]

State = Any  # tensor / tuple / list / None, architecture-defined


def detach_state(state: State) -> State:
    """Detach a state tree from the autograd graph (the TBPTT boundary)."""
    if state is None:
        return None
    if isinstance(state, torch.Tensor):
        return state.detach()
    if isinstance(state, (tuple, list)):
        out = [detach_state(s) for s in state]
        return tuple(out) if isinstance(state, tuple) else out
    if isinstance(state, dict):
        return {k: detach_state(v) for k, v in state.items()}
    raise TypeError(f"cannot detach state of type {type(state)!r}")


def state_to_device(state: State, device: torch.device | str) -> State:
    if state is None:
        return None
    if isinstance(state, torch.Tensor):
        return state.to(device)
    if isinstance(state, (tuple, list)):
        out = [state_to_device(s, device) for s in state]
        return tuple(out) if isinstance(state, tuple) else out
    if isinstance(state, dict):
        return {k: state_to_device(v, device) for k, v in state.items()}
    raise TypeError(f"cannot move state of type {type(state)!r}")


class ConditioningEncoder(nn.Module):
    """Turn a :class:`ParamBatch` into a single conditioning vector.

    Continuous controls pass through as their normalised ``[-1, 1]`` values;
    categorical controls index a learned embedding table.  The concatenation is
    optionally projected by a small MLP (``proj_dim``), which is what the
    conditioning-mechanism literature calls a hyper-conditioning network.

    Emits ``[B, 0]`` for an unconditioned schema, so downstream code can
    concatenate unconditionally.
    """

    def __init__(self, schema: ConditioningSchema, proj_dim: int | None = None) -> None:
        super().__init__()
        self.schema = schema
        self.embeddings = nn.ModuleList(
            [nn.Embedding(p.num_categories, p.embedding_dim) for p in schema.categorical]
        )
        self.raw_dim = schema.cond_dim
        self.proj: nn.Module | None = None
        if proj_dim and self.raw_dim > 0:
            self.proj = nn.Sequential(
                nn.Linear(self.raw_dim, proj_dim),
                nn.Tanh(),
            )
            self.out_dim = proj_dim
        else:
            self.out_dim = self.raw_dim

    def forward(self, params: ParamBatch | None, batch_size: int) -> torch.Tensor:
        device = next((p.device for p in self.parameters()), None)
        if self.raw_dim == 0:
            ref = torch.zeros(0, device=device)
            return ref.new_zeros((batch_size, 0))
        if params is None:
            params = self.schema.empty_batch(batch_size, device=device or "cpu")
        params = params.expand_to(batch_size)
        pieces = [params.continuous.to(dtype=torch.float32)]
        for i, emb in enumerate(self.embeddings):
            pieces.append(emb(params.categorical[:, i]))
        cond = torch.cat(pieces, dim=-1)
        if self.proj is not None:
            cond = self.proj(cond)
        return cond


class FiLM(nn.Module):
    """Feature-wise linear modulation: ``y = gamma(c) * x + beta(c)``.

    Applied per channel and constant over the block, i.e. the *static* FiLM of
    the conditioning-mechanism literature.  Time-varying variants (TFiLM,
    TVFiLM) need a pooled summary over a window, which is a lookahead-free but
    block-size dependent operation -- deliberately not used in the realtime
    baseline; see README.
    """

    def __init__(self, cond_dim: int, channels: int) -> None:
        super().__init__()
        self.channels = channels
        self.enabled = cond_dim > 0
        if self.enabled:
            self.to_scale_shift = nn.Linear(cond_dim, 2 * channels)
            nn.init.zeros_(self.to_scale_shift.weight)
            nn.init.zeros_(self.to_scale_shift.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """``x`` is ``[B, C, T]``, ``cond`` is ``[B, cond_dim]``."""
        if not self.enabled or cond.shape[-1] == 0:
            return x
        scale_shift = self.to_scale_shift(cond)
        gamma, beta = scale_shift.chunk(2, dim=-1)
        return x * (1.0 + gamma).unsqueeze(-1) + beta.unsqueeze(-1)


class StreamingModel(nn.Module, abc.ABC):
    """Base class for every causal, stateful FBMX model."""

    #: registry key, set by :func:`fbmx.models.register_model`
    model_type: str = "abstract"

    def __init__(
        self,
        *,
        sample_rate: int = 48000,
        schema: ConditioningSchema | None = None,
        channels: int = 1,
    ) -> None:
        super().__init__()
        if channels != 1:
            raise NotImplementedError("FBMX v0 models are mono")
        self.sample_rate = int(sample_rate)
        self.channels = channels
        self.schema = schema or ConditioningSchema()

    # -- streaming contract --------------------------------------------
    @abc.abstractmethod
    def init_state(
        self,
        batch_size: int = 1,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> State:
        """A zeroed state, equivalent to "silence forever before now"."""

    @abc.abstractmethod
    def forward(
        self,
        x: torch.Tensor,
        params: ParamBatch | None = None,
        state: State = None,
    ) -> tuple[torch.Tensor, State]:
        """Process ``x`` ``[B, 1, T]`` and return ``(y [B, 1, T], new_state)``."""

    def forward_aux(
        self,
        x: torch.Tensor,
        params: ParamBatch | None = None,
        state: State = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], State]:
        """Like :meth:`forward`, but also returning auxiliary predictions.

        A dynamics model can be asked to predict more than audio -- the gain
        trajectory, the sidechain control voltage -- when the teacher exports
        those traces.  Predictions are keyed ``pred_<trace>`` to match the
        dataset's ``aux[<trace>]``, which is what
        :class:`fbmx.losses.aux.AuxTraceLoss` pairs up.

        The audio output stays mandatory; auxiliary heads are optional, are
        configured per model, and never affect the audio path at inference --
        the exporter writes their weights, and the Rust runtime ignores them.
        """
        y, new_state = self.forward(x, params, state)
        return y, {}, new_state

    @property
    def aux_heads(self) -> tuple[str, ...]:
        """Names of the auxiliary traces this model predicts."""
        return ()

    @property
    def receptive_field(self) -> int:
        """Samples of input history needed for a sample of output.

        Recurrent models report ``1``: with a carried state they need no input
        history at all.  Convolutional models report their true window, which
        the streaming runner uses to size its input cache.
        """
        return 1

    @property
    def is_recurrent(self) -> bool:
        return True

    def detach_state(self, state: State) -> State:
        return detach_state(state)

    # -- description ----------------------------------------------------
    def hparams(self) -> dict[str, Any]:
        """Constructor keyword arguments, enough to rebuild this model."""
        return {"sample_rate": self.sample_rate}

    def num_parameters(self, trainable_only: bool = True) -> int:
        return sum(
            p.numel() for p in self.parameters() if p.requires_grad or not trainable_only
        )

    def export_spec(self) -> dict[str, Any]:
        """Everything the ``.fbmx`` writer needs about this architecture."""
        return {
            "model_type": self.model_type,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "causal": True,
            "recurrent": self.is_recurrent,
            "receptive_field": self.receptive_field,
            "parameter_count": self.num_parameters(trainable_only=False),
            "hparams": self.hparams(),
            "conditioning": self.schema.to_dict(),
            "output_heads": ["audio", *self.aux_heads],
        }

    # -- helpers for subclasses ----------------------------------------
    @staticmethod
    def _check_input(x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:  # [B, T] -> [B, 1, T], a common caller slip
            x = x.unsqueeze(1)
        if x.dim() != 3 or x.shape[1] != 1:
            raise ValueError(f"expected mono audio [B, 1, T], got {tuple(x.shape)}")
        return x

    @classmethod
    def from_config(
        cls, cfg: Mapping[str, Any], schema: ConditioningSchema | None = None
    ) -> "StreamingModel":
        kwargs = {k: v for k, v in cfg.items() if k != "type"}
        return cls(schema=schema, **kwargs)


def sequence_lengths(total: int, chunk: int) -> Sequence[int]:
    """Split ``total`` into ``chunk``-sized pieces, last one possibly shorter."""
    out = [chunk] * (total // chunk)
    if total % chunk:
        out.append(total % chunk)
    return out
