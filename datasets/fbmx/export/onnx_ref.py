"""Optional ONNX export, for cross-checking only.

ONNX is **not** part of the delivery path.  Training is PyTorch, the model is
``.fbmx``, and production inference is a pure Rust runtime.  What ONNX is good
for here is a second opinion: if ``onnxruntime`` happens to be installed, an
exported graph gives an independent implementation to compare the Rust runtime
against while it is being written.

Nothing imports this module by default and neither ``onnx`` nor ``onnxruntime``
is a dependency of this package.  Calling it without them raises with an
instruction rather than crashing obscurely.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from fbmx.conditioning import ParamBatch
from fbmx.models.base import StreamingModel

__all__ = ["StatefulONNXWrapper", "export_onnx", "onnx_available"]


def onnx_available() -> bool:
    try:  # pragma: no cover - environment dependent
        import onnx  # noqa: F401

        return True
    except Exception:
        return False


class StatefulONNXWrapper(nn.Module):
    """Flattens the state tuple into explicit tensor inputs and outputs.

    ONNX has no notion of "the state object this architecture happens to use",
    so a recurrent model becomes ``(x, h, c, cont, cat) -> (y, h', c')``.  Only
    the recurrent models are supported; the TCN's variable-length cache list is
    exportable in principle and not worth the effort for a reference check.
    """

    def __init__(self, model: StreamingModel) -> None:
        super().__init__()
        if not model.is_recurrent:
            raise NotImplementedError("ONNX reference export covers recurrent models only")
        self.model = model.eval()
        self.has_cell = getattr(model, "has_cell_state", True)

    def forward(self, x, h, c, continuous, categorical):
        state = (h, c) if self.has_cell else h
        params = ParamBatch(continuous, categorical)
        y, new_state = self.model(x, params, state)
        if self.has_cell:
            return y, new_state[0], new_state[1]
        return y, new_state, c


def export_onnx(
    model: StreamingModel,
    path: str | Path,
    *,
    block_size: int = 128,
    opset: int = 17,
) -> Path:
    """Write an ONNX graph with a dynamic time axis.  Reference use only."""
    if not onnx_available():
        raise RuntimeError(
            "ONNX export needs the optional 'onnx' package: pip install onnx "
            "(the FBMX runtime never requires it)"
        )
    wrapper = StatefulONNXWrapper(model)
    schema = model.schema
    state = model.init_state(1)
    h, c = state if isinstance(state, tuple) else (state, torch.zeros_like(state))
    params = schema.empty_batch(1)
    args = (
        torch.zeros(1, 1, block_size),
        h,
        c,
        params.continuous,
        params.categorical,
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper,
        args,
        str(path),
        input_names=["audio_in", "h_in", "c_in", "cond_continuous", "cond_categorical"],
        output_names=["audio_out", "h_out", "c_out"],
        dynamic_axes={"audio_in": {2: "time"}, "audio_out": {2: "time"}},
        opset_version=opset,
    )
    return path
