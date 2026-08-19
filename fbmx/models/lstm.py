"""LSTM baseline -- the required V0 architecture.

``LSTM-32`` means: one layer, 32 hidden units, mono, causal, stateful, 48 kHz
reference rate, residual output head.  Roughly 4.5k parameters unconditioned,
which is small enough to run per-sample in a Rust audio callback without
vectorisation heroics.

Why an RNN and not a TCN for V0: a recurrent state is an unbounded memory at
O(1) cost per sample, and the streaming implementation is trivially exact --
carry ``(h, c)`` across the block boundary and the result is bit-comparable to
offline processing.  A TCN needs a receptive-field-sized input cache and, for
the long release times of a compressor, that window gets expensive.
"""

from __future__ import annotations

import torch.nn as nn

from fbmx.models.registry import register_model
from fbmx.models.rnn import RecurrentModel

__all__ = ["LSTMModel"]


@register_model("lstm")
class LSTMModel(RecurrentModel):
    rnn_class = nn.LSTM
    has_cell_state = True
