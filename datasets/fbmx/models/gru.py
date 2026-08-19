"""GRU variant -- experimental, untuned.

Same interface and same conditioning as :class:`~fbmx.models.lstm.LSTMModel`,
one fewer gate and no cell state (~25% fewer parameters at equal width).
Provided so the comparison can be run later; LSTM-32 remains the baseline.
"""

from __future__ import annotations

import torch.nn as nn

from fbmx.models.registry import register_model
from fbmx.models.rnn import RecurrentModel

__all__ = ["GRUModel"]


@register_model("gru")
class GRUModel(RecurrentModel):
    rnn_class = nn.GRU
    has_cell_state = False
