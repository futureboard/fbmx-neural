"""Model implementations and the registry that maps config strings to them."""

from fbmx.models.base import (
    ConditioningEncoder,
    FiLM,
    StreamingModel,
    detach_state,
    state_to_device,
)
from fbmx.models.registry import (
    MODEL_REGISTRY,
    build_model,
    get_model_class,
    register_model,
)

# Importing these registers them.  Add new architectures here.
from fbmx.models.gru import GRUModel  # noqa: F401,E402
from fbmx.models.lstm import LSTMModel  # noqa: F401,E402
from fbmx.models.tcn import TCNModel  # noqa: F401,E402

__all__ = [
    "StreamingModel",
    "ConditioningEncoder",
    "FiLM",
    "detach_state",
    "state_to_device",
    "MODEL_REGISTRY",
    "register_model",
    "get_model_class",
    "build_model",
    "LSTMModel",
    "GRUModel",
    "TCNModel",
]
