"""FBMX -- Futureboard neural DSP research stack.

A small, deliberately boring framework for training causal, stateful neural
models of audio effects and exporting them to the ``.fbmx`` container that the
Rust runtime will eventually load.

Layering (nothing above may import from something below it in reverse):

    datasets  ->  models  ->  losses  ->  training  ->  export / streaming

Nothing in this package is allowed to assume a particular effect.  Parameter
sets are described by a :class:`fbmx.conditioning.ConditioningSchema` that is
carried from the dataset, through the model, into the exported file.
"""

__version__ = "0.1.0"

# The container format version, bumped independently of the package version.
FBMX_FORMAT_VERSION = 1

from fbmx.conditioning import (  # noqa: E402
    CategoricalParam,
    ConditioningSchema,
    ContinuousParam,
    ParamBatch,
)
from fbmx.device import auto_device, describe_device  # noqa: E402

__all__ = [
    "__version__",
    "FBMX_FORMAT_VERSION",
    "ConditioningSchema",
    "ContinuousParam",
    "CategoricalParam",
    "ParamBatch",
    "auto_device",
    "describe_device",
]
