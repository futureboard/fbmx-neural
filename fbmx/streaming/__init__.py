"""Block-wise, stateful inference."""

from fbmx.streaming.inference import (
    StreamingProcessor,
    block_schedule,
    process_blocked,
    process_offline,
    realtime_factor,
    streaming_equivalence,
)

__all__ = [
    "StreamingProcessor",
    "process_offline",
    "process_blocked",
    "streaming_equivalence",
    "realtime_factor",
    "block_schedule",
]
