"""Dataset adapters.

Everything here yields the same thing -- :class:`PairedItem` sequences plus a
:class:`DatasetInfo` -- regardless of whether the audio was generated, rendered
by a teacher DSP, or captured from hardware.
"""

from typing import Any, Mapping

from fbmx.datasets.base import (
    SOURCE_TYPES,
    DatasetInfo,
    PairedItem,
    PairedSequenceDataset,
    collate_pairs,
    dump_provenance,
    file_sha256,
)
from fbmx.datasets.manifest import DatasetManifest, ManifestEntry
from fbmx.datasets.paired_audio import PairedAudioDataset, read_audio, write_wav
from fbmx.datasets.synthetic import (
    SIGNAL_FAMILIES,
    SYNTHETIC_SCHEMA,
    SyntheticSmokeDataset,
    SyntheticTeacher,
)

__all__ = [
    "SOURCE_TYPES",
    "DatasetInfo",
    "PairedItem",
    "PairedSequenceDataset",
    "collate_pairs",
    "dump_provenance",
    "file_sha256",
    "DatasetManifest",
    "ManifestEntry",
    "PairedAudioDataset",
    "read_audio",
    "write_wav",
    "SyntheticSmokeDataset",
    "SyntheticTeacher",
    "SIGNAL_FAMILIES",
    "SYNTHETIC_SCHEMA",
    "build_dataset",
]

DATASET_REGISTRY = {
    "synthetic_smoke": SyntheticSmokeDataset,
    "paired_audio": PairedAudioDataset,
}


def build_dataset(cfg: Mapping[str, Any], split: str | None = None) -> PairedSequenceDataset:
    """Build from ``{"type": "synthetic_smoke", ...}``.

    ``split`` from the caller overrides the config, so one dataset block can
    serve both the train and the validation loader.
    """
    cfg = dict(cfg)
    name = cfg.pop("type", None)
    if name is None:
        raise ValueError("dataset config needs a 'type' key")
    if name not in DATASET_REGISTRY:
        raise KeyError(f"unknown dataset {name!r}; known: {sorted(DATASET_REGISTRY)}")
    if split is not None:
        cfg["split"] = split
    return DATASET_REGISTRY[name](**cfg)
