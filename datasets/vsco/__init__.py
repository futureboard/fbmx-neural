"""Reproducible VSCO 2 CE ingestion and preprocessing for Solfege/FBMX."""

from .metadata import (
    PIPELINE_VERSION,
    ParsedMetadata,
    detect_pitch,
    midi_to_hz,
    parse_sample_metadata,
    stable_sample_id,
)
from .fbmx_pairs import build_vsco_fbmx_pairs, velocity_for_dynamic

__all__ = [
    "PIPELINE_VERSION",
    "ParsedMetadata",
    "detect_pitch",
    "midi_to_hz",
    "parse_sample_metadata",
    "stable_sample_id",
    "build_vsco_fbmx_pairs",
    "velocity_for_dynamic",
]
