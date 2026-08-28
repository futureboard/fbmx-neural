"""Turn note records into the vectors the Performer reads and predicts.

Two vocabularies, deliberately separate:

**Inputs** are things a score knows. Pitch, written duration, tempo, where the
note falls in the bar, what came before and after, where it sits in the phrase.
Nothing here is derived from the recording, because at runtime there is no
recording — the whole point is to play a score that has never been performed.

**Targets** are things only a performance knows, and every one of them is
something section 4's record actually measured. Vibrato carries a mask: a
sixteenth note has no vibrato rate to predict, and training a regression head
against a fabricated zero would teach the model that short notes are notes with
no vibrato rather than notes where the question does not arise.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

FEATURE_SCHEMA_VERSION = 1

#: Order is the contract between training, export, and the Rust runtime.
INPUT_FEATURES: tuple[str, ...] = (
    "pitch_norm",
    "pitch_class_sin",
    "pitch_class_cos",
    "log_duration_beats",
    "log_tempo",
    "beat_phase_sin",
    "beat_phase_cos",
    "previous_interval",
    "next_interval",
    "rest_before",
    "rest_after",
    "phrase_progress",
    "log_phrase_length",
    "velocity",
    "is_phrase_start",
    "is_phrase_end",
)

#: Regression targets, in output order.
TARGETS: tuple[str, ...] = (
    "onset_deviation",
    "log_duration_ratio",
    "pitch_offset",
    "entry_offset",
    "intensity",
    "vibrato_rate",
    "vibrato_depth",
    "vibrato_delay",
)

#: Index of the classification output, appended after the regressions.
VIBRATO_PRESENT_INDEX = len(TARGETS)
OUTPUT_SIZE = len(TARGETS) + 1

#: Scales that put each target roughly in [-1, 1] so one loss weight per target
#: is not secretly a weighting by unit. Stored with the model.
TARGET_SCALES: dict[str, float] = {
    "onset_deviation": 0.100,  # seconds
    "log_duration_ratio": 1.0,  # log2 ratio
    "pitch_offset": 50.0,  # cents
    "entry_offset": 100.0,  # cents
    "intensity": 1.0,  # normalised level
    "vibrato_rate": 8.0,  # Hz
    "vibrato_depth": 50.0,  # cents peak-to-peak
    "vibrato_delay": 0.500,  # seconds
}


def _clamp(value: float, low: float, high: float) -> float:
    return float(min(max(value, low), high))


def note_input_vector(record: dict[str, Any]) -> np.ndarray:
    """The score-side description of one note."""

    pitch = float(record["score_pitch"])
    tempo = float(record.get("tempo_bpm") or 120.0)
    beat_seconds = 60.0 / max(tempo, 1e-6)
    duration_beats = float(record["score_duration_seconds"]) / beat_seconds
    numerator = float((record.get("time_signature") or [4, 4])[0]) or 4.0
    beat_in_bar = float(record.get("beat_in_bar") or 0.0)
    phrase_length = max(int(record.get("phrase_length") or 1), 1)
    phrase_position = int(record.get("phrase_position") or 0)

    previous_interval = record.get("previous_interval")
    next_interval = record.get("next_interval")
    rest_before = record.get("rest_before_seconds")
    rest_after = record.get("rest_after_seconds")

    # Bar position as a phase rather than a number: beat 0 and beat 4 of a 4/4
    # bar are the same place musically, and a raw counter says they are as far
    # apart as possible.
    phase = 2.0 * np.pi * (beat_in_bar / numerator)
    pitch_class_phase = 2.0 * np.pi * ((pitch % 12.0) / 12.0)

    values = {
        "pitch_norm": (pitch - 69.0) / 24.0,
        "pitch_class_sin": float(np.sin(pitch_class_phase)),
        "pitch_class_cos": float(np.cos(pitch_class_phase)),
        "log_duration_beats": _clamp(float(np.log2(max(duration_beats, 1e-3))), -6.0, 4.0) / 4.0,
        "log_tempo": _clamp(float(np.log2(tempo / 120.0)), -2.0, 2.0),
        "beat_phase_sin": float(np.sin(phase)),
        "beat_phase_cos": float(np.cos(phase)),
        "previous_interval": _clamp((previous_interval or 0.0) / 12.0, -2.0, 2.0),
        "next_interval": _clamp((next_interval or 0.0) / 12.0, -2.0, 2.0),
        "rest_before": _clamp(float(np.log1p(max(rest_before or 0.0, 0.0))), 0.0, 2.0),
        "rest_after": _clamp(float(np.log1p(max(rest_after or 0.0, 0.0))), 0.0, 2.0),
        "phrase_progress": phrase_position / max(phrase_length - 1, 1),
        "log_phrase_length": _clamp(float(np.log2(phrase_length)), 0.0, 10.0) / 10.0,
        "velocity": float(record.get("score_velocity") or 64) / 127.0,
        "is_phrase_start": 1.0 if phrase_position == 0 else 0.0,
        "is_phrase_end": 1.0 if phrase_position == phrase_length - 1 else 0.0,
    }
    return np.asarray([values[name] for name in INPUT_FEATURES], dtype=np.float32)


def note_target_vector(record: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, float]:
    """Targets, a per-target mask, and the vibrato-present label.

    The mask is what keeps unmeasured quantities out of the loss. A note with no
    vibrato estimate contributes to the *classifier* — it is a real example of
    "no vibrato here" — but not to the rate, depth, or delay regressions, which
    have nothing to regress towards.
    """

    vibrato = record.get("vibrato") or {}
    present = bool(vibrato.get("present"))
    intensity = record.get("intensity") or {}
    transition = record.get("transition") or {}

    raw = {
        "onset_deviation": float(record.get("onset_deviation_seconds") or 0.0),
        "log_duration_ratio": float(
            np.log2(max(float(record.get("duration_ratio") or 1.0), 1e-3))
        ),
        # Relative to the part's own tuning centre: see `build.py`. Falls back
        # to the raw offset for records written before tuning was measured.
        "pitch_offset": float(
            record.get("pitch_offset_cents_relative")
            if record.get("pitch_offset_cents_relative") is not None
            else (record.get("pitch_offset_cents") or 0.0)
        ),
        "entry_offset": float(transition.get("entry_offset_cents") or 0.0),
        "intensity": float(intensity.get("level") or 0.0),
        "vibrato_rate": float(vibrato.get("rate_hz") or 0.0),
        "vibrato_depth": float(vibrato.get("depth_cents") or 0.0),
        "vibrato_delay": float(vibrato.get("onset_delay_seconds") or 0.0),
    }
    mask = {
        "onset_deviation": 1.0,
        "log_duration_ratio": 1.0,
        "pitch_offset": 1.0,
        "entry_offset": 1.0 if transition.get("entry_offset_cents") is not None else 0.0,
        "intensity": 1.0 if intensity else 0.0,
        "vibrato_rate": 1.0 if present else 0.0,
        "vibrato_depth": 1.0 if present else 0.0,
        "vibrato_delay": 1.0 if present else 0.0,
    }
    targets = np.asarray(
        [raw[name] / TARGET_SCALES[name] for name in TARGETS], dtype=np.float32
    )
    mask_vector = np.asarray([mask[name] for name in TARGETS], dtype=np.float32)
    return targets, mask_vector, 1.0 if present else 0.0


@dataclass
class Sequence_:
    """One part's notes, in order."""

    part_id: str
    piece_id: str
    inputs: np.ndarray  # (notes, INPUT_FEATURES)
    targets: np.ndarray  # (notes, TARGETS)
    mask: np.ndarray  # (notes, TARGETS)
    vibrato_present: np.ndarray  # (notes,)
    records: list[dict[str, Any]]


def load_sequences(path: str | Path) -> list[Sequence_]:
    """Group a JSONL split into per-part sequences.

    The Performer reads a phrase, not a bag of notes, so the natural unit is the
    part: its notes are contiguous, ordered, and share a tempo and a player.
    """

    by_part: dict[str, list[dict[str, Any]]] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        by_part.setdefault(record["part_id"], []).append(record)

    sequences: list[Sequence_] = []
    for part_id in sorted(by_part):
        records = sorted(by_part[part_id], key=lambda item: item["note_index"])
        inputs = np.stack([note_input_vector(record) for record in records])
        target_rows, mask_rows, present_rows = [], [], []
        for record in records:
            target, mask, present = note_target_vector(record)
            target_rows.append(target)
            mask_rows.append(mask)
            present_rows.append(present)
        sequences.append(
            Sequence_(
                part_id=part_id,
                piece_id=records[0]["piece_id"],
                inputs=inputs,
                targets=np.stack(target_rows),
                mask=np.stack(mask_rows),
                vibrato_present=np.asarray(present_rows, dtype=np.float32),
                records=records,
            )
        )
    return sequences


def input_normalization(sequences: Iterable[Sequence_]) -> tuple[np.ndarray, np.ndarray]:
    """Mean and standard deviation of each input feature, over training only.

    Stored with the checkpoint and written into the exported model, so the Rust
    runtime normalises exactly as training did rather than re-deriving statistics
    from whatever score it happens to be given.
    """

    stacked = np.concatenate([sequence.inputs for sequence in sequences], axis=0)
    mean = stacked.mean(axis=0)
    std = stacked.std(axis=0)
    # A constant feature has no scale to divide by; leave it alone rather than
    # amplifying its floating-point dust into a large input.
    std[std < 1e-6] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)
