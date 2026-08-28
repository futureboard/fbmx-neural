"""Turn a score into a Solfage performance document.

The output is the performance JSON `solfege-tools` already renders and the
Pitch editor already means the same thing by: notes with cent-deviation pitch
curves and an expression lane. There is deliberately no Performer-only format.
Whatever the model produces is ordinary project data — a curve the user can drag
a point on, a dynamics lane they can redraw — and if the Performer is never run
the same document can be written by hand.

Curve synthesis is where predicted parameters become something an engine can
play. The model predicts a compact description of a note (how late, how long,
how sharp, how much portamento, how loud, whether and how it vibrates) and this
module renders that description onto the control grid, bounded at every step by
the ranges actually measured in URMP. The bounds are not taste: a model asked to
extrapolate will occasionally emit a 12 Hz vibrato or a note three times its
written length, and an instrument that plays those is broken in a way no
listener will attribute to the model.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from ..datasets.urmp.annotations import read_score_notes, score_tempo_bpm, score_time_signature
from .features import (
    TARGET_SCALES,
    TARGETS,
    VIBRATO_PRESENT_INDEX,
    note_input_vector,
)
from .model import Performer, PerformerConfig

PERFORMANCE_FORMAT = "solfage-performance-1"

#: Bounds taken from the measured URMP violin distributions (the p1..p99 range
#: of `datasets/urmp-violin`), so generated performances stay inside what a
#: violinist was actually observed to do.
LIMITS: dict[str, tuple[float, float]] = {
    "onset_deviation": (-0.180, 0.180),  # seconds
    "duration_ratio": (0.26, 2.00),
    "pitch_offset": (-45.0, 45.0),  # cents
    "entry_offset": (-120.0, 120.0),  # cents
    "intensity": (0.10, 1.50),
    "vibrato_rate": (4.0, 7.0),  # Hz
    "vibrato_depth": (8.0, 55.0),  # cents peak-to-peak
    "vibrato_delay": (0.0, 0.45),  # seconds
}

#: How long a portamento takes to resolve into the note, in seconds.
PORTAMENTO_SECONDS = 0.09

#: How long vibrato takes to reach full depth once it starts.
VIBRATO_FADE_SECONDS = 0.15

#: Curve resolution. 100 Hz is URMP's own F0 grid and the rate the engine
#: interpolates from; finer would be inventing detail the model never saw.
CONTROL_RATE_HZ = 100.0


def _clamp(value: float, bounds: tuple[float, float]) -> float:
    low, high = bounds
    if not np.isfinite(value):
        return float(np.clip(0.0, low, high))
    return float(np.clip(value, low, high))


@dataclass
class NotePerformance:
    """One note's predicted performance, in physical units."""

    onset_deviation: float
    duration_ratio: float
    pitch_offset: float
    entry_offset: float
    intensity: float
    vibrato_present: bool
    vibrato_rate: float
    vibrato_depth: float
    vibrato_delay: float


def decode(prediction: np.ndarray, *, vibrato_threshold: float = 0.5) -> list[NotePerformance]:
    """Model outputs to bounded physical quantities."""

    out: list[NotePerformance] = []
    for row in prediction:
        values = {
            name: float(row[index]) * TARGET_SCALES[name] for index, name in enumerate(TARGETS)
        }
        present = 1.0 / (1.0 + np.exp(-float(row[VIBRATO_PRESENT_INDEX]))) >= vibrato_threshold
        out.append(
            NotePerformance(
                onset_deviation=_clamp(values["onset_deviation"], LIMITS["onset_deviation"]),
                duration_ratio=_clamp(
                    float(2.0 ** values["log_duration_ratio"]), LIMITS["duration_ratio"]
                ),
                pitch_offset=_clamp(values["pitch_offset"], LIMITS["pitch_offset"]),
                entry_offset=_clamp(values["entry_offset"], LIMITS["entry_offset"]),
                intensity=_clamp(values["intensity"], LIMITS["intensity"]),
                vibrato_present=bool(present),
                vibrato_rate=_clamp(values["vibrato_rate"], LIMITS["vibrato_rate"]),
                vibrato_depth=_clamp(values["vibrato_depth"], LIMITS["vibrato_depth"]),
                vibrato_delay=_clamp(values["vibrato_delay"], LIMITS["vibrato_delay"]),
            )
        )
    return out


def synthesize_pitch_curve(
    performance: NotePerformance, duration_seconds: float, *, seed: int | None = None
) -> list[dict[str, float]]:
    """Render one note's pitch description onto the control grid.

    Three layers, each independently editable afterwards because each is a
    separate predicted quantity rather than a slice of one latent:

    * the note's steady intonation, held for its length;
    * a portamento that resolves into it over the first ~90 ms;
    * vibrato, starting after its predicted delay and fading in rather than
      switching on — a violinist's vibrato has a beginning.
    """

    if duration_seconds <= 0.0:
        return []
    count = max(2, int(round(duration_seconds * CONTROL_RATE_HZ)) + 1)
    times = np.linspace(0.0, duration_seconds, count)

    cents = np.full(count, performance.pitch_offset, dtype=np.float64)

    if abs(performance.entry_offset) > 1e-3:
        # Cosine ease from the entry offset to the steady pitch: flat where it
        # arrives, which is what a finger sliding into position does.
        resolve = min(PORTAMENTO_SECONDS, duration_seconds)
        ramp = np.clip(times / max(resolve, 1e-6), 0.0, 1.0)
        eased = 0.5 - 0.5 * np.cos(ramp * np.pi)
        cents += performance.entry_offset * (1.0 - eased)

    if performance.vibrato_present and performance.vibrato_depth > 0.0:
        delay = min(performance.vibrato_delay, max(duration_seconds - 0.05, 0.0))
        since = np.clip(times - delay, 0.0, None)
        fade = np.clip(since / VIBRATO_FADE_SECONDS, 0.0, 1.0)
        # Deterministic phase: a fixed seed must reproduce a performance
        # exactly, so nothing here draws from an unseeded generator.
        phase = 0.0 if seed is None else (seed % 628) / 100.0
        cents += (
            0.5
            * performance.vibrato_depth
            * fade
            * np.sin(2.0 * np.pi * performance.vibrato_rate * since + phase)
        )

    return [
        {"t": round(float(t), 4), "cents": round(float(c), 2)}
        for t, c in zip(times, cents)
    ]


def generate(
    model: Performer,
    score_notes: Sequence,
    *,
    tempo_bpm: float,
    time_signature: tuple[int, int],
    articulation: str = "sustain_vibrato",
    seed: int | None = None,
    static: bool = False,
) -> dict[str, Any]:
    """Score in, Solfage performance document out.

    With `static=True` the model is not consulted at all and the score is
    written out as-is. That is the A leg of the A/B test and also the proof that
    the manual path does not depend on the Performer existing.
    """

    beat_seconds = 60.0 / max(tempo_bpm, 1e-6)
    numerator = time_signature[0]

    # Phrase segmentation from the score's own rests, matching how the training
    # records were segmented.
    phrases: list[int] = []
    phrase = 0
    previous_offset: float | None = None
    for note in score_notes:
        if previous_offset is not None and note.onset - previous_offset > 0.35:
            phrase += 1
        phrases.append(phrase)
        previous_offset = note.offset
    lengths = {index: phrases.count(index) for index in set(phrases)}
    positions: dict[int, int] = {}

    records: list[dict[str, Any]] = []
    for index, note in enumerate(score_notes):
        current = phrases[index]
        positions[current] = positions.get(current, 0)
        previous = score_notes[index - 1] if index > 0 else None
        following = score_notes[index + 1] if index + 1 < len(score_notes) else None
        records.append(
            {
                "score_pitch": note.midi_pitch,
                "score_onset_seconds": note.onset,
                "score_duration_seconds": note.duration,
                "score_velocity": note.velocity,
                "score_beat": note.onset / beat_seconds,
                "beat_in_bar": (note.onset / beat_seconds) % max(numerator, 1),
                "tempo_bpm": tempo_bpm,
                "time_signature": list(time_signature),
                "previous_interval": (
                    note.midi_pitch - previous.midi_pitch if previous else None
                ),
                "next_interval": (
                    following.midi_pitch - note.midi_pitch if following else None
                ),
                "rest_before_seconds": (note.onset - previous.offset if previous else None),
                "rest_after_seconds": (following.onset - note.offset if following else None),
                "phrase_position": positions[current],
                "phrase_length": lengths[current],
            }
        )
        positions[current] += 1

    if static:
        performances = [
            NotePerformance(0.0, 1.0, 0.0, 0.0, 0.75, False, 5.5, 0.0, 0.0)
            for _ in records
        ]
    else:
        inputs = np.stack([note_input_vector(record) for record in records])
        model.eval()
        with torch.no_grad():
            prediction = model(torch.from_numpy(inputs).unsqueeze(0)).squeeze(0).numpy()
        performances = decode(prediction)

    notes: list[dict[str, Any]] = []
    expression: list[dict[str, float]] = []
    end_seconds = 0.0
    for index, (note, performance) in enumerate(zip(score_notes, performances)):
        start = max(0.0, note.onset + performance.onset_deviation)
        duration = max(0.02, note.duration * performance.duration_ratio)
        end_seconds = max(end_seconds, start + duration)
        entry: dict[str, Any] = {
            "start": round(start, 5),
            "duration": round(duration, 5),
            "note": int(note.midi_pitch),
            "velocity": round(float(np.clip(performance.intensity, 0.05, 1.0)), 4),
            "articulation": articulation,
        }
        if not static:
            curve = synthesize_pitch_curve(
                performance, duration, seed=None if seed is None else seed + index
            )
            if curve:
                entry["pitch"] = curve
        notes.append(entry)
        expression.append(
            {"t": round(start, 5), "value": round(float(np.clip(performance.intensity, 0.05, 1.0)), 4)}
        )

    return {
        "format": PERFORMANCE_FORMAT,
        "generated_by": "static score" if static else "fbmx-performer",
        "seed": seed,
        "seconds": round(end_seconds + 1.0, 3),
        "block_frames": 64,
        "tempo_bpm": round(tempo_bpm, 4),
        "expression": expression or [{"t": 0.0, "value": 0.75}],
        "notes": notes,
    }


def load_model(checkpoint: str | Path) -> Performer:
    state = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    model = Performer(PerformerConfig.from_dict(state["config"]))
    model.load_state_dict(state["model"])
    model.eval()
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a Solfage performance from a score")
    parser.add_argument("--checkpoint", help="Performer checkpoint; omitted for --static")
    parser.add_argument("--score", required=True, help="score MIDI file")
    parser.add_argument("--track", type=int, default=1, help="MIDI track index of the part")
    parser.add_argument("--output", required=True)
    parser.add_argument("--articulation", default="sustain_vibrato")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument(
        "--static",
        action="store_true",
        help="write the score with no expression at all (the A/B control)",
    )
    arguments = parser.parse_args()

    score_notes = read_score_notes(arguments.score, arguments.track)
    if not score_notes:
        raise SystemExit(f"no notes on track {arguments.track} of {arguments.score}")

    model = None if arguments.static else load_model(arguments.checkpoint)
    document = generate(
        model,
        score_notes,
        tempo_bpm=score_tempo_bpm(arguments.score),
        time_signature=score_time_signature(arguments.score),
        articulation=arguments.articulation,
        seed=arguments.seed,
        static=arguments.static,
    )
    Path(arguments.output).parent.mkdir(parents=True, exist_ok=True)
    Path(arguments.output).write_text(json.dumps(document, indent=1) + "\n", encoding="utf-8")
    print(
        f"{arguments.output}  notes={len(document['notes'])} "
        f"seconds={document['seconds']} source={document['generated_by']}"
    )


if __name__ == "__main__":
    main()
