"""Write a real URMP performance as a Solfage performance document.

This is the third leg of the comparison, and the only one that is not a model
output: the actual notes a violinist played, their actual pitch trace, and their
actual loudness, expressed in exactly the format the Performer emits and the
engine renders.

Having it in the same format matters for two reasons. It makes "how close is the
model to a human" a question about two files that can be diffed rather than a
listening impression, and it proves the format can carry a real performance —
if a human take cannot be written down in it, the representation is too poor for
the model's output to be trusted either.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from ..datasets.urmp.annotations import read_f0, read_performed_notes
from ..datasets.urmp.discovery import discover
from ..datasets.urmp.features import extract_pitch_trace
from .generate import PERFORMANCE_FORMAT, intensity_to_velocity

#: Keep at most this many pitch points per note. URMP's F0 grid is 100 Hz and a
#: long note can carry hundreds of frames; the engine interpolates between
#: breakpoints, so thinning by a fixed stride keeps the file readable without
#: changing what is heard.
MAX_PITCH_POINTS = 400


def human_performance(
    part,
    *,
    articulation: str = "sustain_vibrato",
    reference_percentile: float = 95.0,
) -> dict[str, Any]:
    """One URMP part, as a Solfage performance document."""

    performed = read_performed_notes(part.notes)
    f0 = read_f0(part.f0)
    audio, sample_rate = sf.read(str(part.audio), dtype="float64", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    levels: list[float] = []
    for note in performed:
        start = int(round(note.onset * sample_rate))
        end = int(round(note.offset * sample_rate))
        segment = audio[max(0, start) : min(audio.size, end)]
        levels.append(float(np.sqrt(np.mean(segment**2))) if segment.size > 32 else 0.0)
    reference = float(np.percentile([v for v in levels if v > 0] or [1.0], reference_percentile))

    notes: list[dict[str, Any]] = []
    expression: list[dict[str, float]] = []
    for index, note in enumerate(performed):
        # The performed note's own pitch is the reference the curve is written
        # against, because that is the note the engine will be asked to play.
        # Rounding to the nearest semitone here and putting the rest in the
        # curve is what keeps the document a *performance of a note* rather
        # than an arbitrary frequency.
        nearest = int(round(note.midi_pitch))
        trace = extract_pitch_trace(
            f0.times,
            f0.frequency_hz,
            onset=note.onset,
            duration=note.duration,
            score_midi_pitch=nearest,
        )
        entry: dict[str, Any] = {
            "start": round(note.onset, 5),
            "duration": round(note.duration, 5),
            "note": nearest,
            "velocity": round(intensity_to_velocity(levels[index] / max(reference, 1e-9)), 4),
            "articulation": articulation,
        }
        if trace.cents.size >= 2:
            stride = max(1, trace.cents.size // MAX_PITCH_POINTS)
            entry["pitch"] = [
                {"t": round(float(t), 4), "cents": round(float(c), 2)}
                for t, c in zip(trace.times[::stride], trace.cents[::stride])
            ]
        notes.append(entry)
        expression.append({"t": round(note.onset, 5), "value": entry["velocity"]})

    end_seconds = max((n["start"] + n["duration"] for n in notes), default=0.0)
    return {
        "format": PERFORMANCE_FORMAT,
        "generated_by": "urmp human reference",
        "part_id": part.part_id,
        "seconds": round(end_seconds + 1.0, 3),
        "block_frames": 64,
        "expression": expression or [{"t": 0.0, "value": 0.75}],
        "notes": notes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Write a URMP part as a Solfage performance")
    parser.add_argument("--urmp-root", required=True)
    parser.add_argument("--piece", required=True, help="piece directory name")
    parser.add_argument("--part", type=int, required=True, help="part index within the piece")
    parser.add_argument("--output", required=True)
    parser.add_argument("--articulation", default="sustain_vibrato")
    arguments = parser.parse_args()

    for piece in discover(arguments.urmp_root):
        if piece.piece_id != arguments.piece:
            continue
        for part in piece.parts:
            if part.index != arguments.part:
                continue
            document = human_performance(part, articulation=arguments.articulation)
            Path(arguments.output).parent.mkdir(parents=True, exist_ok=True)
            Path(arguments.output).write_text(
                json.dumps(document, indent=1) + "\n", encoding="utf-8"
            )
            print(
                f"{arguments.output}  notes={len(document['notes'])} "
                f"seconds={document['seconds']} source={document['generated_by']}"
            )
            return
    raise SystemExit(f"part {arguments.part} of {arguments.piece} not found")


if __name__ == "__main__":
    main()
