"""Build the per-note performance dataset from URMP.

One record per confidently aligned note. The record pairs *what was written*
(the model's input at runtime) with *what was played* (the target), and carries
enough phrase context that the model is not asked to predict expression from a
bare pitch and duration.

Nothing here writes into the URMP tree; the source dataset stays immutable.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import soundfile as sf

from . import DATASET_ATTRIBUTION, DATASET_LICENSE, DATASET_NAME, PIPELINE_VERSION
from .align import align
from .annotations import read_f0, read_performed_notes, read_score_notes, score_tempo_bpm, score_time_signature
from .discovery import VIOLIN, discover
from .features import (
    CONTROL_RATE_HZ,
    extract_pitch_trace,
    measure_intensity,
    measure_transition,
    measure_vibrato,
)

RECORD_SCHEMA_VERSION = 1

#: Confident-match threshold below which a note is not written at all.
MIN_NOTE_CONFIDENCE = 0.5

#: A part whose confident coverage falls below this is excluded wholesale: if a
#: third of the part could not be aligned, the third that could is probably
#: aligned to the wrong places too.
MIN_PART_COVERAGE = 0.80


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def performance_groups(parts: list) -> dict[str, str]:
    """Map each part id to a group id shared by identical performances.

    URMP reuses violin recordings across ensemble variants: the same player's
    take appears in `35_Rondeau_vn_vn_va_db`, `36_Rondeau_vn_vn_va_vc` and
    `37_Rondeau_fl_vn_va_cl`. Two of those in training and one in test would
    look like generalisation and be memorisation, so parts are grouped by the
    transitive closure of sharing *any* of their three source files — audio,
    note table, or F0 track. One Pavane pair shares annotations while differing
    in audio, which a hash of the audio alone would miss.
    """

    parent: dict[str, str] = {}

    def find(key: str) -> str:
        parent.setdefault(key, key)
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    def union(a: str, b: str) -> None:
        root_a, root_b = find(a), find(b)
        if root_a != root_b:
            parent[root_b] = root_a

    hashes: dict[str, tuple[str, str, str]] = {}
    for part in parts:
        hashes[part.part_id] = (
            f"a:{_file_hash(part.audio)}",
            f"n:{_file_hash(part.notes)}",
            f"f:{_file_hash(part.f0)}",
        )
    for part_id, keys in hashes.items():
        find(part_id)
        for key in keys:
            union(part_id, key)

    groups: dict[str, str] = {}
    canonical: dict[str, str] = {}
    for part_id in sorted(hashes):
        root = find(part_id)
        if root not in canonical:
            canonical[root] = f"perf{len(canonical):03d}"
        groups[part_id] = canonical[root]
    return groups


def _phrase_boundaries(matches: list, *, gap_seconds: float = 0.35) -> list[int]:
    """Index of the phrase each matched note belongs to.

    A phrase break is a silence longer than `gap_seconds` between the end of one
    note and the start of the next. This is a crude reading of phrasing and it
    is labelled as such — URMP has no phrase marks — but it is derived from the
    performance rather than invented, and it gives the model the "where am I in
    the line" context section 11 asks for.
    """

    phrase = 0
    out: list[int] = []
    previous_offset: float | None = None
    for match in matches:
        if previous_offset is not None and match.performed.onset - previous_offset > gap_seconds:
            phrase += 1
        out.append(phrase)
        previous_offset = match.performed.offset
    return out


def build_part_records(piece, part, *, group_id: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    score = read_score_notes(piece.score_midi, part.index)
    performed = read_performed_notes(part.notes)
    f0 = read_f0(part.f0)
    alignment = align(score, performed)

    diagnostics: dict[str, Any] = {
        "part_id": part.part_id,
        "piece_id": piece.piece_id,
        "group_id": group_id,
        **alignment.diagnostics,
        "coverage": round(alignment.coverage, 4),
    }
    if alignment.coverage < MIN_PART_COVERAGE:
        diagnostics["rejected"] = f"coverage {alignment.coverage:.3f} below {MIN_PART_COVERAGE}"
        return [], diagnostics

    audio, sample_rate = sf.read(str(part.audio), dtype="float64", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    # Reference level for this part: the 95th percentile of per-note RMS, so
    # `level` describes this player's own dynamic range and not the gain the
    # recording happened to be made at.
    note_rms: list[float] = []
    for match in alignment.matches:
        start = int(round(match.performed.onset * sample_rate))
        end = int(round(match.performed.offset * sample_rate))
        segment = audio[max(0, start) : min(audio.size, end)]
        if segment.size > 32:
            note_rms.append(float(np.sqrt(np.mean(segment**2))))
    reference_rms = float(np.percentile(note_rms, 95)) if note_rms else 1.0

    tempo = score_tempo_bpm(piece.score_midi)
    numerator, denominator = score_time_signature(piece.score_midi)
    beat_seconds = 60.0 / max(tempo, 1e-6)
    phrases = _phrase_boundaries(alignment.matches)
    phrase_counts: dict[int, int] = defaultdict(int)
    for phrase in phrases:
        phrase_counts[phrase] += 1
    phrase_position: dict[int, int] = defaultdict(int)

    # Each session has its own tuning. The per-part medians measured across
    # URMP's violin content run from -11 to +17 cents, which is a real
    # difference between recordings and a completely unpredictable one from a
    # score: nothing in the notes says what the oboe tuned to that afternoon.
    # Left in the target it is 40% of the variance of `pitch_offset_cents` and
    # pure noise to regress against, so the tuning is measured, recorded, and
    # subtracted, and what remains is the expressive intonation of individual
    # notes against the player's own centre.
    tuning_offset_cents = (
        float(np.median([m.pitch_offset_cents for m in alignment.matches]))
        if alignment.matches
        else 0.0
    )

    records: list[dict[str, Any]] = []
    vibrato_reasons: dict[str, int] = defaultdict(int)

    for position, match in enumerate(alignment.matches):
        if match.confidence < MIN_NOTE_CONFIDENCE:
            continue
        previous = alignment.matches[position - 1] if position > 0 else None
        following = (
            alignment.matches[position + 1] if position + 1 < len(alignment.matches) else None
        )

        trace = extract_pitch_trace(
            f0.times,
            f0.frequency_hz,
            onset=match.performed.onset,
            duration=match.performed.duration,
            score_midi_pitch=match.score.midi_pitch,
        )
        vibrato = measure_vibrato(trace, hop_seconds=f0.hop_seconds)
        if vibrato.reason:
            vibrato_reasons[vibrato.reason] += 1
        intensity = measure_intensity(
            audio,
            sample_rate,
            onset=match.performed.onset,
            duration=match.performed.duration,
            reference_rms=reference_rms,
        )
        transition = measure_transition(
            previous.performed.offset if previous else None,
            previous.score.midi_pitch if previous else None,
            match.performed.onset,
            float(match.score.midi_pitch),
            trace,
        )

        phrase = phrases[position]
        phrase_position[phrase] += 1
        score_beat = match.score.onset / beat_seconds

        # The pitch curve, decimated to the control grid and stored the way the
        # Solfege pitch editor stores one: (beats-from-note-start, cents).
        curve: list[list[float]] = []
        if trace.is_usable:
            performed_beat_seconds = beat_seconds * match.local_stretch
            for time, cents in zip(trace.times, trace.cents):
                curve.append([round(float(time / max(performed_beat_seconds, 1e-6)), 5), round(float(cents), 2)])

        record: dict[str, Any] = {
            "schema_version": RECORD_SCHEMA_VERSION,
            "piece_id": piece.piece_id,
            "part_id": part.part_id,
            "group_id": group_id,
            "note_index": position,
            # ── what was written: the model's runtime input ──────────────
            "score_pitch": match.score.midi_pitch,
            "score_onset_seconds": round(match.score.onset, 5),
            "score_duration_seconds": round(match.score.duration, 5),
            "score_velocity": match.score.velocity,
            "score_beat": round(score_beat, 5),
            "beat_in_bar": round(score_beat % max(numerator, 1), 5),
            "tempo_bpm": round(tempo, 4),
            "time_signature": [numerator, denominator],
            "previous_interval": (
                match.score.midi_pitch - previous.score.midi_pitch if previous else None
            ),
            "next_interval": (
                following.score.midi_pitch - match.score.midi_pitch if following else None
            ),
            "rest_before_seconds": (
                round(match.score.onset - previous.score.offset, 5) if previous else None
            ),
            "rest_after_seconds": (
                round(following.score.onset - match.score.offset, 5) if following else None
            ),
            "phrase_index": phrase,
            "phrase_position": phrase_position[phrase] - 1,
            "phrase_length": phrase_counts[phrase],
            # ── what was played: the targets ─────────────────────────────
            "onset_deviation_seconds": round(match.onset_residual, 5),
            "duration_ratio": round(match.duration_ratio, 5),
            "pitch_offset_cents": round(match.pitch_offset_cents, 3),
            "pitch_offset_cents_relative": round(
                match.pitch_offset_cents - tuning_offset_cents, 3
            ),
            "part_tuning_offset_cents": round(tuning_offset_cents, 3),
            "local_stretch": round(match.local_stretch, 5),
            "pitch_curve_cents": curve,
            "pitch_curve_voiced_fraction": round(trace.voiced_fraction, 4),
            "vibrato": vibrato.to_dict(),
            "transition": transition,
            "alignment_confidence": round(match.confidence, 4),
            "control_rate_hz": CONTROL_RATE_HZ,
        }
        if intensity is not None:
            record["intensity"] = {
                "level": round(intensity.level, 5),
                "rms_db": round(intensity.rms_db, 3),
                "peak_db": round(intensity.peak_db, 3),
                "centroid_hz": round(intensity.centroid_hz, 2),
                "attack_slope_db_per_second": round(intensity.attack_slope_db_per_second, 3),
            }
            record["intensity_curve"] = [round(float(v), 6) for v in intensity.curve]
        records.append(record)

    diagnostics["records"] = len(records)
    diagnostics["tuning_offset_cents"] = round(tuning_offset_cents, 3)
    diagnostics["vibrato_rejections"] = dict(vibrato_reasons)
    diagnostics["reference_rms"] = round(reference_rms, 6)
    return records, diagnostics


def build(root: str | Path, output_dir: str | Path, *, instrument: str = VIOLIN) -> dict[str, Any]:
    pieces = discover(root)
    parts = [part for piece in pieces for part in piece.parts_of(instrument) if part.is_complete]
    groups = performance_groups(parts)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    records_path = out / "notes.jsonl"

    all_diagnostics: list[dict[str, Any]] = []
    total = 0
    with records_path.open("w", encoding="utf-8") as handle:
        for piece in pieces:
            for part in piece.parts_of(instrument):
                if not part.is_complete or piece.score_midi is None:
                    continue
                records, diagnostics = build_part_records(
                    piece, part, group_id=groups[part.part_id]
                )
                all_diagnostics.append(diagnostics)
                for record in records:
                    handle.write(json.dumps(record, separators=(",", ":")) + "\n")
                    total += 1

    summary = {
        "schema_version": RECORD_SCHEMA_VERSION,
        "pipeline_version": PIPELINE_VERSION,
        "dataset": DATASET_NAME,
        "license": DATASET_LICENSE,
        "attribution": DATASET_ATTRIBUTION,
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "instrument": instrument,
        "control_rate_hz": CONTROL_RATE_HZ,
        "parts_considered": len(parts),
        "parts_kept": sum(1 for d in all_diagnostics if not d.get("rejected")),
        "parts_rejected": [d for d in all_diagnostics if d.get("rejected")],
        "performance_groups": len(set(groups.values())),
        "group_of_part": groups,
        "total_notes": total,
        "records_path": records_path.name,
    }
    (out / "build-report.json").write_text(
        json.dumps({"summary": summary, "parts": all_diagnostics}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary
