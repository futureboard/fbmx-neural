"""Measured inventory of URMP's violin content.

Nothing here is estimated. Durations come from the WAV headers, note counts
from the annotation tables, and score availability from actually parsing the
MIDI. A part that cannot be read is reported as rejected, with the reason.
"""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from . import DATASET_ATTRIBUTION, DATASET_LICENSE, DATASET_NAME, PIPELINE_VERSION
from .annotations import hz_to_midi, read_f0, read_performed_notes, read_score_notes, score_tempo_bpm
from .discovery import INSTRUMENTS, VIOLIN, Piece, discover

SCAN_SCHEMA_VERSION = 1


def _audio_info(path: Path) -> dict[str, Any]:
    info = sf.info(str(path))
    return {
        "sample_rate": int(info.samplerate),
        "channels": int(info.channels),
        "duration_seconds": float(info.duration),
        "subtype": str(info.subtype),
        "frames": int(info.frames),
    }


def scan_part(piece: Piece, part, *, score_notes: list | None) -> dict[str, Any]:
    """Everything measurable about one part, or the reason it is unusable."""

    record: dict[str, Any] = {
        "piece_id": piece.piece_id,
        "piece_number": piece.number,
        "title": piece.title,
        "part_index": part.index,
        "instrument": part.instrument,
        "instrument_name": INSTRUMENTS.get(part.instrument, part.instrument),
        "part_id": part.part_id,
    }
    missing = part.missing()
    if missing:
        record["usable"] = False
        record["reject_reason"] = f"missing {', '.join(missing)}"
        return record

    try:
        record["audio"] = _audio_info(part.audio)
    except Exception as error:  # noqa: BLE001 - report, do not crash the scan
        record["usable"] = False
        record["reject_reason"] = f"unreadable audio: {error}"
        return record

    performed = read_performed_notes(part.notes)
    f0 = read_f0(part.f0)
    record["performed_note_count"] = len(performed)
    record["f0_frames"] = int(f0.times.size)
    record["f0_hop_seconds"] = round(f0.hop_seconds, 6)
    record["f0_voiced_frames"] = int(np.count_nonzero(f0.voiced))
    record["f0_voiced_fraction"] = (
        round(float(np.count_nonzero(f0.voiced) / f0.times.size), 6) if f0.times.size else 0.0
    )

    if performed:
        pitches = np.asarray([note.midi_pitch for note in performed])
        durations = np.asarray([note.duration for note in performed])
        record["performed_pitch_midi_min"] = round(float(pitches.min()), 3)
        record["performed_pitch_midi_max"] = round(float(pitches.max()), 3)
        record["performed_duration_seconds"] = {
            "total": round(float(durations.sum()), 3),
            "min": round(float(durations.min()), 4),
            "median": round(float(np.median(durations)), 4),
            "max": round(float(durations.max()), 4),
        }
        record["performed_span_seconds"] = round(
            float(performed[-1].offset - performed[0].onset), 3
        )

    if score_notes is None:
        record["score_note_count"] = None
        record["usable"] = False
        record["reject_reason"] = "no score MIDI track for this part"
        return record

    record["score_note_count"] = len(score_notes)
    record["score_vs_performed_note_delta"] = len(score_notes) - len(performed)

    if not performed:
        record["usable"] = False
        record["reject_reason"] = "no performed notes"
        return record
    if not score_notes:
        record["usable"] = False
        record["reject_reason"] = "score track has no notes"
        return record
    if f0.times.size == 0:
        record["usable"] = False
        record["reject_reason"] = "empty F0 track"
        return record

    record["usable"] = True
    return record


def scan(root: str | Path, *, instrument: str = VIOLIN) -> dict[str, Any]:
    pieces = discover(root)
    records: list[dict[str, Any]] = []

    for piece in pieces:
        wanted = piece.parts_of(instrument)
        if not wanted:
            continue
        score_by_track: dict[int, list] = {}
        tempo = None
        if piece.score_midi is not None:
            tempo = score_tempo_bpm(piece.score_midi)
            for part in wanted:
                try:
                    score_by_track[part.index] = read_score_notes(piece.score_midi, part.index)
                except Exception:  # noqa: BLE001
                    score_by_track[part.index] = []
        for part in wanted:
            record = scan_part(
                piece,
                part,
                score_notes=score_by_track.get(part.index) if piece.score_midi else None,
            )
            if tempo is not None:
                record["score_tempo_bpm"] = round(tempo, 4)
            record["ensemble_size"] = len(piece.parts)
            record["ensemble"] = [p.instrument for p in piece.parts]
            records.append(record)

    usable = [record for record in records if record.get("usable")]
    rejected = [record for record in records if not record.get("usable")]

    total_seconds = sum(record["audio"]["duration_seconds"] for record in usable)
    total_notes = sum(record["performed_note_count"] for record in usable)
    sample_rates = Counter(record["audio"]["sample_rate"] for record in usable)
    channels = Counter(record["audio"]["channels"] for record in usable)
    hops = Counter(record["f0_hop_seconds"] for record in usable)
    deltas = Counter(record["score_vs_performed_note_delta"] for record in usable)

    pitch_min = min((record["performed_pitch_midi_min"] for record in usable), default=None)
    pitch_max = max((record["performed_pitch_midi_max"] for record in usable), default=None)

    summary = {
        "schema_version": SCAN_SCHEMA_VERSION,
        "pipeline_version": PIPELINE_VERSION,
        "dataset": DATASET_NAME,
        "license": DATASET_LICENSE,
        "attribution": DATASET_ATTRIBUTION,
        "scanned_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "instrument": instrument,
        "instrument_name": INSTRUMENTS.get(instrument, instrument),
        "pieces_total": len(pieces),
        "pieces_with_instrument": len({record["piece_id"] for record in records}),
        "pieces_usable": len({record["piece_id"] for record in usable}),
        "parts_total": len(records),
        "parts_usable": len(usable),
        "parts_rejected": len(rejected),
        "total_audio_seconds": round(total_seconds, 3),
        "total_audio_minutes": round(total_seconds / 60.0, 3),
        "total_performed_notes": total_notes,
        "performed_pitch_midi_range": [pitch_min, pitch_max],
        "sample_rates": dict(sample_rates),
        "channel_counts": dict(channels),
        "f0_hop_seconds": dict(hops),
        "score_vs_performed_note_delta": dict(sorted(deltas.items())),
        "reject_reasons": dict(Counter(record["reject_reason"] for record in rejected)),
    }
    return {"summary": summary, "parts": records}


def write_scan_report(report: dict[str, Any], path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out
