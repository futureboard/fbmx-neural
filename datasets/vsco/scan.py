"""Recursive VSCO WAV discovery, metadata extraction, and pitch validation."""

from __future__ import annotations

import json
import math
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

# The existing FBMX package is intentionally a source-tree package and its
# internal imports are absolute (``fbmx.*``). Make that package root visible
# when this module is launched as ``python -m neural.datasets.vsco``.
_NEURAL_ROOT = Path(__file__).resolve().parents[2]
if str(_NEURAL_ROOT) not in sys.path:
    sys.path.insert(0, str(_NEURAL_ROOT))

from .metadata import (
    DATASET_LICENSE,
    DATASET_NAME,
    PIPELINE_VERSION,
    content_hash,
    detect_pitch,
    midi_to_hz,
    parse_sample_metadata,
    read_wav_header,
    stable_sample_id,
)

from fbmx.datasets.paired_audio import read_audio


SCAN_SCHEMA_VERSION = 1
_SFZ_SAMPLE_RE = re.compile(r"(?:^|\s)sample\s*=\s*([^\s]+)", re.I)
_SFZ_KEY_RE = re.compile(r"(?:^|\s)(?:key|lokey)\s*=\s*(-?\d+)", re.I)


def discover_wavs(source: str | Path) -> list[Path]:
    """Return every WAV below ``source`` in deterministic relative order."""

    root = Path(source)
    if not root.is_dir():
        raise FileNotFoundError(f"source dataset directory does not exist: {root}")
    return sorted(
        (path for path in root.rglob("*") if path.is_file() and path.suffix.casefold() == ".wav"),
        key=lambda path: path.relative_to(root).as_posix().casefold(),
    )


def is_solo_violin(relative_path: str | Path) -> bool:
    """Select the solo violin directory without including violin sections."""

    parts = [part.casefold().replace("_", " ").strip() for part in Path(relative_path).parts]
    return "solo violin" in parts


def parse_sfz_pitch_map(source: str | Path) -> dict[str, int]:
    """Read simple ``sample=... key=...`` mappings when an SFZ is present."""

    root = Path(source)
    mapping: dict[str, int] = {}
    for sfz in sorted(root.rglob("*.sfz"), key=lambda p: p.as_posix().casefold()):
        try:
            lines = sfz.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        region_sample: str | None = None
        region_key: int | None = None

        def commit_region() -> None:
            if region_sample is None or region_key is None:
                return
            sample_path = (sfz.parent / region_sample).resolve()
            try:
                relative = sample_path.relative_to(root.resolve()).as_posix()
            except ValueError:
                return
            mapping[relative.casefold()] = region_key

        for line in lines + ["<region>"]:
            if "<region>" in line.casefold():
                commit_region()
                region_sample = None
                region_key = None
            sample = _SFZ_SAMPLE_RE.search(line)
            key = _SFZ_KEY_RE.search(line)
            if sample:
                region_sample = sample.group(1)
            if key:
                region_key = int(key.group(1))
    return mapping


def _stats(audio: np.ndarray) -> tuple[float, float, float]:
    audio = np.asarray(audio, dtype=np.float64)
    if audio.size == 0:
        return 0.0, 0.0, 0.0
    finite = np.isfinite(audio)
    if not finite.all():
        return float("nan"), float("nan"), float("nan")
    return (
        float(np.max(np.abs(audio))),
        float(np.sqrt(np.mean(audio * audio))),
        float(np.mean(audio)),
    )


def _warning_list(record: dict[str, Any]) -> list[str]:
    warnings: list[str] = []
    if record["articulation"] == "unknown":
        warnings.append("unknown_articulation")
    if record["midi_note"] is None:
        warnings.append("unknown_pitch")
    if record["peak"] >= 0.999:
        warnings.append("clipping_or_near_clipping")
    if record["duration_seconds"] < 0.05:
        warnings.append("very_short_sample")
    if record["duration_seconds"] > 60.0:
        warnings.append("very_long_sample")
    pitch_error = record.get("pitch_error_cents")
    confidence = record.get("pitch_confidence")
    if pitch_error is not None and abs(pitch_error) > 50.0:
        warnings.append("unexpected_pitch_error")
    if confidence is not None and confidence < 0.25:
        warnings.append("low_pitch_confidence")
    return warnings


def _rejection(relative_path: str, reason: str, detail: str = "") -> dict[str, str]:
    result = {"relative_path": relative_path, "reason": reason}
    if detail:
        result["detail"] = detail
    return result


def scan_source(
    source: str | Path,
    *,
    pitch_detection_enabled: bool = True,
    max_files: int | None = None,
) -> dict[str, Any]:
    """Scan a source tree and return a deterministic report payload.

    Non-target WAV files are recorded as ``not_solo_violin`` rejections. This
    makes the scope decision auditable when scanning the complete VSCO tree.
    """

    root = Path(source).resolve()
    all_wavs = discover_wavs(root)
    sfz_map = parse_sfz_pitch_map(root)
    records: list[dict[str, Any]] = []
    rejections: list[dict[str, str]] = []
    hashes: dict[str, str] = {}
    target_seen = 0
    for path in all_wavs:
        relative = path.relative_to(root).as_posix()
        if not is_solo_violin(relative):
            rejections.append(_rejection(relative, "not_solo_violin"))
            continue
        if max_files is not None and target_seen >= max_files:
            rejections.append(_rejection(relative, "scan_limit"))
            continue
        target_seen += 1
        parsed = parse_sample_metadata(relative)
        try:
            header = read_wav_header(path)
            audio, audio_rate = read_audio(path)
            if audio_rate != header.sample_rate:
                raise ValueError(f"reader rate {audio_rate} differs from header {header.sample_rate}")
            peak, rms, dc_offset = _stats(audio)
            if not all(math.isfinite(value) for value in (peak, rms, dc_offset)):
                raise ValueError("audio contains NaN or Inf")
            if header.frame_count <= 0 or audio.shape[1] <= 0:
                raise ValueError("zero-frame WAV")
        except Exception as exc:  # record and continue so one bad file is visible
            rejections.append(_rejection(relative, "unreadable_or_invalid_wav", str(exc)))
            continue

        source_hash = _sha256(path)
        hashes[relative] = source_hash
        mapped_midi = sfz_map.get(relative.casefold())
        expected_midi = parsed.midi_note if parsed.midi_note is not None else mapped_midi
        expected_hz = midi_to_hz(expected_midi) if expected_midi is not None else None
        pitch = (
            detect_pitch(audio.mean(axis=0), header.sample_rate, expected_midi)
            if pitch_detection_enabled
            else None
        )
        pitch_error = None
        if expected_midi is not None and pitch and pitch.detected_midi_float is not None:
            pitch_error = float((pitch.detected_midi_float - expected_midi) * 100.0)
        record: dict[str, Any] = {
            "id": stable_sample_id(DATASET_NAME, relative, source_hash),
            "source": DATASET_NAME,
            "license": DATASET_LICENSE,
            "source_path": relative,
            "relative_path": relative,
            "source_file_sha256": source_hash,
            "file_size": path.stat().st_size,
            "instrument": parsed.instrument,
            "articulation": parsed.articulation,
            "articulation_original": parsed.articulation_original,
            "midi_note": expected_midi,
            "pitch_name": parsed.pitch_name,
            "pitch_hz": expected_hz,
            "expected_midi": expected_midi,
            "expected_hz": expected_hz,
            "detected_pitch_hz": pitch.detected_hz if pitch else None,
            "detected_midi_float": pitch.detected_midi_float if pitch else None,
            "pitch_error_cents": pitch_error,
            "pitch_confidence": pitch.pitch_confidence if pitch else None,
            "dynamic": parsed.dynamic,
            "velocity": parsed.velocity,
            "velocity_layer": parsed.velocity_layer,
            "round_robin": parsed.round_robin,
            "microphone": parsed.microphone,
            "performer": parsed.performer,
            "sample_rate": header.sample_rate,
            "channels": header.channels,
            "sample_width_bits": header.sample_width_bits,
            "frame_count": header.frame_count,
            "duration_seconds": float(header.frame_count / header.sample_rate),
            "peak": peak,
            "rms": rms,
            "dc_offset": dc_offset,
            "source_sample_rate": header.sample_rate,
            "processed_sample_rate": None,
            "processed_path": None,
            "processed_frame_count": None,
            "processed_duration_seconds": None,
            "processed_peak": None,
            "processed_rms": None,
            "processed_dc_offset": None,
            "original_gain_db": 0.0,
            "processing_gain_db": 0.0,
            "trim_start_frames": 0,
            "trim_end_frames": 0,
            "reference_id": stable_sample_id(DATASET_NAME, relative, source_hash),
            "target_pitch_hz": expected_hz,
            "gesture_hint": None,
            "physical_config": None,
            "alignment_offset": None,
        }
        record["warnings"] = _warning_list(record)
        records.append(record)

    records.sort(key=lambda record: record["id"])
    source_hash = content_hash(hashes)
    return {
        "schema_version": SCAN_SCHEMA_VERSION,
        "pipeline_version": PIPELINE_VERSION,
        "dataset": DATASET_NAME,
        "license": DATASET_LICENSE,
        "instrument": "violin",
        "source_root": str(root),
        "source_dataset_hash": source_hash,
        "sfz_pitch_mappings": len(sfz_map),
        "counts": {
            "wav_files_discovered": len(all_wavs),
            "target_candidates": len(records) + sum(
                1 for item in rejections if item["reason"] == "unreadable_or_invalid_wav"
            ),
            "accepted_violin": len(records),
            "rejected": len(rejections),
        },
        "records": records,
        "rejections": rejections,
        # This field is deliberately outside deterministic records/manifests.
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def _sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def write_scan_report(report: dict[str, Any], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def summarize_records(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(records)
    durations = [row["duration_seconds"] for row in rows if row.get("duration_seconds") is not None]
    pitches = [row["midi_note"] for row in rows if row.get("midi_note") is not None]
    return {
        "total_files": len(rows),
        "unknown_articulation": sum(row["articulation"] == "unknown" for row in rows),
        "unknown_pitch": sum(row["midi_note"] is None for row in rows),
        "sample_rate_distribution": dict(sorted(Counter(str(row["sample_rate"]) for row in rows).items())),
        "channel_distribution": dict(sorted(Counter(str(row["channels"]) for row in rows).items())),
        "duration_seconds": {
            "min": min(durations) if durations else None,
            "max": max(durations) if durations else None,
        },
        "pitch_midi_range": {
            "min": min(pitches) if pitches else None,
            "max": max(pitches) if pitches else None,
        },
        "articulation_distribution": dict(
            sorted(Counter(row["articulation"] for row in rows).items())
        ),
        "dynamic_distribution": dict(
            sorted(Counter(row["dynamic"] or "unknown" for row in rows).items())
        ),
    }
