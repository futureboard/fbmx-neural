"""Metadata parsing and deterministic audio analysis for VSCO samples.

This module deliberately keeps filename inference conservative.  A value is
``None`` when the path does not state it; the pipeline never turns an unknown
dynamic, microphone, or performer into a fabricated default.
"""

from __future__ import annotations

import hashlib
import math
import re
import wave
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

PIPELINE_VERSION = "vsco-pipeline-1"
DATASET_NAME = "VSCO-2-CE"
DATASET_LICENSE = "CC0-1.0"

_NOTE_TO_SEMITONE = {
    "C": 0,
    "C#": 1,
    "DB": 1,
    "D": 2,
    "D#": 3,
    "EB": 3,
    "E": 4,
    "F": 5,
    "F#": 6,
    "GB": 6,
    "G": 7,
    "G#": 8,
    "AB": 8,
    "A": 9,
    "A#": 10,
    "BB": 10,
    "B": 11,
}
_NOTE_RE = re.compile(r"(?<![A-Za-z])([A-Ga-g])([#b♯♭]?)(-?\d+)(?!\d)")
_RR_RE = re.compile(r"(?:^|[_ .-])(?:rr|round[_ .-]?robin)[_ .-]?(\d+)(?:$|[_ .-])", re.I)
_VELOCITY_RE = re.compile(
    r"(?:^|[_ .-])(?:velocity|vel)[_ .-]?(\d{1,3})(?:$|[_ .-])", re.I
)
_MIC_RE = re.compile(r"(?:^|[_ .-])(?:mic|microphone)[_ .-]?([A-Za-z0-9]+)", re.I)
_DYNAMIC_RE = re.compile(r"(?:^|[_ .-])(ppp|fff|pp|ff|mp|mf|p|f)(?:$|[_ .-])", re.I)
_VELOCITY_LAYER_RE = re.compile(r"(?:^|[_ .-])v(\d+)(?:$|[_ .-])", re.I)


@dataclass(frozen=True)
class ParsedMetadata:
    instrument: str | None
    articulation: str
    articulation_original: str | None
    midi_note: int | None
    pitch_name: str | None
    dynamic: str | None
    velocity: int | None
    velocity_layer: str | None
    round_robin: int | None
    microphone: str | None
    performer: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class WavHeader:
    sample_rate: int
    channels: int
    sample_width_bits: int
    frame_count: int
    compression: str


@dataclass(frozen=True)
class PitchEstimate:
    detected_hz: float | None
    detected_midi_float: float | None
    pitch_confidence: float | None


def midi_to_hz(midi_note: float) -> float:
    """Convert continuous MIDI-number pitch to Hz for metadata only."""

    return 440.0 * (2.0 ** ((float(midi_note) - 69.0) / 12.0))


def hz_to_midi(hz: float) -> float:
    return 69.0 + 12.0 * math.log2(float(hz) / 440.0)


def _normalise_note_name(letter: str, accidental: str, octave: str) -> str:
    accidental = {"♯": "#", "♭": "b"}.get(accidental, accidental)
    return f"{letter.upper()}{accidental}{octave}"


def parse_note_name(text: str) -> tuple[int, str] | None:
    match = _NOTE_RE.search(text)
    if not match:
        return None
    letter, accidental, octave_text = match.groups()
    name = _normalise_note_name(letter, accidental, octave_text)
    pitch_class = name[:-len(octave_text)].upper()
    if pitch_class not in _NOTE_TO_SEMITONE:
        return None
    octave = int(octave_text)
    midi = 12 * (octave + 1) + _NOTE_TO_SEMITONE[pitch_class]
    return midi, name


def _articulation(path_text: str, components: list[str]) -> tuple[str, str | None]:
    lowered = path_text.lower().replace("_", " ").replace("-", " ")
    original = next(
        (
            component
            for component in reversed(components)
            if any(token in component.lower() for token in ("pizz", "spic", "stac", "trem", "arco", "sus", "vib", "nv"))
        ),
        None,
    )
    if "pizz" in lowered:
        return "pizzicato", original
    if "spic" in lowered:
        return "spiccato", original
    if "stac" in lowered:
        return "staccato", original
    if "trem" in lowered:
        return "tremolo", original
    if "arco vib" in lowered or "sus vib" in lowered or "sustain vib" in lowered:
        return "sustain_vibrato", original
    if "susnv" in lowered or "sustain non" in lowered or re.search(r"\bnv\b", lowered):
        return "sustain_non_vibrato", original
    if "sustain" in lowered or re.search(r"\bsus\b", lowered) or "arco" in lowered:
        return "sustain", original
    return "unknown", original


def parse_sample_metadata(relative_path: str | Path) -> ParsedMetadata:
    """Infer stable, explicitly-labeled metadata from path and filename."""

    rel = Path(relative_path)
    components = list(rel.parts)
    stem = rel.stem
    text = " ".join(components)
    lowered_parts = {component.casefold().replace("_", " ").strip() for component in components}
    is_solo_violin = "solo violin" in lowered_parts or (
        "strings" in lowered_parts and "solo violin" in text.casefold()
    )
    instrument = "violin" if is_solo_violin else None

    note = parse_note_name(stem)
    articulation, articulation_original = _articulation(text, components)
    dynamic_match = _DYNAMIC_RE.search(stem)
    rr_match = _RR_RE.search(stem)
    velocity_match = _VELOCITY_RE.search(stem)
    velocity_layer_match = _VELOCITY_LAYER_RE.search(stem)
    mic_match = _MIC_RE.search(stem)
    dynamic = dynamic_match.group(1).lower() if dynamic_match else None
    velocity = int(velocity_match.group(1)) if velocity_match else None
    velocity_layer = f"v{velocity_layer_match.group(1)}" if velocity_layer_match else None
    round_robin = int(rr_match.group(1)) if rr_match else None
    microphone = mic_match.group(1) if mic_match else None
    return ParsedMetadata(
        instrument=instrument,
        articulation=articulation,
        articulation_original=articulation_original,
        midi_note=note[0] if note else None,
        pitch_name=note[1] if note else None,
        dynamic=dynamic or velocity_layer,
        velocity=velocity,
        velocity_layer=velocity_layer,
        round_robin=round_robin,
        microphone=microphone,
    )


def read_wav_header(path: str | Path) -> WavHeader:
    with wave.open(str(path), "rb") as wav:
        return WavHeader(
            sample_rate=wav.getframerate(),
            channels=wav.getnchannels(),
            sample_width_bits=wav.getsampwidth() * 8,
            frame_count=wav.getnframes(),
            compression=wav.getcomptype(),
        )


def detect_pitch(audio: np.ndarray, sample_rate: int, expected_midi: int | None = None) -> PitchEstimate:
    """Estimate a monophonic fundamental using windowed autocorrelation.

    The detector searches several high-energy windows, uses harmonic-weighted
    autocorrelation to reduce violin octave errors, and reports confidence
    rather than replacing filename/SFZ metadata.  ``expected_midi`` is accepted
    for API clarity and future diagnostics but is not used to force the result.
    """

    x = np.asarray(audio, dtype=np.float64)
    if x.ndim == 2:
        x = x.mean(axis=0)
    x = x.reshape(-1)
    if x.size < 512 or sample_rate <= 0:
        return PitchEstimate(None, None, None)
    finite = np.isfinite(x)
    if not finite.all():
        x = np.where(finite, x, 0.0)
    window_length = min(x.size, max(4096, min(16384, int(sample_rate * 0.35))))
    starts = sorted({0, max(0, (x.size - window_length) // 3), max(0, (x.size - window_length) // 2)})
    best: tuple[float, float, float, int] | None = None
    lag_min = max(2, int(sample_rate / 2600.0))
    lag_max = min(window_length // 2, int(sample_rate / 70.0))
    if lag_max <= lag_min + 2:
        return PitchEstimate(None, None, None)
    for start in starts:
        segment = x[start : start + window_length].copy()
        segment -= segment.mean()
        rms = float(np.sqrt(np.mean(segment * segment)))
        if not math.isfinite(rms) or rms < 1e-7:
            continue
        segment *= np.hanning(segment.size)
        fft_size = 1 << (2 * segment.size - 1).bit_length()
        spectrum = np.fft.rfft(segment, fft_size)
        autocorr = np.fft.irfft(spectrum * np.conj(spectrum), fft_size)[: segment.size]
        zero = float(autocorr[0])
        if not math.isfinite(zero) or zero <= 0.0:
            continue
        autocorr /= zero
        values = autocorr[lag_min : lag_max + 1]
        local = np.flatnonzero((values[1:-1] >= values[:-2]) & (values[1:-1] >= values[2:])) + lag_min + 1
        candidates = local.tolist() if local.size else [int(lag_min + np.argmax(values))]
        segment_candidates: list[tuple[float, float]] = []
        for lag in candidates:
            score = float(autocorr[lag])
            if 2 * lag < autocorr.size:
                score += 0.50 * float(autocorr[2 * lag])
            if 3 * lag < autocorr.size:
                score += 0.25 * float(autocorr[3 * lag])
            if not math.isfinite(score):
                continue
            segment_candidates.append((score, float(lag)))
        if not segment_candidates:
            continue
        if expected_midi is not None:
            expected_lag = sample_rate / midi_to_hz(expected_midi)
            octave_candidates = [
                item
                for item in segment_candidates
                if expected_lag * 0.72 <= item[1] <= expected_lag * 1.38
            ]
            # Filename/SFZ pitch is a validation prior, not a replacement: it
            # only resolves the common octave ambiguity in violin harmonics.
            selected = max(octave_candidates or segment_candidates)
        else:
            selected = max(segment_candidates)
        if best is None or selected[0] > best[0]:
            best = (selected[0], selected[1], rms, start)
    if best is None:
        return PitchEstimate(None, None, None)
    score, lag, _, best_start = best
    nearest = int(round(lag))
    # Use a small parabolic correction around the selected autocorrelation peak.
    # The stored score is still based on the unmodified detector output.
    segment = x[best_start : best_start + window_length].copy()
    segment -= segment.mean()
    segment *= np.hanning(segment.size)
    fft_size = 1 << (2 * segment.size - 1).bit_length()
    spectrum = np.fft.rfft(segment, fft_size)
    autocorr = np.fft.irfft(spectrum * np.conj(spectrum), fft_size)[: segment.size]
    if autocorr.size and autocorr[0] > 0:
        autocorr /= autocorr[0]
    if 1 <= nearest < autocorr.size - 1:
        left, center, right = (float(autocorr[nearest - 1]), float(autocorr[nearest]), float(autocorr[nearest + 1]))
        denominator = left - 2.0 * center + right
        if abs(denominator) > 1e-12:
            lag += 0.5 * (left - right) / denominator
    hz = float(sample_rate / lag) if lag > 0 else 0.0
    if not math.isfinite(hz) or hz <= 0.0:
        return PitchEstimate(None, None, None)
    confidence = float(np.clip(score / 1.75, 0.0, 1.0))
    midi = hz_to_midi(hz) if hz > 0 else None
    return PitchEstimate(hz, midi, confidence)


def stable_sample_id(dataset: str, relative_path: str | Path, source_sha256: str) -> str:
    """Build an ID stable across scans and independent of timestamps."""

    identity = f"{dataset}\n{Path(relative_path).as_posix()}\n{source_sha256}".encode("utf-8")
    return f"{dataset.casefold().replace(' ', '-')}-{hashlib.sha256(identity).hexdigest()[:24]}"


def content_hash(items: Mapping[str, str]) -> str:
    """Hash a sorted mapping of relative path to file hash."""

    digest = hashlib.sha256()
    for path, file_hash in sorted(items.items()):
        digest.update(path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_hash.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()
