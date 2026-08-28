"""Turn one aligned note into measurable performance targets.

Every quantity here is derived from something URMP actually ships — the note
table, the F0 track, or the isolated stem. Nothing is a label invented to fill a
field. Where a quantity cannot be measured for a given note (a note too short to
carry vibrato, a stretch of F0 the tracker left unvoiced) the field is absent
and its confidence is zero, rather than defaulted to a plausible-looking number.

In particular there is no bow pressure, bow speed, or bow position here. URMP is
audio and annotation; it has no ground truth for what the arm was doing, and a
regression target fabricated from a loudness envelope would teach the Performer
a relationship that was never observed. What is measured instead is what the ear
can hear — energy, brightness, attack sharpness — and the mapping from those to
physical gestures belongs downstream, in the engine, where it is a stated
modelling choice rather than a fake label.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

#: Control rate for every performance curve, in Hz. URMP's F0 grid is already
#: 100 Hz, so this resamples nothing and invents no detail between frames.
CONTROL_RATE_HZ = 100.0

#: Violin vibrato lives here. Slower than this is a drift or a portamento;
#: faster is a tracking artefact, not an arm.
VIBRATO_MIN_HZ = 3.5
VIBRATO_MAX_HZ = 9.0

#: Cycles of vibrato a note must have room for before its rate estimate means
#: anything. Two is the minimum at which a periodicity is distinguishable from
#: a single swell.
VIBRATO_MIN_CYCLES = 2.0

#: Window used to remove portamento and slow drift before looking for vibrato,
#: in seconds. Long enough to leave a 4 Hz oscillation untouched, short enough
#: to follow a slide into the note.
VIBRATO_DETREND_SECONDS = 0.25


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if window < 2 or values.size < 2:
        return values.copy()
    window = min(window, values.size)
    if window % 2 == 0:
        window += 1
    padded = np.pad(values, window // 2, mode="edge")
    kernel = np.ones(window) / window
    return np.convolve(padded, kernel, mode="valid")[: values.size]


@dataclass
class PitchTrace:
    """A note's performed pitch, as cents from the written pitch."""

    times: np.ndarray
    cents: np.ndarray
    voiced_fraction: float

    @property
    def is_usable(self) -> bool:
        return self.cents.size >= 3 and self.voiced_fraction >= 0.5


def extract_pitch_trace(
    f0_times: np.ndarray,
    f0_hz: np.ndarray,
    *,
    onset: float,
    duration: float,
    score_midi_pitch: int,
) -> PitchTrace:
    """Performed pitch inside one note, in cents from the notated pitch.

    Cents from the *written* pitch, not from the note's own average: that is the
    representation the Solfege pitch curve already uses, so a trace extracted
    here drops into `PitchCurve` without a second conversion, and it survives
    transposition the same way a hand-drawn curve does.
    """

    end = onset + duration
    window = (f0_times >= onset) & (f0_times < end)
    times = f0_times[window]
    values = f0_hz[window]
    if times.size == 0:
        return PitchTrace(np.empty(0), np.empty(0), 0.0)

    voiced = values > 1e-6
    voiced_fraction = float(np.count_nonzero(voiced) / voiced.size)
    if not voiced.any():
        return PitchTrace(np.empty(0), np.empty(0), 0.0)

    reference_hz = 440.0 * 2.0 ** ((score_midi_pitch - 69) / 12.0)
    cents = 1200.0 * np.log2(values[voiced] / reference_hz)
    return PitchTrace(
        times=times[voiced] - onset,
        cents=cents,
        voiced_fraction=voiced_fraction,
    )


@dataclass
class Vibrato:
    present: bool
    rate_hz: float | None = None
    depth_cents: float | None = None
    onset_delay_seconds: float | None = None
    confidence: float = 0.0
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"present": self.present, "confidence": round(self.confidence, 4)}
        if self.rate_hz is not None:
            payload["rate_hz"] = round(self.rate_hz, 4)
        if self.depth_cents is not None:
            payload["depth_cents"] = round(self.depth_cents, 3)
        if self.onset_delay_seconds is not None:
            payload["onset_delay_seconds"] = round(self.onset_delay_seconds, 4)
        if self.reason is not None:
            payload["reason"] = self.reason
        return payload


def measure_vibrato(trace: PitchTrace, *, hop_seconds: float) -> Vibrato:
    """Estimate vibrato rate and depth from a note's pitch trace.

    The trace is detrended first. Without that step a scoop into the note or a
    slide out of it is a large low-frequency excursion, and any peak-picking
    over the raw trace reports it as enormous slow vibrato — which is the
    failure mode section 8 warns about. After detrending, what is left in the
    3.5-9 Hz band is periodic pitch modulation and very little else.

    Depth is reported peak-to-peak, the convention violinists use.
    """

    if not trace.is_usable:
        return Vibrato(present=False, reason="insufficient voiced pitch")

    span = float(trace.times[-1] - trace.times[0]) if trace.times.size > 1 else 0.0
    if span <= 0.0:
        return Vibrato(present=False, reason="zero span")

    # A note must be long enough for the slowest vibrato we accept to complete
    # the minimum number of cycles, or the estimate is unfounded.
    if span < VIBRATO_MIN_CYCLES / VIBRATO_MIN_HZ:
        return Vibrato(present=False, reason="note too short for a rate estimate")

    detrend_window = max(3, int(round(VIBRATO_DETREND_SECONDS / max(hop_seconds, 1e-6))))
    baseline = _moving_average(trace.cents, detrend_window)
    residual = trace.cents - baseline
    if residual.size < 8:
        return Vibrato(present=False, reason="too few frames after detrending")

    # Find a candidate rate from the spectrum, then *fit* a sinusoid at that
    # rate and judge it by how much of the note's pitch motion it actually
    # explains. Judging the spectral peak against the summed spectrum instead
    # is not a measure of anything: zero-padding for frequency resolution
    # multiplies the number of bins in the sum, so the same vibrato scores
    # lower the more finely it is resolved.
    windowed = residual * np.hanning(residual.size)
    padded = 1 << int(np.ceil(np.log2(max(residual.size * 8, 64))))
    spectrum = np.abs(np.fft.rfft(windowed, padded))
    frequencies = np.fft.rfftfreq(padded, hop_seconds)

    band = (frequencies >= VIBRATO_MIN_HZ) & (frequencies <= VIBRATO_MAX_HZ)
    if not band.any() or spectrum[band].size == 0:
        return Vibrato(present=False, reason="vibrato band unrepresentable at this hop")

    candidate = float(frequencies[band][int(np.argmax(spectrum[band]))])

    times = np.arange(residual.size) * hop_seconds
    total_variance = float(np.var(residual))
    if total_variance <= 1e-12:
        return Vibrato(present=False, reason="pitch is flat after detrending")

    def fit(frequency: float) -> tuple[float, float]:
        """Amplitude in cents and fraction of variance explained."""

        design = np.column_stack(
            [
                np.cos(2.0 * np.pi * frequency * times),
                np.sin(2.0 * np.pi * frequency * times),
                np.ones_like(times),
            ]
        )
        coefficients, *_ = np.linalg.lstsq(design, residual, rcond=None)
        model = design @ coefficients
        amplitude = float(np.hypot(coefficients[0], coefficients[1]))
        explained = 1.0 - float(np.var(residual - model)) / total_variance
        return amplitude, explained

    # Refine around the spectral peak: the FFT grid is coarse relative to how
    # precisely a rate can be stated once the note is only a few cycles long.
    best_frequency, best_amplitude, best_explained = candidate, 0.0, -np.inf
    for frequency in np.linspace(
        max(VIBRATO_MIN_HZ, candidate - 0.75), min(VIBRATO_MAX_HZ, candidate + 0.75), 25
    ):
        amplitude, explained = fit(float(frequency))
        if explained > best_explained:
            best_frequency, best_amplitude, best_explained = float(frequency), amplitude, explained

    peak_frequency = best_frequency
    depth_peak_to_peak = 2.0 * best_amplitude
    confidence = float(np.clip(best_explained, 0.0, 1.0))

    cycles = peak_frequency * span
    if cycles < VIBRATO_MIN_CYCLES:
        return Vibrato(
            present=False,
            reason=f"only {cycles:.1f} cycles at the detected rate",
        )

    # A depth below a few cents is inside the noise of the F0 tracker.
    if depth_peak_to_peak < 5.0:
        return Vibrato(
            present=False,
            depth_cents=depth_peak_to_peak,
            confidence=confidence,
            reason="depth below tracker noise",
        )

    # Half the pitch motion has to be this one oscillation before it is called
    # vibrato rather than an unsteady note.
    if confidence < 0.5:
        return Vibrato(
            present=False,
            rate_hz=peak_frequency,
            depth_cents=depth_peak_to_peak,
            confidence=confidence,
            reason="modulation not concentrated at one rate",
        )

    # Where the vibrato starts: the first point at which the running envelope
    # of the residual passes half its eventual depth.
    envelope = _moving_average(np.abs(residual), detrend_window)
    threshold = 0.5 * float(np.max(envelope)) if envelope.size else 0.0
    reached = np.nonzero(envelope >= threshold)[0]
    onset_delay = float(trace.times[reached[0]]) if reached.size else 0.0

    return Vibrato(
        present=True,
        rate_hz=peak_frequency,
        depth_cents=depth_peak_to_peak,
        onset_delay_seconds=onset_delay,
        confidence=confidence,
    )


@dataclass
class Intensity:
    """Measured expressive intensity descriptors for one note.

    Loudness alone is not dynamics on a violin: a forte note is brighter as well
    as louder, and the two move together but not identically. The raw
    descriptors are kept so later research can re-derive whatever it needs, and
    a single normalised `level` is provided for the engine to drive expression
    with.
    """

    rms_db: float
    peak_db: float
    centroid_hz: float
    attack_slope_db_per_second: float
    level: float
    brightness: float
    curve: np.ndarray = field(repr=False, default_factory=lambda: np.empty(0))
    centroid_curve: np.ndarray = field(repr=False, default_factory=lambda: np.empty(0))


def measure_intensity(
    audio: np.ndarray,
    sample_rate: int,
    *,
    onset: float,
    duration: float,
    control_rate_hz: float = CONTROL_RATE_HZ,
    reference_rms: float = 1.0,
) -> Intensity | None:
    """Energy and brightness of one note, at control rate."""

    start = int(round(onset * sample_rate))
    end = int(round((onset + duration) * sample_rate))
    start = max(0, min(start, audio.size))
    end = max(start, min(end, audio.size))
    segment = audio[start:end]
    if segment.size < 32:
        return None

    hop = max(1, int(round(sample_rate / control_rate_hz)))
    frame = max(hop * 2, 256)
    frames = max(1, (segment.size - frame) // hop + 1)

    rms_curve = np.empty(frames)
    centroid_curve = np.empty(frames)
    window = np.hanning(frame)
    frequencies = np.fft.rfftfreq(frame, 1.0 / sample_rate)
    for index in range(frames):
        chunk = segment[index * hop : index * hop + frame]
        if chunk.size < frame:
            chunk = np.pad(chunk, (0, frame - chunk.size))
        rms_curve[index] = float(np.sqrt(np.mean(chunk**2)))
        spectrum = np.abs(np.fft.rfft(chunk * window))
        total = float(spectrum.sum())
        centroid_curve[index] = float((spectrum * frequencies).sum() / total) if total > 0 else 0.0

    rms = float(np.sqrt(np.mean(segment**2)))
    peak = float(np.max(np.abs(segment)))
    to_db = lambda value: 20.0 * np.log10(max(value, 1e-9))  # noqa: E731

    # Attack: how fast the level rises over the first 50 ms, in dB/s. A
    # measurable stand-in for how decisively the note was started.
    attack_frames = max(2, int(round(0.05 * control_rate_hz)))
    head = rms_curve[: min(attack_frames, rms_curve.size)]
    if head.size >= 2 and head[0] > 0 and head[-1] > 0:
        attack_slope = (to_db(head[-1]) - to_db(head[0])) / (head.size / control_rate_hz)
    else:
        attack_slope = 0.0

    return Intensity(
        rms_db=to_db(rms),
        peak_db=to_db(peak),
        centroid_hz=float(np.mean(centroid_curve)) if centroid_curve.size else 0.0,
        attack_slope_db_per_second=float(attack_slope),
        # Level normalised against the part's own loudest material, so a quiet
        # recording and a loud one describe their own dynamic range rather than
        # the engineer's gain staging.
        level=float(np.clip(rms / max(reference_rms, 1e-9), 0.0, 4.0)),
        brightness=float(np.mean(centroid_curve)) if centroid_curve.size else 0.0,
        curve=rms_curve,
        centroid_curve=centroid_curve,
    )


def measure_transition(
    previous_offset: float | None,
    previous_pitch: float | None,
    onset: float,
    pitch: float,
    trace: PitchTrace,
) -> dict[str, Any]:
    """How this note joins the one before it."""

    if previous_offset is None or previous_pitch is None:
        return {"kind": "phrase_start", "gap_seconds": None, "interval_semitones": None}

    gap = onset - previous_offset
    interval = pitch - previous_pitch

    # Portamento: how far the pitch was still travelling at the very start of
    # the note. Measured as the first trace value, which is the distance from
    # the written pitch at the moment the note begins.
    entry_cents = float(trace.cents[0]) if trace.cents.size else 0.0

    if gap > 0.08:
        kind = "detached"
    elif gap < -0.01:
        kind = "overlapped"
    else:
        kind = "connected"

    return {
        "kind": kind,
        "gap_seconds": round(float(gap), 5),
        "interval_semitones": round(float(interval), 4),
        "entry_offset_cents": round(entry_cents, 3),
        "direction": int(np.sign(interval)),
    }
