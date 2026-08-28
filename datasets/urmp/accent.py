"""Measure how strongly a human performer emphasised each note.

URMP has no accent label. It has recordings, an F0 track, a note table, and a
score, and that is all — so nothing here reads a field called `accent`, because
no such field exists. What is measured instead is *evidence*: four independent
families of descriptor that a listener uses to decide a note was emphasised,
each computed against the note's own neighbourhood rather than against the
recording as a whole.

    attack evidence     how decisively the note was started
    dynamic evidence    how much louder it is than the notes around it
    agogic evidence     how much extra time it was given
    timbre evidence     how much brighter it is than the notes around it

The local comparison is the whole point. A forte phrase is not a phrase of
accents, and a note that measures 6 dB above the recording's average is
unremarkable if every note around it does too. Every descriptor below is
therefore reduced to a robust z-score against a window of neighbouring notes
(median and MAD, not mean and standard deviation, so one mis-aligned note does
not move the reference), and it is those z-scores — never the raw dB — that
become targets.

What is deliberately *not* here: bow pressure, bow speed, bow position, or any
other physical quantity. URMP is audio; it never observed an arm. Brightness is
called brightness.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

#: Analysis hop for the attack window, in seconds. 5 ms resolves a violin
#: attack (10-80 ms) into a dozen frames; the 10 ms grid the rest of the
#: pipeline uses would resolve it into three.
ATTACK_HOP_SECONDS = 0.005

#: Frame length for the attack analysis, in seconds. Short enough not to smear
#: the transient across the note before it, long enough to give the FFT
#: something to say below 1 kHz.
ATTACK_FRAME_SECONDS = 0.023

#: How much of the preceding audio the flux calculation sees. Spectral flux is
#: a *difference* between frames, so measuring the flux at an onset needs the
#: frame before the onset — which is silence, or the tail of the previous note.
PRE_ROLL_SECONDS = 0.030

#: The "early window": how much of a note counts as its attack.
EARLY_SECONDS = 0.060

#: The body of a note starts after the attack has settled.
BODY_SKIP_SECONDS = 0.060

#: Boundary between "the body of the note" and "its upper partials", as a
#: multiple of the note's own fundamental.
#:
#: A fixed cutoff in Hz cannot do this job. Violin fundamentals in URMP span
#: G3 to roughly E7, three and a half octaves, so a 2 kHz line sits above the
#: sixth harmonic of a low note and below the fundamental of a high one — and
#: any "brightness" measured against it is mostly a measurement of pitch. The
#: first version of this file did exactly that, and the result was a timbre
#: evidence that correlated **-0.44** with loudness: louder notes are lower in
#: a violin part more often than not, and lower notes have more harmonics under
#: 2 kHz. Six harmonics is the same question at every pitch.
HF_HARMONIC = 6.0

#: Band the spectral descriptors are computed over, as multiples of the
#: fundamental. Below the fundamental is room and bleed; above the 24th
#: harmonic a violin has little but noise, and on a low note that is already
#: past 5 kHz.
BAND_LOW_HARMONIC = 0.8
BAND_HIGH_HARMONIC = 24.0

#: Absolute ceiling on the analysis band, in Hz, whatever the harmonic maths
#: says. Above this the microphone and the room dominate.
BAND_CEILING_HZ = 10_000.0

#: Notes either side of the target used as its comparison neighbourhood.
#: Six each side is about two bars of moving quarter notes — long enough for a
#: median to mean something, short enough that a crescendo does not become the
#: reference for its own peak.
LOCAL_WINDOW_NOTES = 6

#: Minimum neighbours (excluding the note itself) before a local statistic is
#: trusted. Below this the evidence is reported as unmeasured rather than
#: compared against one or two notes.
MIN_LOCAL_NEIGHBOURS = 4

_EPS = 1e-9

#: Smallest spread a local window may claim, per descriptor.
#:
#: A window of thirteen even quarter notes has almost no spread in gap or
#: duration, and dividing by that spread turns a 3 ms difference into a z-score
#: of forty. Each floor below is **half the 10th-percentile whole-part MAD**
#: measured across the 33 URMP violin parts — that is, half the spread of the
#: least varied complete performance in the corpus — so a local window can be
#: tighter than a whole part but not arbitrarily tighter. Two are raised above
#: that rule to a perceptual threshold instead, and say so.
#:
#: Measured 2026-08-28 on `datasets/urmp-violin`; re-derive with
#: `scripts/accent_report.py --scales` if the corpus changes.
MAD_FLOORS: dict[str, float] = {
    "onset_flux": 0.075,  # ratio; part MAD p10 0.154
    "attack_slope": 40.0,  # dB/s; part MAD p10 79.6
    "rise": 0.007,  # seconds; part MAD p10 0.0148
    "level_db": 1.8,  # dB; part MAD p10 3.64
    "log_duration_ratio": 0.047,  # log2; part MAD p10 0.094
    "onset_residual": 0.011,  # seconds; part MAD p10 0.0224
    # Perceptual, not statistical: the part MAD p10 rule gives 3.7 ms, and a
    # 3.7 ms difference in separation is not something a listener hears as
    # emphasis. Ten milliseconds is about where a gap becomes audible.
    "extra_gap": 0.010,  # seconds
    "centroid_ratio": 0.30,  # harmonics; part MAD p10 0.61
    "hf_ratio": 0.010,  # ratio; part MAD p10 0.021
    "spectral_tilt": 0.35,  # dB/octave; part MAD p10 0.70
}


def _to_db(value: float) -> float:
    return 20.0 * float(np.log10(max(float(value), 1e-9)))


@dataclass
class AttackAcoustics:
    """Raw, un-normalised descriptors of one performed note.

    Everything here is an absolute measurement of one note in one recording.
    None of it is a target: gain staging, microphone distance, and the room all
    move these numbers, and none of those is a playing decision. They become
    targets only after :func:`localize`.
    """

    #: RMS of the first `EARLY_SECONDS`, in dB relative to the part's reference.
    early_rms_db: float
    #: RMS of the note body, same reference.
    body_rms_db: float
    #: Peak sample of the note, same reference.
    peak_db: float
    #: dB/s rise over the first 50 ms, from the existing intensity measurement's
    #: definition so the two agree.
    attack_slope_db_per_second: float
    #: 10%-to-90% rise time of the level envelope, in seconds. `None` when the
    #: note never resolves a clean rise (a slurred entry inside a legato line
    #: often does not).
    rise_seconds: float | None
    #: Half-wave-rectified spectral flux summed over the onset window, divided
    #: by the part's reference level so it is a ratio rather than a gain.
    onset_flux: float
    #: Fraction of the early window's in-band energy above `HF_HARMONIC`.
    early_hf_ratio: float
    #: Spectral centroid of the early window, as a multiple of the fundamental.
    early_centroid_ratio: float
    #: Spectral centroid of the body, as a multiple of the fundamental.
    body_centroid_ratio: float
    #: Fraction of the body's in-band energy above `HF_HARMONIC`.
    body_hf_ratio: float
    #: Least-squares slope of body power against log frequency, in dB per
    #: octave above the fundamental. More negative is darker. Pitch-invariant
    #: in a way dB-per-kHz is not: a note an octave up has the same harmonic
    #: series stretched across twice the bandwidth, so its dB/kHz halves while
    #: its dB/octave does not move.
    body_spectral_slope_db_per_octave: float
    #: The fundamental every ratio above is relative to, in Hz.
    fundamental_hz: float
    #: Whether the note was long enough to separate an attack from a body.
    body_measured: bool

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "early_rms_db": round(self.early_rms_db, 3),
            "body_rms_db": round(self.body_rms_db, 3),
            "peak_db": round(self.peak_db, 3),
            "attack_slope_db_per_second": round(self.attack_slope_db_per_second, 3),
            "onset_flux": round(self.onset_flux, 5),
            "early_hf_ratio": round(self.early_hf_ratio, 5),
            "early_centroid_ratio": round(self.early_centroid_ratio, 4),
            "body_centroid_ratio": round(self.body_centroid_ratio, 4),
            "body_hf_ratio": round(self.body_hf_ratio, 5),
            "body_spectral_slope_db_per_octave": round(self.body_spectral_slope_db_per_octave, 4),
            "fundamental_hz": round(self.fundamental_hz, 3),
            "body_measured": self.body_measured,
        }
        if self.rise_seconds is not None:
            payload["rise_seconds"] = round(self.rise_seconds, 5)
        return payload


def measure_attack_acoustics(
    audio: np.ndarray,
    sample_rate: int,
    *,
    onset: float,
    duration: float,
    reference_rms: float,
    fundamental_hz: float,
) -> AttackAcoustics | None:
    """Descriptors of one note's attack and body.

    `reference_rms` is the part's own 95th-percentile note RMS — the same
    reference `measure_intensity` uses — so every dB here is relative to how
    loud this player played on this take, not to full scale.

    `fundamental_hz` is the note's *written* pitch in Hz. Every spectral
    descriptor is expressed relative to it, so brightness means "how much
    energy is in the upper partials" at every pitch rather than "how high is
    this note".
    """

    if duration <= 0.0 or not np.isfinite(fundamental_hz) or fundamental_hz <= 0.0:
        return None
    reference = max(float(reference_rms), 1e-9)

    hop = max(1, int(round(ATTACK_HOP_SECONDS * sample_rate)))
    frame = max(hop * 2, int(round(ATTACK_FRAME_SECONDS * sample_rate)))
    pre = int(round(PRE_ROLL_SECONDS * sample_rate))

    onset_sample = int(round(onset * sample_rate))
    end_sample = int(round((onset + duration) * sample_rate))
    start = max(0, onset_sample - pre)
    stop = min(audio.size, max(end_sample, onset_sample + frame))
    if stop - start < frame + hop:
        return None

    segment = audio[start:stop]
    window = np.hanning(frame)
    frequencies = np.fft.rfftfreq(frame, 1.0 / sample_rate)
    fundamental = float(fundamental_hz)
    band = (frequencies >= BAND_LOW_HARMONIC * fundamental) & (
        frequencies <= min(BAND_HIGH_HARMONIC * fundamental, BAND_CEILING_HZ)
    )
    # A note so high that fewer than eight bins fall in its harmonic band has
    # no spectrum to describe at this frame length. Widen to everything above
    # the fundamental rather than reporting a shape fitted to four points.
    if int(band.sum()) < 8:
        band = frequencies >= BAND_LOW_HARMONIC * fundamental
    hf = band & (frequencies >= HF_HARMONIC * fundamental)

    frames = 1 + (segment.size - frame) // hop
    if frames < 3:
        return None
    magnitudes = np.empty((frames, frequencies.size))
    levels = np.empty(frames)
    for index in range(frames):
        chunk = segment[index * hop : index * hop + frame]
        levels[index] = float(np.sqrt(np.mean(chunk**2)))
        magnitudes[index] = np.abs(np.fft.rfft(chunk * window))

    # Frame index whose *centre* is nearest the onset. Flux is read across this
    # boundary, so being a frame out puts the measurement in the note before.
    frame_times = (np.arange(frames) * hop + frame / 2.0) / sample_rate + (start / sample_rate)
    onset_frame = int(np.argmin(np.abs(frame_times - onset)))

    # ── attack: flux across the onset ────────────────────────────────────
    # Half-wave rectified so a note *ending* does not read as an attack, and
    # summed over 30 ms because a bow does not change the spectrum in 5.
    flux_span = max(1, int(round(0.030 / ATTACK_HOP_SECONDS)))
    lo = max(1, onset_frame)
    hi = min(frames, onset_frame + flux_span + 1)
    if hi <= lo:
        return None
    difference = magnitudes[lo:hi] - magnitudes[lo - 1 : hi - 1]
    onset_flux = float(np.maximum(difference, 0.0).sum() / (frame * reference))

    # ── level envelope ───────────────────────────────────────────────────
    early_frames = max(1, int(round(EARLY_SECONDS / ATTACK_HOP_SECONDS)))
    early_slice = slice(onset_frame, min(frames, onset_frame + early_frames))
    early_levels = levels[early_slice]
    if early_levels.size == 0:
        return None

    body_start_frame = onset_frame + int(round(BODY_SKIP_SECONDS / ATTACK_HOP_SECONDS))
    end_frame = min(frames, int(round((end_sample - start) / hop)))
    body_measured = end_frame - body_start_frame >= 3
    body_slice = slice(body_start_frame, end_frame) if body_measured else early_slice
    body_levels = levels[body_slice]
    if body_levels.size == 0:
        body_levels = early_levels
        body_slice = early_slice
        body_measured = False

    early_rms = float(np.sqrt(np.mean(early_levels**2)))
    body_rms = float(np.sqrt(np.mean(body_levels**2)))
    peak = float(np.max(np.abs(audio[max(0, onset_sample) : max(onset_sample + 1, end_sample)])))

    # Rise time over the first 200 ms: 10% to 90% of whatever peak the note
    # reaches in that stretch. A note that is already sounding when it is
    # annotated (a slur) never crosses 10% from below, and reports nothing.
    rise_frames = max(2, int(round(0.200 / ATTACK_HOP_SECONDS)))
    head = levels[onset_frame : min(frames, onset_frame + rise_frames)]
    rise_seconds: float | None = None
    if head.size >= 3:
        peak_head = float(head.max())
        # A note that is already at half its eventual level in its first frame
        # was not started from silence — it is a slurred or re-articulated entry
        # inside a sounding line. There is no rise to time, and calling that a
        # rise of zero seconds would make the softest joins in the corpus read
        # as its sharpest attacks.
        started_from_rest = peak_head > 1e-7 and head[0] < 0.5 * peak_head
        if started_from_rest:
            low = np.nonzero(head >= 0.1 * peak_head)[0]
            high = np.nonzero(head >= 0.9 * peak_head)[0]
            if low.size and high.size and high[0] >= low[0]:
                rise_seconds = float((high[0] - low[0]) * ATTACK_HOP_SECONDS)

    # Attack slope in dB/s over the first 50 ms, matching `measure_intensity`.
    slope_frames = max(2, int(round(0.050 / ATTACK_HOP_SECONDS)))
    slope_head = levels[onset_frame : min(frames, onset_frame + slope_frames)]
    if slope_head.size >= 2 and slope_head[0] > 0 and slope_head[-1] > 0:
        attack_slope = (_to_db(slope_head[-1]) - _to_db(slope_head[0])) / (
            slope_head.size * ATTACK_HOP_SECONDS
        )
    else:
        attack_slope = 0.0

    # ── spectra ──────────────────────────────────────────────────────────
    def spectral(rows: np.ndarray) -> tuple[float, float, float]:
        """Centroid (as a harmonic number), upper-partial ratio, and dB/octave.

        All three are computed inside the harmonic band, so the answer to
        "how bright is this note" does not change when the note changes pitch.
        """

        power = (rows**2).mean(axis=0)
        in_band = power[band]
        total = float(in_band.sum())
        if total <= 0.0:
            return 0.0, 0.0, 0.0
        centroid = float((in_band * frequencies[band]).sum() / total) / fundamental
        ratio = float(power[hf].sum() / total)
        if int(band.sum()) >= 8:
            decibels = 10.0 * np.log10(np.maximum(in_band, 1e-20))
            octaves = np.log2(frequencies[band] / fundamental)
            slope = float(np.polyfit(octaves, decibels, 1)[0])
        else:
            slope = 0.0
        return centroid, ratio, slope

    early_centroid, early_hf, _ = spectral(magnitudes[early_slice])
    body_centroid, body_hf, body_slope = spectral(magnitudes[body_slice])

    return AttackAcoustics(
        early_rms_db=_to_db(early_rms / reference),
        body_rms_db=_to_db(body_rms / reference),
        peak_db=_to_db(peak / reference),
        attack_slope_db_per_second=float(attack_slope),
        rise_seconds=rise_seconds,
        onset_flux=onset_flux,
        early_hf_ratio=early_hf,
        early_centroid_ratio=early_centroid,
        body_centroid_ratio=body_centroid,
        body_hf_ratio=body_hf,
        body_spectral_slope_db_per_octave=body_slope,
        fundamental_hz=fundamental,
        body_measured=body_measured,
    )


# ── local normalisation ──────────────────────────────────────────────────


def robust_local_z(
    values: Sequence[float | None],
    *,
    window: int = LOCAL_WINDOW_NOTES,
    minimum_neighbours: int = MIN_LOCAL_NEIGHBOURS,
    mad_floor: float = 0.0,
) -> list[float | None]:
    """Each value against the median and MAD of its neighbours.

    The note itself is excluded from its own reference. Including it pulls the
    median toward the note being judged, which is precisely backwards: the
    question is how this note compares with the ones around it, and a note that
    is one of seven contributors to its own baseline is partly compared with
    itself.

    `None` in, `None` out. `None` out also when fewer than `minimum_neighbours`
    usable neighbours exist — at the start of a part, or where the tracker lost
    several notes in a row — because a z-score against two notes is noise.

    MAD is scaled by 1.4826 so that on normally distributed data it estimates
    the standard deviation, which makes the resulting z comparable with the
    familiar one.
    """

    count = len(values)
    out: list[float | None] = [None] * count
    for index in range(count):
        value = values[index]
        if value is None or not np.isfinite(value):
            continue
        lo = max(0, index - window)
        hi = min(count, index + window + 1)
        neighbours = [
            float(values[other])
            for other in range(lo, hi)
            if other != index and values[other] is not None and np.isfinite(values[other])
        ]
        if len(neighbours) < minimum_neighbours:
            continue
        array = np.asarray(neighbours, dtype=np.float64)
        median = float(np.median(array))
        mad = 1.4826 * float(np.median(np.abs(array - median)))
        scale = max(mad, mad_floor, _EPS)
        out[index] = float((float(value) - median) / scale)
    return out


#: Instances of a written pitch a part must contain before that pitch gets its
#: own brightness reference. Below this the reference is read off a straight
#: line fitted through the pitches that do qualify.
MIN_PITCH_INSTANCES = 4


def center_per_pitch(
    pitches: Sequence[int | None],
    values: Sequence[float | None],
    *,
    minimum_instances: int = MIN_PITCH_INSTANCES,
) -> list[float | None]:
    """Remove the instrument's own response at each written pitch.

    A violin body is not flat. It has strong resonances, and a note whose
    fundamental lands on one is both **louder and relatively lower in
    centroid** — its energy piles into the first partial. Measured across all
    7113 URMP violin notes that showed up as a -0.46 correlation between local
    loudness and local brightness: louder notes measuring *darker*, which is
    the opposite of what bow force does and was therefore not a fact about
    playing at all. The same response also made notes *higher than their
    neighbours* measure as less emphasised (-0.08 against local pitch), which
    would have given the shipped analyser a melodic-height penalty it has no
    business having.

    It is present in every piece in the corpus, strongest in the two-instrument
    ones, and is unmoved by controlling for duration, for pitch linearly, or by
    subtracting a stationary noise floor — so it is neither ensemble bleed nor
    a recording artefact.

    The fix is to compare a note against *the same written pitch elsewhere in
    the same part*, and it goes on the **magnitude** descriptors — the two RMS
    levels and the onset flux — rather than on the spectral shapes. Those are
    the quantities the body's response scales directly; a spectral centroid
    expressed in harmonics and a slope expressed in dB per octave are already
    largely dimensionless, and an attack slope in dB/s is a ratio of two levels
    with the response cancelling out.

    Which side to centre was measured, not assumed, on both of the criteria
    that matter:

    ==========================  ================  ==================
    centred                     dynamic~timbre    prominence~pitch
    ==========================  ================  ==================
    spectral only                        -0.136              -0.108
    **magnitudes only**                  **-0.115**          **+0.003**
    both                                 -0.212              +0.003
    ==========================  ================  ==================

    Centring the magnitudes removes the confound from *both* sides at once,
    because once the loudness side no longer varies with the instrument's
    response there is nothing left for the brightness side's response to
    correlate with. Centring both is worse than centring either: two noisy
    per-pitch medians estimated from the same notes share their error.

    A cost worth stating: this also removes a recurring motif note that really
    is always played strongly. On a corpus with 7113 notes and no repeated-motif
    annotation there is no way to keep one and not the other.
    """

    grouped: dict[int, list[float]] = {}
    for pitch, value in zip(pitches, values):
        if pitch is None or value is None or not np.isfinite(value):
            continue
        grouped.setdefault(int(pitch), []).append(float(value))
    medians = {
        pitch: float(np.median(items))
        for pitch, items in grouped.items()
        if len(items) >= minimum_instances
    }

    known = sorted(medians)
    if len(known) >= 3:
        fit = np.polyfit(
            np.asarray(known, dtype=np.float64),
            np.asarray([medians[pitch] for pitch in known]),
            1,
        )
        fallback = lambda pitch: float(np.polyval(fit, float(pitch)))  # noqa: E731
    else:
        present = [v for v in values if v is not None and np.isfinite(v)]
        constant = float(np.median(present)) if present else 0.0
        fallback = lambda _pitch: constant  # noqa: E731

    out: list[float | None] = []
    for pitch, value in zip(pitches, values):
        if pitch is None or value is None or not np.isfinite(value):
            out.append(None)
            continue
        reference = medians.get(int(pitch))
        if reference is None:
            reference = fallback(pitch)
        out.append(float(value) - reference)
    return out


def squash(z: float | None, *, half_width: float = 2.0) -> float | None:
    """A robust z-score onto ``0..1``, with 0.5 meaning "like its neighbours".

    ``tanh`` rather than a clip so that being three deviations above the
    neighbourhood is still distinguishable from being six, and so a target
    never sits exactly on the boundary of its own range where a regression head
    can only approach it from one side. `half_width` sets how many deviations
    reach roughly 0.88.
    """

    if z is None or not np.isfinite(z):
        return None
    return float(0.5 + 0.5 * np.tanh(float(z) / max(half_width, 1e-6)))


@dataclass
class AccentEvidence:
    """One note's four evidence components, each ``0..1`` or absent.

    Absent means *not measured*, never *not accented*. A note whose body was
    too short to give a spectrum has no timbre evidence, and training a head
    against a fabricated 0.5 would teach the model that short notes are notes
    played without colour.
    """

    attack: float | None = None
    dynamic: float | None = None
    agogic: float | None = None
    timbre: float | None = None
    #: The z-scores the four came from, kept because the squash is lossy and a
    #: later study may want a different one.
    z: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        for name in ("attack", "dynamic", "agogic", "timbre"):
            value = getattr(self, name)
            if value is not None:
                payload[name] = round(value, 5)
        if self.z:
            payload["z"] = {key: round(value, 4) for key, value in self.z.items()}
        return payload


def part_evidence(
    acoustics: Sequence[AttackAcoustics | None],
    *,
    pitches: Sequence[int | None],
    duration_ratios: Sequence[float | None],
    onset_residuals: Sequence[float | None],
    extra_gaps: Sequence[float | None],
) -> list[AccentEvidence]:
    """Turn one part's raw measurements into locally-normalised evidence.

    Each family is built from more than one descriptor, because each descriptor
    on its own has a failure mode:

    * **attack** — flux alone fires on any spectral change including a change
      of note, and rise time alone is undefined for a slurred entry. Combining
      flux, rise (inverted: faster is stronger), and the dB/s slope means a
      note needs at least two of the three to agree before it reads as sharply
      attacked.
    * **dynamic** — the body level rather than the peak, so a click in the
      annotation does not become a fortissimo, averaged with the early level so
      a note that is loud only at its start still counts.
    * **agogic** — extra length against the local tempo, late placement, and
      *unwritten* extra silence in front of the note. All three are ways of
      giving a note time. Placement is signed: a note dragged late is
      emphasised, a note rushed is not, and a phrase where every note is 20 ms
      late is a tempo offset the local median removes. The gap term is the
      performed gap *minus the written rest at the local tempo*, because a
      written rest is the composer's phrasing and not the player's emphasis;
      without that subtraction every note after a rest would score as accented
      by construction, which is exactly the fake result section 41 asks the
      analyser to derive rather than assume.
    * **timbre** — brightness by three routes that agree on what brightness
      is: where the energy sits (centroid as a harmonic number), how much of it
      is above the sixth harmonic, and how fast the spectrum falls per octave.
      All three are expressed relative to the note's own fundamental, so a
      passage that climbs an octave does not read as a crescendo of brightness.
      What none of them can remove is the *timbre* of a violin's registers —
      the G string is not the E string — so a large leap between strings still
      contaminates this measurement. That is stated in the report rather than
      fixed, and it is the reason the residual dynamic/timbre correlation is
      -0.12 rather than zero.
    """

    count = len(acoustics)

    def column(getter, *, requires_body: bool = False) -> list[float | None]:
        out: list[float | None] = []
        for item in acoustics:
            if item is None or (requires_body and not item.body_measured):
                out.append(None)
                continue
            value = getter(item)
            out.append(None if value is None else float(value))
        return out

    # Magnitudes are judged against the same written pitch elsewhere in the
    # part before they are judged against the neighbours; see
    # `center_per_pitch`. The slope is not, because dB/s is already a ratio of
    # two levels and the instrument's response cancels out of it.
    flux_z = robust_local_z(
        center_per_pitch(pitches, column(lambda a: a.onset_flux)),
        mad_floor=MAD_FLOORS["onset_flux"],
    )
    slope_z = robust_local_z(
        column(lambda a: a.attack_slope_db_per_second), mad_floor=MAD_FLOORS["attack_slope"]
    )
    # A *shorter* rise is a stronger attack, so the sign is flipped before the
    # families are averaged rather than after, where it would be easy to miss.
    rise_z = robust_local_z(column(lambda a: a.rise_seconds), mad_floor=MAD_FLOORS["rise"])
    rise_z = [None if value is None else -value for value in rise_z]

    body_level_z = robust_local_z(
        center_per_pitch(pitches, column(lambda a: a.body_rms_db)),
        mad_floor=MAD_FLOORS["level_db"],
    )
    early_level_z = robust_local_z(
        center_per_pitch(pitches, column(lambda a: a.early_rms_db)),
        mad_floor=MAD_FLOORS["level_db"],
    )

    length_z = robust_local_z(
        [None if r is None else float(np.log2(max(float(r), 1e-3))) for r in duration_ratios],
        mad_floor=MAD_FLOORS["log_duration_ratio"],
    )
    placement_z = robust_local_z(
        list(onset_residuals), mad_floor=MAD_FLOORS["onset_residual"]
    )
    gap_z = robust_local_z(list(extra_gaps), mad_floor=MAD_FLOORS["extra_gap"])

    # The spectral shapes are left as measured: they are already expressed
    # relative to the note's own fundamental, and centring them as well made
    # the dynamic/timbre correlation worse rather than better.
    centroid_z = robust_local_z(
        column(lambda a: a.body_centroid_ratio, requires_body=True),
        mad_floor=MAD_FLOORS["centroid_ratio"],
    )
    hf_z = robust_local_z(
        column(lambda a: a.body_hf_ratio, requires_body=True),
        mad_floor=MAD_FLOORS["hf_ratio"],
    )
    tilt_z = robust_local_z(
        column(lambda a: a.body_spectral_slope_db_per_octave, requires_body=True),
        mad_floor=MAD_FLOORS["spectral_tilt"],
    )

    def blend(parts: Sequence[list[float | None]], index: int) -> float | None:
        present = [column[index] for column in parts if column[index] is not None]
        if not present:
            return None
        return float(np.mean(present))

    out: list[AccentEvidence] = []
    for index in range(count):
        attack_z = blend([flux_z, slope_z, rise_z], index)
        dynamic_z = blend([body_level_z, early_level_z], index)
        agogic_z = blend([length_z, placement_z, gap_z], index)
        timbre_z = blend([centroid_z, hf_z, tilt_z], index)
        zs = {
            name: float(value)
            for name, value in (
                ("attack", attack_z),
                ("dynamic", dynamic_z),
                ("agogic", agogic_z),
                ("timbre", timbre_z),
            )
            if value is not None
        }
        out.append(
            AccentEvidence(
                attack=squash(attack_z),
                dynamic=squash(dynamic_z),
                agogic=squash(agogic_z),
                timbre=squash(timbre_z),
                z=zs,
            )
        )
    return out
