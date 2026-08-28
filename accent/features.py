"""What the Accent Analyzer reads, and what it is asked to predict.

Inputs are **score facts only**. Nothing here touches a recording, an F0 track,
or any measurement of a performance, because at runtime there is no performance
— the question is which notes of a score that has never been played should feel
important. The audio-derived quantities in `datasets.urmp.accent` exist solely
to build targets and must never appear on this side of the line; the parity
test in `tests/test_accent.py` asserts that no input name matches an evidence
name, which is a crude guard but catches the copy-paste that would make the
whole model a fraud.

Score velocity is deliberately **not** an input. URMP's score MIDI carries real
dynamic markings (velocities 35..127, standard deviation 20), so a model given
that column could learn "accent is velocity" and score well doing it — which is
the one implementation the brief forbids. Leaving it out makes the requirement
structural rather than aspirational: identical velocities cannot change the
analyser's answer because the analyser cannot see them. What velocity achieves
on its own is measured separately, as one of the baselines.

Articulation is not an input either, for a duller reason: URMP has no
articulation annotation, so the column would be constant through training and
pure extrapolation at runtime. Articulation enters downstream, where an accent
of a given strength is *realised* differently on a pizzicato than on a sustain.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from .meter import Meter, beat_strength, meter, syncopation

ACCENT_FEATURE_SCHEMA_VERSION = 1

#: A written rest at least this long, in quarter-note beats, starts a new
#: phrase.
#:
#: Chosen from the corpus rather than from taste. URMP's scores are notated
#: with each note's off at the next note's on inside a line, so 95% of notes
#: have a written gap under 0.027 beats and the distribution then jumps
#: straight to 0.5 beats at the 97th percentile: there is almost nothing in
#: between, and a sixteenth rest sits in the empty part of it. At this
#: threshold the corpus averages ~26 notes to a phrase. At one beat it averages
#: 82, which makes "where am I in the phrase" mean "where am I in the piece",
#: and the performance-derived segmentation the Performer's records use is
#: worse still at 155.
PHRASE_GAP_BEATS = 0.25

#: A note at least this many times the local median duration ends its group.
#:
#: Two is the smallest ratio that means "notably longer" rather than "dotted",
#: and it is what takes the corpus from 119-note phrases to 22-note ones. This
#: is a grouping approximation in the spirit of a grouping preference rule, not
#: a phrase analysis: it has no idea about cadences, harmony, or repetition,
#: and it will cut a line in two at a long note in the middle of it. Section 8
#: of the brief permits exactly that, and the alternative — a real phrase
#: segmenter — would be a research project rather than a feature.
PHRASE_ARRIVAL_RATIO = 2.0

#: Notes either side used for the "compared with its neighbours" score features.
#: Matches `datasets.urmp.accent.LOCAL_WINDOW_NOTES`, so a feature that says
#: "longer than its neighbours" is talking about the same neighbourhood the
#: target was measured against.
LOCAL_WINDOW_NOTES = 6

#: Order is the contract between training, export, the Rust runtime, and the
#: parity fixture. Append only; never reorder, never reuse a slot.
ACCENT_INPUT_FEATURES: tuple[str, ...] = (
    # ── meter ────────────────────────────────────────────────────────────
    "metrical_strength",
    "is_downbeat",
    "bar_progress_sin",
    "bar_progress_cos",
    "beat_offset",
    "syncopation",
    "spans_stronger_beat",
    "meter_beats_norm",
    "meter_is_compound",
    "meter_is_irregular",
    # ── duration ─────────────────────────────────────────────────────────
    "log_duration_beats",
    "duration_vs_local",
    "long_after_short",
    # ── separation ───────────────────────────────────────────────────────
    "rest_before",
    "rest_after",
    "starts_after_rest",
    # ── melody ───────────────────────────────────────────────────────────
    "previous_interval",
    "next_interval",
    "leap_into",
    "is_local_peak",
    "is_local_trough",
    "pitch_vs_local",
    "contour_turn",
    "repeated_note",
    "is_phrase_peak",
    "is_window_peak",
    "is_window_trough",
    # ── phrase ───────────────────────────────────────────────────────────
    "phrase_progress",
    "is_phrase_start",
    "is_phrase_end",
    "log_phrase_length",
    # ── global ───────────────────────────────────────────────────────────
    "pitch_norm",
    "log_tempo",
)

#: Regression heads, in output order. `confidence` is not here: it is not a
#: regression against a measured quantity but the model's own uncertainty about
#: `prominence`, produced by a separate head and trained by a separate term.
ACCENT_TARGETS: tuple[str, ...] = ("prominence", "attack", "agogic", "timbre")

#: Index of the log-variance head, appended after the regressions.
CONFIDENCE_INDEX = len(ACCENT_TARGETS)
ACCENT_OUTPUT_SIZE = len(ACCENT_TARGETS) + 1

#: Every target is already ``0..1``, so unlike the Performer there is nothing
#: to rescale. Recentring on 0.5 means an untrained head starts at "average
#: prominence" rather than at zero, which is the bottom of the range.
TARGET_CENTER = 0.5


def _clamp(value: float, low: float, high: float) -> float:
    return float(min(max(float(value), low), high))


@dataclass
class NoteContext:
    """One note of a phrase, as the analyser sees it.

    Deliberately not the URMP record type. Both the training pipeline and the
    DAW construct these, from a JSONL row and from clip state respectively, and
    keeping the analyser's input a plain struct is what lets the same feature
    code serve both without either one importing the other's world.
    """

    #: MIDI note number.
    pitch: int
    #: Quarter-note beats from the start of the piece.
    onset_beats: float
    #: Length in quarter-note beats.
    duration_beats: float
    #: Beats of rest before this note (0 when it follows the previous note
    #: immediately, negative clamped to 0 for overlaps). `None` at a part start.
    rest_before_beats: float | None
    rest_after_beats: float | None
    #: Index within its phrase, and the phrase's length in notes.
    phrase_position: int
    phrase_length: int
    #: Highest pitch anywhere in this note's phrase.
    phrase_peak_pitch: int
    tempo_bpm: float
    time_signature: tuple[int, int]


def phrase_contexts(
    pitches: Sequence[int],
    onsets: Sequence[float],
    durations: Sequence[float],
    *,
    tempo_bpm: float,
    time_signature: tuple[int, int],
    phrase_gap_beats: float,
) -> list[NoteContext]:
    """Build one context per note, with phrases cut at rests.

    Two rules, both read off the score: a written rest longer than
    `phrase_gap_beats`, and a note at least `PHRASE_ARRIVAL_RATIO` times the
    local median duration (the group ends *after* it). It is a crude reading of
    phrasing and is labelled as such wherever it appears; URMP has no phrase
    marks, so a music-theoretic segmenter would be inventing labels rather than
    reading them.
    """

    count = len(pitches)
    if count == 0:
        return []

    rests_before: list[float | None] = [None]
    for index in range(1, count):
        previous_end = onsets[index - 1] + durations[index - 1]
        rests_before.append(max(onsets[index] - previous_end, 0.0))
    rests_after: list[float | None] = []
    for index in range(count):
        if index + 1 >= count:
            rests_after.append(None)
        else:
            rests_after.append(max(onsets[index + 1] - (onsets[index] + durations[index]), 0.0))

    phrase_of: list[int] = []
    phrase = 0
    for index in range(count):
        gap = rests_before[index]
        boundary = gap is not None and gap > phrase_gap_beats
        # A note markedly longer than the ones around it ends a group even
        # without a rest after it. This is the other half of the segmentation
        # and on this corpus it is the half that does the work: URMP's scores
        # notate a line with every note-off on the next note-on, so cutting at
        # rests alone leaves a "phrase" averaging 119 notes — position in the
        # phrase would mean position in the piece. Adding the long-note rule
        # brings that to 22 notes, which is a length a musician would recognise.
        if not boundary and index > 0:
            previous_duration = durations[index - 1]
            low = max(0, index - 1 - LOCAL_WINDOW_NOTES)
            high = min(count, index + LOCAL_WINDOW_NOTES)
            neighbours = [durations[other] for other in range(low, high) if other != index - 1]
            if neighbours:
                boundary = previous_duration >= PHRASE_ARRIVAL_RATIO * float(
                    np.median(neighbours)
                )
        if boundary:
            phrase += 1
        phrase_of.append(phrase)

    lengths: dict[int, int] = {}
    peaks: dict[int, int] = {}
    positions: dict[int, int] = {}
    for index, current in enumerate(phrase_of):
        lengths[current] = lengths.get(current, 0) + 1
        peaks[current] = max(peaks.get(current, -1), int(pitches[index]))

    out: list[NoteContext] = []
    for index, current in enumerate(phrase_of):
        position = positions.get(current, 0)
        positions[current] = position + 1
        out.append(
            NoteContext(
                pitch=int(pitches[index]),
                onset_beats=float(onsets[index]),
                duration_beats=float(durations[index]),
                rest_before_beats=rests_before[index],
                rest_after_beats=rests_after[index],
                phrase_position=position,
                phrase_length=lengths[current],
                phrase_peak_pitch=peaks[current],
                tempo_bpm=float(tempo_bpm),
                time_signature=time_signature,
            )
        )
    return out


def _local_median(values: Sequence[float], index: int, window: int) -> float:
    low = max(0, index - window)
    high = min(len(values), index + window + 1)
    neighbours = [values[other] for other in range(low, high) if other != index]
    if not neighbours:
        return float(values[index])
    return float(np.median(neighbours))


def note_feature_vector(
    notes: Sequence[NoteContext], index: int, *, meter_: Meter | None = None
) -> np.ndarray:
    """The score-side description of one note, in `ACCENT_INPUT_FEATURES` order."""

    note = notes[index]
    grid = meter_ or meter(*note.time_signature)

    previous = notes[index - 1] if index > 0 else None
    following = notes[index + 1] if index + 1 < len(notes) else None

    strength = beat_strength(note.onset_beats, grid)
    displaced = syncopation(note.onset_beats, note.duration_beats, grid)

    bar = grid.bar_beats
    bar_position = (note.onset_beats % bar) / bar if bar > 0.0 else 0.0
    phase = 2.0 * np.pi * bar_position
    # Distance to the nearest beat, as a fraction of the local beat. Zero on a
    # beat, 0.5 exactly between two, and it says something `metrical_strength`
    # does not: a note 1/16 late of beat 3 and one exactly on the offbeat both
    # score 0.25 for strength but are different gestures.
    starts = grid.beat_starts() + [bar]
    within = note.onset_beats % bar if bar > 0.0 else 0.0
    nearest = min(starts, key=lambda start: abs(within - start))
    beat_length = max(grid.groups[0] * grid.unit_beats, 1e-6)
    beat_offset = _clamp(abs(within - nearest) / beat_length, 0.0, 1.0)

    durations = [other.duration_beats for other in notes]
    local_duration = _local_median(durations, index, LOCAL_WINDOW_NOTES)
    duration_vs_local = _clamp(
        float(np.log2(max(note.duration_beats, 1e-4) / max(local_duration, 1e-4))), -3.0, 3.0
    ) / 3.0
    # "Long note after short notes" as a single flag: this note is at least
    # half again as long as the one before it, which is the shape that makes a
    # note land as an arrival.
    long_after_short = (
        1.0
        if previous is not None and note.duration_beats >= 1.5 * max(previous.duration_beats, 1e-6)
        else 0.0
    )

    rest_before = note.rest_before_beats or 0.0
    rest_after = note.rest_after_beats or 0.0

    previous_interval = float(note.pitch - previous.pitch) if previous else 0.0
    next_interval = float(following.pitch - note.pitch) if following else 0.0
    peak = 1.0 if previous and following and note.pitch > previous.pitch and note.pitch > following.pitch else 0.0
    trough = 1.0 if previous and following and note.pitch < previous.pitch and note.pitch < following.pitch else 0.0

    pitches = [float(other.pitch) for other in notes]
    local_pitch = _local_median(pitches, index, LOCAL_WINDOW_NOTES)
    pitch_vs_local = _clamp((note.pitch - local_pitch) / 12.0, -2.0, 2.0)

    low = max(0, index - LOCAL_WINDOW_NOTES)
    high = min(len(notes), index + LOCAL_WINDOW_NOTES + 1)
    neighbourhood = [notes[other].pitch for other in range(low, high) if other != index]
    window_peak = 1.0 if neighbourhood and note.pitch > max(neighbourhood) else 0.0
    window_trough = 1.0 if neighbourhood and note.pitch < min(neighbourhood) else 0.0

    # A change of direction, whatever its size: the note where a rising line
    # turns over is heard, and `is_local_peak` alone misses the case where the
    # turn is not also a maximum of the immediate three.
    contour_turn = (
        1.0
        if previous is not None
        and following is not None
        and np.sign(previous_interval) != 0
        and np.sign(next_interval) != 0
        and np.sign(previous_interval) != np.sign(next_interval)
        else 0.0
    )

    values = {
        "metrical_strength": strength,
        "is_downbeat": 1.0 if strength >= 0.999 else 0.0,
        "bar_progress_sin": float(np.sin(phase)),
        "bar_progress_cos": float(np.cos(phase)),
        "beat_offset": beat_offset,
        "syncopation": _clamp(displaced, 0.0, 1.0),
        "spans_stronger_beat": 1.0 if displaced > 1e-6 else 0.0,
        "meter_beats_norm": _clamp(grid.beat_count / 6.0, 0.0, 2.0),
        "meter_is_compound": 1.0 if grid.is_compound else 0.0,
        "meter_is_irregular": 1.0 if grid.is_irregular else 0.0,
        "log_duration_beats": _clamp(float(np.log2(max(note.duration_beats, 1e-3))), -6.0, 4.0)
        / 4.0,
        "duration_vs_local": duration_vs_local,
        "long_after_short": long_after_short,
        "rest_before": _clamp(float(np.log1p(max(rest_before, 0.0))), 0.0, 3.0) / 3.0,
        "rest_after": _clamp(float(np.log1p(max(rest_after, 0.0))), 0.0, 3.0) / 3.0,
        # A sixteenth of separation is not a rest. An eighth is.
        "starts_after_rest": 1.0 if rest_before >= 0.25 else 0.0,
        "previous_interval": _clamp(previous_interval / 12.0, -2.0, 2.0),
        "next_interval": _clamp(next_interval / 12.0, -2.0, 2.0),
        "leap_into": _clamp(abs(previous_interval) / 12.0, 0.0, 2.0),
        "is_local_peak": peak,
        "is_local_trough": trough,
        "pitch_vs_local": pitch_vs_local,
        "contour_turn": contour_turn,
        "repeated_note": 1.0 if previous is not None and previous_interval == 0.0 else 0.0,
        "is_phrase_peak": 1.0 if note.pitch >= note.phrase_peak_pitch else 0.0,
        # Highest / lowest in a fixed neighbourhood, which `is_local_peak`
        # (three notes) and `is_phrase_peak` (a whole phrase) both miss: the top
        # of a four-bar arch is neither of those and is exactly the note a
        # player leans on.
        "is_window_peak": window_peak,
        "is_window_trough": window_trough,
        "phrase_progress": note.phrase_position / max(note.phrase_length - 1, 1),
        "is_phrase_start": 1.0 if note.phrase_position == 0 else 0.0,
        "is_phrase_end": 1.0 if note.phrase_position == note.phrase_length - 1 else 0.0,
        "log_phrase_length": _clamp(float(np.log2(max(note.phrase_length, 1))), 0.0, 10.0) / 10.0,
        "pitch_norm": (note.pitch - 69.0) / 24.0,
        "log_tempo": _clamp(float(np.log2(max(note.tempo_bpm, 1e-6) / 120.0)), -2.0, 2.0),
    }
    return np.asarray([values[name] for name in ACCENT_INPUT_FEATURES], dtype=np.float32)


def phrase_feature_matrix(notes: Sequence[NoteContext]) -> np.ndarray:
    """`(notes, features)` for a whole phrase."""

    if not notes:
        return np.zeros((0, len(ACCENT_INPUT_FEATURES)), dtype=np.float32)
    grid = meter(*notes[0].time_signature)
    return np.stack(
        [note_feature_vector(notes, index, meter_=grid) for index in range(len(notes))]
    )


# ── the derived prominence target ────────────────────────────────────────


#: How the four evidence families combine into the derived prominence target.
#:
#: **Equal weight on all four**, then z-scored on training and squashed. The
#: recipe is deliberately the dullest one available, and it was chosen against a
#: measured alternative rather than by default.
#:
#: The alternative was the first principal component of the four columns, which
#: is what "let the data decide the weights" means and what the first version of
#: this file used. Measured on the training split, PC1 loads:
#:
#:     attack +0.686   dynamic +0.713   agogic -0.044   timbre -0.140
#:
#: — that is, it is (attack + dynamic) and nothing else, and it is *negatively*
#: related to the other two. That is a true statement about the covariance of
#: these measurements and a bad definition of prominence, because the corpus
#: says the two groups measure different playing decisions: on these
#: performances a long note, a melodic peak, and a note after a leap are
#: emphasised by **taking time** (agogic correlates +0.09, +0.04, +0.31 with
#: those features) while short quick notes get the sharper attack and the
#: higher local level. A user-facing Accent control built on PC1 would
#: therefore have told a musician that the long arrival note they just wrote is
#: the *least* prominent note in the phrase.
#:
#: Perceptual prominence is the union of the ways a player can emphasise a
#: note, not the dominant direction of their covariance. Under equal weights
#: every component correlates positively with the result (attack 0.73, dynamic
#: 0.70, agogic 0.37, timbre 0.33) and the distribution keeps the components'
#: own scale. Two further variants were measured and rejected: a soft-OR
#: saturates (mean 0.87 of a 0..1 range) and a per-note maximum throws away
#: three of the four measurements.
#:
#: PC1's loadings are still fitted and recorded in the report, so the rejected
#: alternative stays checkable rather than becoming a claim in a comment.
PROMINENCE_BASIS_KEY = "prominence_basis"

EVIDENCE_NAMES: tuple[str, ...] = ("attack", "dynamic", "agogic", "timbre")


@dataclass
class ProminenceBasis:
    """The documented v1 recipe for turning four evidences into one number."""

    loadings: dict[str, float]
    #: Mean of each evidence column on training, subtracted before projection.
    means: dict[str, float]
    #: Standard deviation of the projection on training.
    scale: float
    #: Fraction of the four columns' variance PC1 explains. Recorded for the
    #: report; the shipped recipe does not use PC1 (see PROMINENCE_BASIS_KEY).
    explained_variance: float
    #: The rejected alternative's loadings, kept so the comparison in the
    #: report can be re-checked rather than taken on trust.
    pc1_loadings: dict[str, float] = field(default_factory=dict)

    def project(self, evidence: dict[str, float | None]) -> float | None:
        """One note's evidence to a prominence target in ``0..1``.

        Absent evidence is treated as "like its neighbours" — the column's own
        training mean — rather than dropping the note. A note with three of four
        measurements is still a usable example of prominence; a note with none
        is not, and returns `None`.
        """

        present = [name for name in EVIDENCE_NAMES if evidence.get(name) is not None]
        if not present:
            return None
        total = 0.0
        for name in EVIDENCE_NAMES:
            value = evidence.get(name)
            centred = (float(value) if value is not None else self.means[name]) - self.means[name]
            total += self.loadings[name] * centred
        # The same tanh squash the individual evidences use, so `prominence`
        # and `attack` are on one scale and a user dragging one of them in the
        # editor is dragging the same kind of number.
        return float(0.5 + 0.5 * np.tanh(total / max(self.scale, 1e-9) / 2.0))

    def to_dict(self) -> dict[str, Any]:
        return {
            "loadings": {k: round(v, 6) for k, v in self.loadings.items()},
            "means": {k: round(v, 6) for k, v in self.means.items()},
            "scale": round(self.scale, 6),
            "recipe": (
                "equal-weight mean of (attack, dynamic, agogic, timbre) evidence, "
                "centred on the training means, divided by the training standard "
                "deviation, squashed by 0.5 + 0.5*tanh(z/2)"
            ),
            "rejected_alternative": {
                "recipe": "first principal component of the same four columns",
                "loadings": {k: round(v, 6) for k, v in self.pc1_loadings.items()},
                "explained_variance": round(self.explained_variance, 6),
                "why_rejected": (
                    "loads agogic at -0.04 and timbre at -0.14, so a note "
                    "emphasised by taking time would read as unemphasised"
                ),
            },
        }

    @staticmethod
    def from_dict(payload: dict[str, Any]) -> "ProminenceBasis":
        rejected = payload.get("rejected_alternative") or {}
        return ProminenceBasis(
            loadings={k: float(v) for k, v in payload["loadings"].items()},
            means={k: float(v) for k, v in payload["means"].items()},
            scale=float(payload["scale"]),
            explained_variance=float(rejected.get("explained_variance", float("nan"))),
            pc1_loadings={k: float(v) for k, v in (rejected.get("loadings") or {}).items()},
        )


def fit_prominence_basis(records: Iterable[dict[str, Any]]) -> ProminenceBasis:
    """Derive the prominence recipe's scaling from the training records.

    The *weights* are fixed at equal — see `PROMINENCE_BASIS_KEY` for why, and
    for the principal component that was measured and rejected. What is fitted
    here is the centring and the scale, so that a prominence of 0.5 means "like
    its neighbourhood" on this corpus rather than on an arbitrary one, and PC1,
    which is recorded for comparison.

    Only notes with **all four** evidences measured contribute to the fit: a
    covariance estimated from rows with different columns missing is not a
    covariance of anything, and a mean estimated that way is not comparable
    across columns. Notes with some evidence still *receive* a target, through
    `ProminenceBasis.project`.
    """

    rows: list[list[float]] = []
    for record in records:
        evidence = record.get("accent_evidence") or {}
        if any(evidence.get(name) is None for name in EVIDENCE_NAMES):
            continue
        rows.append([float(evidence[name]) for name in EVIDENCE_NAMES])
    if len(rows) < 32:
        raise ValueError(f"only {len(rows)} fully-measured notes; cannot fit a prominence basis")

    matrix = np.asarray(rows, dtype=np.float64)
    means = matrix.mean(axis=0)
    centred = matrix - means

    weights = np.full(len(EVIDENCE_NAMES), 1.0 / len(EVIDENCE_NAMES))
    projection = centred @ weights

    # The rejected alternative, fitted only so the report can show it.
    eigenvalues, eigenvectors = np.linalg.eigh(np.cov(centred, rowvar=False))
    order = int(np.argmax(eigenvalues))
    component = eigenvectors[:, order]
    if component[EVIDENCE_NAMES.index("dynamic")] < 0:
        component = -component

    return ProminenceBasis(
        loadings={name: float(weights[i]) for i, name in enumerate(EVIDENCE_NAMES)},
        means={name: float(means[i]) for i, name in enumerate(EVIDENCE_NAMES)},
        scale=float(projection.std()),
        explained_variance=float(eigenvalues[order] / max(eigenvalues.sum(), 1e-12)),
        pc1_loadings={name: float(component[i]) for i, name in enumerate(EVIDENCE_NAMES)},
    )


# ── sequences ────────────────────────────────────────────────────────────


@dataclass
class AccentSequence:
    """One part's notes, features, targets, and per-target masks."""

    part_id: str
    piece_id: str
    inputs: np.ndarray  # (notes, ACCENT_INPUT_FEATURES)
    targets: np.ndarray  # (notes, ACCENT_TARGETS)
    mask: np.ndarray  # (notes, ACCENT_TARGETS)
    records: list[dict[str, Any]]
    #: The rule analyser's prediction for these notes, `(notes, ACCENT_TARGETS)`.
    #:
    #: Filled by `attach_rule_base` after the rule has been fitted on training
    #: data, and zero until then. The network is trained on `targets - base`, so
    #: this is not a diagnostic: it is half the model.
    base: np.ndarray | None = None

    def residual_targets(self) -> np.ndarray:
        """What the network is asked to predict: the rule's error."""

        if self.base is None:
            return self.targets
        return self.targets - self.base


def contexts_from_records(records: Sequence[dict[str, Any]]) -> list[NoteContext]:
    """Rebuild score contexts from URMP records.

    Phrases are segmented **from the score**, by `phrase_contexts`, and not
    copied from the record's `phrase_index`. The record's phrasing was cut at
    silences in the *recording*, which is a perfectly good way to segment a
    performance and a completely unavailable one at runtime: a DAW clip has no
    recording to look at. Training on performance-derived phrase boundaries and
    running on score-derived ones would give the model a different feature at
    runtime from the one it was fitted to, which is the quiet kind of
    train/serve skew that shows up as "the model is worse in the app".
    """

    if not records:
        return []
    tempo = float(records[0].get("tempo_bpm") or 120.0)
    beat_seconds = 60.0 / max(tempo, 1e-6)
    signature = tuple(records[0].get("time_signature") or (4, 4))
    return phrase_contexts(
        [int(record["score_pitch"]) for record in records],
        [float(record["score_onset_seconds"]) / beat_seconds for record in records],
        [float(record["score_duration_seconds"]) / beat_seconds for record in records],
        tempo_bpm=tempo,
        time_signature=(int(signature[0]), int(signature[1])),
        phrase_gap_beats=PHRASE_GAP_BEATS,
    )


def load_accent_sequences(path: str | Path, basis: ProminenceBasis) -> list[AccentSequence]:
    """Group a JSONL split into per-part sequences with accent targets."""

    by_part: dict[str, list[dict[str, Any]]] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        by_part.setdefault(record["part_id"], []).append(record)

    sequences: list[AccentSequence] = []
    for part_id in sorted(by_part):
        records = sorted(by_part[part_id], key=lambda item: item["note_index"])
        contexts = contexts_from_records(records)
        inputs = phrase_feature_matrix(contexts)

        targets = np.zeros((len(records), len(ACCENT_TARGETS)), dtype=np.float32)
        mask = np.zeros_like(targets)
        for row, record in enumerate(records):
            evidence = record.get("accent_evidence") or {}
            prominence = basis.project(evidence)
            values = {
                "prominence": prominence,
                "attack": evidence.get("attack"),
                "agogic": evidence.get("agogic"),
                "timbre": evidence.get("timbre"),
            }
            for column, name in enumerate(ACCENT_TARGETS):
                value = values[name]
                if value is None:
                    continue
                targets[row, column] = float(value)
                mask[row, column] = 1.0

        sequences.append(
            AccentSequence(
                part_id=part_id,
                piece_id=records[0]["piece_id"],
                inputs=inputs,
                targets=targets,
                mask=mask,
                records=records,
            )
        )
    return sequences


def input_normalization(
    sequences: Iterable[AccentSequence],
) -> tuple[np.ndarray, np.ndarray]:
    """Per-feature mean and standard deviation over the training split."""

    stacked = np.concatenate([sequence.inputs for sequence in sequences], axis=0)
    mean = stacked.mean(axis=0)
    std = stacked.std(axis=0)
    std[std < 1e-6] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)
