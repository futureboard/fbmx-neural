"""Align what was written to what was played.

The score and the performance are two note sequences over the same music, but
they are not the same length in seconds and they do not always have the same
number of notes: a player repeats, drops, or splits notes, and URMP's automatic
note annotation occasionally splits one long note in two. So the mapping has to
be *found*, not assumed, and a note whose match is doubtful must be marked
rather than fed to training as if it were certain.

Two stages:

1. **Match.** Dynamic time warping over the two note sequences, scored on pitch
   agreement and on how far the pairing drags time out of shape. Only one-to-one
   steps become matches; insertions and deletions are recorded and dropped.

2. **Separate tempo from microtiming.** A performance of a 44-second score that
   lasts 63 seconds is not "19 seconds late". Subtracting raw score onsets would
   measure the tempo difference and call it expression. Instead a smooth warp is
   fitted through the matched pairs — that is the tempo, including whatever
   rubato the player shaped over a phrase — and the *residual* from that warp is
   the note's own timing, which is what section 6 asks to learn.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .annotations import PerformedNote, ScoreNote

#: Pitch disagreement, in semitones, beyond which a pairing is not a match at
#: all. A violinist's intonation strays tens of cents, and URMP's note tracker
#: occasionally lands an octave out; one semitone separates those two cases.
MAX_MATCH_SEMITONES = 1.0

#: Cost charged for skipping a note on either side. Set above the cost of a
#: poor-but-plausible match so the warp prefers to pair notes when it can.
SKIP_COST = 1.2

#: Notes either side used when fitting the local tempo. Wide enough to average
#: out one note's expression, narrow enough to follow a ritardando.
TEMPO_WINDOW_NOTES = 8

#: A matched pair whose residual exceeds this is reported but marked
#: low-confidence: at a quarter second the note is more likely mis-paired than
#: expressively placed.
IMPLAUSIBLE_RESIDUAL_SECONDS = 0.25


@dataclass(frozen=True)
class NoteMatch:
    """One score note paired with the performed note that realised it."""

    score_index: int
    performed_index: int
    score: ScoreNote
    performed: PerformedNote
    #: Performed onset minus the smooth tempo warp's prediction, in seconds.
    #: Negative is early.
    onset_residual: float
    #: Performed duration over the duration the tempo warp predicts.
    duration_ratio: float
    #: Performed pitch relative to the written pitch, in cents.
    pitch_offset_cents: float
    #: Local seconds-per-score-second, i.e. how stretched this moment is.
    local_stretch: float
    confidence: float

    @property
    def is_confident(self) -> bool:
        return self.confidence >= 0.5


@dataclass
class Alignment:
    matches: list[NoteMatch]
    unmatched_score: list[int]
    unmatched_performed: list[int]
    #: Whole-part seconds of performance per second of score.
    global_stretch: float
    #: Fraction of score notes that found a confident match.
    coverage: float
    diagnostics: dict


def _dtw_match(
    score: Sequence[ScoreNote],
    performed: Sequence[PerformedNote],
    *,
    global_stretch: float,
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Monotonic one-to-one pairing of the two sequences."""

    n, m = len(score), len(performed)
    if n == 0 or m == 0:
        return [], list(range(n)), list(range(m))

    # Score onsets pushed into performed time by the global stretch, so the
    # time term of the cost compares like with like.
    score_time = np.asarray([note.onset for note in score]) * global_stretch
    score_pitch = np.asarray([float(note.midi_pitch) for note in score])
    performed_time = np.asarray([note.onset for note in performed])
    performed_pitch = np.asarray([note.midi_pitch for note in performed])

    # Time disagreement is normalised by the whole part so it stays a
    # dimensionless nudge; pitch is what actually decides a pairing.
    span = max(float(performed_time[-1] - performed_time[0]), 1e-6)

    pitch_cost = np.abs(score_pitch[:, None] - performed_pitch[None, :])
    time_cost = np.abs(score_time[:, None] - performed_time[None, :]) / span
    cost = pitch_cost + 0.5 * time_cost
    cost[pitch_cost > MAX_MATCH_SEMITONES] = np.inf

    accumulated = np.full((n + 1, m + 1), np.inf)
    accumulated[0, 0] = 0.0
    # 0 = match, 1 = skip score, 2 = skip performed
    choice = np.zeros((n + 1, m + 1), dtype=np.int8)
    for i in range(n + 1):
        for j in range(m + 1):
            if i == 0 and j == 0:
                continue
            best, best_choice = np.inf, 1
            if i > 0 and j > 0 and np.isfinite(cost[i - 1, j - 1]):
                candidate = accumulated[i - 1, j - 1] + cost[i - 1, j - 1]
                if candidate < best:
                    best, best_choice = candidate, 0
            if i > 0:
                candidate = accumulated[i - 1, j] + SKIP_COST
                if candidate < best:
                    best, best_choice = candidate, 1
            if j > 0:
                candidate = accumulated[i, j - 1] + SKIP_COST
                if candidate < best:
                    best, best_choice = candidate, 2
            accumulated[i, j] = best
            choice[i, j] = best_choice

    pairs: list[tuple[int, int]] = []
    unmatched_score: list[int] = []
    unmatched_performed: list[int] = []
    i, j = n, m
    while i > 0 or j > 0:
        step = choice[i, j]
        if step == 0 and i > 0 and j > 0:
            pairs.append((i - 1, j - 1))
            i, j = i - 1, j - 1
        elif step == 1 and i > 0:
            unmatched_score.append(i - 1)
            i -= 1
        elif j > 0:
            unmatched_performed.append(j - 1)
            j -= 1
        else:  # pragma: no cover - guard against a malformed table
            break
    pairs.reverse()
    unmatched_score.reverse()
    unmatched_performed.reverse()
    return pairs, unmatched_score, unmatched_performed


def _local_warp(
    score_times: np.ndarray, performed_times: np.ndarray, *, window: int
) -> tuple[np.ndarray, np.ndarray]:
    """Smooth monotonic map from score time to performed time.

    A local straight-line fit over a window of neighbouring matched notes. The
    slope it returns is the local tempo ratio, and the value it returns is where
    a note would fall if the player were keeping that tempo exactly — so the
    difference between it and the real onset is the note's own placement.
    """

    count = score_times.size
    predicted = np.empty(count)
    slope = np.empty(count)
    for index in range(count):
        low = max(0, index - window)
        high = min(count, index + window + 1)
        xs = score_times[low:high]
        ys = performed_times[low:high]
        if xs.size >= 2 and float(xs.max() - xs.min()) > 1e-9:
            a, b = np.polyfit(xs, ys, 1)
        else:
            a, b = 1.0, (ys.mean() - xs.mean() if xs.size else 0.0)
        slope[index] = a
        predicted[index] = a * score_times[index] + b
    # Keep the warp monotonic: a local fit can invert across a big rubato.
    predicted = np.maximum.accumulate(predicted)
    return predicted, slope


def align(
    score: Sequence[ScoreNote],
    performed: Sequence[PerformedNote],
    *,
    tempo_window: int = TEMPO_WINDOW_NOTES,
) -> Alignment:
    if not score or not performed:
        return Alignment(
            matches=[],
            unmatched_score=list(range(len(score))),
            unmatched_performed=list(range(len(performed))),
            global_stretch=1.0,
            coverage=0.0,
            diagnostics={"reason": "empty sequence"},
        )

    score_span = max(score[-1].offset - score[0].onset, 1e-6)
    performed_span = max(performed[-1].offset - performed[0].onset, 1e-6)
    global_stretch = performed_span / score_span

    pairs, unmatched_score, unmatched_performed = _dtw_match(
        score, performed, global_stretch=global_stretch
    )
    if not pairs:
        return Alignment(
            matches=[],
            unmatched_score=list(range(len(score))),
            unmatched_performed=list(range(len(performed))),
            global_stretch=global_stretch,
            coverage=0.0,
            diagnostics={"reason": "no pitch-compatible pairing"},
        )

    score_times = np.asarray([score[i].onset for i, _ in pairs])
    performed_times = np.asarray([performed[j].onset for _, j in pairs])
    predicted, slope = _local_warp(score_times, performed_times, window=tempo_window)
    residual = performed_times - predicted

    matches: list[NoteMatch] = []
    for position, (i, j) in enumerate(pairs):
        score_note, performed_note = score[i], performed[j]
        stretch = float(slope[position]) if slope[position] > 1e-6 else global_stretch
        expected_duration = score_note.duration * stretch
        duration_ratio = (
            performed_note.duration / expected_duration if expected_duration > 1e-6 else 1.0
        )
        pitch_offset = 1200.0 * np.log2(
            performed_note.frequency_hz / (440.0 * 2.0 ** ((score_note.midi_pitch - 69) / 12.0))
        )

        # Confidence blends the three things that can be wrong: the pitch does
        # not agree, the note sits implausibly far from the tempo, or its
        # length bears no relation to what was written.
        pitch_term = max(0.0, 1.0 - abs(pitch_offset) / 100.0)
        timing_term = max(0.0, 1.0 - abs(residual[position]) / IMPLAUSIBLE_RESIDUAL_SECONDS)
        duration_term = max(0.0, 1.0 - abs(np.log2(max(duration_ratio, 1e-3))) / 2.0)
        confidence = float(0.5 * pitch_term + 0.3 * timing_term + 0.2 * duration_term)

        matches.append(
            NoteMatch(
                score_index=i,
                performed_index=j,
                score=score_note,
                performed=performed_note,
                onset_residual=float(residual[position]),
                duration_ratio=float(duration_ratio),
                pitch_offset_cents=float(pitch_offset),
                local_stretch=stretch,
                confidence=confidence,
            )
        )

    confident = [match for match in matches if match.is_confident]
    diagnostics = {
        "score_notes": len(score),
        "performed_notes": len(performed),
        "paired": len(matches),
        "confident": len(confident),
        "unmatched_score": len(unmatched_score),
        "unmatched_performed": len(unmatched_performed),
        "global_stretch": round(global_stretch, 5),
        "onset_residual_ms": {
            "mean_abs": round(float(np.mean(np.abs(residual)) * 1000.0), 3),
            "median_abs": round(float(np.median(np.abs(residual)) * 1000.0), 3),
            "p95_abs": round(float(np.percentile(np.abs(residual), 95) * 1000.0), 3),
            "max_abs": round(float(np.max(np.abs(residual)) * 1000.0), 3),
        },
        "pitch_offset_cents": {
            "mean_abs": round(
                float(np.mean([abs(m.pitch_offset_cents) for m in matches])), 3
            ),
            "median_abs": round(
                float(np.median([abs(m.pitch_offset_cents) for m in matches])), 3
            ),
        },
    }

    return Alignment(
        matches=matches,
        unmatched_score=unmatched_score,
        unmatched_performed=unmatched_performed,
        global_stretch=global_stretch,
        coverage=len(confident) / len(score),
        diagnostics=diagnostics,
    )
