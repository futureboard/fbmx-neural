"""Read URMP's annotation tables and score MIDI.

Three sources, three different things:

``Notes_*.txt``
    What the player actually played, one line per note:
    ``onset_seconds  frequency_hz  duration_seconds``. This is the *performance*
    and it is the target side of every pair this pipeline builds.

``F0s_*.txt``
    The continuous pitch of that same performance:
    ``time_seconds  frequency_hz`` on a fixed hop, with ``0.0`` meaning
    unvoiced. The hop is measured from the file rather than assumed.

``Sco_*.mid``
    What was written. Track ``index`` of the MIDI is part ``index`` of the
    recording — track 0 carries the tempo map and no notes.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

#: Frequencies at or below this are URMP's "unvoiced" marker, not a pitch.
UNVOICED_HZ = 1e-6


def hz_to_midi(hz: np.ndarray | float) -> np.ndarray | float:
    """Fractional MIDI note number. A4 = 69 = 440 Hz."""

    return 69.0 + 12.0 * np.log2(np.asarray(hz, dtype=np.float64) / 440.0)


def midi_to_hz(midi: np.ndarray | float) -> np.ndarray | float:
    return 440.0 * np.exp2((np.asarray(midi, dtype=np.float64) - 69.0) / 12.0)


def cents_between(hz: np.ndarray | float, reference_hz: float) -> np.ndarray | float:
    """Deviation of ``hz`` from ``reference_hz``, in cents."""

    return 1200.0 * np.log2(np.asarray(hz, dtype=np.float64) / reference_hz)


@dataclass(frozen=True)
class PerformedNote:
    """One note as played, straight from ``Notes_*.txt``."""

    onset: float
    frequency_hz: float
    duration: float

    @property
    def offset(self) -> float:
        return self.onset + self.duration

    @property
    def midi_pitch(self) -> float:
        return float(hz_to_midi(self.frequency_hz))


@dataclass(frozen=True)
class ScoreNote:
    """One note as written, from the score MIDI."""

    onset: float
    duration: float
    midi_pitch: int
    velocity: int

    @property
    def offset(self) -> float:
        return self.onset + self.duration


@dataclass(frozen=True)
class F0Track:
    """Continuous performed pitch on a fixed time grid."""

    times: np.ndarray
    frequency_hz: np.ndarray
    hop_seconds: float

    @property
    def voiced(self) -> np.ndarray:
        return self.frequency_hz > UNVOICED_HZ

    def slice_seconds(self, start: float, end: float) -> tuple[np.ndarray, np.ndarray]:
        """Times and frequencies inside ``[start, end)``."""

        mask = (self.times >= start) & (self.times < end)
        return self.times[mask], self.frequency_hz[mask]


def read_performed_notes(path: str | Path) -> list[PerformedNote]:
    notes: list[PerformedNote] = []
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        fields = line.split()
        if len(fields) < 3:
            continue
        try:
            onset, frequency, duration = (float(fields[0]), float(fields[1]), float(fields[2]))
        except ValueError:
            continue
        if not (onset >= 0.0 and frequency > UNVOICED_HZ and duration > 0.0):
            continue
        notes.append(PerformedNote(onset=onset, frequency_hz=frequency, duration=duration))
    notes.sort(key=lambda note: note.onset)
    return notes


def read_f0(path: str | Path) -> F0Track:
    times: list[float] = []
    values: list[float] = []
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        fields = line.split()
        if len(fields) < 2:
            continue
        try:
            times.append(float(fields[0]))
            values.append(float(fields[1]))
        except ValueError:
            continue
    time_array = np.asarray(times, dtype=np.float64)
    value_array = np.asarray(values, dtype=np.float64)
    # Measure the hop rather than assuming 10 ms: a file with a different grid
    # would otherwise silently rescale every rate this pipeline derives.
    hop = float(np.median(np.diff(time_array))) if time_array.size > 1 else 0.0
    return F0Track(times=time_array, frequency_hz=value_array, hop_seconds=hop)


def read_score_notes(path: str | Path, track_index: int) -> list[ScoreNote]:
    """Notes of one score track, in seconds, with the tempo map applied.

    ``mido``'s per-message ``time`` is a delta in ticks; converting to seconds
    needs the tempo map, which URMP keeps on track 0 while the parts sit on
    tracks 1..n. Merging the file flattens both together so a tempo change
    partway through a piece is honoured.
    """

    import mido

    midi = mido.MidiFile(str(path))
    if track_index >= len(midi.tracks):
        return []

    ticks_per_beat = midi.ticks_per_beat
    # Tempo map from the whole file, as (tick, microseconds_per_beat).
    tempo_changes: list[tuple[int, int]] = []
    for track in midi.tracks:
        tick = 0
        for message in track:
            tick += message.time
            if message.type == "set_tempo":
                tempo_changes.append((tick, message.tempo))
    tempo_changes.sort(key=lambda item: item[0])
    if not tempo_changes or tempo_changes[0][0] > 0:
        tempo_changes.insert(0, (0, 500_000))

    def tick_to_seconds(target: int) -> float:
        seconds = 0.0
        previous_tick, previous_tempo = tempo_changes[0]
        for tick, tempo in tempo_changes[1:]:
            if tick >= target:
                break
            seconds += (tick - previous_tick) / ticks_per_beat * (previous_tempo / 1e6)
            previous_tick, previous_tempo = tick, tempo
        seconds += (target - previous_tick) / ticks_per_beat * (previous_tempo / 1e6)
        return seconds

    open_notes: dict[tuple[int, int], tuple[int, int]] = {}
    notes: list[ScoreNote] = []
    tick = 0
    for message in midi.tracks[track_index]:
        tick += message.time
        if message.type == "note_on" and message.velocity > 0:
            open_notes[(message.channel, message.note)] = (tick, message.velocity)
        elif message.type == "note_off" or (message.type == "note_on" and message.velocity == 0):
            key = (message.channel, message.note)
            started = open_notes.pop(key, None)
            if started is None:
                continue
            start_tick, velocity = started
            onset = tick_to_seconds(start_tick)
            offset = tick_to_seconds(tick)
            if offset <= onset:
                continue
            notes.append(
                ScoreNote(
                    onset=onset,
                    duration=offset - onset,
                    midi_pitch=message.note,
                    velocity=velocity,
                )
            )
    notes.sort(key=lambda note: (note.onset, note.midi_pitch))
    return notes


def score_tempo_bpm(path: str | Path) -> float:
    """The score's initial tempo in beats per minute."""

    import mido

    midi = mido.MidiFile(str(path))
    for track in midi.tracks:
        for message in track:
            if message.type == "set_tempo":
                return 60_000_000.0 / message.tempo
    return 120.0


def score_time_signature(path: str | Path) -> tuple[int, int]:
    import mido

    midi = mido.MidiFile(str(path))
    for track in midi.tracks:
        for message in track:
            if message.type == "time_signature":
                return (message.numerator, message.denominator)
    return (4, 4)
