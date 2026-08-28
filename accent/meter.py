"""Metrical structure: where a note sits in its bar, and what that is worth.

A bar is not a list of equal slots. Beat 1 of a 4/4 bar is not beat 2, beat 3
is not beat 2 either, and the eighth between them is weaker again — that
hierarchy is what makes a note on a weak subdivision *sound* like it is pushing
against something. Nothing here is specific to 4/4: the hierarchy is derived
from the time signature, so 3/4, 6/8, 5/8 and 7/8 each get the grid their own
grouping implies, and 3/2 gets a six-quarter bar rather than the three-quarter
bar that reading the numerator alone would give.

This module is mirrored, function for function and constant for constant, by
`solfege::accent::meter` in Futureboard. The two are checked against a shared
fixture (`scripts/accent_parity.py`) because a model trained on one grid and
run against another is a model given different music than it was taught on.
"""

from __future__ import annotations

from dataclasses import dataclass

def default_grouping(numerator: int, denominator: int) -> tuple[int, ...]:
    """Accent grouping in denominator units, mirroring Futureboard.

    This is a transcription of `default_time_signature_grouping` in
    `SphereUIComponents/src/components/timeline/state/time_signature.rs`, and
    it is a transcription rather than an independent choice on purpose. That
    function is what the DAW already shows in its bar ruler, and the user can
    override it per time-signature marker — so it, and not a table invented
    here, is the product's answer to "how is a 7/8 bar grouped". A model
    trained on a different convention from the one the editor draws would
    disagree with the ruler in front of the user.

    A single-entry result means a simple meter with no internal grouping: the
    beat is the denominator unit. More than one entry is a compound or additive
    meter whose beats start at the cumulative group boundaries.
    """

    num = max(int(numerator), 1)
    den = max(int(denominator), 1)
    table: dict[tuple[int, int], tuple[int, ...]] = {
        (2, 4): (2,),
        (3, 4): (3,),
        (4, 4): (4,),
        (5, 8): (2, 3),
        (6, 8): (3, 3),
        (7, 8): (2, 2, 3),
        (9, 8): (3, 3, 3),
        (12, 8): (3, 3, 3, 3),
    }
    if (num, den) in table:
        return table[(num, den)]
    if den == 8 and num % 2 == 1 and num > 3:
        pairs = (num - 3) // 2
        return (2,) * pairs + (3,)
    return (num,)

#: Strength of each metrical level, strongest first. Halving per level is the
#: usual reading of metrical weight and it keeps the numbers interpretable: a
#: beat is worth half a bar line, an offbeat half a beat.
LEVEL_STRENGTHS: tuple[float, ...] = (1.0, 0.75, 0.5, 0.25, 0.125, 0.0625)

#: Strength given to a position that matches no level of the grid at all.
OFF_GRID_STRENGTH = 0.03125

#: A note counts as "on" a metrical position when it is within half the finest
#: grid step of it.
#:
#: Scores are exact and would not need this. Played MIDI is not: a performance
#: recorded into the DAW lands a few milliseconds either side of every beat,
#: and a strength function that demanded exactness would return the off-grid
#: floor for every note in the take and quietly switch meter off.
GRID_TOLERANCE_FRACTION = 0.5


@dataclass(frozen=True)
class Meter:
    """One time signature, resolved into a metrical grid.

    Positions are in **quarter-note beats from the bar line**, which is the
    unit both the URMP records and the DAW timeline use. The denominator is
    what converts: a 3/2 bar is three half notes and therefore six quarters,
    and a 6/8 bar is six eighths and therefore three.
    """

    numerator: int
    denominator: int
    #: Bar length in quarter-note beats.
    bar_beats: float
    #: One denominator unit in quarter-note beats.
    unit_beats: float
    #: How many denominator units each beat contains, in order. `(1, 1, 1, 1)`
    #: for 4/4, `(3, 3)` for 6/8, `(2, 2, 3)` for 7/8.
    groups: tuple[int, ...]

    @property
    def beat_count(self) -> int:
        return len(self.groups)

    @property
    def is_compound(self) -> bool:
        return all(group == 3 for group in self.groups) and self.groups != (1,) * len(self.groups)

    @property
    def is_irregular(self) -> bool:
        return len(set(self.groups)) > 1

    def beat_starts(self) -> list[float]:
        """Start of each beat, in quarter-note beats from the bar line."""

        starts: list[float] = []
        position = 0.0
        for group in self.groups:
            starts.append(position)
            position += group * self.unit_beats
        return starts

    def levels(self) -> list[list[float]]:
        """Grid positions per metrical level, strongest level first.

        Coarser levels are *not* removed from finer ones; the lookup takes the
        strongest level a position matches, so a downbeat that also appears in
        the beat level still reads 1.0.
        """

        bar = self.bar_beats
        starts = self.beat_starts()

        bar_line = [0.0]
        # A half-bar level exists only where the bar really divides in two:
        # 4/4 and 12/8 have a secondary accent in the middle, 3/4 and 7/8 do
        # not, and inventing one for them would put a stress where players
        # place none.
        half_bar = (
            [bar / 2.0]
            if self.beat_count >= 4 and self.beat_count % 2 == 0 and not self.is_irregular
            else []
        )
        beats = list(starts)

        # First division of each beat: into its own denominator units for a
        # compound or additive beat (three eighths under a dotted quarter),
        # into halves for a simple one.
        divisions: list[float] = []
        for start, group in zip(starts, self.groups):
            length = group * self.unit_beats
            parts = group if group > 1 else 2
            for step in range(1, parts):
                divisions.append(start + length * step / parts)

        # Two further halvings. Positions already on a coarser level are
        # harmless duplicates — the lookup takes the strongest match.
        finer: list[list[float]] = []
        previous = sorted(set(beats + divisions + half_bar + bar_line))
        for _ in range(2):
            midpoints: list[float] = []
            extended = previous + [bar]
            for left, right in zip(extended, extended[1:]):
                midpoints.append((left + right) / 2.0)
            finer.append(midpoints)
            previous = sorted(set(previous + midpoints))

        return [bar_line, half_bar, beats, divisions, *finer]

    def finest_step(self) -> float:
        """Spacing of the finest level, in quarter-note beats."""

        levels = self.levels()
        positions = sorted({0.0, *(p for level in levels for p in level), self.bar_beats})
        gaps = [b - a for a, b in zip(positions, positions[1:]) if b - a > 1e-9]
        return min(gaps) if gaps else self.bar_beats


def meter(numerator: int, denominator: int) -> Meter:
    """Resolve a time signature into its grid."""

    num = max(int(numerator), 1)
    den = max(int(denominator), 1)
    unit = 4.0 / den
    accent_groups = default_grouping(num, den)
    # One group is "no internal grouping", which means the beat is the
    # denominator unit: 4/4 has four beats, 3/2 has three. More than one is a
    # compound or additive meter and the groups *are* the beats.
    groups = accent_groups if len(accent_groups) > 1 else (1,) * num
    return Meter(
        numerator=num,
        denominator=den,
        bar_beats=num * unit,
        unit_beats=unit,
        groups=groups,
    )


def beat_strength(position_beats: float, meter_: Meter) -> float:
    """Metrical weight of a position, in ``0..1``.

    `position_beats` is quarter-note beats from the *start of the piece*; the
    bar position is taken modulo the bar length here rather than by the caller,
    so a caller cannot get the modulus wrong for a meter whose bar is not
    `numerator` beats long.
    """

    bar = meter_.bar_beats
    if bar <= 0.0:
        return OFF_GRID_STRENGTH
    position = float(position_beats) % bar
    tolerance = meter_.finest_step() * GRID_TOLERANCE_FRACTION
    # A note a hair before the bar line belongs to the bar line, not to the
    # last subdivision of the bar before it.
    if bar - position <= tolerance:
        return LEVEL_STRENGTHS[0]
    for strength, level in zip(LEVEL_STRENGTHS, meter_.levels()):
        for grid in level:
            if abs(position - grid) <= tolerance:
                return strength
    return OFF_GRID_STRENGTH


def syncopation(position_beats: float, duration_beats: float, meter_: Meter) -> float:
    """How much stronger a metrical position this note covers than it starts on.

    Zero for a note that starts on the strongest position it touches — which is
    every note that begins on a downbeat, and every short note inside a beat.
    Positive when a note begins somewhere weak and *holds through* somewhere
    strong: the note is sounding when the strong beat arrives, nothing new
    articulates it, and the ear hears the note as displacing the beat. That is
    the case section 6 asks for, and it is why "strong beat = accent" is not
    enough on its own — the note that gets the emphasis here is the one *before*
    the strong beat.

    Only beat-level positions and stronger count as something to displace.
    Sustaining across an offbeat eighth is what every quarter note does.
    """

    if duration_beats <= 0.0:
        return 0.0
    start_strength = beat_strength(position_beats, meter_)
    bar = meter_.bar_beats
    if bar <= 0.0:
        return 0.0

    tolerance = meter_.finest_step() * GRID_TOLERANCE_FRACTION
    end = float(position_beats) + float(duration_beats)
    # Candidate strong positions are the bar lines, half-bars, and beats of
    # every bar the note touches.
    strongest = 0.0
    first_bar = int(float(position_beats) // bar)
    last_bar = int(end // bar)
    for bar_index in range(first_bar, last_bar + 1):
        origin = bar_index * bar
        for strength, level in list(zip(LEVEL_STRENGTHS, meter_.levels()))[:3]:
            for grid in level:
                absolute = origin + grid
                # Strictly inside: a position the note starts on is not one it
                # displaces, and one it merely reaches the end of is not held
                # through.
                if absolute > float(position_beats) + tolerance and absolute < end - tolerance:
                    strongest = max(strongest, strength)
    return max(0.0, strongest - start_strength)
