"""Guards on the Accent Analyzer's contracts.

Not a re-measurement of the model — that is what `accent.crossval` is for. These
are the things that would make every measurement meaningless if they silently
stopped being true.
"""

from __future__ import annotations

import numpy as np
import pytest

from accent.baselines import RULE_FEATURES, fit_rule
from accent.calibration import fit_calibration
from accent.features import (
    ACCENT_INPUT_FEATURES,
    ACCENT_TARGETS,
    PHRASE_GAP_BEATS,
    fit_prominence_basis,
    note_feature_vector,
    phrase_contexts,
    phrase_feature_matrix,
)
from accent.meter import beat_strength, default_grouping, meter, syncopation
from datasets.urmp.accent import center_per_pitch, robust_local_z, squash

EVIDENCE_NAMES = ("attack", "dynamic", "agogic", "timbre")


def phrase(pitches, onsets, durations, signature=(4, 4), tempo=120.0):
    return phrase_contexts(
        pitches,
        onsets,
        durations,
        tempo_bpm=tempo,
        time_signature=signature,
        phrase_gap_beats=PHRASE_GAP_BEATS,
    )


# ── the line between input and target ────────────────────────────────────


def test_no_input_feature_is_named_for_a_performance_measurement():
    """The whole system is a fraud if an audio-derived quantity is an input.

    Crude, and it catches the copy-paste that would do it: every evidence family
    and every word that only appears on the target side.
    """

    forbidden = set(EVIDENCE_NAMES) | {
        "rms",
        "centroid",
        "flux",
        "onset_deviation",
        "duration_ratio",
        "vibrato",
        "intensity",
        "slope",
    }
    for name in ACCENT_INPUT_FEATURES:
        for word in forbidden:
            assert word not in name, f"{name} looks like a performance measurement"


def test_velocity_and_articulation_are_not_inputs():
    """Section 43 made structural: the analyser cannot copy what it cannot see."""

    for name in ACCENT_INPUT_FEATURES:
        assert "velocity" not in name
        assert "articulation" not in name


def test_the_feature_order_has_no_duplicates():
    assert len(set(ACCENT_INPUT_FEATURES)) == len(ACCENT_INPUT_FEATURES)


def test_the_rule_only_uses_features_that_exist():
    for name in RULE_FEATURES:
        assert name in ACCENT_INPUT_FEATURES


# ── meter ────────────────────────────────────────────────────────────────


def test_a_half_note_meter_measures_its_bar_in_quarter_beats():
    """The bug this replaced: reading the numerator as the bar length folded a
    3/2 bar in half and put a downbeat on beat 3 of every other one."""

    three_two = meter(3, 2)
    assert three_two.bar_beats == 6.0
    assert beat_strength(0.0, three_two) == 1.0
    assert beat_strength(2.0, three_two) == 0.5
    assert beat_strength(6.0, three_two) == 1.0
    assert beat_strength(3.0, three_two) == 0.25


def test_four_four_has_a_secondary_accent_and_three_four_does_not():
    four = meter(4, 4)
    assert [beat_strength(b, four) for b in (0.0, 1.0, 2.0, 3.0)] == [1.0, 0.5, 0.75, 0.5]
    three = meter(3, 4)
    assert [beat_strength(b, three) for b in (0.0, 1.0, 2.0)] == [1.0, 0.5, 0.5]


def test_the_grouping_table_matches_futureboards():
    """Transcribed from the DAW's `default_time_signature_grouping`; if the two
    drift, the model is trained on a different bar from the one the ruler draws."""

    assert default_grouping(4, 4) == (4,)
    assert default_grouping(3, 4) == (3,)
    assert default_grouping(6, 8) == (3, 3)
    assert default_grouping(5, 8) == (2, 3)
    assert default_grouping(7, 8) == (2, 2, 3)
    assert default_grouping(12, 8) == (3, 3, 3, 3)
    assert default_grouping(11, 8) == (2, 2, 2, 2, 3)


def test_a_note_held_across_a_stronger_beat_is_syncopated():
    four = meter(4, 4)
    assert syncopation(1.0, 2.0, four) == pytest.approx(0.25)
    assert syncopation(1.5, 1.5, four) == pytest.approx(0.5)
    assert syncopation(0.0, 1.0, four) == 0.0
    assert syncopation(0.5, 0.5, four) == 0.0


# ── features ─────────────────────────────────────────────────────────────


def test_identical_notes_on_different_beats_get_different_features():
    contexts = phrase([60] * 8, [float(i) for i in range(8)], [1.0] * 8)
    matrix = phrase_feature_matrix(contexts)
    strength = ACCENT_INPUT_FEATURES.index("metrical_strength")
    assert matrix[0, strength] > matrix[2, strength] > matrix[1, strength]


def test_changing_the_meter_changes_the_features():
    pitches = [60] * 12
    onsets = [float(i) for i in range(12)]
    durations = [1.0] * 12
    four = phrase_feature_matrix(phrase(pitches, onsets, durations, (4, 4)))
    three = phrase_feature_matrix(phrase(pitches, onsets, durations, (3, 4)))
    assert not np.allclose(four, three)


def test_every_feature_of_a_degenerate_phrase_is_finite():
    contexts = phrase([0], [0.0], [1e-6], tempo=1e-6)
    vector = note_feature_vector(contexts, 0)
    assert np.all(np.isfinite(vector)), dict(zip(ACCENT_INPUT_FEATURES, vector))


def test_a_long_note_ends_its_group():
    pitches = [60] * 6 + [72] + [60] * 6
    onsets = [i * 0.5 for i in range(6)] + [3.0] + [5.0 + i * 0.5 for i in range(6)]
    durations = [0.5] * 6 + [2.0] + [0.5] * 6
    contexts = phrase(pitches, onsets, durations)
    assert contexts[7].phrase_position == 0


# ── evidence and targets ─────────────────────────────────────────────────


def test_a_local_z_score_excludes_the_note_from_its_own_reference():
    """A note that is one of thirteen contributors to its own baseline is
    partly being compared with itself."""

    values = [0.0] * 12 + [10.0]
    z = robust_local_z(values, window=6, minimum_neighbours=4, mad_floor=1.0)
    assert z[12] == pytest.approx(10.0)


def test_a_local_z_score_reports_nothing_when_it_has_too_few_neighbours():
    assert robust_local_z([1.0, 2.0], window=6, minimum_neighbours=4) == [None, None]


def test_the_squash_maps_zero_to_the_middle_and_is_monotone():
    assert squash(0.0) == pytest.approx(0.5)
    assert squash(-3.0) < squash(-1.0) < squash(0.0) < squash(1.0) < squash(3.0)
    # Strictly inside the range across everything the corpus produces: the
    # measured z-scores run to about +/-4 at the 1st and 99th percentiles, and a
    # target sitting exactly on a boundary is one a regression head can only
    # approach from one side.
    assert 0.0 < squash(-5.0) and squash(5.0) < 1.0
    # Far outside it `tanh` saturates and the result is exactly 0 or 1. That is
    # correct rather than a bug — a note fifty deviations from its neighbourhood
    # is at the end of the scale — and it is recorded here so nobody
    # "fixes" it into an epsilon.
    assert squash(-100.0) == 0.0 and squash(100.0) == 1.0
    assert squash(None) is None


def test_per_pitch_centring_removes_a_pitch_dependent_offset():
    """A violin body is not flat, and the offset it adds at each pitch is not a
    playing decision. This is the transform that takes it out."""

    pitches = [60, 62, 60, 62, 60, 62, 60, 62]
    # A constant +6 on every 62 and nothing on any 60.
    values = [0.0, 6.0, 0.0, 6.0, 0.0, 6.0, 0.0, 6.0]
    centred = center_per_pitch(pitches, values, minimum_instances=4)
    assert all(abs(value) < 1e-9 for value in centred)


def test_the_prominence_basis_weights_all_four_evidences_positively():
    """The rejected alternative loaded agogic at -0.04. The shipped recipe must
    not: a note emphasised by taking time has to read as emphasised."""

    records = [
        {
            "accent_evidence": {
                name: float(value)
                for name, value in zip(
                    EVIDENCE_NAMES, np.random.default_rng(seed).random(4)
                )
            }
        }
        for seed in range(64)
    ]
    basis = fit_prominence_basis(records)
    assert set(basis.loadings) == set(EVIDENCE_NAMES)
    for name, loading in basis.loadings.items():
        assert loading > 0.0, f"{name} loads {loading}"


def test_a_note_with_no_evidence_gets_no_prominence_target():
    records = [
        {"accent_evidence": {name: 0.5 for name in EVIDENCE_NAMES}} for _ in range(64)
    ]
    basis = fit_prominence_basis(records)
    assert basis.project({}) is None
    assert basis.project({"dynamic": 0.9}) is not None


# ── calibration ──────────────────────────────────────────────────────────


def test_the_spread_calibration_preserves_every_ranking():
    """It changes the numbers and must not change the order: 'which note of this
    phrase is the most prominent' has to have the same answer either side."""

    rng = np.random.default_rng(7)
    predicted = 0.5 + 0.02 * rng.standard_normal((256, len(ACCENT_TARGETS)))
    actual = 0.5 + 0.2 * rng.standard_normal((256, len(ACCENT_TARGETS)))
    mask = np.ones_like(actual)

    calibration = fit_calibration(predicted, actual, mask)
    calibrated = calibration.apply(predicted)
    for column in range(len(ACCENT_TARGETS)):
        assert np.array_equal(
            np.argsort(predicted[:, column]), np.argsort(calibrated[:, column])
        )
        assert calibrated[:, column].std() > predicted[:, column].std()


def test_the_spread_calibration_never_shrinks():
    """Narrowing an already-narrow prediction would only make the lane less
    usable, so the gain has a floor of one."""

    rng = np.random.default_rng(11)
    predicted = 0.5 + 0.4 * rng.standard_normal((256, len(ACCENT_TARGETS)))
    actual = 0.5 + 0.01 * rng.standard_normal((256, len(ACCENT_TARGETS)))
    calibration = fit_calibration(predicted, actual, np.ones_like(actual))
    assert all(gain >= 1.0 for gain in calibration.gains.values())
