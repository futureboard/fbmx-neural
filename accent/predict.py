"""Run the shipped analyser the way the DAW runs it.

The training code fits things; this is the inference path, and it exists so that
the offline pipeline — the A/B renders, the ablations, the Performer's runtime
inputs — goes through exactly the composition the runtime does:

    score features ──> linear rule ──> + FBMX correction ──> spread calibration

Getting this wrong in a way that only shows up offline is easy and invisible:
apply the calibration before the correction, or forget it, and every number in
an evaluation is about a predictor the product does not contain. The Rust
implementation of the same three steps is `solfege::accent::analyzer`, and the
two are checked against a shared fixture.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .baselines import RuleAnalyzer
from .calibration import SpreadCalibration
from .features import (
    ACCENT_TARGETS,
    PHRASE_GAP_BEATS,
    NoteContext,
    phrase_contexts,
    phrase_feature_matrix,
)

#: What every component reads for a note the analyser declines to judge.
NEUTRAL = 0.5


@dataclass
class ShippedAnalyzer:
    """The rule, the calibration, and optionally the trained correction."""

    rule: RuleAnalyzer
    calibration: SpreadCalibration
    #: A torch `AccentAnalyzer`, or `None` for the rule alone.
    model: Any = None

    @staticmethod
    def load(rule_path: str | Path, checkpoint: str | Path | None = None) -> "ShippedAnalyzer":
        payload = json.loads(Path(rule_path).read_text(encoding="utf-8"))
        rule = RuleAnalyzer.from_dict(payload)
        calibration = SpreadCalibration.from_dict(payload.get("calibration") or {})
        model = None
        if checkpoint is not None:
            import torch

            from .model import AccentAnalyzer, AccentConfig

            state = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
            model = AccentAnalyzer(AccentConfig.from_dict(state["config"]))
            model.load_state_dict(state["model"])
            model.eval()
        return ShippedAnalyzer(rule=rule, calibration=calibration, model=model)

    def analyze(self, contexts: Sequence[NoteContext]) -> list[dict[str, float]]:
        """One accent per note, in the editor's `AccentState` shape."""

        if not contexts:
            return []
        features = phrase_feature_matrix(contexts)
        components = self.rule.predict(features)

        confidence = np.full(len(contexts), self.rule.confidence, dtype=np.float64)
        if self.model is not None:
            import torch

            from .model import CONFIDENCE_INDEX, LOG_VARIANCE_RANGE

            with torch.no_grad():
                out = (
                    self.model(torch.from_numpy(features).unsqueeze(0)).squeeze(0).numpy()
                )
            components = components + out[:, : len(ACCENT_TARGETS)]
            sigma = np.exp(
                0.5 * np.clip(out[:, CONFIDENCE_INDEX], *LOG_VARIANCE_RANGE)
            )
            confidence = np.clip(1.0 - sigma / 0.25, 0.0, 1.0)

        # Calibration last, and after the correction: calibrating the rule and
        # then adding an uncalibrated correction would add two quantities on
        # different scales.
        components = self.calibration.apply(components)
        return [
            {
                **{
                    name: float(np.clip(components[index, position], 0.0, 1.0))
                    for position, name in enumerate(ACCENT_TARGETS)
                },
                "confidence": float(confidence[index]),
            }
            for index in range(len(contexts))
        ]


def contexts_from_score(score_notes: Sequence, *, tempo_bpm: float, time_signature) -> list:
    """Build analyser contexts from `annotations.ScoreNote`s."""

    beat_seconds = 60.0 / max(float(tempo_bpm), 1e-6)
    return phrase_contexts(
        [note.midi_pitch for note in score_notes],
        [note.onset / beat_seconds for note in score_notes],
        [note.duration / beat_seconds for note in score_notes],
        tempo_bpm=float(tempo_bpm),
        time_signature=(int(time_signature[0]), int(time_signature[1])),
        phrase_gap_beats=PHRASE_GAP_BEATS,
    )


def neutral_accents(count: int) -> list[dict[str, float]]:
    """What the Performer reads when accent is switched off.

    The A leg of the A/B, and the control for every ablation: not "no accent
    input", which would be a different model, but the accent a note has when
    nothing has been analysed. Every component neutral, confidence zero.
    """

    return [
        {name: NEUTRAL for name in ACCENT_TARGETS} | {"confidence": 0.0}
        for _ in range(count)
    ]
