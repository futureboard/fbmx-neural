"""Procedural smoke-test dataset.

**This is plumbing, not a modelling target.**  The teacher below is a toy: a
one-pole level detector driving a gain, followed by a static shaper.  It is
deterministic, causal, and has ~80 ms of memory, which is precisely enough to
prove that

    dataset -> dataloader -> training -> loss decreases -> checkpoint ->
    resume -> export -> streaming inference

all work end to end.  A model that fits it has learned a first-order envelope
and a tanh.  No acoustic claim of any kind may be made from a model trained
here, and the exported ``.fbmx`` records ``model_source_type: synthetic`` so
that stays visible downstream.

The signal inventory covers the cases that break stateful models: impulses and
transient bursts (attack behaviour), amplitude steps and silence transitions
(release behaviour and state decay), chirps (frequency dependence), noise
(broadband), sines and multitones (steady state).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from fbmx.conditioning import (
    CategoricalParam,
    ConditioningSchema,
    ContinuousParam,
)
from fbmx.datasets.base import DatasetInfo, PairedItem, PairedSequenceDataset

__all__ = [
    "SIGNAL_FAMILIES",
    "SYNTHETIC_SCHEMA",
    "SyntheticTeacher",
    "SyntheticSmokeDataset",
]

SIGNAL_FAMILIES = (
    "sine",
    "multitone",
    "chirp",
    "noise",
    "impulse",
    "transient_burst",
    "amplitude_step",
    "silence_transition",
)

#: The smoke dataset exercises both parameter kinds on purpose: two continuous
#: controls and one categorical one, so the embedding path is covered by every
#: pipeline test rather than only by a unit test.
SYNTHETIC_SCHEMA = ConditioningSchema(
    continuous=(
        ContinuousParam("drive", 0.0, 1.0, 0.5, "", "toy pre-gain into the shaper"),
        ContinuousParam("mix", 0.0, 1.0, 1.0, "", "dry/wet blend of the toy effect"),
    ),
    categorical=(
        CategoricalParam(
            "mode",
            ("soft", "hard"),
            "soft",
            embedding_dim=4,
            description="shaper family; discrete on purpose, see conditioning.py",
        ),
    ),
)


# ---------------------------------------------------------------------------
# signal generation
# ---------------------------------------------------------------------------
def _render_signal(family: str, n: int, sr: int, rng: np.random.Generator) -> np.ndarray:
    t = np.arange(n, dtype=np.float64) / sr
    if family == "sine":
        f = float(rng.uniform(40.0, 4000.0))
        x = np.sin(2 * np.pi * f * t)
    elif family == "multitone":
        x = np.zeros(n)
        for _ in range(int(rng.integers(2, 5))):
            f = float(rng.uniform(50.0, 6000.0))
            x += np.sin(2 * np.pi * f * t + rng.uniform(0, 2 * np.pi))
        x /= np.abs(x).max() + 1e-9
    elif family == "chirp":
        f0, f1 = 30.0, float(rng.uniform(2000.0, 12000.0))
        k = (f1 - f0) / max(t[-1], 1e-9)
        x = np.sin(2 * np.pi * (f0 * t + 0.5 * k * t * t))
    elif family == "noise":
        x = rng.standard_normal(n) * 0.3
    elif family == "impulse":
        x = np.zeros(n)
        for pos in rng.integers(0, n, size=int(rng.integers(3, 8))):
            x[int(pos)] = float(rng.choice([-1.0, 1.0]))
    elif family == "transient_burst":
        x = rng.standard_normal(n) * 0.5
        env = np.zeros(n)
        for _ in range(int(rng.integers(2, 5))):
            start = int(rng.integers(0, max(n - 1, 1)))
            length = int(rng.integers(sr // 400, sr // 40))
            end = min(start + length, n)
            env[start:end] = np.exp(-np.linspace(0, 6, end - start))
        x *= env
    elif family == "amplitude_step":
        x = np.sin(2 * np.pi * float(rng.uniform(80.0, 900.0)) * t)
        env = np.ones(n) * float(rng.uniform(0.05, 0.2))
        edges = sorted(rng.integers(0, n, size=2).tolist())
        env[edges[0] : edges[1]] = float(rng.uniform(0.6, 1.0))
        x *= env
    elif family == "silence_transition":
        x = np.sin(2 * np.pi * float(rng.uniform(60.0, 1200.0)) * t)
        cut = int(n * float(rng.uniform(0.3, 0.7)))
        x[cut:] = 0.0
        if rng.random() < 0.5:
            x = x[::-1].copy()  # silence -> signal as well as signal -> silence
    else:  # pragma: no cover - guarded by the caller
        raise ValueError(f"unknown signal family {family!r}")

    peak = float(np.abs(x).max())
    if peak > 0:
        x = x / peak * float(rng.uniform(0.15, 0.95))
    return x.astype(np.float64)


# ---------------------------------------------------------------------------
# toy teacher
# ---------------------------------------------------------------------------
class SyntheticTeacher:
    """A deliberately simple, deliberately non-physical dynamic shaper.

    ``env[n] = a * env[n-1] + (1 - a) * |x[n]|`` with separate attack and
    release coefficients, a gain of ``1 / (1 + k * env)``, then a static
    nonlinearity.  Nothing here is derived from any real circuit and no
    parameter has a physical unit.  It exists to give the pipeline a target
    with memory, so a stateless model cannot fit it and streaming-equality
    tests actually test something.
    """

    def __init__(
        self,
        sample_rate: int = 48000,
        attack_ms: float = 5.0,
        release_ms: float = 80.0,
        knee: float = 4.0,
    ) -> None:
        self.sample_rate = sample_rate
        self.attack_ms = attack_ms
        self.release_ms = release_ms
        self.knee = knee
        self.a_att = float(np.exp(-1.0 / (sample_rate * attack_ms * 1e-3)))
        self.a_rel = float(np.exp(-1.0 / (sample_rate * release_ms * 1e-3)))

    def config(self) -> dict[str, Any]:
        return {
            "teacher": "SyntheticTeacher",
            "sample_rate": self.sample_rate,
            "attack_ms": self.attack_ms,
            "release_ms": self.release_ms,
            "knee": self.knee,
        }

    def __call__(
        self,
        dry: np.ndarray,  # [N, T]
        drive: np.ndarray,  # [N]
        mix: np.ndarray,  # [N]
        mode_index: np.ndarray,  # [N] 0 = soft, 1 = hard
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(wet [N, T], gain_reduction [N, T])``.

        The gain trace is returned as an auxiliary target: a future
        gain-reduction loss (see ``fbmx.losses.auxiliary``) needs exactly this, and
        the smoke dataset is where that plumbing gets exercised first.
        """
        n_seq, n_samples = dry.shape
        rectified = np.abs(dry)
        env = np.zeros(n_seq)
        env_trace = np.empty_like(dry)
        # Vectorised over sequences; the recursion over time cannot be, and
        # does not need to be -- the smoke set is tiny by construction.
        for i in range(n_samples):
            r = rectified[:, i]
            coeff = np.where(r > env, self.a_att, self.a_rel)
            env = coeff * env + (1.0 - coeff) * r
            env_trace[:, i] = env

        drive_c = drive[:, None]
        pre = 1.0 + 3.0 * drive_c
        gain = 1.0 / (1.0 + self.knee * (1.0 + 2.0 * drive_c) * env_trace)
        driven = pre * dry * gain

        soft = np.tanh(driven)
        hard = np.clip(driven, -0.7, 0.7) / 0.7
        shaped = np.where(mode_index[:, None] == 1, hard, soft)

        mix_c = mix[:, None]
        wet = mix_c * shaped + (1.0 - mix_c) * dry
        return wet, gain


# ---------------------------------------------------------------------------
# dataset
# ---------------------------------------------------------------------------
class SyntheticSmokeDataset(PairedSequenceDataset):
    """Fully deterministic dry/wet pairs generated in memory.

    Same ``seed`` and ``split`` always produce the same audio, on any machine,
    so a failing test is reproducible without shipping any data.
    """

    def __init__(
        self,
        num_sequences: int = 16,
        sequence_length: int = 12288,
        sample_rate: int = 48000,
        seed: int = 0,
        split: str = "train",
        teacher: SyntheticTeacher | None = None,
        with_aux: bool = True,
    ) -> None:
        self.teacher = teacher or SyntheticTeacher(sample_rate=sample_rate)
        self.split = split
        self.seed = seed
        self.num_sequences = int(num_sequences)
        self.sequence_length = int(sequence_length)
        self.with_aux = with_aux

        info = DatasetInfo(
            name="fbmx-synthetic-smoke",
            source="fbmx.datasets.synthetic.SyntheticSmokeDataset",
            source_type="synthetic",
            license="CC0-1.0",
            version=f"1-{split}-{seed}",
            sample_rate=sample_rate,
            attribution="",
            redistributable=True,
            notes=(
                "Procedurally generated pipeline smoke test. The teacher is a toy "
                "envelope-driven shaper and is not a model of any real device. No "
                "acoustic conclusions may be drawn from models trained on it."
            ),
            extra={
                "families": list(SIGNAL_FAMILIES),
                "num_sequences": self.num_sequences,
                "sequence_length": self.sequence_length,
                **self.teacher.config(),
            },
        )
        super().__init__(info, SYNTHETIC_SCHEMA)
        self._generate()
        # The "checksum" of a procedural dataset is a hash of what it produced.
        import hashlib

        h = hashlib.sha256()
        h.update(self._dry.numpy().tobytes())
        h.update(self._wet.numpy().tobytes())
        self.info.checksum = h.hexdigest()

    # -- generation ------------------------------------------------------
    def _generate(self) -> None:
        # split is folded into the seed so train and val never overlap
        split_offset = {"train": 0, "val": 1_000_003, "test": 2_000_003}.get(
            self.split, abs(hash(self.split)) % 1_000_000
        )
        rng = np.random.default_rng(self.seed + split_offset)

        dry = np.empty((self.num_sequences, self.sequence_length), dtype=np.float64)
        families: list[str] = []
        for i in range(self.num_sequences):
            family = SIGNAL_FAMILIES[i % len(SIGNAL_FAMILIES)]
            families.append(family)
            dry[i] = _render_signal(family, self.sequence_length, self.sample_rate, rng)

        drive = rng.uniform(0.0, 1.0, size=self.num_sequences)
        mix = rng.uniform(0.5, 1.0, size=self.num_sequences)
        mode_index = rng.integers(0, 2, size=self.num_sequences)

        wet, gain = self.teacher(dry, drive, mix, mode_index)

        self._families = families
        self._dry = torch.from_numpy(dry).float().unsqueeze(1)  # [N, 1, T]
        self._wet = torch.from_numpy(wet).float().unsqueeze(1)
        self._gain = torch.from_numpy(gain).float().unsqueeze(1)
        self._settings = [
            {
                "drive": float(drive[i]),
                "mix": float(mix[i]),
                "mode": SYNTHETIC_SCHEMA.categorical[0].categories[int(mode_index[i])],
            }
            for i in range(self.num_sequences)
        ]

    # -- Dataset ---------------------------------------------------------
    def __len__(self) -> int:
        return self.num_sequences

    def __getitem__(self, index: int) -> PairedItem:
        aux = {"gain": self._gain[index]} if self.with_aux else {}
        return PairedItem(
            dry=self._dry[index],
            wet=self._wet[index],
            params=self.schema.encode(self._settings[index]),
            key=f"{self.split}/{index:03d}/{self._families[index]}",
            aux=aux,
        )

    def settings(self, index: int) -> dict[str, Any]:
        return dict(self._settings[index])
