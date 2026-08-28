"""Build voicebank -> VSCO residual training pairs for FBMX.

What the model is actually for
------------------------------
At runtime the DAW renders a Solfege track with ``SfmMode::VoicebankOnly`` and
applies the FBMX model to each output channel. The model's job is therefore a
narrow one: given the voicebank's own playback of a note, produce what the
original recording of that note sounded like. Everything in this module exists
to make the training pairs *be that*, because a residual model is only as good
as the difference it is asked to learn.

Four properties the earlier pairs did not have, each of which was measured to
matter more than any change to the model:

1. **The dry is what the runtime produces.** It was rendered with
   ``SfmMode::Hybrid``, which sums the bowed-string physical layer — an
   uncorrelated signal of comparable level and a ~40 Hz spectral centroid —
   into every training input. Measured over the pairs, that alone took the
   residual from 40 % of the target's energy to 101 % of it.

2. **The dry and the target are at the same level.** The voicebank renders a
   note at ``velocity`` scaled amplitude while the reference sits at unity, so
   the "residual" was dominated by a deterministic per-note gain of 1.5x to 7x
   (median 3.1x). A 5.7k-parameter causal LSTM asked to apply a broadband gain
   it cannot observe will smear rather than scale. After a single least-squares
   gain match the residual drops to ESR 0.001 — that is, the level was very
   nearly the entire "residual" the model was being trained on.

3. **Pairs are per channel, not downmixed.** The runtime never sees ``(L+R)/2``.
   The two channels of this source are only weakly correlated (median 0.36,
   some negative), so the old mono target was a comb-filtered signal losing a
   median of 1.75 dB and up to 5.6 dB — a spectrum the model would be corrected
   toward and then never encounter. Splitting per channel also doubles the
   number of training sequences.

4. **Every pair is verified before it is used.** Sample alignment, polarity and
   residual size are measured, and a pair that fails is rejected with a reason
   rather than quietly poisoning the objective.

The manifest records the per-pair diagnostics so a later run can see exactly
what it is training on.
"""

from __future__ import annotations

import json
import math
import os
import struct
import subprocess
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from fbmx.conditioning import CategoricalParam, ConditioningSchema, ContinuousParam
from fbmx.datasets.base import DatasetInfo
from fbmx.datasets.manifest import DatasetManifest, ManifestEntry
from fbmx.datasets.paired_audio import read_audio, write_wav

from .preprocess import read_jsonl

__all__ = ["build_vsco_fbmx_pairs", "velocity_for_dynamic", "align_and_match"]

SAMPLE_RATE = 48_000
PAIR_VERSION = "vsco2-ce-solo-violin-fbmx-v3-voicebank-perchannel"

#: Longest pair kept, in seconds. The cap bounds training cost; the tail of a
#: 17 s sustain teaches the model nothing the first seconds did not. The cut is
#: faded rather than hard, because a step discontinuity at the end of a target
#: is a broadband impulse that the loss will chase.
MAX_PAIR_SECONDS = 6.0
#: Fade applied at the cut only — never at the start, where it would erase the
#: attack transient that carries the instrument's identity.
CUT_FADE_MS = 20.0

#: A pair whose channels disagree in time by more than this is rejected. The
#: renderer starts the entry at frame 0 and the reference is the same recording,
#: so the honest expectation is exactly zero; anything else means the entry
#: mapping is wrong and the pair would teach the model to smear.
MAX_ALIGNMENT_SAMPLES = 1
#: A pair whose residual still carries more energy than this fraction of the
#: target, after alignment and gain matching, is not a correction problem and is
#: rejected rather than trained on.
MAX_RESIDUAL_ESR = 0.25

#: Largest transposition, in semitones, used to build a correction pair.
#:
#: The bank holds 19 distinct pitches across MIDI 43..96 with gaps of 2 to 4
#: semitones, so the resolver's worst stand-in is 2 semitones away and its
#: typical one is 1. Generating beyond that would teach the model to repair a
#: shift the runtime never asks for.
MAX_TRANSPOSE_SEMITONES = 2

#: Longest interpolation pair, in seconds.
#:
#: Shorter than `MAX_PAIR_SECONDS` because interpolation error is stationary:
#: it is a property of the read positions, not of where in the note they fall,
#: so three seconds of it says everything six would. The reference resample is
#: the expensive step in building these and this halves it.
INTERP_PAIR_SECONDS = 3.0

#: An interpolation pair whose error is smaller than this is genuinely
#: transparent and only dilutes the objective.
#:
#: Calibrated, not guessed: measured across the shifted renders, the engine's
#: 4-point Hermite read sits at ESR 1.8e-04 to 7.6e-04 (median 2.8e-04) against
#: the 64-tap float64 reference — about 35 dB down. The threshold sits an order
#: of magnitude below the median so the ordinary case is kept and only the
#: genuinely null pairs are dropped.
MIN_TRANSPOSED_ESR = 2e-5


def velocity_for_dynamic(dynamic: str | None) -> float:
    """Map the source's conservative dynamic labels to render velocity."""

    return {
        "p": 0.35,
        "f": 0.75,
        "v1": 0.85,
        "v2": 0.65,
    }.get(str(dynamic or "").casefold(), 0.7)


def _default_renderer(workspace_root: Path) -> Path:
    candidates = (
        workspace_root / "solfege-engine" / "target" / "release" / "solfage-model.exe",
        workspace_root / "solfege-engine" / "target" / "debug" / "solfage-model.exe",
        workspace_root / "solfege-engine" / "target" / "release" / "solfage-model",
        workspace_root / "solfege-engine" / "target" / "debug" / "solfage-model",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "solfage-model not found; build it with "
        "cargo build -p solfege-tools --release --manifest-path solfege-engine/Cargo.toml"
    )


def _voicebank_entry_ids(sfm_path: Path) -> dict[str, int]:
    """Read the compiler's source-record mapping from the SFM ACOU section."""

    raw = sfm_path.read_bytes()
    if raw[:4] != b"SFM\0":
        raise ValueError(f"invalid SFM magic in {sfm_path}")
    count = struct.unpack_from("<I", raw, 12)[0]
    table_offset = struct.unpack_from("<Q", raw, 16)[0]
    for index in range(count):
        entry = table_offset + index * 56
        if raw[entry : entry + 4] != b"ACOU":
            continue
        offset, size = struct.unpack_from("<QQ", raw, entry + 8)
        payload = json.loads(raw[offset : offset + size])
        return {
            str(row["source_record_id"]): int(row["id"])
            for row in payload.get("entries", [])
            if row.get("source_record_id") is not None
        }
    raise ValueError(f"SFM ACOU section is missing from {sfm_path}")


def _render_transposed(
    renderer: Path,
    sfm_path: Path,
    jobs: list[dict[str, Any]],
    jobs_path: Path,
    output_dir: Path,
    max_seconds: float,
) -> None:
    """Render "this recorded entry, played at that pitch" for every job."""

    jobs_path.parent.mkdir(parents=True, exist_ok=True)
    jobs_path.write_text(json.dumps(jobs, indent=1), encoding="utf-8")
    completed = subprocess.run(
        [
            str(renderer),
            "render-transposed",
            str(sfm_path),
            str(jobs_path),
            str(output_dir),
            str(max_seconds),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()
        raise RuntimeError(f"transposed render failed: {detail}")


def _render_batch(
    renderer: Path,
    sfm_path: Path,
    dataset_root: Path,
    manifest_path: Path,
    output_dir: Path,
    max_seconds: float,
) -> None:
    command = [
        str(renderer),
        "render-batch",
        str(sfm_path),
        str(dataset_root),
        str(manifest_path),
        str(output_dir),
        str(max_seconds),
        # The mode the DAW actually plays. See the module docstring.
        "voicebank",
    ]
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()
        raise RuntimeError(f"voicebank batch render failed: {detail}")


def _best_lag(target: np.ndarray, candidate: np.ndarray, max_lag: int) -> int:
    """Integer sample lag maximising the cross-correlation of the pair."""

    n = min(target.size, candidate.size)
    if n < 32 or max_lag <= 0:
        return 0
    a = target[:n] - target[:n].mean()
    b = candidate[:n] - candidate[:n].mean()
    size = 1 << int(math.ceil(math.log2(2 * n)))
    correlation = np.fft.irfft(np.fft.rfft(a, size) * np.conj(np.fft.rfft(b, size)), size)
    lags = np.concatenate([np.arange(0, max_lag + 1), np.arange(size - max_lag, size)])
    peak = int(lags[int(np.argmax(correlation[lags]))])
    return peak - size if peak > size // 2 else peak


def align_and_match(dry: np.ndarray, wet: np.ndarray) -> dict[str, Any]:
    """Measure how far a pair is from being a correction problem.

    Returns the integer lag, the least-squares gain that best maps ``dry`` onto
    ``wet``, and the residual ESR that remains after applying it. The gain is
    reported rather than applied here so the caller decides which side of the
    pair moves — see ``build_vsco_fbmx_pairs``, which scales the *target* down
    to the dry's level so the model never has to learn a gain.
    """

    n = min(dry.size, wet.size)
    dry, wet = dry[:n], wet[:n]
    lag = _best_lag(wet, dry, max_lag=min(4800, n // 2))
    denominator = float(np.dot(dry, dry))
    gain = float(np.dot(wet, dry) / denominator) if denominator > 1e-20 else 0.0
    matched = gain * dry
    energy = float(np.sum(wet.astype(np.float64) ** 2))
    residual = wet - matched
    esr = float(np.sum(residual.astype(np.float64) ** 2) / energy) if energy > 1e-20 else float("inf")
    return {
        "lag_samples": lag,
        "gain": gain,
        "residual_esr": esr,
        "dry_rms": float(np.sqrt(np.mean(dry**2))),
        "wet_rms": float(np.sqrt(np.mean(wet**2))),
    }


def resample_by_ratio(x: np.ndarray, ratio: float, taps: int = 64) -> np.ndarray:
    """Read ``x`` at ``position = k * ratio`` with a long windowed sinc, in f64.

    This is the *reference* interpolator: the answer the realtime engine's
    4-point Hermite read is trying to approximate when it transposes a recording.
    64 taps and float64 arithmetic make its own error negligible next to the
    error being measured, and it uses exactly the same read positions the engine
    uses (cursor starts at 0 and advances by ``ratio``), so the two outputs are
    sample-aligned by construction rather than by a correlation search.
    """
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    if n == 0 or ratio <= 0.0:
        return np.zeros(0, dtype=np.float32)
    out_len = max(1, int((n - taps) / ratio))
    half = taps // 2
    positions = np.arange(out_len, dtype=np.float64) * ratio
    centers = np.floor(positions).astype(np.int64)
    offsets = np.arange(-half + 1, half + 1, dtype=np.int64)
    padded = np.pad(x, (half, half), mode="edge")
    out = np.empty(out_len, dtype=np.float64)
    # Chunked so the tap matrix stays cache friendly on long recordings.
    cutoff = min(1.0, 1.0 / ratio)
    for start in range(0, out_len, 8192):
        stop = min(out_len, start + 8192)
        taps_index = centers[start:stop, None] + offsets[None, :]
        distance = taps_index - positions[start:stop, None]
        window = 0.5 + 0.5 * np.cos(np.pi * np.clip(distance / half, -1.0, 1.0))
        weights = np.sinc(distance * cutoff) * cutoff * window
        weights /= np.sum(weights, axis=1, keepdims=True).clip(1e-12)
        out[start:stop] = np.sum(padded[taps_index + half] * weights, axis=1)
    return out.astype(np.float32)


def _entry_metadata(sfm_path: Path) -> dict[str, dict[str, Any]]:
    """`source_record_id -> {entry id, midi note, articulation, dynamic}`."""

    raw = sfm_path.read_bytes()
    count = struct.unpack_from("<I", raw, 12)[0]
    table_offset = struct.unpack_from("<Q", raw, 16)[0]
    for index in range(count):
        entry = table_offset + index * 56
        if raw[entry : entry + 4] != b"ACOU":
            continue
        offset, size = struct.unpack_from("<QQ", raw, entry + 8)
        payload = json.loads(raw[offset : offset + size])
        return {
            str(row["source_record_id"]): row
            for row in payload.get("entries", [])
            if row.get("source_record_id") is not None
        }
    raise ValueError(f"SFM ACOU section is missing from {sfm_path}")


def _interpolation_jobs(
    records: list[Mapping[str, Any]],
    entry_meta: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Ask the engine to play each recording at a shifted pitch.

    **Why not pair a transposed recording against the real recording of the
    target pitch?** Because they are different takes. Measured over 736 such
    pairs, the best cross-correlation lag ranged from 184 to 1056 samples: the
    bow attack, the vibrato phase and the noise are simply different
    performances, and no sample-accurate map exists between them. Training a
    waveform model on that target asks it to predict what is unpredictable, and
    the least-error answer is a phase-incoherent average — which is precisely
    what "watery" and "smeared" sound like. That is the formulation error behind
    the current model, not a hyperparameter.

    The well-posed question keeps the take fixed and varies only the
    interpolator: play *this* recording at *that* pitch with the realtime
    engine's 4-point Hermite read, and compare against the same recording read
    at the same positions with a 64-tap sinc in float64. Same take, same read
    positions, aligned by construction — the difference is exactly the realtime
    interpolator's error, which is deterministic, small, and worth removing.
    """

    jobs: list[dict[str, Any]] = []
    for row in sorted(records, key=lambda item: str(item["id"])):
        key = str(row["id"])
        meta = entry_meta.get(key)
        if meta is None:
            continue
        note = int(row.get("midi_note") or 60)
        for shift in range(-MAX_TRANSPOSE_SEMITONES, MAX_TRANSPOSE_SEMITONES + 1):
            target_note = note + shift
            if shift == 0 or not 0 <= target_note <= 127:
                continue
            jobs.append(
                {
                    "out": f"{key}__shift{shift:+d}.wav",
                    "entry_id": int(meta["id"]),
                    "note": target_note,
                    "velocity": velocity_for_dynamic(row.get("dynamic")),
                    "articulation": row.get("articulation") or "sustain_vibrato",
                    # Carried for the manifest; the renderer ignores extra keys.
                    "source_id": key,
                    "shift": shift,
                    "midi_note": target_note,
                    "dynamic": row.get("dynamic") or "unknown",
                    "split": str(row.get("split") or "train"),
                }
            )
    return jobs


def _fade_out(audio: np.ndarray, sample_rate: int, milliseconds: float) -> np.ndarray:
    frames = min(int(round(sample_rate * milliseconds / 1000.0)), audio.size // 4)
    if frames <= 1:
        return audio
    faded = audio.copy()
    faded[-frames:] *= np.linspace(1.0, 0.0, frames, dtype=np.float32)
    return faded


def _schema(records: list[Mapping[str, Any]]) -> ConditioningSchema:
    articulations = sorted({str(row.get("articulation") or "unknown") for row in records})
    dynamics = sorted({str(row.get("dynamic") or "unknown") for row in records})
    pitches = [int(row["midi_note"]) for row in records if row.get("midi_note") is not None]
    return ConditioningSchema(
        continuous=(
            ContinuousParam(
                "midi_note",
                float(min(pitches or [0])),
                float(max(pitches or [127])),
                69.0,
                "MIDI note",
                "Reference pitch label; the renderer receives the corresponding MIDI note.",
            ),
            ContinuousParam(
                "velocity", 0.0, 1.0, 0.7, "", "Render velocity inferred from the source dynamic."
            ),
        ),
        categorical=(
            CategoricalParam(
                "articulation",
                tuple(articulations),
                articulations[0] if articulations else "unknown",
                embedding_dim=4,
                description="VSCO source articulation.",
            ),
            CategoricalParam(
                "dynamic",
                tuple(dynamics),
                dynamics[0] if dynamics else "unknown",
                embedding_dim=3,
                description="VSCO source dynamic/velocity layer.",
            ),
        ),
    )


def build_vsco_fbmx_pairs(
    dataset_root: str | Path,
    *,
    sfm: str | Path,
    renderer: str | Path | None = None,
    force: bool = False,
    max_seconds: float | None = MAX_PAIR_SECONDS,
    include_identity: bool = True,
    include_transposed: bool = True,
) -> dict[str, Any]:
    """Render and manifest per-channel voicebank -> reference pairs.

    Two kinds of pair, and they answer different questions:

    ``identity``
        The note played from its own recorded entry. After level matching the
        residual here is ~0 (median ESR 1.8e-05 measured on this source), which
        is the *point*: it is what teaches the model to leave a correctly
        resolved note alone instead of colouring everything it touches.

    ``transposed``
        The note played from a neighbouring entry, shifted by the resampler,
        against the true recording of the target pitch. This is the only place
        in this dataset where a learned correction has real work to do, and it
        is exactly the situation the runtime is in for every pitch the bank does
        not contain.

    Training on the identity pairs alone yields a model with nothing to learn;
    training on the transposed pairs alone yields one that "corrects" notes that
    were already right. Both are needed.
    """

    root = Path(dataset_root).resolve()
    source_manifest = root / "manifests" / "violin-all.jsonl"
    if not source_manifest.exists():
        raise FileNotFoundError(f"missing {source_manifest}; run the VSCO prepare step first")
    records = [
        row
        for row in read_jsonl(source_manifest)
        if str(row.get("instrument", "")).casefold() == "violin"
    ]
    if not records:
        raise ValueError(f"no violin records in {source_manifest}")

    workspace_root = Path(__file__).resolve().parents[3]
    renderer_path = Path(renderer).resolve() if renderer else _default_renderer(workspace_root)
    sfm_path = Path(sfm).resolve()
    if not sfm_path.exists():
        raise FileNotFoundError(f"missing voicebank SFM: {sfm_path}")
    entry_ids = _voicebank_entry_ids(sfm_path)

    dry_dir = root / "fbmx" / "dry"
    wet_dir = root / "fbmx" / "wet"
    dry_dir.mkdir(parents=True, exist_ok=True)
    wet_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "manifests" / "violin-fbmx.json"
    manifest_dir = manifest_path.parent

    cap = float(max_seconds) if max_seconds is not None else MAX_PAIR_SECONDS
    render_dir = root / "fbmx" / "render"
    render_dir.mkdir(parents=True, exist_ok=True)
    if force or any(not (render_dir / f"{row['id']}.wav").exists() for row in records):
        _render_batch(renderer_path, sfm_path, root, source_manifest, render_dir, cap)

    entries: list[ManifestEntry] = []
    rejected: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []

    def emit(
        pair_id: str,
        render_path: Path,
        reference_path: Path,
        params: dict[str, Any],
        split: str,
        kind: str,
        note: str,
        min_esr: float = 0.0,
    ) -> None:
        """Verify one render/reference pair per channel and write what survives."""

        try:
            reference, reference_rate = read_audio(reference_path)
            rendered, render_rate = read_audio(render_path)
        except OSError as error:
            rejected.append({"key": pair_id, "kind": kind, "reason": str(error)})
            return
        if reference_rate != SAMPLE_RATE or render_rate != SAMPLE_RATE:
            rejected.append({"key": pair_id, "kind": kind, "reason": "unexpected sample rate"})
            return

        frames = min(reference.shape[1], rendered.shape[1], int(round(cap * SAMPLE_RATE)))
        if frames < SAMPLE_RATE // 10:
            rejected.append({"key": pair_id, "kind": kind, "reason": f"only {frames} frames"})
            return

        for channel in range(min(reference.shape[0], rendered.shape[0])):
            key = f"{pair_id}#{channel}"
            dry = np.ascontiguousarray(rendered[channel, :frames], dtype=np.float32)
            wet = np.ascontiguousarray(reference[channel, :frames], dtype=np.float32)
            report = align_and_match(dry, wet)
            report["kind"] = kind

            if abs(report["lag_samples"]) > MAX_ALIGNMENT_SAMPLES:
                rejected.append(
                    {"key": key, "kind": kind, "reason": f"lag {report['lag_samples']} samples"}
                )
                continue
            if report["gain"] <= 0.0:
                # A non-positive best-fit gain means the render is inverted
                # relative to the reference, or uncorrelated with it. Either way
                # the difference is not a correction the model should learn.
                rejected.append(
                    {"key": key, "kind": kind, "reason": f"fit gain {report['gain']:.3f}"}
                )
                continue
            if not math.isfinite(report["residual_esr"]) or report["residual_esr"] > MAX_RESIDUAL_ESR:
                rejected.append(
                    {"key": key, "kind": kind, "reason": f"ESR {report['residual_esr']:.3f}"}
                )
                continue
            if report["residual_esr"] < min_esr:
                rejected.append(
                    {"key": key, "kind": kind, "reason": f"ESR {report['residual_esr']:.5f} too small"}
                )
                continue

            # Bring the TARGET to the dry's level rather than the other way
            # round. The dry is what the runtime will hand the model, at
            # whatever level the velocity produced; teaching the model to
            # preserve that level and correct only the timbre is what makes the
            # correction generalise across dynamics.
            dry_out = _fade_out(dry, SAMPLE_RATE, CUT_FADE_MS)
            wet_out = _fade_out(wet / report["gain"], SAMPLE_RATE, CUT_FADE_MS)
            dry_path = dry_dir / f"{pair_id}-ch{channel}.wav"
            wet_path = wet_dir / f"{pair_id}-ch{channel}.wav"
            write_wav(dry_path, dry_out[None, :], SAMPLE_RATE)
            write_wav(wet_path, wet_out[None, :], SAMPLE_RATE)

            diagnostics.append({"key": key, **report})
            entries.append(
                ManifestEntry(
                    key=key,
                    dry=Path(os.path.relpath(dry_path, manifest_dir)).as_posix(),
                    wet=Path(os.path.relpath(wet_path, manifest_dir)).as_posix(),
                    split=split,
                    params=params,
                    notes=note,
                )
            )

    def emit_interpolation_pair(job: dict[str, Any], shifted_dir: Path) -> None:
        """Pair the engine's transposed read against the reference interpolator."""

        pair_id = f"{job['source_id']}__shift{job['shift']:+d}"
        try:
            shifted, shifted_rate = read_audio(shifted_dir / job["out"])
            unity, unity_rate = read_audio(render_dir / f"{job['source_id']}.wav")
        except OSError as error:
            rejected.append({"key": pair_id, "kind": "interp", "reason": str(error)})
            return
        if shifted_rate != SAMPLE_RATE or unity_rate != SAMPLE_RATE:
            rejected.append({"key": pair_id, "kind": "interp", "reason": "unexpected sample rate"})
            return

        ratio = 2.0 ** (job["shift"] / 12.0)
        # Skip the engine's own attack ramp: it is applied in real time and so
        # is not itself resampled, which would otherwise show up as an envelope
        # difference the model would try to learn.
        skip = int(SAMPLE_RATE * 0.03)
        for channel in range(min(shifted.shape[0], unity.shape[0])):
            key = f"{pair_id}#{channel}"
            reference = resample_by_ratio(unity[channel], ratio)
            frames = min(
                shifted.shape[1], reference.size, int(round(INTERP_PAIR_SECONDS * SAMPLE_RATE))
            )
            if frames - skip < SAMPLE_RATE // 10:
                rejected.append({"key": key, "kind": "interp", "reason": "too short"})
                continue
            dry = np.ascontiguousarray(shifted[channel, skip:frames], dtype=np.float32)
            wet = np.ascontiguousarray(reference[skip:frames], dtype=np.float32)

            report = align_and_match(dry, wet)
            report["kind"] = "interp"
            report["shift"] = job["shift"]
            if abs(report["lag_samples"]) > MAX_ALIGNMENT_SAMPLES:
                # Aligned by construction; a nonzero lag means the read
                # positions diverged and the pair is not what it claims to be.
                rejected.append(
                    {"key": key, "kind": "interp", "reason": f"lag {report['lag_samples']}"}
                )
                continue
            if report["residual_esr"] < MIN_TRANSPOSED_ESR:
                rejected.append(
                    {
                        "key": key,
                        "kind": "interp",
                        "reason": f"ESR {report['residual_esr']:.6f} already transparent",
                    }
                )
                continue
            if not math.isfinite(report["residual_esr"]) or report["residual_esr"] > MAX_RESIDUAL_ESR:
                rejected.append(
                    {"key": key, "kind": "interp", "reason": f"ESR {report['residual_esr']:.3f}"}
                )
                continue

            dry_out = _fade_out(dry, SAMPLE_RATE, CUT_FADE_MS)
            wet_out = _fade_out(wet, SAMPLE_RATE, CUT_FADE_MS)
            dry_path = dry_dir / f"{pair_id}-ch{channel}.wav"
            wet_path = wet_dir / f"{pair_id}-ch{channel}.wav"
            write_wav(dry_path, dry_out[None, :], SAMPLE_RATE)
            write_wav(wet_path, wet_out[None, :], SAMPLE_RATE)

            diagnostics.append({"key": key, **report})
            entries.append(
                ManifestEntry(
                    key=key,
                    dry=Path(os.path.relpath(dry_path, manifest_dir)).as_posix(),
                    wet=Path(os.path.relpath(wet_path, manifest_dir)).as_posix(),
                    split=job["split"],
                    params={
                        "midi_note": int(job["midi_note"]),
                        "velocity": float(job["velocity"]),
                        "articulation": job["articulation"],
                        "dynamic": job["dynamic"],
                    },
                    notes=(
                        f"The engine's 4-point Hermite read of one recording transposed "
                        f"{job['shift']:+d} semitones, against a 64-tap float64 sinc read of the "
                        "same recording at the same positions. The difference is the realtime "
                        "interpolator's error and nothing else."
                    ),
                )
            )

    if include_identity:
        for row in sorted(records, key=lambda item: str(item["id"])):
            key = str(row["id"])
            if key not in entry_ids:
                rejected.append({"key": key, "kind": "identity", "reason": "no voicebank entry"})
                continue
            emit(
                pair_id=key,
                render_path=render_dir / f"{key}.wav",
                reference_path=root / str(row["processed_path"]),
                params={
                    "midi_note": int(row.get("midi_note") or 60),
                    "velocity": velocity_for_dynamic(row.get("dynamic")),
                    "articulation": row.get("articulation") or "unknown",
                    "dynamic": row.get("dynamic") or "unknown",
                },
                split=str(row.get("split") or "train"),
                kind="identity",
                note=(
                    "The note played from its own recorded entry, one channel, exactly as the "
                    "runtime renders it. Target is the same channel of the source recording at "
                    "the input's level. Teaches the model to leave a correct note alone."
                ),
            )

    if include_transposed:
        entry_meta = _entry_metadata(sfm_path)
        jobs = _interpolation_jobs(records, entry_meta)
        shifted_dir = root / "fbmx" / "shifted"
        shifted_dir.mkdir(parents=True, exist_ok=True)
        if jobs and (force or any(not (shifted_dir / job["out"]).exists() for job in jobs)):
            _render_transposed(
                renderer_path,
                sfm_path,
                jobs,
                root / "fbmx" / "transposition-jobs.json",
                shifted_dir,
                INTERP_PAIR_SECONDS,
            )
        for job in jobs:
            emit_interpolation_pair(job, shifted_dir)

    if not entries:
        raise ValueError(f"every pair was rejected: {rejected[:5]}")

    def stats(kind: str | None) -> dict[str, float]:
        rows = [d for d in diagnostics if kind is None or d["kind"] == kind]
        if not rows:
            return {"pairs": 0}
        esr = np.array([d["residual_esr"] for d in rows])
        gains = np.array([d["gain"] for d in rows])
        return {
            "pairs": len(rows),
            "residual_esr_mean": float(esr.mean()),
            "residual_esr_median": float(np.median(esr)),
            "residual_esr_p90": float(np.percentile(esr, 90)),
            "level_gain_median": float(np.median(gains)),
            "level_gain_p10": float(np.percentile(gains, 10)),
            "level_gain_p90": float(np.percentile(gains, 90)),
        }

    summary = {
        "pairs": len(entries),
        "rejected": len(rejected),
        "all": stats(None),
        "identity": stats("identity"),
        "interp": stats("interp"),
    }

    source_summary = root / "metadata" / "scan-report.json"
    source_hash = ""
    if source_summary.exists():
        source_hash = str(
            json.loads(source_summary.read_text(encoding="utf-8")).get("source_dataset_hash", "")
        )
    info = DatasetInfo(
        name="vsco2-ce-solo-violin-fbmx",
        source="VSCO-2-CE + embedded Solfage voicebank (VoicebankOnly render)",
        source_type="hybrid",
        license="CC0-1.0",
        version=PAIR_VERSION,
        sample_rate=SAMPLE_RATE,
        license_url="https://creativecommons.org/publicdomain/zero/1.0/",
        attribution="VSCO 2 Community Edition; CC0-1.0 source data.",
        checksum=source_hash,
        redistributable=True,
        notes=(
            "Per-channel voicebank->reference pairs, level matched and alignment verified. "
            "The SFM owns the source material at runtime."
        ),
        extra={
            "source_manifest": "manifests/violin-all.jsonl",
            "sfm": sfm_path.name,
            "renderer": "solfage-model render-batch (voicebank)",
            "max_pair_seconds": cap,
            "pair_diagnostics": summary,
            "rejected_pairs": rejected[:64],
        },
    )
    manifest = DatasetManifest(
        info=info,
        entries=entries,
        schema=_schema(records),
        root=manifest_dir,
    )
    manifest.fill_checksums()
    manifest.validate(check_files=True, check_checksums=True)
    manifest.save(manifest_path)
    (root / "metadata" / "fbmx-pair-diagnostics.json").write_text(
        json.dumps({"summary": summary, "pairs": diagnostics, "rejected": rejected}, indent=2),
        encoding="utf-8",
    )
    return {
        "manifest": str(manifest_path),
        "renderer": str(renderer_path),
        "entries": len(entries),
        "rejected": len(rejected),
        "diagnostics": summary,
        "splits": {
            split: sum(entry.split == split for entry in entries)
            for split in ("train", "valid", "test")
        },
    }
