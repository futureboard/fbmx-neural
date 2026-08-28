"""Render the same score twice — accent off, accent on — and measure what moved.

Section 45 asks for an A/B through the *same* voicebank with nothing else
changed, and section 46 for an ablation that proves each accent component
reaches its own dimension rather than all four secretly controlling gain. This
does both, and it measures rather than asserts:

* per-note, what the Performer did differently (placement, length, velocity);
* per-render, how much of the difference is simply loudness.

That last number is the one that decides whether the feature is real. If the
whole difference between A and B is a level change, the accent system is a gain
knob with extra steps, and the report has to say so.

    python scripts/accent_ab.py \\
        --model <SoloViolin.sfm> --score <Sco.mid> --track 1 \\
        --checkpoint <performer/best.pt> --accent-rule <rule_coefficients.json> \\
        --out <artifacts/ab>
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

import _bootstrap  # noqa: F401

from accent.features import ACCENT_TARGETS
from accent.predict import ShippedAnalyzer, contexts_from_score, neutral_accents
from datasets.urmp.annotations import read_score_notes, score_tempo_bpm, score_time_signature
from performer.generate import generate, load_model


def _render(tool: Path, model: Path, document: Path, wav: Path) -> None:
    result = subprocess.run(
        [str(tool), "render-perf", str(model), str(document), str(wav), "voicebank"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise SystemExit(f"render failed: {result.stderr.strip() or result.stdout.strip()}")


def _read_mono(path: Path) -> tuple[np.ndarray, int]:
    audio, rate = sf.read(str(path), dtype="float64", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio, rate


def _level_matched(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, float]:
    """`b` scaled to `a`'s RMS, and the gain that took to do.

    The point of the exercise: subtract the loudness difference and see what is
    left. If almost nothing is, the accent system is a volume control.
    """

    rms_a = float(np.sqrt(np.mean(a**2)))
    rms_b = float(np.sqrt(np.mean(b**2)))
    gain = rms_a / max(rms_b, 1e-12)
    return b * gain, gain


def _spectral_centroid(audio: np.ndarray, rate: int, frame: int = 2048) -> float:
    hop = frame // 2
    frames = max(1, 1 + (audio.size - frame) // hop)
    window = np.hanning(frame)
    frequencies = np.fft.rfftfreq(frame, 1.0 / rate)
    total = 0.0
    weighted = 0.0
    for index in range(frames):
        chunk = audio[index * hop : index * hop + frame]
        if chunk.size < frame:
            break
        power = np.abs(np.fft.rfft(chunk * window)) ** 2
        energy = float(power.sum())
        if energy <= 0.0:
            continue
        total += energy
        weighted += float((power * frequencies).sum())
    return weighted / total if total > 0.0 else 0.0


def compare_documents(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    """What the Performer did differently, note by note."""

    pairs = list(zip(a["notes"], b["notes"]))
    onset = np.asarray([n2["start"] - n1["start"] for n1, n2 in pairs])
    duration = np.asarray(
        [n2["duration"] / max(n1["duration"], 1e-9) for n1, n2 in pairs]
    )
    velocity = np.asarray(
        [float(n2.get("velocity") or 0) - float(n1.get("velocity") or 0) for n1, n2 in pairs]
    )
    return {
        "notes": len(pairs),
        "onset_shift_ms": {
            "mean_abs": float(np.abs(onset).mean() * 1000.0),
            "max_abs": float(np.abs(onset).max() * 1000.0),
            "p95_abs": float(np.percentile(np.abs(onset), 95) * 1000.0),
        },
        "duration_ratio": {
            "mean": float(duration.mean()),
            "min": float(duration.min()),
            "max": float(duration.max()),
        },
        "velocity_delta": {
            "mean_abs": float(np.abs(velocity).mean()),
            "max_abs": float(np.abs(velocity).max()),
        },
        "notes_moved": int((np.abs(onset) > 1e-4).sum()),
        "notes_relengthened": int((np.abs(duration - 1.0) > 1e-4).sum()),
        "notes_revoiced": int((np.abs(velocity) > 1e-4).sum()),
    }


def compare_audio(a_path: Path, b_path: Path) -> dict[str, Any]:
    a, rate = _read_mono(a_path)
    b, rate_b = _read_mono(b_path)
    if rate != rate_b:
        raise SystemExit(f"sample rates differ: {rate} and {rate_b}")
    length = min(a.size, b.size)
    a, b = a[:length], b[:length]

    matched, gain = _level_matched(a, b)
    difference = a - matched
    energy = float(np.sum(a**2))
    return {
        "seconds": length / rate,
        "rms_db_a": float(20.0 * np.log10(max(np.sqrt(np.mean(a**2)), 1e-12))),
        "rms_db_b": float(20.0 * np.log10(max(np.sqrt(np.mean(b**2)), 1e-12))),
        "loudness_difference_db": float(-20.0 * np.log10(max(gain, 1e-12))),
        # What is left once the two are the same loudness, as a fraction of the
        # signal's energy. A number near zero would mean the whole audible
        # difference was a gain change.
        "residual_after_level_match": float(
            np.sum(difference**2) / max(energy, 1e-12)
        ),
        "centroid_hz_a": _spectral_centroid(a, rate),
        "centroid_hz_b": _spectral_centroid(b, rate),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="SoloViolin.sfm")
    parser.add_argument("--score", required=True, help="score MIDI")
    parser.add_argument("--track", type=int, default=1)
    parser.add_argument("--checkpoint", required=True, help="Performer best.pt")
    parser.add_argument("--accent-rule", required=True)
    parser.add_argument("--accent-checkpoint")
    parser.add_argument("--tool", required=True, help="solfage-model executable")
    parser.add_argument("--out", required=True)
    parser.add_argument("--articulation", default="sustain_vibrato")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--notes", type=int, default=0, help="truncate to N notes")
    arguments = parser.parse_args()

    out = Path(arguments.out)
    out.mkdir(parents=True, exist_ok=True)
    tool = Path(arguments.tool)
    model_path = Path(arguments.model)

    score_notes = read_score_notes(arguments.score, arguments.track)
    if arguments.notes:
        score_notes = score_notes[: arguments.notes]
    if not score_notes:
        raise SystemExit(f"no notes on track {arguments.track}")
    tempo = score_tempo_bpm(arguments.score)
    signature = score_time_signature(arguments.score)

    performer = load_model(arguments.checkpoint)
    analyzer = ShippedAnalyzer.load(arguments.accent_rule, arguments.accent_checkpoint)
    contexts = contexts_from_score(
        score_notes, tempo_bpm=tempo, time_signature=signature
    )
    analysed = analyzer.analyze(contexts)

    legs: dict[str, list[dict[str, float]] | None] = {
        # A: the Performer with every accent neutral. Not "no accent input" —
        # that would be a different model — but the accent a clip nobody has
        # analysed carries.
        "a-accent-off": neutral_accents(len(score_notes)),
        "b-accent-on": analysed,
    }
    # One leg per component, each holding the other three neutral. This is the
    # ablation: if all four legs move the same thing, the components are not
    # separate.
    for component in ACCENT_TARGETS:
        neutral = neutral_accents(len(score_notes))
        legs[f"ablate-{component}"] = [
            {**flat, component: entry[component]}
            for entry, flat in zip(analysed, neutral)
        ]

    documents: dict[str, dict[str, Any]] = {}
    for name, accents in legs.items():
        document = generate(
            performer,
            score_notes,
            tempo_bpm=tempo,
            time_signature=signature,
            articulation=arguments.articulation,
            seed=arguments.seed,
            accent_override=accents,
        )
        path = out / f"{name}.json"
        path.write_text(json.dumps(document, indent=1) + "\n", encoding="utf-8")
        documents[name] = document
        _render(tool, model_path, path, out / f"{name}.wav")
        print(f"rendered {name}.wav")

    report: dict[str, Any] = {
        "score": str(arguments.score),
        "track": arguments.track,
        "notes": len(score_notes),
        "tempo_bpm": tempo,
        "time_signature": list(signature),
        "articulation": arguments.articulation,
        "accent_distribution": {
            name: {
                "mean": float(np.mean([entry[name] for entry in analysed])),
                "std": float(np.std([entry[name] for entry in analysed])),
                "min": float(np.min([entry[name] for entry in analysed])),
                "max": float(np.max([entry[name] for entry in analysed])),
            }
            for name in ACCENT_TARGETS
        },
        "ab": {
            "document": compare_documents(
                documents["a-accent-off"], documents["b-accent-on"]
            ),
            "audio": compare_audio(out / "a-accent-off.wav", out / "b-accent-on.wav"),
        },
        "ablation": {
            component: compare_documents(
                documents["a-accent-off"], documents[f"ablate-{component}"]
            )
            for component in ACCENT_TARGETS
        },
    }
    (out / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, default=float) + "\n", encoding="utf-8"
    )

    ab = report["ab"]
    print(f"\nA/B over {report['notes']} notes")
    print(f"  onsets moved      {ab['document']['notes_moved']} notes, "
          f"mean |shift| {ab['document']['onset_shift_ms']['mean_abs']:.1f} ms, "
          f"max {ab['document']['onset_shift_ms']['max_abs']:.1f} ms")
    print(f"  lengths changed   {ab['document']['notes_relengthened']} notes, "
          f"ratio {ab['document']['duration_ratio']['min']:.3f}..{ab['document']['duration_ratio']['max']:.3f}")
    print(f"  velocities        {ab['document']['notes_revoiced']} notes, "
          f"mean |delta| {ab['document']['velocity_delta']['mean_abs']:.4f}")
    print(f"  loudness          {ab['audio']['loudness_difference_db']:+.2f} dB")
    print(f"  residual after level match  {ab['audio']['residual_after_level_match']:.4f}")
    print(f"  centroid          {ab['audio']['centroid_hz_a']:.0f} -> {ab['audio']['centroid_hz_b']:.0f} Hz")
    print("\nablation: what each component alone changes")
    print(f"  {'component':12s} {'moved':>6s} {'relengthened':>13s} {'revoiced':>9s}")
    for component in ACCENT_TARGETS:
        row = report["ablation"][component]
        print(
            f"  {component:12s} {row['notes_moved']:6d} {row['notes_relengthened']:13d} "
            f"{row['notes_revoiced']:9d}"
        )


if __name__ == "__main__":
    main()
