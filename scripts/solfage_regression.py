"""Render the fixed Solfage validation phrase and report objective metrics.

One command, one directory of WAVs, one JSON of numbers — so a claim about the
instrument's sound is a diff between two runs of this script rather than an
opinion. It renders the same phrase in every configuration that matters:

    voicebank.wav        the base renderer alone, neural correction bypassed
    voicebank-fbmx.wav   what the DAW plays: VoicebankOnly + the embedded FBMX
    physical.wav         the bowed-string model alone
    hybrid.wav           voicebank + physical, no neural correction
    hybrid-fbmx.wav      the same with the embedded FBMX residual applied

`voicebank` and `voicebank-fbmx` differ in exactly one thing, so the reported
difference between them *is* the neural correction on the shipping path — which
is what attributes an artefact to the model rather than to the base renderer.
`hybrid`/`hybrid-fbmx` is the same comparison with the physical layer present,
kept because the model tools render that way.

and reports, per file: level and DC health, NaN/Inf counts, spectral centroid
and high-frequency ratio (a model that averages away bow noise moves both down),
pitch accuracy and drift on the held note, harmonic amplitude modulation and
frame-to-frame pitch jitter on the same window, and — against a chosen
reference render — ESR, MAE, multi-resolution spectral convergence, onset timing
and attack slope.

Usage:
    python scripts/solfage_regression.py --sfm PATH.sfm --out DIR [--label before]
    python scripts/solfage_regression.py --compare DIR_A DIR_B
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.audio_report import compare, describe, read_wav  # noqa: E402

#: The one window every stability number is measured on: the long held A4 in the
#: validation phrase, after its vibrato figure and before its wide bend. A
#: number measured anywhere else is not comparable between runs.
HELD_NOTE_WINDOW = (26.5, 31.5)

DEFAULT_PHRASE = (
    Path(__file__).resolve().parents[2]
    / "solfege-engine"
    / "crates"
    / "solfege-tools"
    / "validation"
    / "solo-violin-phrase.json"
)
DEFAULT_RENDERER = (
    Path(__file__).resolve().parents[2]
    / "solfege-engine"
    / "target"
    / "release"
    / "solfage-model.exe"
)


def render(renderer: Path, sfm: Path, phrase: Path, out: Path, mode: str, fbmx: bool) -> str:
    command = [str(renderer), "render-perf", str(sfm), str(phrase), str(out), mode]
    if not fbmx:
        command.append("--no-fbmx")
    done = subprocess.run(command, capture_output=True, text=True)
    if done.returncode != 0:
        raise RuntimeError((done.stderr or done.stdout).strip())
    return done.stdout.strip().splitlines()[0]


def held_note(path: Path) -> dict[str, object]:
    """Describe only the held-note window, where stability is meaningful."""
    from scripts import audio_report

    data, rate = read_wav(path)
    lo, hi = HELD_NOTE_WINDOW
    segment = data[int(lo * rate) : int(hi * rate)]
    out: dict[str, object] = {}
    out.update(audio_report.health(segment))
    out.update(audio_report.band_energy(segment, rate))
    out.update(audio_report.f0_summary(segment, rate))
    out.update(audio_report.stability(segment, rate))
    return out


def run(sfm: Path, out_dir: Path, phrase: Path, renderer: Path) -> dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    renders: dict[str, Path] = {}
    lines: dict[str, str] = {}
    for name, mode, fbmx in (
        ("voicebank", "voicebank", False),
        ("voicebank-fbmx", "voicebank", True),
        ("physical", "physical", False),
        ("hybrid", "hybrid", False),
        ("hybrid-fbmx", "hybrid", True),
    ):
        path = out_dir / f"{name}.wav"
        lines[name] = render(renderer, sfm, phrase, path, mode, fbmx)
        renders[name] = path

    report: dict[str, object] = {
        "format": "solfage-regression-v1",
        "sfm": str(sfm),
        "phrase": str(phrase),
        "held_note_window_s": list(HELD_NOTE_WINDOW),
        "render_lines": lines,
        "files": {},
        "held_note": {},
    }
    for name, path in renders.items():
        report["files"][name] = describe(path)
        report["held_note"][name] = held_note(path)

    # The neural correction against the same base, so any difference between
    # them is the model and nothing else.
    for label, dry_name, wet_name in (
        ("fbmx_vs_base", "voicebank", "voicebank-fbmx"),
        ("fbmx_vs_base_hybrid", "hybrid", "hybrid-fbmx"),
    ):
        base, _ = read_wav(renders[dry_name])
        candidate, rate = read_wav(renders[wet_name])
        report[label] = compare(base, candidate, rate)

    text = json.dumps(report, indent=2, sort_keys=True)
    (out_dir / "report.json").write_text(text, encoding="utf-8")
    return report


def summarise(report: dict[str, object], label: str) -> None:
    print(f"\n=== {label} ===")
    print(
        f"{'render':<14}{'rms':>10}{'peak':>10}{'dc':>12}{'centroid':>10}"
        f"{'hf4k':>9}{'NaN':>5}"
    )
    for name, stats in report["files"].items():
        print(
            f"{name:<14}{stats['rms']:>10.5f}{stats['peak']:>10.4f}{stats['dc']:>12.2e}"
            f"{stats['spectral_centroid_hz']:>10.0f}{stats['hf_ratio_4k']:>9.4f}"
            f"{stats['nan_count'] + stats['inf_count']:>5}"
        )
    lo, hi = report["held_note_window_s"]
    print(f"\nheld note {lo}-{hi}s (A4 = 440.000 Hz expected)")
    print(f"{'render':<14}{'f0 Hz':>10}{'cents':>8}{'jitter':>9}{'harm AM':>9}{'clicks':>8}")
    for name, stats in report["held_note"].items():
        f0 = stats.get("f0_median_hz", float("nan"))
        cents = 1200.0 * math.log2(f0 / 440.0) if f0 == f0 and f0 > 0 else float("nan")
        print(
            f"{name:<14}{f0:>10.2f}{cents:>8.2f}"
            f"{stats.get('f0_jitter_cents', float('nan')):>9.2f}"
            f"{stats.get('harmonic_am_depth', float('nan')):>9.3f}"
            f"{stats.get('click_count', 0):>8}"
        )
    for label, title in (
        ("fbmx_vs_base", "shipping path (voicebank)"),
        ("fbmx_vs_base_hybrid", "tools path (hybrid)      "),
    ):
        fbmx = report.get(label)
        if not fbmx:
            continue
        print(
            f"\nneural correction, {title}: esr={fbmx['esr']:.4f} "
            f"mae={fbmx['mae']:.6f} spec_conv={fbmx['spectral_convergence']:.4f} "
            f"hf_ratio_kept={fbmx['hf_loss_ratio']:.3f} dc_error={fbmx['dc_error']:.2e}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sfm", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--phrase", type=Path, default=DEFAULT_PHRASE)
    parser.add_argument("--renderer", type=Path, default=DEFAULT_RENDERER)
    parser.add_argument("--label", default="run")
    parser.add_argument(
        "--compare",
        nargs=2,
        type=Path,
        metavar=("BEFORE", "AFTER"),
        help="print two existing report directories side by side",
    )
    args = parser.parse_args()

    if args.compare:
        before = json.loads((args.compare[0] / "report.json").read_text(encoding="utf-8"))
        after = json.loads((args.compare[1] / "report.json").read_text(encoding="utf-8"))
        summarise(before, f"BEFORE  {args.compare[0]}")
        summarise(after, f"AFTER   {args.compare[1]}")
        return 0

    if not args.sfm or not args.out:
        parser.error("--sfm and --out are required unless --compare is given")
    report = run(args.sfm, args.out, args.phrase, args.renderer)
    summarise(report, args.label)
    print(f"\nwrote {args.out / 'report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
