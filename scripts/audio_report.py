"""Objective comparison of a rendered audio file against a reference.

Answers the question the ear is asking when a render sounds "watery": *which*
part of the signal is wrong. Waveform error alone cannot tell a phase problem
from a level problem, so every metric here is reported next to the one that
disambiguates it:

* ``esr`` / ``mae`` / ``rmse``   -- sample-accurate error. High ESR with low
  spectral-magnitude error means the magnitudes are right and the **phase** is
  not: the classic watery/smeared signature.
* ``spectral_convergence`` and ``log_mag_mae`` -- magnitude-only error, so they
  are blind to phase by construction and act as the control for the above.
* ``hf_ratio`` / ``spectral_centroid`` -- where the energy sits. A model that
  averages away bow noise and high harmonics moves both down; that is the
  "artificially clean violin" failure.
* ``transient_*`` -- onset timing and attack slope, which long sustained frames
  dominate in training and therefore lose first.
* ``dc``, ``peak``, ``nan``/``inf`` -- health checks that must be exactly right
  before any of the above is worth reading.

Self-comparison (no reference) is supported and reports the health/spectral
half, so a single render can still be inspected.

Usage:
    python scripts/audio_report.py TARGET.wav [CANDIDATE.wav ...] [--json OUT]
    python scripts/audio_report.py --single FILE.wav [FILE.wav ...]

Depends only on numpy plus the repository's own WAV reader, so it runs anywhere
the training stack runs and adds no plotting/GUI dependency.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    """Mono float32 in [-1, 1] plus the sample rate.

    Multi-channel input is averaged; every metric here is about spectral and
    temporal structure, which the downmix preserves, and the renders under test
    are mono or dual-mono anyway.

    ``soundfile`` is preferred because the dataset's rendered pairs are
    IEEE-float WAV (format tag 3), which the standard library's ``wave`` module
    refuses to open. The integer path below stays as the dependency-free
    fallback for the PCM16 renders the Rust tools emit.
    """
    try:
        import soundfile  # optional; see pyproject's `audio` extra

        data, rate = soundfile.read(str(path), dtype="float32", always_2d=True)
        return np.ascontiguousarray(data.mean(axis=1)), int(rate)
    except ImportError:
        pass

    with wave.open(str(path), "rb") as handle:
        channels = handle.getnchannels()
        width = handle.getsampwidth()
        rate = handle.getframerate()
        raw = handle.readframes(handle.getnframes())
    if width == 2:
        data = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif width == 4:
        data = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    elif width == 3:
        packed = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
        merged = packed[:, 0] | (packed[:, 1] << 8) | (packed[:, 2] << 16)
        merged = np.where(merged & 0x800000, merged - 0x1000000, merged)
        data = merged.astype(np.float32) / 8388608.0
    elif width == 1:
        data = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    else:
        raise ValueError(f"{path}: unsupported sample width {width}")
    if channels > 1:
        data = data.reshape(-1, channels).mean(axis=1)
    return np.ascontiguousarray(data), rate


def stft_mag(x: np.ndarray, n_fft: int, hop: int) -> np.ndarray:
    if x.size < n_fft:
        x = np.pad(x, (0, n_fft - x.size))
    window = np.hanning(n_fft).astype(np.float32)
    frames = 1 + (x.size - n_fft) // hop
    if frames < 1:
        return np.zeros((1, n_fft // 2 + 1), dtype=np.float32)
    strided = np.lib.stride_tricks.as_strided(
        x, shape=(frames, n_fft), strides=(x.strides[0] * hop, x.strides[0])
    )
    return np.abs(np.fft.rfft(strided * window, axis=-1)).astype(np.float32)


def spectral_metrics(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    """Magnitude-domain error at three resolutions.

    Deliberately magnitude-only: paired with the time-domain ESR it separates
    "the spectrum is wrong" from "the spectrum is right but the phase is not".
    """
    sc_terms, log_terms = [], []
    for n_fft, hop in ((512, 128), (1024, 256), (2048, 512)):
        ma, mb = stft_mag(a, n_fft, hop), stft_mag(b, n_fft, hop)
        frames = min(ma.shape[0], mb.shape[0])
        ma, mb = ma[:frames], mb[:frames]
        denom = float(np.linalg.norm(ma)) or 1e-12
        sc_terms.append(float(np.linalg.norm(ma - mb) / denom))
        log_terms.append(float(np.mean(np.abs(np.log(ma + 1e-7) - np.log(mb + 1e-7)))))
    return {
        "spectral_convergence": float(np.mean(sc_terms)),
        "log_mag_mae": float(np.mean(log_terms)),
    }


def band_energy(x: np.ndarray, rate: int) -> dict[str, float]:
    mag = stft_mag(x, 2048, 512)
    freqs = np.fft.rfftfreq(2048, 1.0 / rate)
    power = mag.astype(np.float64) ** 2
    total = float(power.sum()) or 1e-20
    centroid = float((power * freqs).sum() / total) if total > 0 else 0.0
    hf = float(power[:, freqs >= 4000.0].sum() / total)
    vhf = float(power[:, freqs >= 8000.0].sum() / total)
    return {"spectral_centroid_hz": centroid, "hf_ratio_4k": hf, "hf_ratio_8k": vhf}


def envelope(x: np.ndarray, rate: int, ms: float = 5.0) -> np.ndarray:
    win = max(1, int(rate * ms / 1000.0))
    padded = np.pad(np.abs(x), (win // 2, win // 2), mode="edge")
    kernel = np.ones(win, dtype=np.float64) / win
    return np.convolve(padded, kernel, mode="valid")[: x.size]


def transient_metrics(a: np.ndarray, b: np.ndarray, rate: int) -> dict[str, float]:
    """Onset timing and attack slope agreement.

    The onset is taken as the first crossing of 20 % of the envelope peak. The
    slope is the maximum positive envelope derivative, which is what an ear
    reads as "attack" and what an averaging loss flattens first.
    """
    ea, eb = envelope(a, rate), envelope(b, rate)
    out: dict[str, float] = {}
    for label, env in (("target", ea), ("candidate", eb)):
        peak = float(env.max()) if env.size else 0.0
        idx = int(np.argmax(env >= peak * 0.2)) if peak > 0 else 0
        out[f"onset_ms_{label}"] = idx * 1000.0 / rate
        out[f"attack_slope_{label}"] = float(np.max(np.diff(env)) * rate) if env.size > 1 else 0.0
    out["onset_error_ms"] = abs(out["onset_ms_candidate"] - out["onset_ms_target"])
    denom = abs(out["attack_slope_target"]) or 1e-12
    out["attack_slope_error"] = abs(out["attack_slope_candidate"] - out["attack_slope_target"]) / denom
    return out


def f0_track(
    x: np.ndarray,
    rate: int,
    frame_ms: float = 40.0,
    hop_ms: float = 10.0,
    fmin: float = 55.0,
    fmax: float = 2200.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-frame fundamental via normalised autocorrelation (NSDF/McLeod).

    Written here rather than pulled from a library because the pitch acceptance
    test must run wherever the training stack runs, and because the quantity
    under test — "did the rendered pitch actually move to 466 Hz" — needs a
    tracker whose failure modes are visible. Octave errors are suppressed by
    taking the *first* peak above a clarity threshold rather than the global
    maximum, which is the standard fix and matters because a violin's second
    harmonic is often stronger than its fundamental.

    Returns ``(times_seconds, hz)`` with ``nan`` where no confident pitch was
    found, so unvoiced frames cannot silently drag an average.
    """
    frame = max(64, int(rate * frame_ms / 1000.0))
    hop = max(1, int(rate * hop_ms / 1000.0))
    if x.size < frame:
        return np.zeros(0), np.zeros(0)
    min_lag = max(2, int(rate / fmax))
    max_lag = min(frame - 1, int(rate / fmin))
    if max_lag <= min_lag:
        return np.zeros(0), np.zeros(0)

    times, freqs = [], []
    for start in range(0, x.size - frame + 1, hop):
        seg = x[start : start + frame].astype(np.float64)
        seg = seg - seg.mean()
        power = float(np.dot(seg, seg))
        times.append((start + frame / 2) / rate)
        if power < 1e-12:
            freqs.append(np.nan)
            continue
        size = 1 << int(math.ceil(math.log2(2 * frame)))
        spec = np.fft.rfft(seg, size)
        acf = np.fft.irfft(spec * np.conj(spec), size)[: max_lag + 1]
        # NSDF denominator: energy of the two overlapping halves at each lag.
        cumsq = np.concatenate([[0.0], np.cumsum(seg**2)])
        head = cumsq[frame - np.arange(max_lag + 1)] - cumsq[0]
        tail = cumsq[frame] - cumsq[np.arange(max_lag + 1)]
        nsdf = 2.0 * acf / np.maximum(head + tail, 1e-20)
        nsdf[:min_lag] = 0.0
        # First local maximum clearing 0.8 x the global max avoids the
        # octave-below error a plain argmax makes on strong second harmonics.
        peak = float(nsdf[min_lag:].max())
        if peak < 0.3:
            freqs.append(np.nan)
            continue
        threshold = 0.8 * peak
        lag = int(np.argmax(nsdf[min_lag:] >= threshold)) + min_lag
        while lag + 1 <= max_lag and nsdf[lag + 1] > nsdf[lag]:
            lag += 1
        if lag <= min_lag or lag >= max_lag:
            freqs.append(np.nan)
            continue
        # Parabolic interpolation on the NSDF peak for sub-sample lag accuracy;
        # without it the quantisation error at 440 Hz alone is several cents.
        y0, y1, y2 = nsdf[lag - 1], nsdf[lag], nsdf[lag + 1]
        denom = y0 - 2.0 * y1 + y2
        shift = 0.5 * (y0 - y2) / denom if abs(denom) > 1e-20 else 0.0
        freqs.append(rate / (lag + shift))
    return np.asarray(times), np.asarray(freqs)


def f0_summary(x: np.ndarray, rate: int) -> dict[str, float]:
    times, hz = f0_track(x, rate)
    voiced = hz[np.isfinite(hz)]
    if voiced.size == 0:
        return {"f0_median_hz": float("nan"), "f0_voiced_frames": 0}
    return {
        "f0_median_hz": float(np.median(voiced)),
        "f0_mean_hz": float(np.mean(voiced)),
        "f0_p10_hz": float(np.percentile(voiced, 10)),
        "f0_p90_hz": float(np.percentile(voiced, 90)),
        "f0_voiced_frames": int(voiced.size),
        "f0_total_frames": int(hz.size),
    }


def stability(x: np.ndarray, rate: int) -> dict[str, float]:
    """Metrics that put a number on "watery".

    A held note on a real instrument has a steady fundamental and harmonics
    whose levels drift slowly. The three failure modes this catches:

    * ``f0_jitter_cents`` -- frame-to-frame pitch wander. Reading-cursor
      quantisation and precision loss show up here first, as fast jitter rather
      than as a slow drift.
    * ``harmonic_am_depth`` -- how much the harmonic levels are modulated across
      the note, measured as the median relative deviation of each of the first
      six harmonics' envelopes. Summing two decorrelated takes of the same note
      produces exactly this: a comb whose notches sweep, heard as chorus or
      "phasey".
    * ``click_count`` -- samples whose second difference is a large multiple of
      the local norm. A lost loop point, a hard voice steal, or an unramped gain
      change all land here.

    Reported together because each alone is ambiguous: vibrato also moves f0,
    and a real tremolo also modulates harmonics. Only the combination, compared
    against the same phrase rendered before a change, identifies a defect.
    """
    out: dict[str, float] = {}
    _, hz = f0_track(x, rate, frame_ms=40.0, hop_ms=5.0)
    voiced = hz[np.isfinite(hz)]
    if voiced.size > 4:
        cents = 1200.0 * np.log2(voiced / np.median(voiced))
        # Differencing removes vibrato and drift, leaving frame-to-frame jitter.
        out["f0_jitter_cents"] = float(np.std(np.diff(cents)))
        out["f0_drift_cents"] = float(
            1200.0 * np.log2(np.median(voiced[-max(4, voiced.size // 8) :]) / np.median(voiced[: max(4, voiced.size // 8)]))
        )
    else:
        out["f0_jitter_cents"] = float("nan")
        out["f0_drift_cents"] = float("nan")

    mag = stft_mag(x, 2048, 256)
    freqs = np.fft.rfftfreq(2048, 1.0 / rate)
    f0 = float(np.median(voiced)) if voiced.size else 0.0
    depths = []
    if f0 > 20.0:
        for harmonic in range(1, 7):
            target = f0 * harmonic
            if target >= rate / 2:
                break
            bin_index = int(np.argmin(np.abs(freqs - target)))
            track = mag[:, max(0, bin_index - 1) : bin_index + 2].max(axis=1)
            track = track[track > track.max() * 0.05] if track.size else track
            if track.size > 8:
                depths.append(float(np.std(track) / (np.mean(track) + 1e-12)))
    out["harmonic_am_depth"] = float(np.median(depths)) if depths else float("nan")

    if x.size > 3:
        d2 = np.abs(np.diff(x, n=2))
        scale = float(np.median(d2[d2 > 0])) if np.any(d2 > 0) else 0.0
        out["click_count"] = int(np.sum(d2 > max(scale * 60.0, 1e-4)))
    else:
        out["click_count"] = 0
    return out


def health(x: np.ndarray) -> dict[str, float]:
    finite = np.isfinite(x)
    return {
        "rms": float(np.sqrt(np.mean(np.square(x[finite]))) if finite.any() else 0.0),
        "peak": float(np.max(np.abs(x[finite])) if finite.any() else 0.0),
        "dc": float(np.mean(x[finite]) if finite.any() else 0.0),
        "nan_count": int(np.isnan(x).sum()),
        "inf_count": int(np.isinf(x).sum()),
        "clipped_samples": int(np.sum(np.abs(x[finite]) >= 0.999)) if finite.any() else 0,
        "samples": int(x.size),
    }


def best_lag(a: np.ndarray, b: np.ndarray, max_lag: int) -> int:
    """Integer sample lag maximising cross-correlation of ``b`` against ``a``.

    Reported rather than silently applied: a nonzero lag between a render and
    its reference means the *training pairs* were misaligned, which is the
    single most destructive preprocessing bug for residual learning.
    """
    n = min(a.size, b.size)
    if n < 16:
        return 0
    a, b = a[:n] - a[:n].mean(), b[:n] - b[:n].mean()
    size = 1 << int(math.ceil(math.log2(2 * n)))
    corr = np.fft.irfft(np.fft.rfft(a, size) * np.conj(np.fft.rfft(b, size)), size)
    lags = np.concatenate([np.arange(0, max_lag + 1), np.arange(size - max_lag, size)])
    values = corr[lags]
    peak = int(lags[int(np.argmax(values))])
    return peak - size if peak > size // 2 else peak


def compare(target: np.ndarray, cand: np.ndarray, rate: int) -> dict[str, object]:
    n = min(target.size, cand.size)
    t, c = target[:n], cand[:n]
    err = t - c
    signal_energy = float(np.sum(t.astype(np.float64) ** 2)) or 1e-20
    metrics: dict[str, object] = {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err.astype(np.float64) ** 2))),
        "esr": float(np.sum(err.astype(np.float64) ** 2) / signal_energy),
        "peak_error": float(np.max(np.abs(err))) if n else 0.0,
        "dc_error": float(np.mean(err)),
        "lag_samples": best_lag(t, c, max_lag=min(4800, n // 2)),
    }
    metrics["esr_db"] = 10.0 * math.log10(max(float(metrics["esr"]), 1e-20))
    metrics.update(spectral_metrics(t, c))
    metrics.update(transient_metrics(t, c, rate))
    ta, ca = band_energy(t, rate), band_energy(c, rate)
    metrics["hf_ratio_4k_target"] = ta["hf_ratio_4k"]
    metrics["hf_ratio_4k_candidate"] = ca["hf_ratio_4k"]
    metrics["hf_loss_ratio"] = ca["hf_ratio_4k"] / (ta["hf_ratio_4k"] or 1e-12)
    metrics["centroid_target_hz"] = ta["spectral_centroid_hz"]
    metrics["centroid_candidate_hz"] = ca["spectral_centroid_hz"]
    return metrics


def describe(path: Path) -> dict[str, object]:
    x, rate = read_wav(path)
    out: dict[str, object] = {"path": str(path), "sample_rate": rate}
    out.update(health(x))
    out.update(band_energy(x, rate))
    out.update(f0_summary(x, rate))
    out.update(stability(x, rate))
    out["duration_s"] = x.size / rate if rate else 0.0
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("files", nargs="+", type=Path)
    parser.add_argument("--single", action="store_true", help="describe each file; do not compare")
    parser.add_argument(
        "--window",
        default=None,
        metavar="FROM:TO",
        help="restrict analysis to these seconds, e.g. 26:38 for a held note",
    )
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    report: dict[str, object] = {"format": "fbmx-audio-report-v1"}
    if args.window:
        lo, hi = (float(part) for part in args.window.split(":"))
        report["window_s"] = [lo, hi]
        original_read = read_wav

        def read_wav_windowed(path: Path) -> tuple[np.ndarray, int]:
            data, rate = original_read(path)
            return data[int(lo * rate) : int(hi * rate)], rate

        globals()["read_wav"] = read_wav_windowed

    if args.single or len(args.files) == 1:
        report["files"] = [describe(path) for path in args.files]
    else:
        target_path, *candidates = args.files
        target, rate = read_wav(target_path)
        report["target"] = describe(target_path)
        entries = []
        for path in candidates:
            cand, cand_rate = read_wav(path)
            entry: dict[str, object] = {"path": str(path), "self": describe(path)}
            if cand_rate != rate:
                entry["error"] = f"sample rate {cand_rate} != target {rate}"
            else:
                entry.update(compare(target, cand, rate))
            entries.append(entry)
        report["candidates"] = entries

    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
