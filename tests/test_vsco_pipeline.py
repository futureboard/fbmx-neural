"""Small deterministic tests for the VSCO ingestion pipeline."""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
import pytest

try:
    from neural.datasets.vsco.metadata import detect_pitch, midi_to_hz, parse_sample_metadata, stable_sample_id
    from neural.datasets.vsco.preprocess import (
        build_manifests,
        config_hash,
        resample_audio,
        trim_silence,
    )
    from neural.datasets.vsco.scan import scan_source
    from neural.datasets.vsco.split import assign_splits, group_key
except ModuleNotFoundError:  # pytest run from inside neural/
    from datasets.vsco.metadata import detect_pitch, midi_to_hz, parse_sample_metadata, stable_sample_id
    from datasets.vsco.preprocess import build_manifests, config_hash, resample_audio, trim_silence
    from datasets.vsco.scan import scan_source
    from datasets.vsco.split import assign_splits, group_key


def _write_pcm16(path: Path, samples: np.ndarray, rate: int) -> None:
    samples = np.asarray(samples, dtype=np.float64)
    if samples.ndim == 1:
        samples = samples[None, :]
    interleaved = np.clip(samples.T, -1.0, 1.0)
    raw = np.round(interleaved * 32767.0).astype("<i2").tobytes()
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(samples.shape[0])
        stream.setsampwidth(2)
        stream.setframerate(rate)
        stream.writeframes(raw)


def test_filename_metadata_and_pitch_conversion():
    parsed = parse_sample_metadata("Strings/Solo Violin/Pizz/LLVln_Pizz_A#4_p_RR2.wav")
    assert parsed.instrument == "violin"
    assert parsed.articulation == "pizzicato"
    assert parsed.midi_note == 70
    assert parsed.dynamic == "p"
    assert parsed.round_robin == 2
    assert midi_to_hz(69) == 440.0


def test_pitch_detector_tracks_sine():
    rate = 48000
    t = np.arange(rate) / rate
    estimate = detect_pitch(np.sin(2 * np.pi * 440.0 * t), rate, expected_midi=69)
    assert estimate.detected_hz is not None
    assert estimate.detected_hz == pytest.approx(440.0, abs=1.0)


def test_stable_ids_and_group_splits():
    digest = "a" * 64
    first = stable_sample_id("VSCO-2-CE", "a.wav", digest)
    assert first == stable_sample_id("VSCO-2-CE", "a.wav", digest)
    assert first != stable_sample_id("VSCO-2-CE", "b.wav", digest)
    rows = [
        {"id": f"{pitch}-{rr}", "midi_note": pitch, "articulation": "pizzicato", "round_robin": rr}
        for pitch in (60, 60, 62, 62, 64, 64, 65, 65, 67, 67)
        for rr in ([1, 2] if pitch in (60, 62, 64, 65, 67) else [1])
    ]
    assigned_a = assign_splits(rows, seed=7)
    assigned_b = assign_splits(rows, seed=7)
    assert assigned_a == assigned_b
    by_group = {}
    for row in assigned_a:
        by_group.setdefault(group_key(row), set()).add(row["split"])
    assert all(len(splits) == 1 for splits in by_group.values())


def test_trim_and_resample_are_deterministic():
    rate = 44100
    body = np.ones(1000, dtype=np.float32) * 0.2
    source = np.concatenate([np.zeros(300), body, np.zeros(400)])[None, :]
    trimmed, start, end = trim_silence(source, rate, threshold_db=-70, pre_roll_ms=0, post_roll_ms=0)
    assert start == 300 and end == 400
    assert trimmed.shape == (1, 1000)
    stereo = np.vstack([source, source * 0.5])
    resampled_a = resample_audio(stereo, rate, 48000)
    resampled_b = resample_audio(stereo, rate, 48000)
    assert resampled_a.shape[0] == 2
    assert resampled_a.shape[1] == round(source.shape[1] * 48000 / rate)
    assert np.array_equal(resampled_a, resampled_b)
    # The hard-edged fixture has ordinary band-limited (Gibbs) overshoot, but
    # must not exhibit the boundary instability that previously reached 1e7.
    assert np.max(np.abs(resampled_a)) <= 0.3


def test_scan_records_unknowns_and_rejections(tmp_path):
    rate = 44100
    t = np.arange(rate // 10) / rate
    _write_pcm16(
        tmp_path / "Strings" / "Solo Violin" / "Pizz" / "LLVln_Pizz_A4_p_RR1.wav",
        np.vstack([np.sin(2 * np.pi * 440 * t), np.sin(2 * np.pi * 440 * t)]),
        rate,
    )
    _write_pcm16(tmp_path / "Brass" / "Trumpet" / "unrelated.wav", np.zeros((1, 100)), rate)
    report = scan_source(tmp_path)
    assert report["counts"]["accepted_violin"] == 1
    assert report["counts"]["rejected"] == 1
    row = report["records"][0]
    assert row["channels"] == 2
    assert row["source_file_sha256"]
    assert row["processed_path"] is None


def test_manifest_build_is_deterministic(tmp_path):
    config = {
        "dataset": "vsco2-ce",
        "split": {"train": 0.8, "valid": 0.1, "test": 0.1, "seed": 9},
    }
    rows = [
        {"id": f"{i:02d}", "midi_note": 60 + i // 2, "articulation": "sustain", "processed_path": f"x{i}.wav"}
        for i in range(10)
    ]
    assert config_hash(config) == config_hash(config)
    assert build_manifests(rows, tmp_path, config) == build_manifests(rows, tmp_path, config)
    assert (tmp_path / "manifests" / "violin-train.jsonl").exists()


def test_resampler_is_a_low_pass_interpolator_not_an_imager():
    """The kernel must be centred on the fractional read position.

    The regression this pins: `distance` was computed against *padded* indices
    while `positions` were in unpadded coordinates, so the sinc and its window
    were evaluated eight taps off centre. That is not an interpolator — it left
    20.8 dB (3 kHz) to 27.9 dB (440 Hz) of inharmonic imaging noise in every
    resampled file, which any model trained on the result would learn to
    reproduce. A correct 16-tap windowed sinc clears 60 dB comfortably.
    """
    for source_rate, target_rate in ((44100, 48000), (48000, 44100), (96000, 48000)):
        t = np.arange(int(source_rate * 0.5)) / source_rate
        for frequency in (440.0, 1000.0, 3000.0):
            tone = np.sin(2.0 * np.pi * frequency * t).astype(np.float32)[None, :]
            out = resample_audio(tone, source_rate, target_rate)[0]
            window = np.hanning(out.size)
            spectrum = np.abs(np.fft.rfft(out * window))
            freqs = np.fft.rfftfreq(out.size, 1.0 / target_rate)
            peak = int(np.argmin(np.abs(freqs - frequency)))
            band = slice(max(0, peak - 4), peak + 5)
            in_band = float((spectrum[band] ** 2).sum())
            total = float((spectrum**2).sum())
            snr_db = 10.0 * np.log10(max(in_band / max(total - in_band, 1e-20), 1e-20))
            assert snr_db > 60.0, (
                f"{source_rate}->{target_rate} Hz at {frequency} Hz left only {snr_db:.1f} dB "
                "of separation; the resampling kernel is off centre"
            )
            # A resampled sine must not gain level either.
            assert np.max(np.abs(out)) < 1.1


def test_resampler_preserves_a_dc_free_signal_without_adding_offset():
    rate_in, rate_out = 44100, 48000
    t = np.arange(rate_in) / rate_in
    tone = np.sin(2.0 * np.pi * 220.0 * t).astype(np.float32)[None, :]
    out = resample_audio(tone, rate_in, rate_out)[0]
    assert abs(float(out.mean())) < 1e-3


def test_a_fade_never_attenuates_an_attack_that_starts_from_silence():
    """A file that already begins at zero must keep its transient intact.

    The regression this pins: the fade-in was unconditional, and every file in
    the VSCO source begins exactly at its onset (`trim_start_frames == 0` for
    all 161) with `x[0] == 0.0`. There was no discontinuity to smooth, but 123
    of the 161 reach above -60 dBFS inside the 3 ms ramp — so the ramp was
    attenuating the bow attack, in the voicebank and in the training target
    alike.
    """
    from datasets.vsco.preprocess import _apply_fades

    rate = 48000
    frames = rate  # one second
    t = np.arange(frames) / rate
    # Silence at sample 0, then a fast attack peaking well inside the 3 ms
    # window — the shape of a real bow or pluck onset.
    envelope = np.clip(t / 0.002, 0.0, 1.0) * np.exp(-t * 3.0)
    audio = (envelope * np.sin(2.0 * np.pi * 440.0 * t)).astype(np.float32)[None, :]
    audio[:, 0] = 0.0

    faded = _apply_fades(audio, rate, 3.0)
    window = int(rate * 0.003)
    assert np.max(np.abs(audio[:, :window])) > 0.1, "the fixture must have a real attack"
    assert np.allclose(
        faded[:, :window], audio[:, :window]
    ), "the attack was attenuated even though the file starts at silence"


def test_a_fade_still_protects_a_boundary_that_would_click():
    """A file cut mid-waveform must still be faded, or it clicks."""
    from datasets.vsco.preprocess import _apply_fades

    rate = 48000
    t = np.arange(rate) / rate
    audio = (0.5 * np.sin(2.0 * np.pi * 440.0 * t + 1.0)).astype(np.float32)[None, :]
    assert abs(float(audio[0, 0])) > 0.1, "the fixture must start mid-waveform"

    faded = _apply_fades(audio, rate, 3.0)
    assert abs(float(faded[0, 0])) < abs(float(audio[0, 0])) * 0.1
    assert abs(float(faded[0, -1])) < abs(float(audio[0, -1])) * 0.1
