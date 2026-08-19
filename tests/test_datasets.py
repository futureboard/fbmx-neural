"""Determinism, provenance discipline, manifest parsing, WAV round-trip."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from fbmx.conditioning import ConditioningSchema, ContinuousParam
from fbmx.datasets import (
    SIGNAL_FAMILIES,
    SYNTHETIC_SCHEMA,
    DatasetInfo,
    DatasetManifest,
    ManifestEntry,
    PairedAudioDataset,
    SyntheticSmokeDataset,
    collate_pairs,
    read_audio,
    write_wav,
)
from fbmx.datasets.manifest import MANIFEST_VERSION


# -- synthetic ---------------------------------------------------------------
def test_synthetic_is_deterministic():
    a = SyntheticSmokeDataset(num_sequences=4, sequence_length=2048, seed=7)
    b = SyntheticSmokeDataset(num_sequences=4, sequence_length=2048, seed=7)
    assert torch.equal(a[0].dry, b[0].dry)
    assert torch.equal(a[2].wet, b[2].wet)
    assert a.info.checksum == b.info.checksum


def test_synthetic_seeds_and_splits_differ():
    a = SyntheticSmokeDataset(num_sequences=2, sequence_length=2048, seed=0, split="train")
    b = SyntheticSmokeDataset(num_sequences=2, sequence_length=2048, seed=1, split="train")
    c = SyntheticSmokeDataset(num_sequences=2, sequence_length=2048, seed=0, split="val")
    assert not torch.equal(a[0].dry, b[0].dry)
    assert not torch.equal(a[0].dry, c[0].dry)


def test_synthetic_covers_every_signal_family():
    ds = SyntheticSmokeDataset(num_sequences=len(SIGNAL_FAMILIES), sequence_length=1024)
    families = {ds[i].key.split("/")[-1] for i in range(len(ds))}
    assert families == set(SIGNAL_FAMILIES)


def test_synthetic_items_are_clean(dataset):
    dataset.validate(limit=None)
    item = dataset[0]
    assert item.dry.shape == item.wet.shape
    assert torch.isfinite(item.wet).all()
    assert float(item.wet.abs().max()) <= 4.0  # toy teacher, but not exploding


def test_teacher_has_memory():
    """A stateless target would make the streaming tests meaningless.

    Two signals with an identical tail but different history must produce
    different outputs over that tail.
    """
    from fbmx.datasets import SyntheticTeacher

    teacher = SyntheticTeacher(sample_rate=48000)
    n = 4800
    # the tail runs ~12 release time constants, so any memory of the burst
    # must have decayed away by the end of it
    tail = 0.2 * np.sin(2 * np.pi * 220 * np.arange(48000) / 48000)
    loud = np.concatenate([np.ones(n) * 0.9, tail])
    quiet = np.concatenate([np.zeros(n), tail])
    dry = np.stack([loud, quiet])
    args = (np.array([0.5, 0.5]), np.array([1.0, 1.0]), np.array([0, 0]))
    wet, _ = teacher(dry, *args)
    assert float(np.abs(wet[0, n : n + 200] - wet[1, n : n + 200]).max()) > 1e-3
    # ...and the memory must fade, or it is not a release, it is a latch
    assert float(np.abs(wet[0, -200:] - wet[1, -200:]).max()) < 1e-3


def test_synthetic_provides_aux_gain_trace(dataset):
    item = dataset[0]
    assert "gain" in item.aux
    assert item.aux["gain"].shape == item.dry.shape


def test_synthetic_declares_provenance(dataset):
    info = dataset.info
    assert info.source_type == "synthetic"
    assert info.license and info.checksum
    assert "not a model of any real device" in info.notes


def test_collate_keeps_params_and_rejects_mixed_lengths(dataset):
    batch = collate_pairs([dataset[0], dataset[1]])
    assert batch["dry"].shape[0] == 2
    assert batch["params"].batch_size == 2
    short = dataset[0]
    short.dry = short.dry[..., :100]
    short.wet = short.wet[..., :100]
    with pytest.raises(ValueError):
        collate_pairs([dataset[1], short])


# -- provenance --------------------------------------------------------------
def test_dataset_info_requires_licence_and_source():
    with pytest.raises(ValueError):
        DatasetInfo(name="x", source="", source_type="synthetic", license="CC0-1.0")
    with pytest.raises(ValueError):
        DatasetInfo(name="x", source="s", source_type="synthetic", license="")


def test_dataset_info_rejects_unknown_source_type():
    with pytest.raises(ValueError):
        DatasetInfo(name="x", source="s", source_type="downloaded", license="MIT")


def test_dataset_info_defaults_to_not_redistributable():
    info = DatasetInfo(name="x", source="s", source_type="hardware_capture", license="proprietary")
    assert info.redistributable is False


# -- manifest ----------------------------------------------------------------
@pytest.fixture
def tiny_corpus(tmp_path):
    """Two 0.1 s dry/wet pairs on disk plus a manifest describing them."""
    sr = 48000
    n = sr // 10
    schema = ConditioningSchema(continuous=(ContinuousParam("drive", 0.0, 1.0, 0.5),))
    entries = []
    for i, drive in enumerate((0.25, 0.75)):
        t = np.arange(n) / sr
        dry = (0.5 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
        wet = np.tanh((1 + 3 * drive) * dry).astype(np.float32)
        write_wav(tmp_path / f"pair{i}_dry.wav", dry, sr)
        write_wav(tmp_path / f"pair{i}_wet.wav", wet, sr)
        entries.append(
            ManifestEntry(
                key=f"pair{i}",
                dry=f"pair{i}_dry.wav",
                wet=f"pair{i}_wet.wav",
                split="train" if i == 0 else "val",
                params={"drive": drive},
            )
        )
    info = DatasetInfo(
        name="unit-test-corpus",
        source="generated by tests/test_datasets.py",
        source_type="circuit_model",
        license="CC0-1.0",
        sample_rate=sr,
        redistributable=True,
    )
    manifest = DatasetManifest.build(info, schema, entries, root=tmp_path)
    manifest.fill_checksums()
    manifest.save(tmp_path / "manifest.json")
    return tmp_path / "manifest.json"


def test_wav_round_trip(tmp_path):
    sr = 48000
    x = (np.random.default_rng(0).standard_normal(1000) * 0.4).astype(np.float32)
    write_wav(tmp_path / "x.wav", x, sr)
    back, rate = read_audio(tmp_path / "x.wav")
    assert rate == sr
    assert back.shape == (1, 1000)
    assert np.allclose(back[0], x, atol=1e-7)


def test_manifest_round_trip(tiny_corpus):
    manifest = DatasetManifest.load(tiny_corpus)
    assert manifest.version == MANIFEST_VERSION
    assert len(manifest) == 2
    assert manifest.splits == ["train", "val"]
    assert manifest.info.source_type == "circuit_model"
    assert manifest.schema.names == ["drive"]
    manifest.validate(check_files=True, check_checksums=True)


def test_manifest_rejects_wrong_version(tmp_path, tiny_corpus):
    data = json.loads(tiny_corpus.read_text(encoding="utf-8"))
    data["fbmx_manifest_version"] = 99
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported manifest version"):
        DatasetManifest.load(bad)


def test_manifest_requires_provenance(tmp_path):
    bad = tmp_path / "noinfo.json"
    bad.write_text(json.dumps({"fbmx_manifest_version": 1, "entries": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="provenance"):
        DatasetManifest.load(bad)


def test_manifest_rejects_unknown_parameter(tiny_corpus):
    manifest = DatasetManifest.load(tiny_corpus)
    manifest.entries[0].params["nonexistent"] = 1.0
    with pytest.raises(ValueError, match="not in the manifest schema"):
        manifest.validate()


def test_manifest_detects_tampered_audio(tiny_corpus):
    manifest = DatasetManifest.load(tiny_corpus)
    target = manifest.root / manifest.entries[0].wet
    write_wav(target, np.zeros(4800, dtype=np.float32), manifest.info.sample_rate)
    with pytest.raises(ValueError, match="checksum mismatch"):
        manifest.validate(check_checksums=True)


def test_paired_audio_dataset_reads_pairs(tiny_corpus):
    ds = PairedAudioDataset(tiny_corpus, split="train")
    assert len(ds) == 1
    item = ds[0]
    assert item.dry.shape == item.wet.shape
    assert ds.schema.decode(item.params)["drive"] == pytest.approx(0.25)
    ds.validate()


def test_paired_audio_segments(tiny_corpus):
    ds = PairedAudioDataset(tiny_corpus, split=None, segment_length=1200)
    assert len(ds) == 2 * (4800 // 1200)
    assert ds[0].dry.shape[-1] == 1200


def test_paired_audio_rejects_sample_rate_mismatch(tmp_path, tiny_corpus):
    manifest = DatasetManifest.load(tiny_corpus)
    target = manifest.root / manifest.entries[0].dry
    dry, _ = read_audio(target)
    write_wav(target, dry, 44100)  # same audio, wrong rate
    manifest.entries[0].dry_sha256 = ""
    manifest.entries[0].wet_sha256 = ""
    with pytest.raises(ValueError, match="resample deliberately"):
        PairedAudioDataset(manifest, split="train")


# ---------------------------------------------------------------------------
# the rendered FA76 corpus, when one is present
# ---------------------------------------------------------------------------
FA76_MANIFEST = Path(__file__).resolve().parents[2] / "datasets" / "fa76-revd-v2" / "manifest.json"


@pytest.mark.skipif(not FA76_MANIFEST.exists(), reason="teacher corpus not rendered here")
def test_the_teacher_corpus_records_its_alignment():
    """The phase-3 fix has to be legible from the manifest alone.

    A consumer must be able to tell whether a corpus was rendered before or
    after the fractional-delay compensation without re-measuring the audio.
    """
    manifest = DatasetManifest.load(FA76_MANIFEST)
    alignment = manifest.info.extra["alignment"]
    assert alignment["whole_samples_removed"] == 4
    assert abs(alignment["residual_after_samples"]) < 0.01, alignment
    # ...and that it is an improvement on what a whole-sample shift leaves.
    assert abs(alignment["residual_before_samples"]) > 0.1
    assert "no EQ compensation" in alignment["method"]


@pytest.mark.skipif(not FA76_MANIFEST.exists(), reason="teacher corpus not rendered here")
def test_the_teacher_corpus_is_gain_reduction_targeted():
    manifest = DatasetManifest.load(FA76_MANIFEST)
    assert manifest.info.source_type == "circuit_model"
    assert manifest.info.extra["teacher"] == "FA76 Circuit"
    assert "gr-targeted" in manifest.info.extra["sampler"]
    assert manifest.info.extra["aux_traces"] == ["gain", "cv"]

    # Matched 20:1 / all-buttons pairs: same dry audio, same dials, both
    # ratios. Without them the model can learn "all-buttons = more".
    pairs: dict[str, list] = {}
    for entry in manifest.entries:
        marker = [p for p in entry.notes.split("|") if "pair" in p]
        if marker:
            pairs.setdefault(marker[0].strip(), []).append(entry)
    assert len(pairs) > 20, f"only {len(pairs)} matched pairs in the corpus"
    for key, entries in pairs.items():
        assert len(entries) == 2, key
        ratios = {e.params["Ratio"] for e in entries}
        assert ratios == {"20:1", "All Buttons"}, key
        # everything except the ratio must be identical
        a, b = entries
        for dial in ("Input", "Attack", "Release"):
            assert a.params[dial] == b.params[dial], f"{key}: {dial} differs"
