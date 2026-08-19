"""Auxiliary dynamics heads, and the manifest plumbing that feeds them.

The audio head is mandatory; a gain-reduction or control-voltage head is
optional, is configured per model, and must never change the audio path.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from fbmx.conditioning import ConditioningSchema, ContinuousParam
from fbmx.datasets import DatasetInfo, DatasetManifest, ManifestEntry, PairedAudioDataset
from fbmx.datasets.paired_audio import write_wav
from fbmx.export.fbmx import read_fbmx, write_fbmx
from fbmx.losses import build_loss
from fbmx.losses.auxiliary import AuxTraceLoss
from fbmx.models import build_model


@pytest.fixture
def schema():
    return ConditioningSchema(continuous=(ContinuousParam("Input", 0.0, 10.0, 5.0),))


@pytest.fixture
def model(schema):
    torch.manual_seed(0)
    return build_model(
        {"type": "lstm", "hidden_size": 16, "aux_heads": ["gain"]}, schema
    ).eval()


def test_aux_head_predicts_a_trace_of_the_right_shape(model, schema):
    x = torch.randn(2, 1, 512) * 0.3
    y, aux, state = model.forward_aux(x, schema.empty_batch(2), None)
    assert y.shape == x.shape
    assert set(aux) == {"pred_gain"}
    assert aux["pred_gain"].shape == x.shape
    assert state is not None


def test_forward_and_forward_aux_agree_on_the_audio(model, schema):
    x = torch.randn(1, 1, 256) * 0.3
    params = schema.empty_batch(1)
    a, _ = model(x, params, None)
    b, _, _ = model.forward_aux(x, params, None)
    assert torch.equal(a, b)


def test_aux_heads_do_not_change_the_audio_path(schema):
    """A model with and without the head must produce the same audio.

    Same seed, same weights for every shared module: the aux head is built
    last, so it can only consume randomness after the audio path is fixed.
    """
    torch.manual_seed(7)
    without = build_model({"type": "lstm", "hidden_size": 16}, schema).eval()
    torch.manual_seed(7)
    with_head = build_model(
        {"type": "lstm", "hidden_size": 16, "aux_heads": ["gain"]}, schema
    ).eval()
    x = torch.randn(1, 1, 512) * 0.3
    params = schema.empty_batch(1)
    with torch.no_grad():
        assert torch.equal(without(x, params, None)[0], with_head(x, params, None)[0])


def test_a_model_without_aux_heads_reports_none(schema):
    model = build_model({"type": "lstm", "hidden_size": 8}, schema)
    assert model.aux_heads == ()
    _, aux, _ = model.forward_aux(torch.zeros(1, 1, 64), None, None)
    assert aux == {}


def test_export_spec_lists_the_output_heads(model):
    assert model.export_spec()["output_heads"] == ["audio", "gain"]


def test_aux_weights_survive_the_fbmx_round_trip(tmp_path, model, schema):
    path = write_fbmx(tmp_path / "m.fbmx", model)
    container = read_fbmx(path)
    names = [t["name"] for t in container.header["tensors"]]
    assert "aux.gain.weight" in names and "aux.gain.bias" in names
    restored = container.build_model("cpu")
    assert restored.aux_heads == ("gain",)
    x = torch.randn(1, 1, 256) * 0.3
    with torch.no_grad():
        a, aux_a, _ = model.forward_aux(x, schema.empty_batch(1), None)
        b, aux_b, _ = restored.forward_aux(x, schema.empty_batch(1), None)
    assert torch.equal(a, b)
    assert torch.equal(aux_a["pred_gain"], aux_b["pred_gain"])


def test_aux_trace_loss_scale_is_applied():
    loss = AuxTraceLoss("gain", scale=0.04)
    aux = {
        "gain": torch.full((1, 1, 8), -25.0),   # dB
        "pred_gain": torch.full((1, 1, 8), -1.0),  # dB/25
    }
    assert float(loss(torch.zeros(1, 1, 8), torch.zeros(1, 1, 8), aux)) == pytest.approx(0.0)


def test_lambda_and_scale_come_from_the_config():
    loss_fn = build_loss([
        {"name": "mae", "weight": 1.0},
        {"name": "aux_trace", "as": "gr", "key": "gain", "weight": 0.5, "scale": 0.04},
    ])
    assert loss_fn.weights == {"mae": 1.0, "gr": 0.5}
    assert loss_fn.terms["gr"].scale == pytest.approx(0.04)


# ---------------------------------------------------------------------------
# manifest side
# ---------------------------------------------------------------------------
@pytest.fixture
def corpus_with_aux(tmp_path):
    sr = 48000
    n = 4800
    schema = ConditioningSchema(continuous=(ContinuousParam("Input", 0.0, 10.0, 5.0),))
    t = np.arange(n) / sr
    dry = (0.5 * np.sin(2 * np.pi * 100 * t)).astype(np.float32)
    wet = np.tanh(2 * dry).astype(np.float32)
    gain = np.linspace(0.0, -12.0, n).astype(np.float32)
    write_wav(tmp_path / "a_dry.wav", dry, sr)
    write_wav(tmp_path / "a_wet.wav", wet, sr)
    write_wav(tmp_path / "a_gain.wav", gain, sr)

    info = DatasetInfo(
        name="aux-test",
        source="tests",
        source_type="circuit_model",
        license="MIT",
        sample_rate=sr,
        extra={"teacher": "FA76 Circuit", "revision": "Rev D"},
    )
    manifest = DatasetManifest.build(
        info,
        schema,
        [
            ManifestEntry(
                key="a",
                dry="a_dry.wav",
                wet="a_wet.wav",
                params={"Input": 7.0},
                aux={"gain": "a_gain.wav"},
            )
        ],
        root=tmp_path,
    )
    manifest.fill_checksums()
    manifest.save(tmp_path / "manifest.json")
    return tmp_path / "manifest.json"


def test_manifest_carries_aux_paths_and_checksums(corpus_with_aux):
    manifest = DatasetManifest.load(corpus_with_aux)
    entry = manifest.entries[0]
    assert entry.aux == {"gain": "a_gain.wav"}
    assert entry.aux_sha256["gain"]
    manifest.validate(check_files=True, check_checksums=True)


def test_tampered_aux_trace_is_detected(corpus_with_aux):
    manifest = DatasetManifest.load(corpus_with_aux)
    write_wav(manifest.root / "a_gain.wav", np.zeros(4800, dtype=np.float32), 48000)
    with pytest.raises(ValueError, match="checksum mismatch"):
        manifest.validate(check_checksums=True)


def test_dataset_loads_the_aux_trace(corpus_with_aux):
    ds = PairedAudioDataset(corpus_with_aux, split="train", aux_traces=["gain"])
    item = ds[0]
    assert "gain" in item.aux
    assert item.aux["gain"].shape == item.dry.shape
    assert float(item.aux["gain"][0, 0]) == pytest.approx(0.0, abs=1e-4)
    assert float(item.aux["gain"][0, -1]) == pytest.approx(-12.0, abs=0.1)


def test_requesting_a_trace_the_teacher_never_exported_fails_loudly(corpus_with_aux):
    with pytest.raises(KeyError, match="cv"):
        PairedAudioDataset(corpus_with_aux, split="train", aux_traces=["cv"])


def test_aux_traces_are_segmented_with_the_audio(corpus_with_aux):
    ds = PairedAudioDataset(
        corpus_with_aux, split="train", segment_length=1200, aux_traces=["gain"]
    )
    assert len(ds) == 4
    for i in range(len(ds)):
        assert ds[i].aux["gain"].shape[-1] == 1200


# ---------------------------------------------------------------------------
# manifest compatibility gate
# ---------------------------------------------------------------------------
def test_expectations_accept_the_right_dataset(corpus_with_aux):
    PairedAudioDataset(
        corpus_with_aux,
        split="train",
        expect={"source_type": "circuit_model", "teacher": "FA76 Circuit", "revision": "Rev D"},
    )


def test_expectations_reject_the_wrong_teacher(corpus_with_aux):
    with pytest.raises(ValueError, match="refusing to train"):
        PairedAudioDataset(
            corpus_with_aux, split="train", expect={"teacher": "some hardware unit"}
        )


def test_expectations_reject_a_missing_field(corpus_with_aux):
    with pytest.raises(ValueError, match="declares no"):
        PairedAudioDataset(corpus_with_aux, split="train", expect={"calibration": "2026"})


def test_expectations_reject_a_hardware_claim_on_circuit_data(corpus_with_aux):
    # The case that matters: a config that says "trained on hardware" must not
    # accept a circuit render, because the resulting model would inherit a
    # provenance claim that is false.
    with pytest.raises(ValueError, match="refusing to train"):
        PairedAudioDataset(
            corpus_with_aux, split="train", expect={"source_type": "hardware_capture"}
        )


def test_manifest_json_is_readable_by_a_third_party(corpus_with_aux):
    """The manifest is the interchange format; keep it plain JSON."""
    raw = json.loads(corpus_with_aux.read_text(encoding="utf-8"))
    assert raw["fbmx_manifest_version"] == 1
    assert raw["info"]["source_type"] == "circuit_model"
    assert raw["entries"][0]["aux"]["gain"] == "a_gain.wav"
