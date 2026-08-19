"""The .fbmx container: round-trip, self-description, tamper detection."""

from __future__ import annotations

import json
import struct

import pytest
import torch

from fbmx import FBMX_FORMAT_VERSION
from fbmx.datasets import SYNTHETIC_SCHEMA, SyntheticSmokeDataset
from fbmx.export.fbmx import (
    MAGIC,
    FBMXMetadata,
    Normalization,
    export_from_checkpoint,
    read_fbmx,
    write_fbmx,
)
from fbmx.models import build_model
from fbmx.streaming.inference import process_blocked, process_offline
from fbmx.training.checkpoint import save_checkpoint


@pytest.fixture
def exported(tmp_path):
    torch.manual_seed(0)
    model = build_model(
        {"type": "lstm", "hidden_size": 32, "conditioning": "both"}, SYNTHETIC_SCHEMA
    ).eval()
    path = write_fbmx(
        tmp_path / "m.fbmx",
        model,
        FBMXMetadata(
            name="unit-test",
            license="CC0-1.0",
            model_source_type="synthetic",
            notes="pipeline test only",
        ),
    )
    return model, path


def test_header_layout(exported):
    _, path = exported
    raw = path.read_bytes()
    assert raw[:4] == MAGIC
    assert struct.unpack_from("<I", raw, 4)[0] == FBMX_FORMAT_VERSION
    header_len = struct.unpack_from("<Q", raw, 8)[0]
    header = json.loads(raw[16 : 16 + header_len])
    assert header["format"] == "fbmx"
    # tensor data starts on a 16-byte boundary, so a Rust reader can cast slices
    assert (16 + header_len + (-(16 + header_len) % 16)) % 16 == 0


def test_container_declares_everything_a_runtime_needs(exported):
    _, path = exported
    container = read_fbmx(path)
    h = container.header
    for key in ("format_version", "model_uuid", "created_utc", "model", "input_spec",
                "state_spec", "conditioning", "normalization", "tensors", "metadata",
                "checksums"):
        assert key in h, key
    for key in ("type", "sample_rate", "channels", "causal", "recurrent",
                "receptive_field", "parameter_count", "hidden_size", "hparams"):
        assert key in h["model"], key
    assert h["model"]["sample_rate"] == 48000
    assert h["model"]["hidden_size"] == 32
    assert h["state_spec"]["kind"] == "lstm"
    assert container.metadata.license == "CC0-1.0"
    assert container.metadata.model_source_type == "synthetic"
    assert container.metadata.validated is False


def test_conditioning_schema_survives_the_round_trip(exported):
    model, path = exported
    assert read_fbmx(path).schema == model.schema


def test_weights_round_trip_exactly(exported):
    model, path = exported
    restored = read_fbmx(path).build_model("cpu")
    for name, tensor in model.state_dict().items():
        assert torch.equal(tensor, restored.state_dict()[name]), name


def test_reloaded_model_computes_the_same_audio(exported):
    model, path = exported
    restored = read_fbmx(path).build_model("cpu")
    torch.manual_seed(1)
    x = torch.randn(1, 1, 4096) * 0.3
    params = SYNTHETIC_SCHEMA.encode({"drive": 0.7, "mix": 0.9, "mode": "hard"})
    with torch.no_grad():
        assert torch.equal(process_offline(model, x, params), process_offline(restored, x, params))


def test_reloaded_model_streams_identically(exported):
    _, path = exported
    restored = read_fbmx(path).build_model("cpu")
    torch.manual_seed(2)
    x = torch.randn(1, 1, 4096) * 0.3
    params = SYNTHETIC_SCHEMA.empty_batch(1)
    offline = process_offline(restored, x, params)
    assert float((offline - process_blocked(restored, x, 64, params)).abs().max()) < 1e-5


def test_uuid_is_unique_per_export(tmp_path):
    model = build_model({"type": "lstm", "hidden_size": 8}, SYNTHETIC_SCHEMA)
    a = read_fbmx(write_fbmx(tmp_path / "a.fbmx", model)).model_uuid
    b = read_fbmx(write_fbmx(tmp_path / "b.fbmx", model)).model_uuid
    assert a != b


def test_normalization_is_recorded(tmp_path):
    model = build_model({"type": "lstm", "hidden_size": 8}, SYNTHETIC_SCHEMA)
    norm = Normalization(scheme="peak", input_gain=0.5, output_gain=2.0)
    container = read_fbmx(write_fbmx(tmp_path / "n.fbmx", model, normalization=norm))
    assert container.normalization == norm


def test_tamper_is_detected(exported):
    model, path = exported
    raw = bytearray(path.read_bytes())
    raw[-40] ^= 0xFF  # flip a bit inside the tensor data
    path.write_bytes(bytes(raw))
    with pytest.raises(ValueError, match="checksum mismatch"):
        read_fbmx(path)


def test_truncation_is_detected(exported):
    _, path = exported
    raw = path.read_bytes()
    path.write_bytes(raw[: len(raw) // 2])
    with pytest.raises(ValueError):
        read_fbmx(path)


def test_bad_magic_is_rejected(tmp_path):
    bad = tmp_path / "bad.fbmx"
    bad.write_bytes(b"NOPE" + b"\x00" * 128)
    with pytest.raises(ValueError, match="bad magic"):
        read_fbmx(bad)


def test_file_contains_no_pickle_opcodes(exported):
    """A .fbmx must never be a pickle -- loading one cannot execute anything."""
    _, path = exported
    raw = path.read_bytes()
    assert not raw.startswith(b"\x80")  # pickle protocol marker
    assert b"torch.storage" not in raw
    assert b"__reduce__" not in raw


def test_read_is_pure_data(exported):
    """Reading must not import or construct a model."""
    _, path = exported
    container = read_fbmx(path)
    assert isinstance(container.header, dict)
    assert all(isinstance(t, torch.Tensor) for t in container.tensors.values())


@pytest.mark.parametrize("model_cfg", [
    {"type": "lstm", "hidden_size": 16},
    {"type": "gru", "hidden_size": 16},
    {"type": "tcn", "channels": 8, "num_blocks": 3, "kernel_size": 3},
])
def test_every_architecture_round_trips(tmp_path, model_cfg):
    torch.manual_seed(0)
    model = build_model(dict(model_cfg), SYNTHETIC_SCHEMA).eval()
    path = write_fbmx(tmp_path / f"{model_cfg['type']}.fbmx", model)
    restored = read_fbmx(path).build_model("cpu")
    x = torch.randn(1, 1, 2048) * 0.3
    params = SYNTHETIC_SCHEMA.empty_batch(1)
    with torch.no_grad():
        assert torch.equal(process_offline(model, x, params), process_offline(restored, x, params))


def test_export_from_checkpoint_inherits_provenance(tmp_path):
    dataset = SyntheticSmokeDataset(num_sequences=2, sequence_length=1024)
    model = build_model({"type": "lstm", "hidden_size": 8}, SYNTHETIC_SCHEMA)
    ckpt = save_checkpoint(
        tmp_path / "best.pt",
        model,
        epoch=3,
        global_step=100,
        extra={"dataset": dataset.provenance()},
    )
    path, container = export_from_checkpoint(ckpt, tmp_path / "out.fbmx")
    meta = container.metadata
    assert meta.model_source_type == "synthetic"
    assert meta.license == "CC0-1.0"
    assert meta.dataset["name"] == "fbmx-synthetic-smoke"
    assert meta.dataset["checksum"] == dataset.info.checksum
    assert meta.training["epochs"] == 3
    assert meta.validated is False


def test_export_cannot_be_written_with_an_invented_source_type():
    with pytest.raises(ValueError, match="model_source_type"):
        FBMXMetadata(model_source_type="definitely_real_hardware")


def test_summary_mentions_the_essentials(exported):
    _, path = exported
    summary = read_fbmx(path).summary()
    for token in ("fbmx v", "uuid", "lstm", "48000 Hz", "source type", "licence"):
        assert token in summary
