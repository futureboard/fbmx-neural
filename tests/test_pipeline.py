"""End-to-end: the four CLI entry points, run for real on a tiny config.

Slow-ish (tens of seconds) but this is the test that would have caught every
integration bug the unit tests missed, so it runs by default.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
CONFIG = ROOT / "configs" / "smoke.yaml"

TINY = [
    "data.num_sequences=4",
    "data.sequence_length=8192",
    "train.epochs=2",
    "train.batch_size=2",
    "train.chunk_size=2048",
    "train.warmup_samples=512",
    "train.log_every=0",
    "train.device=cpu",
]


def run(script: str, *args: str) -> str:
    result = subprocess.run(
        [sys.executable, str(SCRIPTS / script), *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        pytest.fail(f"{script} failed:\n{result.stdout}\n{result.stderr}")
    return result.stdout


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    workdir = tmp_path_factory.mktemp("pipeline")
    ckpt_dir = workdir / "checkpoints"
    out = run("train.py", "--config", str(CONFIG), "--set", *TINY,
              f"train.checkpoint_dir={ckpt_dir}")
    assert (ckpt_dir / "best.pt").exists()
    assert (ckpt_dir / "last.pt").exists()
    return workdir, ckpt_dir, out


def test_train_reports_progress(trained):
    _, _, out = trained
    assert "[trainer] device:" in out
    assert "val_loss=" in out
    assert "[train] done" in out


def test_resume_from_checkpoint(trained):
    workdir, ckpt_dir, _ = trained
    out = run("train.py", "--config", str(CONFIG), "--set", *TINY, "train.epochs=3",
              f"train.checkpoint_dir={ckpt_dir}", "--resume", str(ckpt_dir / "last.pt"))
    assert "architecture rebuilt from" in out
    assert "resumed from" in out
    # epoch 2 already happened; only the third runs
    assert "[epoch 2]" in out and "[epoch 0]" not in out


def test_validate_reports_metrics_and_streaming(trained):
    _, ckpt_dir, _ = trained
    out = run("validate.py", "--checkpoint", str(ckpt_dir / "best.pt"))
    assert "esr" in out and "mae" in out
    assert "streaming equivalence" in out


def test_export_and_inspect(trained):
    workdir, ckpt_dir, _ = trained
    model_path = workdir / "models" / "smoke.fbmx"
    out = run("export_fbmx.py", "--checkpoint", str(ckpt_dir / "best.pt"),
              "--output", str(model_path), "--author", "tests")
    assert model_path.exists()
    assert "reload max |checkpoint - fbmx|" in out

    out = run("inspect_fbmx.py", str(model_path), "--tensors", "--load")
    assert "checksums      verified" in out
    assert "source type    synthetic" in out
    assert "rnn.weight_ih_l0" in out
    assert "load check" in out


def test_inspect_json_is_parseable(trained):
    import json

    workdir, ckpt_dir, _ = trained
    model_path = workdir / "models" / "smoke.fbmx"
    if not model_path.exists():
        run("export_fbmx.py", "--checkpoint", str(ckpt_dir / "best.pt"),
            "--output", str(model_path))
    header = json.loads(run("inspect_fbmx.py", str(model_path), "--json"))
    assert header["format"] == "fbmx"
    assert header["model"]["type"] == "lstm"
