from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fbmx.datasets import SYNTHETIC_SCHEMA, SyntheticSmokeDataset  # noqa: E402
from fbmx.models import build_model  # noqa: E402

#: Every architecture is held to the same contract, so most tests are
#: parameterised over all of them rather than written for the baseline only.
MODEL_CONFIGS = [
    {"type": "lstm", "hidden_size": 32},
    {"type": "lstm", "hidden_size": 8, "conditioning": "both", "head_hidden": 8},
    {"type": "gru", "hidden_size": 16},
    {"type": "tcn", "channels": 8, "num_blocks": 4, "kernel_size": 3},
]

BLOCK_SIZES = (16, 32, 64, 128, 256, 512, 1024)


@pytest.fixture(scope="session")
def schema():
    return SYNTHETIC_SCHEMA


@pytest.fixture(scope="session")
def dataset():
    return SyntheticSmokeDataset(num_sequences=4, sequence_length=8192, seed=0)


@pytest.fixture(params=MODEL_CONFIGS, ids=lambda c: f"{c['type']}{c.get('hidden_size', '')}")
def model(request):
    torch.manual_seed(0)
    return build_model(dict(request.param), SYNTHETIC_SCHEMA).eval()


@pytest.fixture
def lstm32():
    torch.manual_seed(0)
    return build_model({"type": "lstm", "hidden_size": 32}, SYNTHETIC_SCHEMA).eval()


@pytest.fixture
def probe_signal():
    torch.manual_seed(1234)
    x = torch.randn(1, 1, 6000) * 0.3
    x[..., 2000:2100] = 0.0  # a silence transition, where state bugs show up
    x[..., 3000] = 0.95  # and an impulse
    return x


def cuda_devices() -> list[str]:
    return ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
