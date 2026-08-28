"""Export a trained Performer to ``.fbmx`` and check the export round-trips.

The container is the existing one. What is new is a model type the audio
runtime does not claim to understand: a reader that sees ``performer-gru`` and
tries to drive it one sample at a time will fail loudly rather than produce
plausible nonsense, which is the behaviour worth having.

The normalisation statistics travel as tensors rather than as header numbers.
They are per-feature vectors, they are part of the computation, and putting them
anywhere else invites a runtime that forgets to apply them.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from fbmx.export.fbmx import FBMXMetadata, Normalization, read_fbmx, write_fbmx

from .features import INPUT_FEATURES, TARGETS
from .model import Performer, PerformerConfig


def export(
    checkpoint: str | Path,
    output: str | Path,
    *,
    name: str = "Solo Violin Performer",
    dataset_name: str = "urmp-performance",
) -> dict[str, Any]:
    state = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    config = PerformerConfig.from_dict(state["config"])
    model = Performer(config)
    model.load_state_dict(state["model"])
    model.eval()

    metadata = FBMXMetadata.from_dict(
        {
            "name": name,
            "description": (
                "Predicts how a violinist performs a written score: onset "
                "placement, duration, intonation, portamento, dynamics, and "
                "vibrato. Produces no audio."
            ),
            "attribution": (
                "Trained on the University of Rochester Multi-Modal Music "
                "Performance (URMP) dataset."
            ),
            "tags": ["violin", "performance", "expression"],
            "dataset": {
                "name": dataset_name,
                "source": "URMP isolated violin stems, notes, and F0 annotations",
                "source_type": "measured",
                "license": "CC BY-NC-SA 4.0",
                "version": "urmp-violin-1",
            },
            "training": {
                "epochs": int(state.get("epoch", -1)) + 1,
                "monitor": "validation_score",
                "best_metric": float(state.get("validation_score", float("nan"))),
                "torch_version": torch.__version__,
            },
            "notes": (
                "Performance model, not an audio model. Reads a sequence of "
                "note feature vectors and returns one performance vector per "
                "note; the engine renders those onto control curves."
            ),
        }
    )

    path = write_fbmx(
        output,
        model,
        metadata,
        normalization=Normalization(),
        extra={
            "performer": {
                "input_features": list(INPUT_FEATURES),
                "targets": list(TARGETS),
                "feature_schema_version": int(state.get("feature_schema_version", 1)),
            }
        },
    )

    # Round-trip immediately. An export that cannot be read back is not an
    # export, and finding that out at integration time wastes the trip.
    container = read_fbmx(path)
    return {
        "path": str(path),
        "bytes": Path(path).stat().st_size,
        "model_type": container.header["model"]["type"],
        "parameter_count": container.header["model"]["parameter_count"],
        "tensors": sorted(container.tensors),
    }


def parity(checkpoint: str | Path, exported: str | Path, *, notes: int = 128, seed: int = 7) -> dict[str, Any]:
    """Compare PyTorch against the numbers actually stored in the file.

    This is not yet a Python/Rust comparison — it checks that the exported
    tensors reproduce the trained model when the forward pass is recomputed
    from them with plain numpy. If this fails, the Rust runtime has no chance,
    and if it passes, any Rust disagreement is the runtime's arithmetic rather
    than a bad file.
    """

    state = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    config = PerformerConfig.from_dict(state["config"])
    model = Performer(config)
    model.load_state_dict(state["model"])
    model.eval()

    generator = np.random.default_rng(seed)
    sample = generator.standard_normal((notes, config.input_size)).astype(np.float32)

    with torch.no_grad():
        expected = model(torch.from_numpy(sample).unsqueeze(0)).squeeze(0).numpy()

    container = read_fbmx(exported)
    tensors = {name: value.numpy() for name, value in container.tensors.items()}
    actual = numpy_forward(tensors, sample, config)

    difference = np.abs(expected - actual)
    return {
        "notes": notes,
        "max_abs_error": float(difference.max()),
        "mean_abs_error": float(difference.mean()),
        "rms_error": float(np.sqrt((difference**2).mean())),
    }


def _gru_layer(
    x: np.ndarray,
    w_ih: np.ndarray,
    w_hh: np.ndarray,
    b_ih: np.ndarray,
    b_hh: np.ndarray,
    hidden: int,
    *,
    reverse: bool = False,
) -> np.ndarray:
    """One GRU direction, in PyTorch's gate order (reset, update, new).

    Written out rather than called from a library because this is the reference
    the Rust runtime is checked against, and a reference that delegates to the
    same framework it is validating proves nothing.
    """

    steps = x.shape[0]
    h = np.zeros(hidden, dtype=np.float64)
    out = np.zeros((steps, hidden), dtype=np.float64)
    order = range(steps - 1, -1, -1) if reverse else range(steps)
    sigmoid = lambda v: 1.0 / (1.0 + np.exp(-v))  # noqa: E731

    for index in order:
        gi = w_ih @ x[index] + b_ih
        gh = w_hh @ h + b_hh
        r = sigmoid(gi[:hidden] + gh[:hidden])
        z = sigmoid(gi[hidden : 2 * hidden] + gh[hidden : 2 * hidden])
        # The new gate takes the *reset-scaled hidden contribution*, which is
        # why the hidden bias cannot simply be folded into one sum.
        n = np.tanh(gi[2 * hidden :] + r * gh[2 * hidden :])
        h = (1.0 - z) * n + z * h
        out[index] = h
    return out


def numpy_forward(
    tensors: dict[str, np.ndarray], sample: np.ndarray, config: PerformerConfig
) -> np.ndarray:
    hidden = config.hidden_size
    mean = tensors["input_mean"].astype(np.float64)
    std = tensors["input_std"].astype(np.float64)
    x = (sample.astype(np.float64) - mean) / std

    forward = _gru_layer(
        x,
        tensors["rnn.weight_ih_l0"].astype(np.float64),
        tensors["rnn.weight_hh_l0"].astype(np.float64),
        tensors["rnn.bias_ih_l0"].astype(np.float64),
        tensors["rnn.bias_hh_l0"].astype(np.float64),
        hidden,
    )
    if config.bidirectional:
        backward = _gru_layer(
            x,
            tensors["rnn.weight_ih_l0_reverse"].astype(np.float64),
            tensors["rnn.weight_hh_l0_reverse"].astype(np.float64),
            tensors["rnn.bias_ih_l0_reverse"].astype(np.float64),
            tensors["rnn.bias_hh_l0_reverse"].astype(np.float64),
            hidden,
            reverse=True,
        )
        encoded = np.concatenate([forward, backward], axis=-1)
    else:
        encoded = forward

    outputs = []
    for index in range(config.output_size):
        weight = tensors[f"heads.{index}.weight"].astype(np.float64)
        bias = tensors[f"heads.{index}.bias"].astype(np.float64)
        outputs.append(encoded @ weight.T + bias)
    return np.concatenate(outputs, axis=-1).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a Performer checkpoint to .fbmx")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--name", default="Solo Violin Performer")
    arguments = parser.parse_args()

    summary = export(arguments.checkpoint, arguments.output, name=arguments.name)
    print(json.dumps(summary, indent=1))
    print("\nnumpy-vs-torch parity:")
    print(json.dumps(parity(arguments.checkpoint, arguments.output), indent=1))


if __name__ == "__main__":
    main()
