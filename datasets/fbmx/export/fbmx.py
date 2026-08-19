"""The ``.fbmx`` model container (prototype, format version 1).

Design rules, in order of importance:

1. **No pickle, no code execution.**  A ``.fbmx`` file is a header and a block
   of numbers.  Loading one must never be able to run anything.  PyTorch
   checkpoints are development artifacts and stay inside this repository.
2. **Independently readable.**  Everything needed to reconstruct the model --
   architecture, hyper-parameters, tensor names/shapes/dtypes, sample rate,
   conditioning schema, normalisation -- is in the header.  The eventual Rust
   runtime reads it with a JSON parser and a byte-slice cast, no Python
   involved.
3. **Self-declaring provenance.**  Licence, dataset, and ``model_source_type``
   travel *with the weights*.  A model fitted to a hardware capture and one
   fitted to a synthetic teacher are not interchangeable claims, and the file
   has to say which it is.
4. **Verifiable.**  The tensor region is hashed into the header and the whole
   file is hashed into a trailer.

Byte layout::

    offset  size      contents
    0       4         magic, b"FBMX"
    4       4         u32 LE format version
    8       8         u64 LE header length in bytes
    16      H         UTF-8 JSON header
    ...     pad       zero padding to a 16-byte boundary
    D       ...       tensor data, little-endian, C-contiguous, in header order
    EOF-32  32        sha256 of every byte before the trailer

All integers are little-endian.  Tensor ``offset`` values in the header are
relative to ``data_offset``.

Not yet decided, and deliberately left out of v1: quantised weight storage,
multiple sub-models in one file, and any notion of a signature.  Bumping
``format_version`` is cheap; guessing wrong about those is not.
"""

from __future__ import annotations

import hashlib
import json
import struct
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from fbmx import FBMX_FORMAT_VERSION, __version__
from fbmx.conditioning import ConditioningSchema
from fbmx.datasets.base import SOURCE_TYPES
from fbmx.models.base import StreamingModel

__all__ = [
    "MAGIC",
    "FBMXMetadata",
    "Normalization",
    "FBMXFile",
    "write_fbmx",
    "read_fbmx",
    "export_from_checkpoint",
]

MAGIC = b"FBMX"
HEADER_ALIGN = 16
TRAILER_SIZE = 32

#: numpy dtype <-> the short names written into the header.  Restricted on
#: purpose: a runtime should not have to implement every dtype numpy has.
DTYPES = {"f32": np.float32, "f64": np.float64, "f16": np.float16, "i64": np.int64}
DTYPE_NAMES = {np.dtype(v): k for k, v in DTYPES.items()}


@dataclass
class Normalization:
    """Fixed affine pre/post scaling applied around the network.

    ``y = output_gain * f(input_gain * x + input_offset) + output_offset``

    v0 models train on unnormalised audio and export the identity.  The field
    exists now because bolting normalisation on after a runtime exists means
    two incompatible readers.
    """

    scheme: str = "none"
    input_gain: float = 1.0
    input_offset: float = 0.0
    output_gain: float = 1.0
    output_offset: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: Mapping[str, Any] | None) -> "Normalization":
        d = dict(d or {})
        known = set(Normalization.__dataclass_fields__)
        return Normalization(**{k: v for k, v in d.items() if k in known})


@dataclass
class FBMXMetadata:
    """Everything about a model that is not a number in a tensor."""

    name: str = "unnamed"
    description: str = ""
    author: str = ""
    license: str = ""
    license_url: str = ""
    attribution: str = ""
    model_source_type: str = "synthetic"  # one of fbmx.datasets.SOURCE_TYPES
    dataset: dict[str, Any] = field(default_factory=dict)
    training: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    notes: str = ""
    validated: bool = False  # True only for a model that passed a real listening/measurement pass

    def __post_init__(self) -> None:
        if self.model_source_type not in SOURCE_TYPES:
            raise ValueError(
                f"model_source_type must be one of {SOURCE_TYPES}, "
                f"got {self.model_source_type!r}"
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: Mapping[str, Any] | None) -> "FBMXMetadata":
        d = dict(d or {})
        known = set(FBMXMetadata.__dataclass_fields__)
        return FBMXMetadata(**{k: v for k, v in d.items() if k in known})


# ---------------------------------------------------------------------------
# writing
# ---------------------------------------------------------------------------
def _tensor_entries(state: Mapping[str, torch.Tensor]) -> tuple[list[dict[str, Any]], bytes]:
    entries: list[dict[str, Any]] = []
    blobs: list[bytes] = []
    offset = 0
    for name, tensor in state.items():
        array = tensor.detach().cpu().contiguous().numpy()
        dtype_name = DTYPE_NAMES.get(array.dtype)
        if dtype_name is None:
            raise TypeError(
                f"tensor {name!r} has dtype {array.dtype}, which .fbmx v1 does not store; "
                f"supported: {sorted(DTYPES)}"
            )
        raw = array.tobytes(order="C")
        entries.append(
            {
                "name": name,
                "dtype": dtype_name,
                "shape": list(array.shape),
                "offset": offset,
                "nbytes": len(raw),
            }
        )
        blobs.append(raw)
        offset += len(raw)
    return entries, b"".join(blobs)


def write_fbmx(
    path: str | Path,
    model: StreamingModel,
    metadata: FBMXMetadata | Mapping[str, Any] | None = None,
    *,
    normalization: Normalization | None = None,
    model_uuid: str | None = None,
    extra: Mapping[str, Any] | None = None,
) -> Path:
    """Serialise ``model`` to ``path``.  Returns the path written."""
    if not isinstance(metadata, FBMXMetadata):
        metadata = FBMXMetadata.from_dict(metadata)
    normalization = normalization or Normalization()

    spec = model.export_spec()
    state = {k: v for k, v in model.state_dict().items()}
    entries, data = _tensor_entries(state)

    header: dict[str, Any] = {
        "format": "fbmx",
        "format_version": FBMX_FORMAT_VERSION,
        "model_uuid": model_uuid or str(uuid.uuid4()),
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "producer": {
            "library": "fbmx",
            "version": __version__,
            "torch": torch.__version__,
        },
        "model": {
            "type": spec["model_type"],
            "architecture": spec["model_type"],
            "sample_rate": spec["sample_rate"],
            "channels": spec["channels"],
            "causal": spec["causal"],
            "recurrent": spec["recurrent"],
            "receptive_field": spec["receptive_field"],
            "parameter_count": spec["parameter_count"],
            "hidden_size": spec["hparams"].get("hidden_size"),
            "hparams": spec["hparams"],
        },
        "input_spec": spec.get(
            "input_features", {"layout": "BCT", "channels": 1, "order": ["audio"]}
        ),
        "state_spec": spec.get("state_spec", {}),
        "conditioning": spec["conditioning"],
        "normalization": normalization.to_dict(),
        "tensors": entries,
        "metadata": metadata.to_dict(),
        "checksums": {
            "algorithm": "sha256",
            "data_sha256": hashlib.sha256(data).hexdigest(),
            "data_nbytes": len(data),
        },
        "extra": dict(extra or {}),
    }

    header_bytes = json.dumps(header, indent=None, separators=(",", ":")).encode("utf-8")
    prefix = MAGIC + struct.pack("<I", FBMX_FORMAT_VERSION) + struct.pack("<Q", len(header_bytes))
    unpadded = len(prefix) + len(header_bytes)
    pad = (-unpadded) % HEADER_ALIGN
    body = prefix + header_bytes + b"\x00" * pad + data
    digest = hashlib.sha256(body).digest()

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(body + digest)
    tmp.replace(path)
    return path


# ---------------------------------------------------------------------------
# reading
# ---------------------------------------------------------------------------
@dataclass
class FBMXFile:
    """A parsed ``.fbmx``.  Reading never executes anything from the file."""

    header: dict[str, Any]
    tensors: dict[str, torch.Tensor]
    path: Path | None = None

    # -- convenience accessors ------------------------------------------
    @property
    def format_version(self) -> int:
        return int(self.header["format_version"])

    @property
    def model_uuid(self) -> str:
        return self.header["model_uuid"]

    @property
    def model_type(self) -> str:
        return self.header["model"]["type"]

    @property
    def sample_rate(self) -> int:
        return int(self.header["model"]["sample_rate"])

    @property
    def hparams(self) -> dict[str, Any]:
        return dict(self.header["model"]["hparams"])

    @property
    def schema(self) -> ConditioningSchema:
        return ConditioningSchema.from_dict(self.header.get("conditioning"))

    @property
    def metadata(self) -> FBMXMetadata:
        return FBMXMetadata.from_dict(self.header.get("metadata"))

    @property
    def normalization(self) -> Normalization:
        return Normalization.from_dict(self.header.get("normalization"))

    def state_dict(self) -> dict[str, torch.Tensor]:
        return dict(self.tensors)

    def build_model(self, device: torch.device | str = "cpu") -> StreamingModel:
        """Reconstruct a PyTorch model from the file alone.

        This is the reference implementation of what the Rust runtime will do;
        it is used by the round-trip tests to prove the container is complete.
        """
        from fbmx.models import build_model as _build

        cfg = dict(self.hparams)
        cfg["type"] = self.model_type
        model = _build(cfg, self.schema)
        model.load_state_dict(self.state_dict())
        return model.to(device).eval()

    def summary(self) -> str:
        m = self.header["model"]
        meta = self.metadata
        lines = [
            f"format         fbmx v{self.format_version}",
            f"uuid           {self.model_uuid}",
            f"model          {m['type']}  ({m['parameter_count']:,} parameters)",
            f"sample rate    {m['sample_rate']} Hz, {m['channels']} ch, "
            f"causal={m['causal']} recurrent={m['recurrent']} rf={m['receptive_field']}",
            f"hparams        {json.dumps(m['hparams'])}",
            f"conditioning   {self.schema}",
            f"normalization  {json.dumps(self.header.get('normalization', {}))}",
            f"tensors        {len(self.tensors)} "
            f"({self.header['checksums']['data_nbytes']:,} bytes)",
            f"source type    {meta.model_source_type}",
            f"licence        {meta.license or '(unspecified)'}",
            f"validated      {meta.validated}",
        ]
        return "\n".join(lines)


def read_fbmx(path: str | Path, verify: bool = True) -> FBMXFile:
    raw = Path(path).read_bytes()
    if len(raw) < 16 + TRAILER_SIZE:
        raise ValueError(f"{path}: too short to be a .fbmx file")
    if raw[:4] != MAGIC:
        raise ValueError(f"{path}: bad magic {raw[:4]!r}, expected {MAGIC!r}")
    (version,) = struct.unpack_from("<I", raw, 4)
    if version != FBMX_FORMAT_VERSION:
        raise ValueError(
            f"{path}: format version {version}, this build reads {FBMX_FORMAT_VERSION}"
        )
    (header_len,) = struct.unpack_from("<Q", raw, 8)
    header_end = 16 + header_len
    if header_end > len(raw) - TRAILER_SIZE:
        raise ValueError(f"{path}: header length {header_len} runs past end of file")
    header = json.loads(raw[16:header_end].decode("utf-8"))

    data_offset = header_end + ((-header_end) % HEADER_ALIGN)
    body_end = len(raw) - TRAILER_SIZE
    data = raw[data_offset:body_end]

    if verify:
        expected_file = hashlib.sha256(raw[:body_end]).digest()
        if expected_file != raw[body_end:]:
            raise ValueError(f"{path}: file checksum mismatch (truncated or modified)")
        expected_data = header.get("checksums", {}).get("data_sha256")
        if expected_data and hashlib.sha256(data).hexdigest() != expected_data:
            raise ValueError(f"{path}: tensor-data checksum mismatch")

    tensors: dict[str, torch.Tensor] = {}
    for entry in header.get("tensors", []):
        dtype = DTYPES.get(entry["dtype"])
        if dtype is None:
            raise ValueError(f"{path}: tensor {entry['name']!r} has unknown dtype {entry['dtype']!r}")
        start = int(entry["offset"])
        nbytes = int(entry["nbytes"])
        chunk = data[start : start + nbytes]
        if len(chunk) != nbytes:
            raise ValueError(f"{path}: tensor {entry['name']!r} runs past end of data")
        array = np.frombuffer(chunk, dtype=dtype).reshape(entry["shape"])
        tensors[entry["name"]] = torch.from_numpy(array.copy())

    return FBMXFile(header=header, tensors=tensors, path=Path(path))


# ---------------------------------------------------------------------------
# checkpoint -> fbmx
# ---------------------------------------------------------------------------
def export_from_checkpoint(
    checkpoint_path: str | Path,
    output_path: str | Path,
    *,
    metadata: FBMXMetadata | Mapping[str, Any] | None = None,
    normalization: Normalization | None = None,
) -> tuple[Path, FBMXFile]:
    """Load a development checkpoint and write the distributable container.

    Provenance defaults are pulled from the checkpoint's recorded dataset info,
    so an export cannot accidentally claim a cleaner origin than the training
    run had.
    """
    from fbmx.training.checkpoint import load_checkpoint, model_from_checkpoint

    ckpt = load_checkpoint(checkpoint_path, map_location="cpu")
    model = model_from_checkpoint(ckpt, device="cpu").eval()

    dataset_info = (ckpt.get("extra", {}).get("dataset", {}) or {}).get("dataset", {})
    defaults: dict[str, Any] = {
        "name": Path(output_path).stem,
        "license": dataset_info.get("license", ""),
        "license_url": dataset_info.get("license_url", ""),
        "attribution": dataset_info.get("attribution", ""),
        "model_source_type": _source_type_for(dataset_info.get("source_type", "synthetic")),
        "dataset": dataset_info,
        "training": {
            "epochs": ckpt.get("epoch"),
            "global_step": ckpt.get("global_step"),
            "monitor": ckpt.get("monitor"),
            "best_metric": ckpt.get("best_metric"),
            "metrics": ckpt.get("metrics", {}),
            "trainer": ckpt.get("extra", {}).get("trainer", {}),
            "torch_version": ckpt.get("torch_version"),
        },
        "notes": dataset_info.get("notes", ""),
    }
    merged = dict(defaults)
    if metadata is not None:
        provided = metadata.to_dict() if isinstance(metadata, FBMXMetadata) else dict(metadata)
        merged.update({k: v for k, v in provided.items() if v not in (None, "", [], {})})

    path = write_fbmx(
        output_path,
        model,
        FBMXMetadata.from_dict(merged),
        normalization=normalization,
        extra={"checkpoint": str(Path(checkpoint_path))},
    )
    return path, read_fbmx(path)


def _source_type_for(dataset_source_type: str) -> str:
    """A model's source type follows its data's, unless told otherwise."""
    return dataset_source_type if dataset_source_type in SOURCE_TYPES else "synthetic"
