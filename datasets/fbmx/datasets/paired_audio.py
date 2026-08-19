"""Manifest-driven dry/wet audio dataset.

This is the adapter every real corpus goes through -- teacher DSP renders,
hardware captures, licensed or public datasets.  It reads a
:class:`~fbmx.datasets.manifest.DatasetManifest`, so acquiring the audio is
somebody else's problem and the repository never has to contain it.

Deliberate non-features:

* **No downloading.**  Nothing in this package fetches a dataset.  Obtaining
  the audio, and checking that its licence permits what you intend to do with
  the resulting model, is a human step on purpose.
* **No resampling.**  A file whose rate differs from the manifest's is an
  error, not something to silently fix; resampling a nonlinear effect's
  training pair changes the target.
* **No stereo.**  V0 is mono; a multichannel file must be split by the capture
  tooling, which knows whether the channels are two takes or one stereo image.

WAV reading uses a small built-in RIFF parser (16/24/32-bit PCM and 32/64-bit
float) so that ``soundfile``/``libsndfile`` stays an optional dependency.  If
``soundfile`` is installed it is preferred, which brings FLAC and AIFF along.
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from fbmx.datasets.base import PairedItem, PairedSequenceDataset
from fbmx.datasets.manifest import DatasetManifest, ManifestEntry

__all__ = ["PairedAudioDataset", "read_audio", "read_wav", "write_wav"]

try:  # optional, never required
    import soundfile as _soundfile
except Exception:  # pragma: no cover - depends on the environment
    _soundfile = None


def read_wav(path: str | Path) -> tuple[np.ndarray, int]:
    """Minimal RIFF/WAVE reader.  Returns ``(samples [channels, n], rate)``."""
    data = Path(path).read_bytes()
    if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise ValueError(f"{path}: not a RIFF/WAVE file")

    pos = 12
    fmt: tuple[int, int, int, int] | None = None
    audio: np.ndarray | None = None
    while pos + 8 <= len(data):
        chunk_id = data[pos : pos + 4]
        (chunk_size,) = struct.unpack_from("<I", data, pos + 4)
        body = pos + 8
        if chunk_id == b"fmt ":
            audio_format, channels, rate, _, _, bits = struct.unpack_from(
                "<HHIIHH", data, body
            )
            if audio_format == 0xFFFE and chunk_size >= 40:  # WAVE_FORMAT_EXTENSIBLE
                (audio_format,) = struct.unpack_from("<H", data, body + 24)
            fmt = (audio_format, channels, rate, bits)
        elif chunk_id == b"data":
            if fmt is None:
                raise ValueError(f"{path}: data chunk before fmt chunk")
            audio_format, channels, rate, bits = fmt
            raw = data[body : body + chunk_size]
            if audio_format == 1 and bits == 16:
                x = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
            elif audio_format == 1 and bits == 24:
                b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
                packed = (b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)).astype(np.int32)
                packed = np.where(packed & 0x800000, packed - (1 << 24), packed)
                x = packed.astype(np.float32) / 8388608.0
            elif audio_format == 1 and bits == 32:
                x = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
            elif audio_format == 3 and bits == 32:
                x = np.frombuffer(raw, dtype="<f4").astype(np.float32)
            elif audio_format == 3 and bits == 64:
                x = np.frombuffer(raw, dtype="<f8").astype(np.float32)
            else:
                raise ValueError(
                    f"{path}: unsupported WAV format tag {audio_format} @ {bits} bits"
                )
            audio = x.reshape(-1, channels).T.copy()
        pos = body + chunk_size + (chunk_size & 1)  # chunks are word aligned

    if audio is None or fmt is None:
        raise ValueError(f"{path}: no data chunk")
    return audio, fmt[2]


def write_wav(path: str | Path, samples: np.ndarray, sample_rate: int) -> Path:
    """Write 32-bit float WAV.

    Float, not 24-bit PCM: a teacher render or a model output can legitimately
    exceed 0 dBFS, and quietly clipping the training target is the kind of bug
    that costs a week.  Accepts ``[n]`` or ``[channels, n]``.
    """
    samples = np.asarray(samples, dtype=np.float32)
    if samples.ndim == 1:
        samples = samples[None, :]
    channels, n = samples.shape
    interleaved = samples.T.copy(order="C").tobytes()
    block_align = channels * 4
    fmt = struct.pack(
        "<4sIHHIIHH",
        b"fmt ",
        16,
        3,  # IEEE float
        channels,
        sample_rate,
        sample_rate * block_align,
        block_align,
        32,
    )
    data = b"data" + struct.pack("<I", len(interleaved)) + interleaved
    riff = b"RIFF" + struct.pack("<I", 4 + len(fmt) + len(data)) + b"WAVE"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(riff + fmt + data)
    return path


def read_audio(path: str | Path) -> tuple[np.ndarray, int]:
    """``(samples [channels, n], rate)``, via soundfile when available."""
    path = Path(path)
    if _soundfile is not None and path.suffix.lower() != ".wav":
        x, rate = _soundfile.read(str(path), dtype="float32", always_2d=True)
        return x.T.copy(), int(rate)
    return read_wav(path)


class PairedAudioDataset(PairedSequenceDataset):
    """Dry/wet pairs listed by a manifest.

    ``segment_length`` cuts each entry into consecutive equal-length sequences
    (the last partial one is dropped).  Cuts are *contiguous and ordered*: they
    are units of TBPTT, not random crops, and the trainer resets the recurrent
    state at each item boundary, so make them long enough to contain the
    effect's memory several times over.
    """

    def __init__(
        self,
        manifest: DatasetManifest | str | Path,
        split: str | None = "train",
        segment_length: int | None = None,
        max_entries: int | None = None,
        verify_checksums: bool = False,
        aux_traces: Sequence[str] | None = None,
        expect: Mapping[str, Any] | None = None,
    ) -> None:
        if not isinstance(manifest, DatasetManifest):
            manifest = DatasetManifest.load(manifest)
        manifest.validate(check_files=True, check_checksums=verify_checksums)
        super().__init__(manifest.info, manifest.schema)
        self.manifest = manifest
        self.split = split
        self.segment_length = segment_length
        self.aux_traces = list(aux_traces or [])
        _check_expectations(manifest, expect)

        entries = manifest.select(split)
        if max_entries is not None:
            entries = entries[:max_entries]
        if not entries:
            raise ValueError(f"manifest has no entries for split {split!r}")
        self.entries = entries
        self._index: list[tuple[int, int, int]] = []  # (entry, start, length)
        self._cache: dict[int, tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]] = {}
        self._build_index()

    def _load_entry(
        self, i: int
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        if i in self._cache:
            return self._cache[i]
        entry = self.entries[i]
        dry_path, wet_path = entry.resolved(self.manifest.root)
        dry, dry_rate = read_audio(dry_path)
        wet, wet_rate = read_audio(wet_path)
        for rate, path in ((dry_rate, dry_path), (wet_rate, wet_path)):
            if rate != self.sample_rate:
                raise ValueError(
                    f"{entry.key}: {path} is {rate} Hz but the manifest declares "
                    f"{self.sample_rate} Hz; resample deliberately, upstream"
                )
        for arr, path in ((dry, dry_path), (wet, wet_path)):
            if arr.shape[0] != 1:
                raise ValueError(f"{entry.key}: {path} has {arr.shape[0]} channels, need mono")
        n = min(dry.shape[1], wet.shape[1])
        if abs(dry.shape[1] - wet.shape[1]) > self.sample_rate // 100:
            raise ValueError(
                f"{entry.key}: dry/wet lengths differ by more than 10 ms "
                f"({dry.shape[1]} vs {wet.shape[1]}); alignment is a capture-time job"
            )
        aux: dict[str, torch.Tensor] = {}
        for name in self.aux_traces:
            rel = entry.aux.get(name)
            if rel is None:
                raise KeyError(
                    f"{entry.key}: aux trace {name!r} was requested but the manifest "
                    f"only provides {sorted(entry.aux)}; a teacher that cannot export "
                    f"it must not have the corresponding loss enabled"
                )
            trace, trace_rate = read_audio(self.manifest.root / rel)
            if trace_rate != self.sample_rate:
                raise ValueError(f"{entry.key}: aux {name!r} is {trace_rate} Hz, not {self.sample_rate}")
            if trace.shape[1] < n:
                raise ValueError(
                    f"{entry.key}: aux {name!r} is {trace.shape[1]} samples, shorter than "
                    f"the {n}-sample audio; traces must be sample-aligned with the audio"
                )
            aux[name] = torch.from_numpy(trace[:, :n]).float()

        pair = (
            torch.from_numpy(dry[:, :n]).float(),
            torch.from_numpy(wet[:, :n]).float(),
            aux,
        )
        self._cache[i] = pair
        return pair

    def _build_index(self) -> None:
        for i in range(len(self.entries)):
            dry, _, _ = self._load_entry(i)
            n = int(dry.shape[-1])
            if self.segment_length is None:
                self._index.append((i, 0, n))
            else:
                for start in range(0, n - self.segment_length + 1, self.segment_length):
                    self._index.append((i, start, self.segment_length))
        if not self._index:
            raise ValueError("no usable segments; is segment_length longer than the audio?")

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, index: int) -> PairedItem:
        entry_index, start, length = self._index[index]
        entry: ManifestEntry = self.entries[entry_index]
        dry, wet, aux = self._load_entry(entry_index)
        sl = slice(start, start + length)
        return PairedItem(
            dry=dry[:, sl].contiguous(),
            wet=wet[:, sl].contiguous(),
            params=self.schema.encode(entry.params),
            key=f"{entry.key}@{start}",
            aux={name: trace[:, sl].contiguous() for name, trace in aux.items()},
        )

    def provenance(self) -> dict[str, Any]:
        p = super().provenance()
        p["entries"] = [e.key for e in self.entries]
        p["split"] = self.split
        p["aux_traces"] = list(self.aux_traces)
        return p


def _check_expectations(
    manifest: DatasetManifest, expect: Mapping[str, Any] | None
) -> None:
    """Refuse a manifest that is not the dataset the config asked for.

    A config that says "train on the Rev D circuit teacher" must fail loudly
    when pointed at a hardware capture or a different revision, rather than
    training happily and producing a model whose provenance claim is false.
    Keys are matched against ``DatasetInfo`` fields first, then its ``extra``
    block, so ``source_type``, ``teacher``, ``revision`` and
    ``generator_version`` all work.
    """
    for key, wanted in (expect or {}).items():
        if hasattr(manifest.info, key):
            actual = getattr(manifest.info, key)
        elif key in manifest.info.extra:
            actual = manifest.info.extra[key]
        else:
            raise ValueError(
                f"manifest declares no {key!r}; the config expected {wanted!r}. "
                f"Known fields: {sorted(set(manifest.info.to_dict()) | set(manifest.info.extra))}"
            )
        if str(actual) != str(wanted):
            raise ValueError(
                f"manifest {key} is {actual!r} but the config requires {wanted!r}; "
                f"refusing to train on a dataset that is not the one asked for"
            )
