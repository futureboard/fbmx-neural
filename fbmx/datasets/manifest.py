"""Dataset manifests.

A manifest is a JSON file that says which audio files pair with which, at what
parameter setting, in which split, plus the :class:`DatasetInfo` provenance
block.  It is the only thing a dataset adapter needs in order to be reproducible
by somebody who has obtained the audio through the proper channel.

The manifest is checked in; the audio is not.  That is the point -- it lets a
capture session, a licensed corpus and a teacher render all be described in the
repository without redistributing a single sample of anybody's audio.

Format::

    {
      "fbmx_manifest_version": 1,
      "info": { ...DatasetInfo... },
      "conditioning": [ {"kind": "continuous", "name": "drive", ...}, ... ],
      "entries": [
        {"key": "sweep_01", "split": "train",
         "dry": "audio/sweep_01_dry.wav", "wet": "audio/sweep_01_wet.wav",
         "params": {"drive": 0.4, "mode": "soft"},
         "dry_sha256": "...", "wet_sha256": "..."}
      ]
    }

Paths are relative to the manifest file unless absolute.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from fbmx.conditioning import ConditioningSchema
from fbmx.datasets.base import DatasetInfo, file_sha256

__all__ = ["MANIFEST_VERSION", "ManifestEntry", "DatasetManifest"]

MANIFEST_VERSION = 1


@dataclass
class ManifestEntry:
    key: str
    dry: str
    wet: str
    split: str = "train"
    params: dict[str, Any] = field(default_factory=dict)
    #: Optional auxiliary target traces, ``{name: path}`` -- a gain-reduction
    #: or control-voltage trace exported by a teacher that can produce one.
    #: Same length and rate as the audio; see ``fbmx.losses.auxiliary``.
    aux: dict[str, str] = field(default_factory=dict)
    aux_sha256: dict[str, str] = field(default_factory=dict)
    dry_sha256: str = ""
    wet_sha256: str = ""
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: Mapping[str, Any]) -> "ManifestEntry":
        missing = {"key", "dry", "wet"} - set(d)
        if missing:
            raise ValueError(f"manifest entry missing fields: {sorted(missing)}")
        known = set(ManifestEntry.__dataclass_fields__)
        return ManifestEntry(**{k: v for k, v in d.items() if k in known})

    def resolved(self, root: Path) -> tuple[Path, Path]:
        return root / self.dry, root / self.wet

    def resolved_aux(self, root: Path) -> dict[str, Path]:
        return {name: root / rel for name, rel in self.aux.items()}


@dataclass
class DatasetManifest:
    info: DatasetInfo
    entries: list[ManifestEntry] = field(default_factory=list)
    schema: ConditioningSchema = field(default_factory=ConditioningSchema)
    root: Path = field(default_factory=lambda: Path("."))
    version: int = MANIFEST_VERSION

    # -- io --------------------------------------------------------------
    @staticmethod
    def from_dict(d: Mapping[str, Any], root: Path | str = ".") -> "DatasetManifest":
        version = int(d.get("fbmx_manifest_version", 0))
        if version != MANIFEST_VERSION:
            raise ValueError(
                f"unsupported manifest version {version}, expected {MANIFEST_VERSION}"
            )
        if "info" not in d:
            raise ValueError("manifest has no 'info' block; provenance is mandatory")
        return DatasetManifest(
            info=DatasetInfo.from_dict(d["info"]),
            entries=[ManifestEntry.from_dict(e) for e in d.get("entries", [])],
            schema=ConditioningSchema.from_config(d.get("conditioning", [])),
            root=Path(root),
            version=version,
        )

    def to_dict(self) -> dict[str, Any]:
        cond = self.schema.to_dict()
        return {
            "fbmx_manifest_version": self.version,
            "info": self.info.to_dict(),
            "conditioning": cond["continuous"] + cond["categorical"],
            "entries": [e.to_dict() for e in self.entries],
        }

    @staticmethod
    def load(path: str | Path) -> "DatasetManifest":
        path = Path(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        return DatasetManifest.from_dict(data, root=path.parent)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path

    # -- queries ---------------------------------------------------------
    def __len__(self) -> int:
        return len(self.entries)

    def __iter__(self) -> Iterator[ManifestEntry]:
        return iter(self.entries)

    @property
    def splits(self) -> list[str]:
        return sorted({e.split for e in self.entries})

    def select(self, split: str | None = None) -> list[ManifestEntry]:
        if split is None:
            return list(self.entries)
        return [e for e in self.entries if e.split == split]

    # -- integrity -------------------------------------------------------
    def validate(self, *, check_files: bool = False, check_checksums: bool = False) -> None:
        """Structural check; optionally that the audio exists and matches."""
        keys = [e.key for e in self.entries]
        if len(set(keys)) != len(keys):
            dupes = sorted({k for k in keys if keys.count(k) > 1})
            raise ValueError(f"duplicate manifest keys: {dupes}")
        for e in self.entries:
            unknown = set(e.params) - set(self.schema.names)
            if unknown:
                raise ValueError(
                    f"{e.key}: parameters not in the manifest schema: {sorted(unknown)}"
                )
            # Round-trips through the schema, so an out-of-range or misspelled
            # category is caught here rather than at epoch 3.
            self.schema.encode(e.params)
            if check_files or check_checksums:
                for path in (*e.resolved(self.root), *e.resolved_aux(self.root).values()):
                    if not path.exists():
                        raise FileNotFoundError(f"{e.key}: missing audio {path}")
            if check_checksums:
                dry_path, wet_path = e.resolved(self.root)
                targets = [
                    (dry_path, e.dry_sha256, "dry"),
                    (wet_path, e.wet_sha256, "wet"),
                ]
                targets += [
                    (path, e.aux_sha256.get(name, ""), f"aux:{name}")
                    for name, path in e.resolved_aux(self.root).items()
                ]
                for path, expected, which in targets:
                    if not expected:
                        continue
                    actual = file_sha256(path)
                    if actual != expected:
                        raise ValueError(
                            f"{e.key}: {which} checksum mismatch\n"
                            f"  expected {expected}\n  actual   {actual}"
                        )

    def fill_checksums(self) -> "DatasetManifest":
        """Hash every referenced file and record it.  Run once, at capture."""
        for e in self.entries:
            dry_path, wet_path = e.resolved(self.root)
            e.dry_sha256 = file_sha256(dry_path)
            e.wet_sha256 = file_sha256(wet_path)
            e.aux_sha256 = {
                name: file_sha256(path) for name, path in e.resolved_aux(self.root).items()
            }
        return self

    @staticmethod
    def build(
        info: DatasetInfo,
        schema: ConditioningSchema,
        entries: Sequence[ManifestEntry],
        root: Path | str = ".",
    ) -> "DatasetManifest":
        m = DatasetManifest(info=info, entries=list(entries), schema=schema, root=Path(root))
        m.validate()
        return m
