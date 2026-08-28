"""Find the pieces, parts, and files URMP actually ships.

Everything here reads the layout off disk rather than assuming it. URMP's
directory names encode the ensemble (``13_Hark_vn_vn_va``) and its files encode
the part index and instrument (``AuSep_2_vn_13_Hark.wav``), but the only
trustworthy statement about a piece is the set of files that exist in it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

#: URMP's instrument abbreviations, expanded. Taken from the dataset's own
#: naming, not guessed: every token below appears in a shipped directory name.
INSTRUMENTS = {
    "vn": "violin",
    "va": "viola",
    "vc": "cello",
    "db": "double bass",
    "fl": "flute",
    "ob": "oboe",
    "cl": "clarinet",
    "sax": "saxophone",
    "bn": "bassoon",
    "tpt": "trumpet",
    "hn": "horn",
    "tbn": "trombone",
    "tba": "tuba",
}

VIOLIN = "vn"

_PIECE_DIR_RE = re.compile(r"^(?P<number>\d+)_(?P<title>[^_]+)_(?P<parts>.+)$")
_PART_FILE_RE = re.compile(
    r"^(?P<kind>AuSep|F0s|Notes)_(?P<part>\d+)_(?P<instrument>[a-z]+)_(?P<number>\d+)_(?P<title>.+)\.(?P<ext>wav|txt)$"
)


def _is_apple_double(path: Path) -> bool:
    """macOS resource forks shipped inside the archive.

    They sit beside every real file with a ``._`` prefix and are a few hundred
    bytes of AppleDouble metadata. Reading one as a WAV or as an annotation
    table produces nonsense, so they are excluded everywhere by name.
    """

    return path.name.startswith("._")


@dataclass(frozen=True)
class Part:
    """One player's part within one piece."""

    piece_id: str
    index: int
    instrument: str
    audio: Path | None
    f0: Path | None
    notes: Path | None

    @property
    def part_id(self) -> str:
        return f"{self.piece_id}#{self.index}"

    @property
    def is_complete(self) -> bool:
        """Whether this part has everything the performance pipeline needs."""

        return self.audio is not None and self.f0 is not None and self.notes is not None

    def missing(self) -> list[str]:
        return [
            name
            for name, value in (("audio", self.audio), ("f0", self.f0), ("notes", self.notes))
            if value is None
        ]


@dataclass(frozen=True)
class Piece:
    """One ensemble recording."""

    piece_id: str
    number: int
    title: str
    directory: Path
    score_midi: Path | None
    mix_audio: Path | None
    parts: list[Part] = field(default_factory=list)

    def parts_of(self, instrument: str) -> list[Part]:
        return [part for part in self.parts if part.instrument == instrument]


def discover(root: str | Path) -> list[Piece]:
    """Return every piece under ``root``, ordered by piece number.

    Directories that do not parse as a URMP piece (the supplementary images and
    documentation) are skipped rather than reported as broken pieces.
    """

    base = Path(root)
    if not base.is_dir():
        raise FileNotFoundError(f"URMP root does not exist: {base}")

    pieces: list[Piece] = []
    for directory in sorted(base.iterdir(), key=lambda p: p.name):
        if not directory.is_dir() or _is_apple_double(directory):
            continue
        match = _PIECE_DIR_RE.match(directory.name)
        if match is None:
            continue
        number = int(match.group("number"))
        title = match.group("title")
        piece_id = directory.name

        by_index: dict[int, dict[str, Path]] = {}
        instruments: dict[int, str] = {}
        score_midi: Path | None = None
        mix_audio: Path | None = None

        for path in sorted(directory.iterdir(), key=lambda p: p.name):
            if not path.is_file() or _is_apple_double(path):
                continue
            name = path.name
            if name.startswith("Sco_") and path.suffix.casefold() == ".mid":
                score_midi = path
                continue
            if name.startswith("AuMix_") and path.suffix.casefold() == ".wav":
                mix_audio = path
                continue
            part_match = _PART_FILE_RE.match(name)
            if part_match is None:
                continue
            index = int(part_match.group("part"))
            by_index.setdefault(index, {})[part_match.group("kind")] = path
            instruments[index] = part_match.group("instrument")

        parts = [
            Part(
                piece_id=piece_id,
                index=index,
                instrument=instruments[index],
                audio=by_index[index].get("AuSep"),
                f0=by_index[index].get("F0s"),
                notes=by_index[index].get("Notes"),
            )
            for index in sorted(by_index)
        ]
        pieces.append(
            Piece(
                piece_id=piece_id,
                number=number,
                title=title,
                directory=directory,
                score_midi=score_midi,
                mix_audio=mix_audio,
                parts=parts,
            )
        )

    pieces.sort(key=lambda piece: piece.number)
    return pieces


def violin_parts(pieces: list[Piece]) -> list[Part]:
    """Every violin part across every piece, in a deterministic order."""

    return [part for piece in pieces for part in piece.parts_of(VIOLIN)]
