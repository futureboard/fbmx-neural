"""Deterministic group-aware train/valid/test assignment."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import Any, Iterable, Mapping


def group_key(record: Mapping[str, Any]) -> str:
    """Keep pitch/articulation variants, including round robins, together."""

    pitch = record.get("midi_note")
    pitch_key = str(pitch) if pitch is not None else "unknown"
    return f"{pitch_key}|{record.get('articulation') or 'unknown'}"


def assign_splits(
    records: Iterable[Mapping[str, Any]],
    *,
    train: float = 0.8,
    valid: float = 0.1,
    test: float = 0.1,
    seed: int = 20260827,
) -> list[dict[str, Any]]:
    """Assign whole pitch/articulation groups without round-robin leakage."""

    rows = [dict(record) for record in records]
    total = len(rows)
    ratios = [float(train), float(valid), float(test)]
    if any(value < 0 for value in ratios) or sum(ratios) <= 0:
        raise ValueError("split ratios must be non-negative and have a positive sum")
    normalised = [value / sum(ratios) for value in ratios]
    targets = [int(round(total * value)) for value in normalised]
    targets[2] += total - sum(targets)
    names = ("train", "valid", "test")

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[group_key(row)].append(row)
    ordered = sorted(
        groups.items(),
        key=lambda item: (-len(item[1]), hashlib.sha256(f"{seed}:{item[0]}".encode()).hexdigest()),
    )
    counts = [0, 0, 0]
    for key, group in ordered:
        del key
        size = len(group)
        choices = sorted(
            range(3),
            key=lambda index: (
                max(0, counts[index] + size - targets[index]),
                (counts[index] + size) / max(1, targets[index]),
                index,
            ),
        )
        selected = choices[0]
        counts[selected] += size
        for row in group:
            row["split"] = names[selected]
    return sorted(rows, key=lambda row: row["id"])

