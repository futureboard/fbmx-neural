"""Split URMP violin notes without leaking a performance across sets.

The unit of splitting is a *session cluster*: the transitive closure of two
relations over parts. Notes cannot be split individually — neighbouring notes of
one phrase share tempo, player, room, and instrument, so a random note split
measures interpolation inside a recording the model has already seen.

Two things have to travel together:

* **A reused take.** URMP puts the same violin recording under several piece
  names — `35_Rondeau_vn_vn_va_db`, `36_Rondeau_vn_vn_va_vc` and
  `37_Rondeau_fl_vn_va_cl` are one performance — so the groups computed by
  :func:`build.performance_groups` are indivisible.
* **A shared piece.** `02_Sonata_vn_vn` is two violinists playing a duet. They
  are different performances, but they played the same music in the same room at
  the same time, so putting one in training and the other in test would let the
  model meet the piece before it is asked to generalise to it.

Taking the closure of both is what makes "unseen" mean unseen.

The split is deterministic: same groups in, same assignment out, recorded in the
report so a checkpoint can name the split it was trained under.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

SPLIT_SCHEMA_VERSION = 1

#: Target share of notes per split. Approximate — groups are indivisible, so
#: the greedy assignment below lands near these rather than on them.
DEFAULT_TARGETS = {"train": 0.70, "validation": 0.15, "test": 0.15}


def session_clusters(records: Iterable[dict[str, Any]]) -> dict[str, str]:
    """Map each performance group to the cluster it must be split with.

    Unions a group with every piece it appears in, and every piece with every
    group that appears in it, then takes the closure. A group reused across
    three pieces drags all three in; a piece with two players drags both of
    their groups in.
    """

    parent: dict[str, str] = {}

    def find(key: str) -> str:
        parent.setdefault(key, key)
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    def union(a: str, b: str) -> None:
        root_a, root_b = find(a), find(b)
        if root_a != root_b:
            parent[root_b] = root_a

    groups: set[str] = set()
    for record in records:
        group = record["group_id"]
        groups.add(group)
        union(f"g:{group}", f"p:{record['piece_id']}")

    canonical: dict[str, str] = {}
    out: dict[str, str] = {}
    for group in sorted(groups):
        root = find(f"g:{group}")
        if root not in canonical:
            canonical[root] = f"session{len(canonical):03d}"
        out[group] = canonical[root]
    return out


def _cluster_note_counts(
    records: Iterable[dict[str, Any]], clusters: dict[str, str]
) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for record in records:
        counts[clusters[record["group_id"]]] += 1
    return dict(counts)


def _cluster_pieces(
    records: Iterable[dict[str, Any]], clusters: dict[str, str]
) -> dict[str, set[str]]:
    pieces: dict[str, set[str]] = defaultdict(set)
    for record in records:
        pieces[clusters[record["group_id"]]].add(record["piece_id"])
    return {cluster: set(names) for cluster, names in pieces.items()}


def assign(
    records: list[dict[str, Any]],
    *,
    targets: dict[str, float] | None = None,
    holdout_pieces: set[str] | None = None,
) -> dict[str, str]:
    """Map each performance group to a split name.

    Groups are placed largest-first into whichever split is furthest below its
    target share. Largest-first matters: leaving the biggest group until last
    forces it into whatever remains and skews the result badly when one group
    holds a tenth of the corpus.

    `holdout_pieces` forces any group touching those pieces into `test`, which
    is how the unseen-piece evaluation gets a piece the model has never met in
    any ensemble variant.
    """

    targets = dict(targets or DEFAULT_TARGETS)
    clusters = session_clusters(records)
    counts = _cluster_note_counts(records, clusters)
    pieces = _cluster_pieces(records, clusters)
    total = sum(counts.values()) or 1

    cluster_split: dict[str, str] = {}
    placed: dict[str, int] = {name: 0 for name in targets}

    if holdout_pieces:
        for cluster, names in pieces.items():
            if names & holdout_pieces:
                cluster_split[cluster] = "test"
                placed["test"] += counts[cluster]

    remaining = sorted(
        (cluster for cluster in counts if cluster not in cluster_split),
        key=lambda cluster: (-counts[cluster], cluster),
    )
    for cluster in remaining:
        deficits = {name: targets[name] - placed[name] / total for name in targets}
        # Ties break by split name so the result does not depend on dict order.
        best = max(sorted(deficits), key=lambda name: deficits[name])
        cluster_split[cluster] = best
        placed[best] += counts[cluster]

    return {group: cluster_split[cluster] for group, cluster in clusters.items()}


def apply_split(
    records: list[dict[str, Any]], assignment: dict[str, str]
) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        out[assignment[record["group_id"]]].append(record)
    return dict(out)


def write_split(
    dataset_dir: str | Path,
    *,
    targets: dict[str, float] | None = None,
    holdout_pieces: set[str] | None = None,
) -> dict[str, Any]:
    directory = Path(dataset_dir)
    records = [
        json.loads(line)
        for line in (directory / "notes.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assignment = assign(records, targets=targets, holdout_pieces=holdout_pieces)
    parts = apply_split(records, assignment)

    for name, rows in parts.items():
        path = directory / f"notes.{name}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, separators=(",", ":")) + "\n")

    clusters = session_clusters(records)
    counts = _cluster_note_counts(records, clusters)
    total = sum(counts.values()) or 1
    report = {
        "schema_version": SPLIT_SCHEMA_VERSION,
        "strategy": (
            "session cluster: closure of (parts sharing any of audio/notes/f0 "
            "file content) and (parts of the same piece)"
        ),
        "cluster_of_group": dict(sorted(clusters.items())),
        "holdout_pieces": sorted(holdout_pieces or []),
        "targets": targets or DEFAULT_TARGETS,
        "group_assignment": dict(sorted(assignment.items())),
        "splits": {
            name: {
                "notes": len(rows),
                "share": round(len(rows) / total, 4),
                "groups": sorted({row["group_id"] for row in rows}),
                "pieces": sorted({row["piece_id"] for row in rows}),
                "parts": sorted({row["part_id"] for row in rows}),
            }
            for name, rows in sorted(parts.items())
        },
    }

    # A split that shares a piece between sets is not a split. URMP's reuse of
    # takes makes this easy to get wrong, so it is checked rather than assumed.
    seen: dict[str, str] = {}
    for name, detail in report["splits"].items():
        for piece in detail["pieces"]:
            if piece in seen and seen[piece] != name:
                raise ValueError(
                    f"piece {piece} appears in both {seen[piece]} and {name}: "
                    "performance grouping failed to keep a shared take together"
                )
            seen[piece] = name

    (directory / "split-report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report
