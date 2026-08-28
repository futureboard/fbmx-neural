"""Command line entry point for the VSCO 2 CE violin pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .preprocess import (
    build_manifests,
    build_summary,
    config_hash,
    load_config,
    preprocess_records,
    read_jsonl,
    resolve_workspace_path,
    validate_processed_records,
    write_summary,
)
from .scan import scan_source, write_scan_report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare VSCO 2 CE Solo Violin data for Solfege/FBMX")
    parser.add_argument("--config", type=Path, help="versioned YAML configuration")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("scan", "preprocess", "prepare"):
        command = sub.add_parser(name, help={"scan": "scan WAV metadata", "preprocess": "write processed WAVs", "prepare": "run the complete pipeline"}[name])
        command.add_argument("--source", type=Path, help="immutable source dataset root")
        command.add_argument("--dataset-root", type=Path, help="normalized output root")
        if name == "scan":
            command.add_argument("--max-files", type=int)
    manifest = sub.add_parser("build-manifest", help="build JSONL manifests from processed records")
    manifest.add_argument("--dataset-root", type=Path, help="normalized output root")
    return parser


def _print_summary(summary: dict[str, Any]) -> None:
    print(f"dataset: {summary.get('dataset')} ({summary.get('dataset_version')})")
    print(f"processed: {summary.get('total_files')}")
    print(f"scan discovered: {summary.get('scan_discovered_wavs')}")
    print(f"scan rejected: {summary.get('scan_rejected')}")
    print(f"preprocess rejected: {summary.get('preprocess_rejected')}")
    print(f"sample rates: {summary.get('sample_rate_distribution')}")
    print(f"channels: {summary.get('channel_distribution')}")
    print(f"articulations: {summary.get('articulation_distribution')}")
    print(f"dynamics: {summary.get('dynamic_distribution')}")
    print(f"splits: {summary.get('split_counts', {})}")


def _source_and_root(config: dict[str, Any], source: Path | None, dataset_root: Path | None) -> tuple[Path, Path]:
    return (
        resolve_workspace_path(source or config["source"]),
        resolve_workspace_path(dataset_root or config["dataset_root"]),
    )


def _run_scan(config: dict[str, Any], source: Path | None, dataset_root: Path | None, max_files: int | None = None) -> dict[str, Any]:
    source_root, output_root = _source_and_root(config, source, dataset_root)
    report = scan_source(
        source_root,
        pitch_detection_enabled=bool(config.get("pitch_detection", {}).get("enabled", True)),
        max_files=max_files,
    )
    report["config_hash"] = config_hash(config)
    write_scan_report(report, output_root / "metadata" / "scan-report.json")
    print(json.dumps(report["counts"], sort_keys=True))
    return report


def _run_preprocess(config: dict[str, Any], source: Path | None, dataset_root: Path | None) -> dict[str, Any]:
    source_root, output_root = _source_and_root(config, source, dataset_root)
    report_path = output_root / "metadata" / "scan-report.json"
    if report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
    else:
        report = _run_scan(config, source, dataset_root)
    processed, rejected = preprocess_records(report["records"], source_root, output_root, config)
    payload = {
        "schema_version": 1,
        "pipeline_version": report["pipeline_version"],
        "config_hash": config_hash(config),
        "source_dataset_hash": report["source_dataset_hash"],
        "records": sorted(processed, key=lambda row: row["id"]),
        "rejections": rejected,
    }
    output_root.joinpath("metadata").mkdir(parents=True, exist_ok=True)
    output_root.joinpath("metadata", "processed-records.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    errors = validate_processed_records(processed, output_root)
    if errors:
        raise SystemExit(f"processed validation failed: {errors[:3]}")
    summary = build_summary(report, processed, rejected, config)
    write_summary(summary, output_root / "metadata" / "violin-summary.json")
    print(f"processed {len(processed)} files; rejected {len(rejected)}")
    return {"report": report, "processed": processed, "rejected": rejected, "summary": summary}


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = load_config(args.config)
    if args.command == "scan":
        _run_scan(config, args.source, args.dataset_root, args.max_files)
        return 0
    if args.command == "preprocess":
        _run_preprocess(config, args.source, args.dataset_root)
        return 0
    if args.command == "prepare":
        from .preprocess import run_prepare

        result = run_prepare(config, source=args.source, dataset_root=args.dataset_root)
        _print_summary(result["summary"])
        return 0
    if args.command == "build-manifest":
        output_root = resolve_workspace_path(args.dataset_root or config["dataset_root"])
        payload_path = output_root / "metadata" / "processed-records.json"
        if not payload_path.exists():
            raise SystemExit(f"missing {payload_path}; run preprocess or prepare first")
        payload = json.loads(payload_path.read_text(encoding="utf-8"))
        errors = validate_processed_records(payload["records"], output_root)
        if errors:
            raise SystemExit(f"processed validation failed: {errors[:3]}")
        counts = build_manifests(payload["records"], output_root, config)
        print(f"wrote manifests: {counts}")
        return 0
    raise SystemExit(f"unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
