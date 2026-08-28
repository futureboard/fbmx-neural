"""Non-destructive VSCO preprocessing, validation, manifests, and reports."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import numpy as np

_NEURAL_ROOT = Path(__file__).resolve().parents[2]
if str(_NEURAL_ROOT) not in sys.path:
    sys.path.insert(0, str(_NEURAL_ROOT))

from .metadata import PIPELINE_VERSION
from .scan import SCAN_SCHEMA_VERSION, scan_source, summarize_records, write_scan_report
from .split import assign_splits

from fbmx.datasets.paired_audio import read_audio, write_wav


MANIFEST_SCHEMA_VERSION = 1
WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = WORKSPACE_ROOT / "neural" / "configs" / "vsco2_ce_violin.yaml"

DEFAULT_CONFIG: dict[str, Any] = {
    "version": 1,
    "dataset": "vsco2-ce",
    "dataset_version": "1.1.0",
    "source": "solfage-datasets/VSCO-2-CE",
    "dataset_root": "solfage-datasets/vsco2-ce",
    "instrument": "violin",
    "target_sample_rate": 48000,
    "preserve_channels": True,
    "trim": {
        "enabled": True,
        "threshold_db": -70.0,
        "pre_roll_ms": 20.0,
        "post_roll_ms": 100.0,
    },
    "normalization": {"enabled": False, "mode": "peak", "target_db": -1.0},
    "fade_ms": 3.0,
    "pitch_detection": {"enabled": True},
    "split": {"train": 0.8, "valid": 0.1, "test": 0.1, "seed": 20260827},
}


def _deep_merge(base: Mapping[str, Any], update: Mapping[str, Any]) -> dict[str, Any]:
    result = {key: value for key, value in base.items()}
    for key, value in update.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not path.exists():
        return dict(DEFAULT_CONFIG)
    text = path.read_text(encoding="utf-8")
    try:
        import yaml

        data = yaml.safe_load(text) or {}
    except ImportError:  # pragma: no cover - PyYAML is a declared dependency
        data = json.loads(text)
    if not isinstance(data, Mapping):
        raise ValueError(f"configuration must be a mapping: {path}")
    config = _deep_merge(DEFAULT_CONFIG, data)
    if int(config.get("version", 0)) != 1:
        raise ValueError(f"unsupported VSCO pipeline config version: {config.get('version')}")
    return config


def config_hash(config: Mapping[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def resolve_workspace_path(value: str | Path, *, base: Path = WORKSPACE_ROOT) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base / path).resolve()


def _stats(audio: np.ndarray) -> tuple[float, float, float]:
    audio = np.asarray(audio, dtype=np.float64)
    return (
        float(np.max(np.abs(audio))) if audio.size else 0.0,
        float(np.sqrt(np.mean(audio * audio))) if audio.size else 0.0,
        float(np.mean(audio)) if audio.size else 0.0,
    )


def trim_silence(
    audio: np.ndarray,
    sample_rate: int,
    *,
    threshold_db: float = -70.0,
    pre_roll_ms: float = 20.0,
    post_roll_ms: float = 100.0,
) -> tuple[np.ndarray, int, int]:
    """Trim only level below a relative threshold, retaining padded tails."""

    if audio.ndim != 2 or audio.shape[1] == 0:
        return audio, 0, 0
    envelope = np.max(np.abs(audio), axis=0)
    peak = float(np.max(envelope))
    if not math.isfinite(peak) or peak <= 0.0:
        return audio[:, :0], audio.shape[1], 0
    threshold = peak * (10.0 ** (float(threshold_db) / 20.0))
    active = np.flatnonzero(envelope >= threshold)
    if active.size == 0:
        return audio[:, :0], audio.shape[1], 0
    pre = max(0, int(round(sample_rate * pre_roll_ms / 1000.0)))
    post = max(0, int(round(sample_rate * post_roll_ms / 1000.0)))
    start = max(0, int(active[0]) - pre)
    end = min(audio.shape[1], int(active[-1]) + 1 + post)
    return audio[:, start:end], start, audio.shape[1] - end


def resample_audio(audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    """Deterministic 16-tap windowed-sinc resampler for offline preparation."""

    if source_rate == target_rate:
        return np.asarray(audio, dtype=np.float32).copy()
    if source_rate <= 0 or target_rate <= 0:
        raise ValueError("sample rates must be positive")
    channels, input_length = audio.shape
    output_length = max(1, int(round(input_length * target_rate / source_rate)))
    output = np.empty((channels, output_length), dtype=np.float32)
    half_taps = 8
    offsets = np.arange(-half_taps + 1, half_taps + 1, dtype=np.int64)
    ratio = target_rate / source_rate
    cutoff = min(1.0, ratio)
    chunk_size = 8192
    source = np.asarray(audio, dtype=np.float64)
    # Reflective padding keeps the window normalized at both boundaries. A
    # clipped index repeated the last sample can make a windowed-sinc kernel
    # nearly cancel at the final output samples and produce an enormous gain.
    pad_mode = "reflect" if input_length > half_taps else "edge"
    source = np.pad(source, ((0, 0), (half_taps, half_taps)), mode=pad_mode)
    for start in range(0, output_length, chunk_size):
        stop = min(output_length, start + chunk_size)
        positions = np.arange(start, stop, dtype=np.float64) / ratio
        centers = np.floor(positions).astype(np.int64)
        taps = centers[:, None] + offsets[None, :]
        # Indices address the *padded* array; distances must be measured in
        # unpadded source coordinates. Folding `half_taps` into the distance as
        # well evaluates the sinc and its window eight taps off centre, which is
        # not a low-pass interpolator at all: measured on pure tones it left
        # 20.8 dB (3 kHz) to 27.9 dB (440 Hz) of inharmonic imaging noise.
        indices = taps + half_taps
        distance = taps - positions[:, None]
        window = np.where(
            np.abs(distance) < half_taps,
            0.5 + 0.5 * np.cos(np.pi * distance / half_taps),
            0.0,
        )
        weights = np.sinc(distance * cutoff) * cutoff * window
        weights /= np.sum(weights, axis=1, keepdims=True).clip(1e-12)
        output[:, start:stop] = np.sum(source[:, indices] * weights[None, :, :], axis=2)
    return output


#: Level below which a boundary sample cannot click, so no fade is needed.
#: -60 dBFS is roughly the noise floor of the source recordings.
_CLICK_FLOOR = 10.0 ** (-60.0 / 20.0)


def _apply_fades(audio: np.ndarray, sample_rate: int, fade_ms: float) -> np.ndarray:
    """Fade only the boundaries that would otherwise click.

    A fade exists to stop a step discontinuity at a cut. It is not free: a
    3 ms ramp at 48 kHz is 144 samples. Measured on this source, every one of
    the 161 files begins at exactly ``0.0`` — there is no discontinuity to
    smooth — yet 123 of them reach above -60 dBFS *within* that 3 ms window,
    with a peak as high as 0.33. The unconditional fade-in was therefore
    attenuating the bow attack of three quarters of the library: the transient
    that carries the instrument's identity, softened before the audio ever
    reached the voicebank or the training target.

    So each end is faded only when it actually starts or ends on signal. A
    boundary already at the click floor has nothing to smooth away.
    """
    frames = min(int(round(sample_rate * max(0.0, fade_ms) / 1000.0)), audio.shape[1] // 2)
    if frames <= 0:
        return audio
    result = audio.copy()
    ramp = np.linspace(0.0, 1.0, frames, endpoint=True, dtype=np.float32)
    if float(np.max(np.abs(audio[:, 0]))) > _CLICK_FLOOR:
        result[:, :frames] *= ramp[None, :]
    if float(np.max(np.abs(audio[:, -1]))) > _CLICK_FLOOR:
        result[:, -frames:] *= ramp[::-1][None, :]
    return result


def _normalise(audio: np.ndarray, settings: Mapping[str, Any]) -> tuple[np.ndarray, float]:
    if not settings.get("enabled", False) or audio.size == 0:
        return audio, 0.0
    mode = str(settings.get("mode", "peak")).casefold()
    target_db = float(settings.get("target_db", -1.0 if mode == "peak" else -20.0))
    target = 10.0 ** (target_db / 20.0)
    if mode == "peak":
        current = float(np.max(np.abs(audio)))
    elif mode == "rms":
        current = float(np.sqrt(np.mean(audio * audio)))
    else:
        raise ValueError(f"normalization.mode must be 'peak' or 'rms', got {mode!r}")
    if not math.isfinite(current) or current <= 1e-12:
        return audio, 0.0
    gain = target / current
    return (audio * gain).astype(np.float32), float(20.0 * math.log10(gain))


def process_record(
    record: Mapping[str, Any],
    source_root: Path,
    dataset_root: Path,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    source_path = source_root / str(record["relative_path"])
    audio, sample_rate = read_audio(source_path)
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim != 2 or audio.shape[0] == 0:
        raise ValueError("audio must have shape [channels, frames]")
    if not config.get("preserve_channels", True) and audio.shape[0] > 1:
        audio = audio.mean(axis=0, keepdims=True)
    source_peak, source_rms, source_dc = _stats(audio)
    if not all(math.isfinite(value) for value in (source_peak, source_rms, source_dc)):
        raise ValueError("source audio contains NaN or Inf")

    trim_cfg = config.get("trim", {})
    if trim_cfg.get("enabled", True):
        audio, trim_start, trim_end = trim_silence(
            audio,
            sample_rate,
            threshold_db=float(trim_cfg.get("threshold_db", -70.0)),
            pre_roll_ms=float(trim_cfg.get("pre_roll_ms", 20.0)),
            post_roll_ms=float(trim_cfg.get("post_roll_ms", 100.0)),
        )
    else:
        trim_start, trim_end = 0, 0
    if audio.shape[1] == 0:
        raise ValueError("all-silence sample after trimming")
    audio = resample_audio(audio, sample_rate, int(config["target_sample_rate"]))
    audio, gain_db = _normalise(audio, config.get("normalization", {}))
    audio = _apply_fades(audio, int(config["target_sample_rate"]), float(config.get("fade_ms", 3.0)))
    if not np.isfinite(audio).all():
        raise ValueError("processed audio contains NaN or Inf")
    processed_peak, processed_rms, processed_dc = _stats(audio)

    relative_output = Path("processed") / "violin" / Path(str(record["relative_path"]))
    relative_output = relative_output.with_suffix(".wav").as_posix()
    output_path = dataset_root / relative_output
    write_wav(output_path, audio, int(config["target_sample_rate"]))
    result = dict(record)
    result.update(
        {
            "processed_path": relative_output,
            "source_sample_rate": sample_rate,
            "processed_sample_rate": int(config["target_sample_rate"]),
            "sample_rate": int(config["target_sample_rate"]),
            "channels": int(audio.shape[0]),
            "processed_frame_count": int(audio.shape[1]),
            "processed_duration_seconds": float(audio.shape[1] / config["target_sample_rate"]),
            "duration_seconds": float(audio.shape[1] / config["target_sample_rate"]),
            "processed_peak": processed_peak,
            "processed_rms": processed_rms,
            "processed_dc_offset": processed_dc,
            "peak": processed_peak,
            "rms": processed_rms,
            "dc_offset": processed_dc,
            "original_peak": source_peak,
            "original_rms": source_rms,
            "original_dc_offset": source_dc,
            "original_gain_db": 0.0,
            "processing_gain_db": gain_db,
            "trim_start_frames": int(trim_start),
            "trim_end_frames": int(trim_end),
            "validation": {
                "file_exists": True,
                "finite": True,
                "nonzero_duration": True,
                "sample_rate_valid": int(config["target_sample_rate"]) > 0,
                "frame_count_valid": int(audio.shape[1]) > 0,
                "peak_finite": math.isfinite(processed_peak),
                "rms_finite": math.isfinite(processed_rms),
            },
        }
    )
    return result


def preprocess_records(
    records: list[Mapping[str, Any]], source_root: Path, dataset_root: Path, config: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    processed: list[dict[str, Any]] = []
    rejected: list[dict[str, str]] = []
    for record in sorted(records, key=lambda row: str(row["id"])):
        try:
            processed.append(process_record(record, source_root, dataset_root, config))
        except Exception as exc:
            rejected.append({"relative_path": str(record["relative_path"]), "reason": "preprocess_failed", "detail": str(exc)})
    return processed, rejected


def write_jsonl(records: list[Mapping[str, Any]], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
    return path


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def validate_processed_records(records: list[Mapping[str, Any]], dataset_root: Path) -> list[dict[str, Any]]:
    errors: list[dict[str, Any]] = []
    for record in records:
        path = dataset_root / str(record["processed_path"])
        try:
            if not path.exists():
                raise FileNotFoundError(path)
            audio, rate = read_audio(path)
            if rate <= 0 or audio.shape[1] <= 0:
                raise ValueError("invalid rate or frame count")
            if not np.isfinite(audio).all():
                raise ValueError("NaN/Inf audio")
            peak, rms, _ = _stats(audio)
            if not math.isfinite(peak) or not math.isfinite(rms):
                raise ValueError("non-finite peak/RMS")
            if int(record["processed_sample_rate"]) != rate:
                raise ValueError(f"manifest rate {record['processed_sample_rate']} != file rate {rate}")
        except Exception as exc:
            errors.append({"id": record.get("id"), "path": str(path), "reason": str(exc)})
    return errors


def build_manifests(
    records: list[Mapping[str, Any]], dataset_root: Path, config: Mapping[str, Any]
) -> dict[str, int]:
    split_cfg = config.get("split", {})
    assigned = assign_splits(
        records,
        train=float(split_cfg.get("train", 0.8)),
        valid=float(split_cfg.get("valid", 0.1)),
        test=float(split_cfg.get("test", 0.1)),
        seed=int(split_cfg.get("seed", 20260827)),
    )
    hash_value = config_hash(config)
    enriched: list[dict[str, Any]] = []
    for record in assigned:
        row = dict(record)
        row.update(
            {
                "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
                "pipeline_version": PIPELINE_VERSION,
                "config_hash": hash_value,
                "dataset": config["dataset"],
                "source": "VSCO-2-CE",
                "license": "CC0-1.0",
                "license_url": "https://creativecommons.org/publicdomain/zero/1.0/",
            }
        )
        enriched.append(row)
    manifest_root = dataset_root / "manifests"
    write_jsonl(enriched, manifest_root / "violin-all.jsonl")
    counts: dict[str, int] = {}
    for split in ("train", "valid", "test"):
        rows = [row for row in enriched if row["split"] == split]
        counts[split] = len(rows)
        write_jsonl(rows, manifest_root / f"violin-{split}.jsonl")
    return counts


def build_summary(
    scan_report: Mapping[str, Any], processed: list[Mapping[str, Any]], preprocessing_rejections: list[Mapping[str, Any]], config: Mapping[str, Any]
) -> dict[str, Any]:
    summary = summarize_records(processed)
    summary.update(
        {
            "schema_version": 1,
            "pipeline_version": PIPELINE_VERSION,
            "dataset": config["dataset"],
            "dataset_version": config.get("dataset_version"),
            "source": "VSCO-2-CE",
            "license": "CC0-1.0",
            "source_dataset_hash": scan_report.get("source_dataset_hash"),
            "scan_discovered_wavs": scan_report["counts"]["wav_files_discovered"],
            "scan_rejected": scan_report["counts"]["rejected"],
            "preprocess_rejected": len(preprocessing_rejections),
            "rejected_by_reason": dict(
                sorted(
                    Counter(
                        [item["reason"] for item in scan_report.get("rejections", [])]
                        + [item["reason"] for item in preprocessing_rejections]
                    ).items()
                )
            ),
            "warnings": dict(
                sorted(
                    Counter(warning for row in processed for warning in row.get("warnings", [])).items()
                )
            ),
            "config_hash": config_hash(config),
            "split_strategy": "group by midi_note and normalized articulation; all round robins and dynamic variants stay together",
        }
    )
    return summary


def write_summary(summary: Mapping[str, Any], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def copy_license(source_root: Path, dataset_root: Path) -> Path | None:
    candidates = [source_root / "LICENSE", source_root / "license", source_root / "LICENSE.txt"]
    source = next((path for path in candidates if path.exists()), None)
    if source is None:
        return None
    destination = dataset_root / "LICENSE" / "VSCO-2-CE-LICENSE.txt"
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    return destination


def run_prepare(
    config: Mapping[str, Any], *, source: str | Path | None = None, dataset_root: str | Path | None = None
) -> dict[str, Any]:
    source_root = resolve_workspace_path(source or config["source"])
    output_root = resolve_workspace_path(dataset_root or config["dataset_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    scan_report = scan_source(
        source_root,
        pitch_detection_enabled=bool(config.get("pitch_detection", {}).get("enabled", True)),
    )
    scan_report["config_hash"] = config_hash(config)
    write_scan_report(scan_report, output_root / "metadata" / "scan-report.json")
    processed, preprocessing_rejections = preprocess_records(
        scan_report["records"], source_root, output_root, config
    )
    processed.sort(key=lambda row: row["id"])
    processed_payload = {
        "schema_version": SCAN_SCHEMA_VERSION,
        "pipeline_version": PIPELINE_VERSION,
        "config_hash": config_hash(config),
        "source_dataset_hash": scan_report["source_dataset_hash"],
        "records": processed,
        "rejections": preprocessing_rejections,
    }
    (output_root / "metadata").mkdir(parents=True, exist_ok=True)
    (output_root / "metadata" / "processed-records.json").write_text(
        json.dumps(processed_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    errors = validate_processed_records(processed, output_root)
    if errors:
        raise RuntimeError(f"processed validation failed for {len(errors)} samples: {errors[:3]}")
    split_counts = build_manifests(processed, output_root, config)
    summary = build_summary(scan_report, processed, preprocessing_rejections, config)
    summary["split_counts"] = split_counts
    write_summary(summary, output_root / "metadata" / "violin-summary.json")
    copy_license(source_root, output_root)
    return {
        "source_root": source_root,
        "dataset_root": output_root,
        "scan_report": scan_report,
        "processed": processed,
        "preprocessing_rejections": preprocessing_rejections,
        "summary": summary,
    }
