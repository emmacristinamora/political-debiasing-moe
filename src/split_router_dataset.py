# src/split_router_dataset.py


# === IMPORTS ===

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


# router_calibration_config is torch-free.
_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from router_calibration_config import (  # noqa: E402
    RouterCalibrationConfig,
    load_router_calibration_config,
)


# === CONSTANTS ===

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SPLIT_NAMES: tuple[str, ...] = ("train", "val", "test")
RECORDS_FILENAME: str = "records.jsonl"
DEFAULT_REPORT_FILENAME: str = "split_report.json"

FRACTION_SUM_TOLERANCE: float = 1e-6

SUPPORTED_STRATIFY: tuple[str, ...] = ("source", "axis", "source_axis", "none")
UNKNOWN_LABEL: str = "unknown"

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO,
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# === DATACLASS ===

@dataclass
class SplitBuildConfig:
    """
    Effective build-time config after merging YAML defaults and CLI overrides.
    Step 1's SplitConfig calls the stratification key 'split_by'; the spec for
    step 9 calls it 'stratify_by'. build_split_config bridges the two.
    """
    train_fraction: float
    val_fraction:   float
    test_fraction:  float
    seed:           int
    stratify_by:    str
    records_path:   Path
    output_dir:     Path
    report_path:    Path


# === HELPERS — IO ===

def load_records(path: Path) -> list[dict]:
    """Read records.jsonl. Loud on missing/empty/malformed input."""
    if not path.is_file():
        raise FileNotFoundError(f"records file not found: {path}")
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"records file is empty: {path}")
    out: list[dict] = []
    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON at {path}:{lineno}") from exc
        out.append(record)
    return out


# === HELPERS — VALIDATION ===

def validate_split_fractions(train: float, val: float, test: float) -> None:
    """Each fraction > 0; the three must sum to 1 within FRACTION_SUM_TOLERANCE."""
    for name, v in (
        ("train_fraction", train),
        ("val_fraction",   val),
        ("test_fraction",  test),
    ):
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise ValueError(
                f"{name} must be a number, got {type(v).__name__}"
            )
        if v <= 0:
            raise ValueError(f"{name} must be > 0, got {v}")
    total = float(train) + float(val) + float(test)
    if abs(total - 1.0) > FRACTION_SUM_TOLERANCE:
        raise ValueError(
            f"split fractions must sum to 1 within {FRACTION_SUM_TOLERANCE}, "
            f"got {total}"
        )


# === HELPERS — STRATUM KEY ===

def _record_metadata(record: dict) -> dict:
    metadata = record.get("metadata", {})
    return metadata if isinstance(metadata, dict) else {}


def _extract_source(metadata: dict) -> str:
    """
    Resolution chain per spec:
        metadata.source
        metadata.input_metadata.source
        metadata.original_source
        else "unknown"
    """
    src = metadata.get("source")
    if isinstance(src, str) and src.strip():
        return src
    nested = metadata.get("input_metadata")
    if isinstance(nested, dict):
        nested_src = nested.get("source")
        if isinstance(nested_src, str) and nested_src.strip():
            return nested_src
    original = metadata.get("original_source")
    if isinstance(original, str) and original.strip():
        return original
    return UNKNOWN_LABEL


def _extract_axis(metadata: dict) -> str:
    axis = metadata.get("axis")
    if isinstance(axis, str) and axis.strip():
        return axis
    return UNKNOWN_LABEL


def get_stratum_key(record: dict, stratify_by: str) -> str:
    """
    Compute the stratum key for a record per `stratify_by`. Always returns
    a non-empty string so dict-grouping is always well-defined.
    """
    if stratify_by not in SUPPORTED_STRATIFY:
        raise ValueError(
            f"unsupported stratify_by={stratify_by!r}; "
            f"expected one of {list(SUPPORTED_STRATIFY)}"
        )
    if stratify_by == "none":
        return "all"
    metadata = _record_metadata(record)
    if stratify_by == "source":
        return _extract_source(metadata)
    if stratify_by == "axis":
        return _extract_axis(metadata)
    # "source_axis"
    return f"{_extract_source(metadata)}::{_extract_axis(metadata)}"


# === HELPERS — ALLOCATION ===

def allocate_counts(
    n: int,
    fractions: tuple[float, float, float],
) -> tuple[int, int, int]:
    """
    Largest-remainder allocation of n items into 3 buckets per fractions
    (train, val, test). Guarantees:
      n == 0 -> (0, 0, 0)
      n == 1 -> (1, 0, 0)        # always at least one in train
      n == 2 -> (1, 1, 0)        # train > val > test in tie-break
      n >= 3 -> each bucket gets at least 1 (steals from the largest)

    Tie-break in the leftover-distribution step prefers train, then val,
    then test, so deterministic regardless of float-rounding noise.
    """
    if isinstance(n, bool) or not isinstance(n, int):
        raise ValueError(f"n must be an int, got {type(n).__name__}")
    if n < 0:
        raise ValueError(f"n must be >= 0, got {n}")
    if n == 0:
        return (0, 0, 0)
    if n == 1:
        return (1, 0, 0)
    if n == 2:
        return (1, 1, 0)

    train_f, val_f, test_f = fractions
    raw     = [n * float(train_f), n * float(val_f), n * float(test_f)]
    floors  = [int(r) for r in raw]
    leftover = n - sum(floors)
    remainders = [r - f for r, f in zip(raw, floors)]

    # leftover goes to indices with largest fractional remainder; ties broken
    # by canonical order (train < val < test by index 0,1,2).
    order = sorted(range(3), key=lambda i: (-remainders[i], i))
    counts = list(floors)
    for i in range(leftover):
        counts[order[i]] += 1

    # ensure each bucket gets at least 1 by stealing from the largest
    while any(c == 0 for c in counts):
        max_idx  = max(range(3), key=lambda i: counts[i])
        zero_idx = next(i for i in range(3) if counts[i] == 0)
        if counts[max_idx] <= 1:
            break  # cannot satisfy without violating non-empty elsewhere
        counts[max_idx]  -= 1
        counts[zero_idx] += 1

    return (counts[0], counts[1], counts[2])


# === SPLIT ===

def _validate_records_for_split(records: Any) -> None:
    if not isinstance(records, list) or not records:
        raise ValueError("records must be a non-empty list")
    seen_ids: set[str] = set()
    for i, rec in enumerate(records):
        if not isinstance(rec, dict):
            raise ValueError(
                f"records[{i}] must be a dict, got {type(rec).__name__}"
            )
        eid = rec.get("example_id")
        if not isinstance(eid, str) or not eid.strip():
            raise ValueError(
                f"records[{i}] missing or invalid example_id"
            )
        if eid in seen_ids:
            raise ValueError(f"duplicate example_id: {eid!r}")
        seen_ids.add(eid)
        href = rec.get("hidden_representation_ref")
        if not isinstance(href, str) or not href:
            raise ValueError(
                f"[{eid}] missing or invalid hidden_representation_ref"
            )


def split_records(
    records: list[dict],
    cfg: SplitBuildConfig,
) -> tuple[dict[str, list[dict]], dict]:
    """
    Stratified deterministic split of records into train/val/test.

    Determinism contract:
      - same input set + same SplitBuildConfig -> identical splits regardless
        of input row order (records are sorted by example_id within each
        stratum before shuffling)
      - changing seed changes within-stratum shuffling and therefore
        membership, but not the per-stratum count breakdown
    """
    _validate_records_for_split(records)
    validate_split_fractions(cfg.train_fraction, cfg.val_fraction, cfg.test_fraction)
    if cfg.stratify_by not in SUPPORTED_STRATIFY:
        raise ValueError(
            f"unsupported stratify_by={cfg.stratify_by!r}; "
            f"expected one of {list(SUPPORTED_STRATIFY)}"
        )

    # group by stratum
    strata: dict[str, list[dict]] = {}
    for rec in records:
        key = get_stratum_key(rec, cfg.stratify_by)
        strata.setdefault(key, []).append(rec)

    splits: dict[str, list[dict]] = {name: [] for name in SPLIT_NAMES}
    strata_report: dict[str, dict[str, int]] = {}
    warnings: list[str] = []

    rng = random.Random(int(cfg.seed))
    fractions = (cfg.train_fraction, cfg.val_fraction, cfg.test_fraction)

    # iterate strata in sorted-key order (deterministic regardless of dict order)
    for key in sorted(strata.keys()):
        # within-stratum: sort by example_id so input row order doesn't
        # affect the result, then shuffle with the seeded rng
        stratum = sorted(strata[key], key=lambda r: r["example_id"])
        rng.shuffle(stratum)
        n = len(stratum)
        train_n, val_n, test_n = allocate_counts(n, fractions)

        if n == 1:
            warnings.append(
                f"stratum {key!r} has only 1 example; assigned to train"
            )

        splits["train"].extend(stratum[:train_n])
        splits["val"].extend(stratum[train_n : train_n + val_n])
        splits["test"].extend(stratum[train_n + val_n : train_n + val_n + test_n])
        strata_report[key] = {
            "total": n, "train": train_n, "val": val_n, "test": test_n,
        }

    if not splits["train"]:
        raise ValueError("split would yield empty train set")
    if not splits["val"]:
        warnings.append("val split is empty")
    if not splits["test"]:
        warnings.append("test split is empty")

    output_paths = {
        name: str(cfg.output_dir / name / RECORDS_FILENAME) for name in SPLIT_NAMES
    }
    report = {
        "input_path":   str(cfg.records_path),
        "output_paths": output_paths,
        "num_records":  len(records),
        "fractions": {
            "train": cfg.train_fraction,
            "val":   cfg.val_fraction,
            "test":  cfg.test_fraction,
        },
        "seed":        int(cfg.seed),
        "stratify_by": cfg.stratify_by,
        "counts": {
            "train": len(splits["train"]),
            "val":   len(splits["val"]),
            "test":  len(splits["test"]),
        },
        "strata":   strata_report,
        "warnings": warnings,
    }
    return splits, report


# === IO ===

def write_splits(
    splits: dict[str, list[dict]],
    report: dict,
    output_dir: Path,
    report_path: Path,
) -> None:
    """
    Write split records.jsonl files under output_dir/{train,val,test}/ and
    write split_report.json. All writes go through .tmp + atomic rename.
    Parents are created on demand.
    """
    for name in SPLIT_NAMES:
        path = output_dir / name / RECORDS_FILENAME
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            for rec in splits[name]:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        tmp.replace(path)

    report_path.parent.mkdir(parents=True, exist_ok=True)
    rep_tmp = report_path.with_name(report_path.name + ".tmp")
    with rep_tmp.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    rep_tmp.replace(report_path)


# === CONFIG RESOLUTION ===

def build_split_config(
    cfg: RouterCalibrationConfig,
    *,
    records_path: Path | None = None,
    output_dir: Path | None = None,
    report_path: Path | None = None,
    seed: int | None = None,
) -> SplitBuildConfig:
    """
    Resolve effective config from a yaml-loaded RouterCalibrationConfig
    plus optional CLI overrides. Bridges the field-name discrepancy
    between step 1's SplitConfig.split_by and the spec's stratify_by.
    """
    train_f = float(cfg.split.train_fraction)
    val_f   = float(cfg.split.val_fraction)
    test_f  = float(cfg.split.test_fraction)
    validate_split_fractions(train_f, val_f, test_f)

    cfg_seed = int(cfg.split.seed)
    effective_seed = int(seed) if seed is not None else cfg_seed

    # spec uses 'stratify_by'; existing yaml uses 'split_by' — accept either
    stratify_by = (
        getattr(cfg.split, "stratify_by", None)
        or getattr(cfg.split, "split_by", None)
        or "none"
    )
    if stratify_by not in SUPPORTED_STRATIFY:
        raise ValueError(
            f"unsupported stratify_by={stratify_by!r}; "
            f"expected one of {list(SUPPORTED_STRATIFY)}"
        )

    rp = records_path if records_path is not None else cfg.paths.records_path

    od_default = (
        getattr(cfg.paths, "split_output_dir", None) or cfg.paths.output_dir
    )
    od = output_dir if output_dir is not None else od_default

    rep_default = (
        getattr(cfg.paths, "split_report_path", None)
        or cfg.paths.output_dir / DEFAULT_REPORT_FILENAME
    )
    rep = report_path if report_path is not None else rep_default

    return SplitBuildConfig(
        train_fraction=train_f,
        val_fraction=val_f,
        test_fraction=test_f,
        seed=effective_seed,
        stratify_by=stratify_by,
        records_path=rp,
        output_dir=od,
        report_path=rep,
    )


# === CLI ===

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "split router-calibration records.jsonl into deterministic, "
            "stratified train/val/test subsets"
        ),
    )
    p.add_argument("--config",        type=Path, required=True)
    p.add_argument("--records-path",  type=Path, default=None)
    p.add_argument("--output-dir",    type=Path, default=None)
    p.add_argument("--report-path",   type=Path, default=None)
    p.add_argument("--seed",          type=int,  default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_router_calibration_config(args.config)
    log.info("config loaded from %s", args.config)

    split_cfg = build_split_config(
        cfg,
        records_path=args.records_path,
        output_dir=args.output_dir,
        report_path=args.report_path,
        seed=args.seed,
    )

    log.info("loading records from %s", split_cfg.records_path)
    records = load_records(split_cfg.records_path)
    log.info(
        "loaded %d records (stratify_by=%s, seed=%d)",
        len(records), split_cfg.stratify_by, split_cfg.seed,
    )

    splits, report = split_records(records, split_cfg)

    log.info(
        "split sizes — train=%d val=%d test=%d",
        report["counts"]["train"], report["counts"]["val"], report["counts"]["test"],
    )
    for warning in report["warnings"]:
        log.warning("%s", warning)

    write_splits(splits, report, split_cfg.output_dir, split_cfg.report_path)
    for name in SPLIT_NAMES:
        log.info(
            "wrote %s split → %s",
            name, split_cfg.output_dir / name / RECORDS_FILENAME,
        )
    log.info("wrote report → %s", split_cfg.report_path)


if __name__ == "__main__":
    main()
