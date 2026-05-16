# src/19_repartition_chunks.py

# Repartitions already-scored chunks using a calibrated compass center.
# The raw projection scores (score_econ, score_soc) in scored_chunks.jsonl
# are kept as-is; only the derived fields that depend on the origin are
# recomputed: quadrant, threshold_pass, score_abs_econ, score_abs_soc,
# confidence_margin.  The per-quadrant retained.jsonl and report.json are
# then rewritten from scratch.  No GPU or model is required.
#
#   input   data/experts/quadrant-pools/{source}/scored_chunks.jsonl
#           data/compass_center/center.json
#   output  data/experts/quadrant-pools/{source}/scored_chunks.jsonl   (updated in-place)
#           data/experts/quadrant-pools/{source}/{quadrant}/retained.jsonl
#           data/experts/quadrant-pools/{source}/{quadrant}/report.json
#           data/experts/quadrant-pools/{source}/document_summaries.jsonl (updated in-place)


# === IMPORTS ===

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

# === CONFIG ===

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_POOLS_DIR   = PROJECT_ROOT / "data" / "experts" / "quadrant-pools"
DEFAULT_CENTER_PATH = PROJECT_ROOT / "data" / "compass_center" / "center.json"

QUADRANTS = ("right_auth", "left_auth", "left_lib", "right_lib")


# === HELPERS: IO ===

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Repartition existing scored chunks using a calibrated compass center."
    )
    parser.add_argument("--pools-dir",   type=Path, default=DEFAULT_POOLS_DIR)
    parser.add_argument("--center-path", type=Path, default=DEFAULT_CENTER_PATH)
    parser.add_argument("--min-confidence-margin", type=float, default=None,
                        help="Override the confidence-margin threshold from the snapshot. "
                             "If omitted, the value stored in each source's build_config_snapshot.json is used.")
    parser.add_argument("--min-abs-econ", type=float, default=None)
    parser.add_argument("--min-abs-soc",  type=float, default=None)
    parser.add_argument("--dry-run", action="store_true",
                        help="Print statistics but do not overwrite any files.")
    return parser.parse_args()


def load_center(path: Path) -> tuple[float, float]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return float(data["center"]["economic"]), float(data["center"]["social"])


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def save_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def load_thresholds(snapshot_path: Path, cli: argparse.Namespace) -> tuple[float, float, float]:
    """Read thresholds from the original build snapshot, with CLI overrides."""
    defaults = {"min_abs_econ": 0.0, "min_abs_soc": 0.0, "min_confidence_margin": 0.015}
    if snapshot_path.is_file():
        snap = json.loads(snapshot_path.read_text(encoding="utf-8"))
        thresh = snap.get("thresholds", {})
        defaults.update({k: float(v) for k, v in thresh.items() if v is not None})
    min_abs_econ  = cli.min_abs_econ  if cli.min_abs_econ  is not None else defaults["min_abs_econ"]
    min_abs_soc   = cli.min_abs_soc   if cli.min_abs_soc   is not None else defaults["min_abs_soc"]
    min_conf      = cli.min_confidence_margin if cli.min_confidence_margin is not None else defaults["min_confidence_margin"]
    return min_abs_econ, min_abs_soc, min_conf


# === RECLASSIFICATION ===

def quadrant_from_shifted(se: float, ss: float) -> str:
    """Assign quadrant based on the sign of center-shifted scores."""
    if se >= 0 and ss >= 0:
        return "right_auth"
    if se < 0 and ss >= 0:
        return "left_auth"
    if se < 0 and ss < 0:
        return "left_lib"
    return "right_lib"


def reclassify(
    row: dict[str, Any],
    center_econ: float,
    center_soc: float,
    min_abs_econ: float,
    min_abs_soc: float,
    min_conf: float,
) -> dict[str, Any]:
    """Return a copy of the row with derived classification fields updated."""
    se = row["score_econ"] - center_econ
    ss = row["score_soc"]  - center_soc
    abs_se = abs(se)
    abs_ss = abs(ss)
    conf   = min(abs_se, abs_ss)
    passed = abs_se >= min_abs_econ and abs_ss >= min_abs_soc and conf >= min_conf
    quad   = quadrant_from_shifted(se, ss)
    updated = dict(row)
    updated["quadrant"]          = quad
    updated["score_abs_econ"]    = abs_se
    updated["score_abs_soc"]     = abs_ss
    updated["confidence_margin"] = conf
    updated["threshold_pass"]    = passed
    if passed:
        updated["selection_stage"] = "retained"
    elif row.get("selection_stage") == "retained":
        updated["selection_stage"] = "scored"
    updated["example_id"] = f"{quad}_{row['chunk_id']}"
    return updated


# === REPORT BUILDING ===

def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0

def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mu = _mean(values)
    return math.sqrt(sum((v - mu) ** 2 for v in values) / len(values))

def _proportion_dict(counter: Counter) -> dict[str, float]:
    total = sum(counter.values())
    if total == 0:
        return {}
    return {k: v / total for k, v in counter.items()}


def build_report(rows: list[dict[str, Any]], quadrant: str) -> dict[str, Any]:
    source_rows: dict[str, list] = defaultdict(list)
    for r in rows:
        source_rows[r.get("source_family", "unknown")].append(r)

    source_topic_counts: dict[str, dict] = {}
    source_topic_props: dict[str, dict] = {}
    for family, frows in source_rows.items():
        counts = Counter((r.get("topic_primary") or "unlabeled") for r in frows)
        source_topic_counts[family] = dict(counts)
        source_topic_props[family]  = _proportion_dict(counts)

    ranked = sorted(rows, key=lambda r: r["confidence_margin"], reverse=True)[:5]
    top_examples = [
        {
            "example_id":        r["example_id"],
            "document_id":       r["document_id"],
            "topic_primary":     r.get("topic_primary"),
            "source_family":     r.get("source_family"),
            "source_name":       r.get("source_name"),
            "score_econ":        r["score_econ"],
            "score_soc":         r["score_soc"],
            "confidence_margin": r["confidence_margin"],
            "text_preview":      r.get("text", "")[:300],
        }
        for r in ranked
    ]

    summary = {
        "n_rows":                 len(rows),
        "n_documents":            len({r["document_id"] for r in rows}),
        "n_twitter":              sum(1 for r in rows if r.get("twitter_flag")),
        "source_family_counts":   dict(Counter(r.get("source_family", "unknown") for r in rows)),
        "source_name_counts":     dict(Counter(r.get("source_name", "unknown") for r in rows)),
        "topic_counts":           dict(Counter((r.get("topic_primary") or "unlabeled") for r in rows)),
        "mean_abs_econ":          _mean([r["score_abs_econ"] for r in rows]),
        "mean_abs_soc":           _mean([r["score_abs_soc"] for r in rows]),
        "mean_confidence_margin": _mean([r["confidence_margin"] for r in rows]),
        "mean_tokens":            _mean([float(r.get("n_tokens", 0)) for r in rows]),
    }

    return {
        "quadrant":                 quadrant,
        "summary":                  summary,
        "source_proportions":       _proportion_dict(Counter(r.get("source_family", "unknown") for r in rows)),
        "source_topic_counts":      source_topic_counts,
        "source_topic_proportions": source_topic_props,
        "top_examples":             top_examples,
    }


def rebuild_document_summaries(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rebuild document summaries, updating dominant_quadrant to match the new partition."""
    by_doc: dict[str, list] = defaultdict(list)
    for r in rows:
        by_doc[r["document_id"]].append(r)

    summaries = []
    for doc_id, doc_rows in by_doc.items():
        first = doc_rows[0]
        quad_counts = dict(Counter(r["quadrant"] for r in doc_rows))
        dominant = max(quad_counts, key=quad_counts.__getitem__) if quad_counts else None
        summaries.append({
            "document_id":       doc_id,
            "source_name":       first.get("source_name"),
            "source_family":     first.get("source_family"),
            "language":          first.get("language", "en"),
            "title":             first.get("title"),
            "date":              first.get("date"),
            "speaker_or_author": first.get("speaker_or_author"),
            "twitter_flag":      first.get("twitter_flag", False),
            "raw_dataset":       first.get("raw_dataset"),
            "n_chunks":          len(doc_rows),
            "mean_score_econ":   _mean([r["score_econ"] for r in doc_rows]),
            "mean_score_soc":    _mean([r["score_soc"]  for r in doc_rows]),
            "std_score_econ":    _std( [r["score_econ"] for r in doc_rows]),
            "std_score_soc":     _std( [r["score_soc"]  for r in doc_rows]),
            "dominant_quadrant": dominant,
            "quadrant_counts":   quad_counts,
        })
    return summaries


# === MAIN ===

def process_source(
    source_dir: Path,
    center_econ: float,
    center_soc: float,
    cli: argparse.Namespace,
) -> dict[str, int]:
    chunks_path = source_dir / "scored_chunks.jsonl"
    if not chunks_path.is_file():
        print(f"  [skip] no scored_chunks.jsonl in {source_dir.name}")
        return {}

    rows = load_jsonl(chunks_path)
    min_abs_econ, min_abs_soc, min_conf = load_thresholds(
        source_dir / "build_config_snapshot.json", cli
    )

    updated = [reclassify(r, center_econ, center_soc, min_abs_econ, min_abs_soc, min_conf) for r in rows]

    retained_by_quad: dict[str, list] = defaultdict(list)
    for r in updated:
        if r["threshold_pass"]:
            retained_by_quad[r["quadrant"]].append(r)

    counts = {q: len(retained_by_quad[q]) for q in QUADRANTS}

    if not cli.dry_run:
        save_jsonl(chunks_path, updated)
        save_jsonl(source_dir / "document_summaries.jsonl", rebuild_document_summaries(updated))
        for quad in QUADRANTS:
            quad_dir = source_dir / quad
            retained = retained_by_quad[quad]
            save_jsonl(quad_dir / "retained.jsonl", retained)
            save_json(quad_dir / "report.json", build_report(retained, quad))

    return counts


def main() -> None:
    args = parse_args()

    if not args.center_path.is_file():
        raise FileNotFoundError(f"center file not found: {args.center_path}")
    center_econ, center_soc = load_center(args.center_path)
    print(f"compass center: economic={center_econ:+.6f}  social={center_soc:+.6f}")
    if args.dry_run:
        print("[dry-run] no files will be written")

    source_dirs = sorted(p for p in args.pools_dir.iterdir() if p.is_dir())
    if not source_dirs:
        raise FileNotFoundError(f"no source subdirectories found in {args.pools_dir}")

    totals: dict[str, int] = defaultdict(int)
    for source_dir in source_dirs:
        print(f"\n{source_dir.name}")
        counts = process_source(source_dir, center_econ, center_soc, args)
        for q, n in counts.items():
            print(f"  {q:<12} {n:>6} retained")
            totals[q] += n

    print(f"\n=== totals across all sources ===")
    for q in QUADRANTS:
        print(f"  {q:<12} {totals[q]:>7}")
    print(f"\n{'[dry-run] nothing written' if args.dry_run else 'done'}")


if __name__ == "__main__":
    main()
