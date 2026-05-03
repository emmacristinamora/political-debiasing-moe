"""
inspect_quadrant_pools.py

Prints topic × source chunk-count tables for all four quadrants,
after applying the two pipeline filters:
  1. Token filter      : 150 ≤ n_tokens ≤ 700
  2. Within-doc dedupe : greedy selection with min_chunk_index_gap = 2

Run from data/experts/:
    python ../../src/inspect_quadrant_pools.py

Or pass an explicit path:
    python src/inspect_quadrant_pools.py --retained-dir data/experts/quadrant-pools
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

MIN_TOKENS       = 50
MAX_TOKENS       = 700
MIN_CHUNK_GAP    = 2

QUADRANTS = ["right_auth", "left_auth", "left_lib", "right_lib"]

TOPICS = [
    "economy",
    "foreign_policy",
    "governance_institutions",
    "health_education",
    "law_order",
    "immigration",
]

SOURCES = [
    "allsides",
    "reddit_liberal",
    "reddit_conservative",
    "hoc",
    "ec_press",
    "uk_press",
    "ire_press",
]

SOURCE_LABELS = {
    "allsides":            "allsides",
    "reddit_liberal":      "reddit_lib",
    "reddit_conservative": "reddit_con",
    "hoc":                 "hoc",
    "ec_press":            "ec_press",
    "uk_press":            "uk_press",
    "ire_press":           "ire_press",
}


def parse_chunk_index(chunk_id: str) -> int:
    return int(chunk_id.rsplit("_chunk", 1)[1])


def dedupe(chunks: list[dict]) -> list[dict]:
    by_doc: dict[str, list[dict]] = defaultdict(list)
    for c in chunks:
        by_doc[c["document_id"]].append(c)

    result: list[dict] = []
    for doc_chunks in by_doc.values():
        sorted_chunks = sorted(doc_chunks, key=lambda c: c["chunk_index"])
        last = -MIN_CHUNK_GAP
        for c in sorted_chunks:
            if c["chunk_index"] >= last + MIN_CHUNK_GAP:
                result.append(c)
                last = c["chunk_index"]
    return result


def load_quadrant(retained_dir: Path, quadrant: str) -> list[dict]:
    chunks: list[dict] = []
    for source_dir in sorted(retained_dir.iterdir()):
        if not source_dir.is_dir():
            continue
        source = source_dir.name
        path = source_dir / quadrant / "retained.jsonl"
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                n_tok = int(d["n_tokens"])
                if not (MIN_TOKENS <= n_tok <= MAX_TOKENS):
                    continue
                chunks.append({
                    "document_id": d["document_id"],
                    "chunk_index": parse_chunk_index(d["chunk_id"]),
                    "topic":       d["topic_primary"],
                    "source":      source,
                })
    return dedupe(chunks)


def build_table(chunks: list[dict]) -> dict[tuple[str, str], int]:
    counts: dict[tuple[str, str], int] = defaultdict(int)
    for c in chunks:
        counts[(c["topic"], c["source"])] += 1
    return counts


def print_table(quadrant: str, counts: dict[tuple[str, str], int]) -> None:
    topic_w  = max(len(t) for t in TOPICS)
    col_w    = 10

    header_labels = [SOURCE_LABELS[s] for s in SOURCES] + ["TOTAL"]
    header = f"  {'topic':<{topic_w}}  " + "  ".join(f"{h:>{col_w}}" for h in header_labels)
    sep    = "  " + "-" * (topic_w + 2 + (col_w + 2) * len(header_labels) - 2)

    print(f"\n{'=' * len(sep)}")
    print(f"  QUADRANT: {quadrant}")
    print(sep)
    print(header)
    print(sep)

    col_totals = defaultdict(int)
    for topic in TOPICS:
        row_vals = [counts.get((topic, src), 0) for src in SOURCES]
        row_total = sum(row_vals)
        for src, v in zip(SOURCES, row_vals):
            col_totals[src] += v
        col_totals["TOTAL"] += row_total
        cells = "  ".join(f"{v:>{col_w},}" for v in row_vals)
        print(f"  {topic:<{topic_w}}  {cells}  {row_total:>{col_w},}")

    print(sep)
    total_vals = [col_totals[src] for src in SOURCES]
    total_cells = "  ".join(f"{v:>{col_w},}" for v in total_vals)
    print(f"  {'TOTAL':<{topic_w}}  {total_cells}  {col_totals['TOTAL']:>{col_w},}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--retained-dir", type=Path, default=Path("quadrant-pools"),
        help="Root of retained pools: <dir>/<source>/<quadrant>/retained.jsonl "
             "(default: quadrant-pools, relative to CWD)",
    )
    args = parser.parse_args()

    if not args.retained_dir.exists():
        raise SystemExit(
            f"retained-dir not found: {args.retained_dir.resolve()}\n"
            "Run from data/experts/ or pass --retained-dir explicitly."
        )

    print(f"Retained dir : {args.retained_dir.resolve()}")
    print(f"Token filter : {MIN_TOKENS}–{MAX_TOKENS} tokens")
    print(f"Dedupe gap   : {MIN_CHUNK_GAP} (min chunk_index gap within a document)")
    print(f"Topics shown : {', '.join(TOPICS)}")

    for quadrant in QUADRANTS:
        chunks = load_quadrant(args.retained_dir, quadrant)
        counts = build_table(chunks)
        print_table(quadrant, counts)


if __name__ == "__main__":
    main()
