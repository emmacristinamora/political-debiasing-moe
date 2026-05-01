# src/06_validate_experts_datasets.py


# === IMPORTS ===

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


# === CONSTANTS ===

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH  = PROJECT_ROOT / "config" / "config.yaml"

QUADRANTS        = ["right_auth", "left_auth", "left_lib", "right_lib"]
TEMPLATE_QUADRANT = "left_auth"   # smallest pool; sampled first as binding constraint
OTHER_QUADRANTS   = [q for q in QUADRANTS if q != TEMPLATE_QUADRANT]

VALID_SOURCES = {
    "allsides", "ec_press", "ire_press", "uk_press",
    "hoc", "reddit_liberal", "reddit_conservative",
}

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO,
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# === DATACLASSES ===

@dataclass(frozen=True)
class Chunk:
    chunk_id:    str
    document_id: str
    chunk_index: int
    text:        str
    source:      str       # directory key: allsides, ec_press, …
    topic_label: str       # topic_primary in raw data
    s_econ:      float     # score_econ in raw data
    s_soc:       float     # score_soc in raw data
    conf_margin: float
    n_tokens:    int
    quadrant:    str


@dataclass
class CellTarget:
    quadrant:     str
    topic:        str
    source:       str
    target_count: int


@dataclass
class SamplingPlan:
    target_n_per_quadrant:  int
    source_cap_pct:         float
    n_topics:               int
    n_sources_per_quadrant: dict   # quadrant -> int (excluding held-out source)
    cell_targets:           list   # list[CellTarget]
    q2_realized:            dict = field(default_factory=dict)


@dataclass
class QuadrantSplit:
    quadrant:   str
    train:      list   # list[Chunk]
    val_indist: list
    val_source: list
    val_topic:  list


# === CONFIG LOADING ===

_REQUIRED_KEYS = {
    "viable_topics", "held_out_topic", "target_n_per_quadrant",
    "source_cap_pct", "min_cell_size", "held_out_sources",
    "val_pct", "random_seed", "dedupe", "confidence_bins", "sanity_checks",
    "length_filter",
}


def load_config(path: Path) -> dict:
    """
    Reads config.yaml, validates the validate_expert_datasets section,
    and returns the full config dict.
    """
    with path.open(encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    s = cfg.get("validate_expert_datasets", {})
    missing = _REQUIRED_KEYS - set(s.keys())
    if missing:
        raise ValueError(
            f"config.yaml missing keys under validate_expert_datasets: {sorted(missing)}"
        )

    hos = s["held_out_sources"]
    missing_q = set(QUADRANTS) - set(hos.keys())
    if missing_q:
        raise ValueError(
            f"held_out_sources missing quadrants: {sorted(missing_q)}. "
            f"Expected keys: {QUADRANTS}"
        )

    if s["held_out_topic"] in s["viable_topics"]:
        raise ValueError(
            f"held_out_topic '{s['held_out_topic']}' is also listed in viable_topics — contradiction."
        )

    if not 0 < s["val_pct"] < 0.5:
        raise ValueError(f"val_pct must be in (0, 0.5), got {s['val_pct']}")

    if not 0 < s["source_cap_pct"] < 1:
        raise ValueError(f"source_cap_pct must be in (0, 1), got {s['source_cap_pct']}")

    edges = s["confidence_bins"]["edges"]
    if any(edges[i] >= edges[i + 1] for i in range(len(edges) - 1)):
        raise ValueError(
            f"confidence_bins.edges must be strictly increasing, got: {edges}"
        )

    return cfg


# === I/O HELPERS ===

def _parse_chunk_index(chunk_id: str) -> int:
    """
    Extracts chunk_index from chunk_id.
    Expected format: <document_id>_chunk<NNNN>  (e.g. 'hoc_1989_0034692_chunk0002' -> 2).
    Raises ValueError with a clear message if the format is unexpected.
    """
    try:
        return int(chunk_id.rsplit("_chunk", 1)[1])
    except (IndexError, ValueError) as exc:
        raise ValueError(
            f"Cannot parse chunk_index from chunk_id '{chunk_id}'. "
            "chunk_id must end with '_chunk<digits>' (e.g. '_chunk0003'). "
            "Regenerate the retained pool to include chunk_index, or verify chunk_id format."
        ) from exc


def load_quadrant_pool(
    retained_dir: Path,
    quadrant: str,
    min_tokens: int = 0,
    max_tokens: int = 10_000,
) -> list[Chunk]:
    """
    Loads all retained.jsonl files for one quadrant across all source subdirectories.
    Input layout: <retained_dir>/<source>/<quadrant>/retained.jsonl

    Field mapping from raw data:
        topic_primary  -> topic_label
        score_econ     -> s_econ
        score_soc      -> s_soc
        confidence_margin -> conf_margin
        chunk_id suffix -> chunk_index (parsed)
        directory name  -> source

    Chunks outside [min_tokens, max_tokens] are dropped at load time.
    """
    chunks: list[Chunk] = []
    n_dropped_length = 0
    for source_dir in sorted(retained_dir.iterdir()):
        if not source_dir.is_dir():
            continue
        source = source_dir.name
        path = source_dir / quadrant / "retained.jsonl"
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSON at {path}:{lineno}: {exc}"
                    ) from exc

                n_tok = int(d["n_tokens"])
                if not (min_tokens <= n_tok <= max_tokens):
                    n_dropped_length += 1
                    continue

                chunk_id    = d["chunk_id"]
                chunk_index = _parse_chunk_index(chunk_id)

                chunks.append(Chunk(
                    chunk_id    = chunk_id,
                    document_id = d["document_id"],
                    chunk_index = chunk_index,
                    text        = d["text"],
                    source      = source,
                    topic_label = d["topic_primary"],
                    s_econ      = float(d["score_econ"]),
                    s_soc       = float(d["score_soc"]),
                    conf_margin = float(d["confidence_margin"]),
                    n_tokens    = n_tok,
                    quadrant    = quadrant,
                ))

    log.info(
        f"Loaded {len(chunks):,} retained chunks for '{quadrant}' "
        f"({n_dropped_length:,} dropped by length filter [{min_tokens}–{max_tokens}])"
    )
    return chunks


def write_jsonl(chunks: list[Chunk], path: Path) -> None:
    """Writes chunks to a JSONL file, dropping the quadrant field (implicit in path)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for chunk in chunks:
            row = asdict(chunk)
            row.pop("quadrant")
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_report(report: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)


# === STAGE 1: TOPIC FILTER ===

def apply_topic_filter(
    pools: dict[str, list[Chunk]],
    viable_topics: list[str],
    held_out_topic: str,
) -> dict[str, list[Chunk]]:
    """
    Keeps only chunks whose topic is in viable_topics or equals held_out_topic.
    All other topics are dropped here (they are neither used for training nor for eval).
    """
    keep_set = set(viable_topics) | {held_out_topic}
    result: dict[str, list[Chunk]] = {}
    for q, chunks in pools.items():
        filtered = [c for c in chunks if c.topic_label in keep_set]
        dropped  = len(chunks) - len(filtered)
        log.info(f"Stage 1 | {q}: {len(chunks):,} → {len(filtered):,} ({dropped:,} dropped)")
        result[q] = filtered
    return result


# === STAGE 2: HELD-OUT CARVE-OUTS ===

def carve_held_out_topic(
    pools: dict[str, list[Chunk]],
    held_out_topic: str,
) -> tuple[dict, dict]:
    """
    Separates held-out topic chunks from the working pools.
    Topic carve-out happens first so that val_source contains only viable-topic chunks.

    Returns:
        (working_pools, val_topic_pools) — both dicts keyed by quadrant
    """
    working:  dict[str, list[Chunk]] = {}
    val_topic: dict[str, list[Chunk]] = {}
    for q, chunks in pools.items():
        working[q]   = [c for c in chunks if c.topic_label != held_out_topic]
        val_topic[q] = [c for c in chunks if c.topic_label == held_out_topic]
        log.info(
            f"Stage 2a | {q}: working={len(working[q]):,}, "
            f"val_topic={len(val_topic[q]):,} ({held_out_topic})"
        )
    return working, val_topic


def carve_held_out_source(
    working_pools: dict[str, list[Chunk]],
    held_out_sources: dict[str, str],
) -> tuple[dict, dict]:
    """
    Separates held-out source chunks per quadrant from the working pools.
    Each quadrant has a different held-out source (configured per-quadrant).

    Returns:
        (final_working_pools, val_source_pools)
    """
    working:    dict[str, list[Chunk]] = {}
    val_source: dict[str, list[Chunk]] = {}
    for q, chunks in working_pools.items():
        src = held_out_sources[q]
        working[q]    = [c for c in chunks if c.source != src]
        val_source[q] = [c for c in chunks if c.source == src]
        log.info(
            f"Stage 2b | {q}: held_out_source='{src}' → "
            f"working={len(working[q]):,}, val_source={len(val_source[q]):,}"
        )
    return working, val_source


# === STAGE 3: TARGET SIZE COMPUTATION ===

def compute_cell_targets(
    working_pools: dict[str, list[Chunk]],
    config_sampling: dict,
) -> SamplingPlan:
    """
    Computes target chunk counts for each (quadrant, topic, source) cell.

    Logic:
        - per-topic target  = target_n_per_quadrant / n_topics
        - per-cell cap      = floor(per-topic target × source_cap_pct)
        - per-cell target   = ceil(per-topic target / n_sources), capped at per-cell cap
    """
    viable_topics  = config_sampling["viable_topics"]
    target_per_q   = config_sampling["target_n_per_quadrant"]
    source_cap_pct = config_sampling["source_cap_pct"]
    n_topics       = len(viable_topics)

    per_topic_target = target_per_q / n_topics
    per_cell_cap     = math.floor(per_topic_target * source_cap_pct)

    n_sources_per_q: dict[str, int] = {}
    cell_targets: list[CellTarget]  = []

    for q, chunks in working_pools.items():
        sources_present = sorted({c.source for c in chunks})
        n_src = len(sources_present)
        n_sources_per_q[q] = n_src

        per_cell_target = math.ceil(per_topic_target / n_src) if n_src > 0 else 0
        per_cell_target = min(per_cell_target, per_cell_cap)

        for topic in viable_topics:
            for src in sources_present:
                cell_targets.append(CellTarget(
                    quadrant=q, topic=topic, source=src, target_count=per_cell_target,
                ))

        log.info(
            f"Stage 3 | {q}: {n_src} sources, "
            f"per-topic={per_topic_target:.1f}, per-cell={per_cell_target}, cap={per_cell_cap}"
        )

    return SamplingPlan(
        target_n_per_quadrant  = target_per_q,
        source_cap_pct         = source_cap_pct,
        n_topics               = n_topics,
        n_sources_per_quadrant = n_sources_per_q,
        cell_targets           = cell_targets,
    )


# === STAGE 4: WITHIN-DOCUMENT DEDUPE ===

def select_non_overlapping_chunks(
    chunks: list[Chunk],
    min_index_gap: int,
) -> list[Chunk]:
    """
    Greedy selection of non-overlapping chunks from a single document.

    Args:
        chunks: chunks all sharing the same document_id
        min_index_gap: minimum gap between selected chunk indices

    Logic:
        Sort by chunk_index, then greedily pick chunks that are at least
        min_index_gap apart from the last selected chunk.
    """
    sorted_chunks = sorted(chunks, key=lambda c: c.chunk_index)
    selected: list[Chunk] = []
    last_selected = -(min_index_gap)  # ensures first chunk is always a candidate
    for c in sorted_chunks:
        if c.chunk_index >= last_selected + min_index_gap:
            selected.append(c)
            last_selected = c.chunk_index
    return selected


def apply_dedupe(chunks: list[Chunk], min_index_gap: int) -> list[Chunk]:
    """
    Applies within-document dedupe across a mixed list from many documents.
    Groups by document_id first, then applies greedy deduplication within each group.
    """
    by_doc: dict[str, list[Chunk]] = defaultdict(list)
    for c in chunks:
        by_doc[c.document_id].append(c)

    result: list[Chunk] = []
    for doc_chunks in by_doc.values():
        result.extend(select_non_overlapping_chunks(doc_chunks, min_index_gap))
    return result


# === STAGE 4.5: CELL DIAGNOSTIC ===

def print_cell_diagnostics(
    pools: dict[str, list[Chunk]],
    viable_topics: list[str],
    held_out_topic: str,
    min_cell_size: int,
) -> dict:
    """
    Prints a (topic × source) cell-count table for every quadrant after dedupe.
    Also reports per-topic and per-source marginals.

    This is the binding information for sampling: any cell below min_cell_size
    will be skipped or trigger redistribution. Called after dedupe, before
    cell-target computation.

    Returns the diagnostic data so it can be stored in the report.
    """
    import sys as _sys
    emit = lambda *args, **kw: print(*args, **kw, file=_sys.stderr)

    all_topics  = sorted({c.topic_label for chunks in pools.values() for c in chunks})
    all_sources = sorted({c.source      for chunks in pools.values() for c in chunks})
    col_w = 7
    src_w = max(len(s) for s in all_sources) if all_sources else 10

    diagnostic: dict = {}

    for q, chunks in pools.items():
        cell:         dict[tuple[str, str], int] = defaultdict(int)
        topic_total:  dict[str, int]             = defaultdict(int)
        source_total: dict[str, int]             = defaultdict(int)
        for c in chunks:
            cell[(c.topic_label, c.source)] += 1
            topic_total[c.topic_label]      += 1
            source_total[c.source]          += 1

        q_sources    = sorted({c.source for c in chunks})
        train_topics = [t for t in all_topics if t in viable_topics]

        emit(f"\n{'─' * 72}")
        emit(f"  Stage 4.5 | {q}  ({len(chunks):,} chunks after dedupe)")
        emit(f"{'─' * 72}")

        # Per-topic marginals
        emit(f"\n  {'Topic':<30}  {'Total':>{col_w}}  {'Below':>{col_w}}")
        emit(f"  {'─'*30}  {'─'*col_w}  {'─'*col_w}")
        for t in all_topics:
            n      = topic_total.get(t, 0)
            n_below = sum(1 for src in q_sources if cell.get((t, src), 0) < min_cell_size)
            marker = ("  ← held-out" if t == held_out_topic
                      else "  ← training" if t in viable_topics
                      else "  ← dropped")
            emit(f"  {t:<30}  {n:>{col_w},}  {n_below:>{col_w}}{marker}")

        # Per-source marginals
        emit(f"\n  {'Source':<{src_w}}  {'Total':>{col_w}}")
        emit(f"  {'─'*src_w}  {'─'*col_w}")
        for src in q_sources:
            emit(f"  {src:<{src_w}}  {source_total.get(src, 0):>{col_w},}")

        # Cell matrix (training topics only)
        if train_topics and q_sources:
            col_headers = "  ".join(f"{src[:col_w]:>{col_w}}" for src in q_sources)
            emit(f"\n  (topic × source) cells  [min_cell_size={min_cell_size}]")
            emit(f"  {' ' * src_w}  {col_headers}")
            emit(f"  {'─'*src_w}  {'─'*(col_w * len(q_sources) + 2 * (len(q_sources) - 1))}")
            for t in train_topics:
                row_vals = [
                    f"{'!' if cell.get((t, src), 0) < min_cell_size else ' '}"
                    f"{cell.get((t, src), 0):>{col_w - 1},}"
                    for src in q_sources
                ]
                emit(f"  {t:<{src_w}}  {'  '.join(row_vals)}")
            emit(f"\n  ! = below min_cell_size ({min_cell_size})")

        diagnostic[q] = {
            "total_chunks": len(chunks),
            "per_topic":    dict(topic_total),
            "per_source":   dict(source_total),
            "cells":        {f"{t}|{s}": n for (t, s), n in cell.items()},
            "cells_below_min_size": [
                {"topic": t, "source": s, "n": cell.get((t, s), 0)}
                for t in train_topics
                for s in q_sources
                if cell.get((t, s), 0) < min_cell_size
            ],
        }

    emit(f"\n{'─' * 72}\n")
    return diagnostic


# === STAGE 5: Q2 SAMPLING (TEMPLATE) ===

def confidence_bin_indices(chunks: list[Chunk], edges: list[float]) -> list[int]:
    """
    Assigns each chunk to a confidence bin (0 to n_bins-1) based on conf_margin.
    A chunk with conf_margin >= edges[-1] falls into the last bin.
    """
    n_bins = len(edges) - 1
    result: list[int] = []
    for c in chunks:
        assigned = n_bins - 1
        for i in range(n_bins):
            if edges[i] <= c.conf_margin < edges[i + 1]:
                assigned = i
                break
        result.append(assigned)
    return result


def sample_cell_stratified(
    candidates: list[Chunk],
    target_count: int,
    bin_edges: list[float],
    rng: random.Random,
) -> list[Chunk]:
    """
    Samples up to target_count chunks, stratified by confidence bin.

    Logic:
        - return all candidates if count <= target
        - otherwise assign bins, allocate evenly, then redistribute deficit from
          short bins to bins that have spare capacity
    """
    if len(candidates) <= target_count:
        return list(candidates)

    n_bins      = len(bin_edges) - 1
    bin_indices = confidence_bin_indices(candidates, bin_edges)

    bins: list[list[Chunk]] = [[] for _ in range(n_bins)]
    for chunk, b in zip(candidates, bin_indices):
        bins[b].append(chunk)
    for b in bins:
        rng.shuffle(b)

    # Even allocation with remainder distributed to first bins
    base    = target_count // n_bins
    rem     = target_count % n_bins
    allocs  = [base + (1 if i < rem else 0) for i in range(n_bins)]

    # First pass: take min(alloc, available), collect deficit
    pointers = [min(a, len(b)) for a, b in zip(allocs, bins)]
    deficit  = sum(a - p for a, p in zip(allocs, pointers))

    # Second pass: fill deficit from bins with spare capacity
    if deficit > 0:
        for b_idx in range(n_bins):
            spare = len(bins[b_idx]) - pointers[b_idx]
            extra = min(spare, deficit)
            pointers[b_idx] += extra
            deficit -= extra
            if deficit == 0:
                break

    sampled: list[Chunk] = []
    for b_idx in range(n_bins):
        sampled.extend(bins[b_idx][:pointers[b_idx]])
    return sampled


def _sample_topic_cells(
    topic: str,
    topic_cells: dict[str, list[Chunk]],   # source -> candidates
    topic_targets: dict[str, int],           # source -> target count
    per_cell_cap: int,
    bin_edges: list[float],
    min_cell_size: int,
    rng: random.Random,
) -> tuple[list[Chunk], dict[str, int], list[dict]]:
    """
    Samples one topic's cells, redistributing budget from sources below min_cell_size
    to viable sources within the same topic.

    Returns:
        (sampled_chunks, realized_counts, cells_below_min)
    """
    cells_below: list[dict] = []
    viable:      list[str]  = []
    deficit                  = 0

    for src, target in topic_targets.items():
        n_avail = len(topic_cells.get(src, []))
        if n_avail < min_cell_size:
            deficit += target
            cells_below.append({"topic": topic, "source": src, "n_available": n_avail})
        else:
            viable.append(src)

    result:   list[Chunk]     = []
    realized: dict[str, int]  = {}

    if not viable:
        return result, realized, cells_below

    # Redistribute deficit evenly across viable sources
    extra     = deficit // len(viable)
    remainder = deficit % len(viable)

    for i, src in enumerate(viable):
        cands = topic_cells[src]
        want  = topic_targets.get(src, 0) + extra + (1 if i < remainder else 0)
        want  = min(want, per_cell_cap)
        selected       = sample_cell_stratified(cands, want, bin_edges, rng)
        realized[src]  = len(selected)
        result.extend(selected)

    return result, realized, cells_below


def sample_q2_template(
    q2_pool: list[Chunk],
    plan: SamplingPlan,
    config_sampling: dict,
    rng: random.Random,
) -> tuple[list[Chunk], dict]:
    """
    Samples the template quadrant (left_auth) according to cell targets.
    The realized (topic, source) counts become the binding template for Q1/Q3/Q4.

    Returns:
        (sampled_chunks, realized_template)
        realized_template: "<topic>|<source>" -> realized count
    """
    viable_topics    = config_sampling["viable_topics"]
    bin_edges        = config_sampling["confidence_bins"]["edges"]
    min_cell_size    = config_sampling["min_cell_size"]
    per_topic_target = plan.target_n_per_quadrant / plan.n_topics
    per_cell_cap     = math.floor(per_topic_target * plan.source_cap_pct)

    # Group pool by (topic, source)
    by_ts: dict[str, dict[str, list[Chunk]]] = defaultdict(lambda: defaultdict(list))
    for c in q2_pool:
        if c.topic_label in viable_topics:
            by_ts[c.topic_label][c.source].append(c)

    # Build per-cell targets for TEMPLATE_QUADRANT
    q_targets: dict[tuple[str, str], int] = {
        (ct.topic, ct.source): ct.target_count
        for ct in plan.cell_targets
        if ct.quadrant == TEMPLATE_QUADRANT
    }

    all_sampled:       list[Chunk]     = []
    realized_template: dict[str, int]  = {}
    all_cells_below:   list[dict]      = []

    for topic in viable_topics:
        topic_cells   = dict(by_ts.get(topic, {}))
        topic_targets = {src: q_targets.get((topic, src), 0) for src in topic_cells}

        sampled, realized, cells_below = _sample_topic_cells(
            topic, topic_cells, topic_targets, per_cell_cap, bin_edges, min_cell_size, rng
        )
        all_sampled.extend(sampled)
        all_cells_below.extend(cells_below)
        for src, n in realized.items():
            realized_template[f"{topic}|{src}"] = n

    log.info(
        f"Stage 5 | {TEMPLATE_QUADRANT} template: "
        f"{len(all_sampled):,} chunks, {len(all_cells_below)} cells below min_size"
    )
    return all_sampled, realized_template


# === STAGE 6: Q1/Q3/Q4 SAMPLING (MATCH Q2) ===

def sample_quadrant_to_template(
    pool: list[Chunk],
    quadrant: str,
    template: dict,
    config_sampling: dict,
    rng: random.Random,
) -> tuple[list[Chunk], list[dict]]:
    """
    Samples a non-template quadrant to match Q2's realized cell counts.
    If a source from the template is absent in this pool, its budget is redistributed
    within the same topic to other available sources, exactly as in Stage 5.

    Returns:
        (sampled_chunks, deviations_from_template)
    """
    viable_topics    = config_sampling["viable_topics"]
    bin_edges        = config_sampling["confidence_bins"]["edges"]
    min_cell_size    = config_sampling["min_cell_size"]
    per_topic_target = config_sampling["target_n_per_quadrant"] / len(viable_topics)
    per_cell_cap     = math.floor(per_topic_target * config_sampling["source_cap_pct"])

    # Group pool by (topic, source)
    by_ts: dict[str, dict[str, list[Chunk]]] = defaultdict(lambda: defaultdict(list))
    for c in pool:
        if c.topic_label in viable_topics:
            by_ts[c.topic_label][c.source].append(c)

    # Parse template into per-topic dicts: topic -> {source: count}
    template_by_topic: dict[str, dict[str, int]] = defaultdict(dict)
    for key, count in template.items():
        topic, src = key.split("|", 1)
        template_by_topic[topic][src] = count

    all_sampled: list[Chunk] = []
    deviations:  list[dict]  = []

    for topic in viable_topics:
        # Template targets for this topic — may name sources absent in this quadrant
        topic_template = template_by_topic.get(topic, {})
        # Pool only has the sources actually present; template sources not present get 0 candidates
        topic_cells = dict(by_ts.get(topic, {}))

        sampled, realized, _ = _sample_topic_cells(
            topic, topic_cells, topic_template, per_cell_cap, bin_edges, min_cell_size, rng
        )
        all_sampled.extend(sampled)

        # Record per-source deviations from template
        for src, wanted in topic_template.items():
            got = realized.get(src, 0)
            if got != wanted:
                deviations.append({
                    "quadrant": quadrant, "topic": topic, "source": src,
                    "template": wanted, "realized": got, "delta": got - wanted,
                })

    log.info(
        f"Stage 6 | {quadrant}: {len(all_sampled):,} chunks, "
        f"{len(deviations)} deviations from template"
    )
    return all_sampled, deviations


# === STAGE 7: TRAIN / VAL_INDIST SPLIT ===

def document_level_split(
    chunks: list[Chunk],
    val_pct: float,
    rng: random.Random,
) -> tuple[list[Chunk], list[Chunk]]:
    """
    Splits chunks into train and val_indist at the document level.
    No document_id appears in both splits — guarantees no leakage.

    Logic:
        Shuffle unique document_ids, place val_pct of them in val,
        assign each chunk by its document_id.
    """
    doc_ids   = list({c.document_id for c in chunks})
    rng.shuffle(doc_ids)
    n_val     = max(1, math.ceil(len(doc_ids) * val_pct))
    val_docs  = set(doc_ids[:n_val])

    train  = [c for c in chunks if c.document_id not in val_docs]
    val_in = [c for c in chunks if c.document_id in val_docs]
    return train, val_in


# === STAGE 8: SANITY CHECKS ===

def _check_source_cap(
    splits: dict[str, QuadrantSplit], max_pct: float
) -> list[dict]:
    results = []
    for q, split in splits.items():
        counts: dict[str, int] = defaultdict(int)
        for c in split.train:
            counts[c.source] += 1
        total = len(split.train) or 1
        worst_src = max(counts, key=counts.get) if counts else "n/a"
        worst_pct = counts.get(worst_src, 0) / total
        passed    = worst_pct <= max_pct
        results.append({
            "check_name": "SOURCE_CAP", "quadrant": q, "passed": passed,
            "details": {"worst_source": worst_src, "pct": round(worst_pct, 4), "limit": max_pct},
        })
    return results


def _kl(p: list[float], q_: list[float], eps: float = 1e-9) -> float:
    return sum(pi * math.log((pi + eps) / (qi + eps)) for pi, qi in zip(p, q_))


def _check_topic_kl(
    splits: dict[str, QuadrantSplit], viable_topics: list[str], max_kl: float
) -> list[dict]:
    dists: dict[str, list[float]] = {}
    for q, split in splits.items():
        counts: dict[str, int] = defaultdict(int)
        for c in split.train:
            if c.topic_label in viable_topics:
                counts[c.topic_label] += 1
        total = sum(counts.values()) or 1
        dists[q] = [counts.get(t, 0) / total for t in viable_topics]

    results = []
    qs = list(splits.keys())
    for i in range(len(qs)):
        for j in range(i + 1, len(qs)):
            kl     = _kl(dists[qs[i]], dists[qs[j]])
            passed = kl <= max_kl
            results.append({
                "check_name": "TOPIC_KL", "passed": passed,
                "details": {"q_a": qs[i], "q_b": qs[j], "kl": round(kl, 6), "limit": max_kl},
            })
    return results


def _check_length_ratio(
    splits: dict[str, QuadrantSplit], max_ratio: float
) -> list[dict]:
    means = {
        q: sum(c.n_tokens for c in s.train) / len(s.train)
        for q, s in splits.items() if s.train
    }
    if not means:
        return [{"check_name": "LENGTH_RATIO", "passed": False, "details": "no train data"}]
    ratio  = max(means.values()) / max(min(means.values()), 1e-9)
    passed = ratio <= max_ratio
    return [{
        "check_name": "LENGTH_RATIO", "passed": passed,
        "details": {"means": {k: round(v, 1) for k, v in means.items()},
                    "ratio": round(ratio, 4), "limit": max_ratio},
    }]


def _check_conf_ratio(
    splits: dict[str, QuadrantSplit], max_ratio: float
) -> list[dict]:
    means = {
        q: sum(c.conf_margin for c in s.train) / len(s.train)
        for q, s in splits.items() if s.train
    }
    if not means:
        return [{"check_name": "CONF_MARGIN_RATIO", "passed": False, "details": "no train data"}]
    ratio  = max(means.values()) / max(min(means.values()), 1e-9)
    passed = ratio <= max_ratio
    return [{
        "check_name": "CONF_MARGIN_RATIO", "passed": passed,
        "details": {"means": {k: round(v, 6) for k, v in means.items()},
                    "ratio": round(ratio, 4), "limit": max_ratio},
    }]


def _check_no_doc_leakage(splits: dict[str, QuadrantSplit]) -> list[dict]:
    results = []
    for q, split in splits.items():
        train_docs = {c.document_id for c in split.train}
        for val_name, val_chunks in [
            ("val_indist", split.val_indist),
            ("val_source", split.val_source),
        ]:
            overlap = train_docs & {c.document_id for c in val_chunks}
            results.append({
                "check_name": "NO_DOC_LEAKAGE", "quadrant": q, "val_split": val_name,
                "passed": len(overlap) == 0,
                "details": {"n_leaked_docs": len(overlap)},
            })
    return results


def _check_val_topic_purity(
    splits: dict[str, QuadrantSplit], held_out_topic: str
) -> list[dict]:
    results = []
    for q, split in splits.items():
        impure = [c for c in split.val_topic if c.topic_label != held_out_topic]
        results.append({
            "check_name": "VAL_TOPIC_PURITY", "quadrant": q,
            "passed": len(impure) == 0,
            "details": {"n_impure": len(impure), "expected_topic": held_out_topic},
        })
    return results


def _check_val_source_purity(
    splits: dict[str, QuadrantSplit], held_out_sources: dict[str, str]
) -> list[dict]:
    results = []
    for q, split in splits.items():
        expected = held_out_sources[q]
        impure   = [c for c in split.val_source if c.source != expected]
        results.append({
            "check_name": "VAL_SOURCE_PURITY", "quadrant": q,
            "passed": len(impure) == 0,
            "details": {"n_impure": len(impure), "expected_source": expected},
        })
    return results


def _check_non_empty(splits: dict[str, QuadrantSplit], min_size: int) -> list[dict]:
    results = []
    for q, split in splits.items():
        for split_name, chunks in [
            ("train",      split.train),
            ("val_indist", split.val_indist),
            ("val_source", split.val_source),
            ("val_topic",  split.val_topic),
        ]:
            results.append({
                "check_name": "NON_EMPTY", "quadrant": q, "split": split_name,
                "passed": len(chunks) >= min_size,
                "details": {"n": len(chunks), "min": min_size},
            })
    return results


def run_sanity_checks(
    splits: dict[str, QuadrantSplit],
    config_sampling: dict,
) -> tuple[bool, list[dict]]:
    """
    Runs all eight sanity checks. Returns (all_passed, check_results).

    Checks:
        SOURCE_CAP        — no source > max_source_pct_train in any train split
        TOPIC_KL          — pairwise KL between quadrant topic distributions
        LENGTH_RATIO      — max / min mean n_tokens across quadrants
        CONF_MARGIN_RATIO — max / min mean conf_margin across quadrants
        NO_DOC_LEAKAGE    — zero document_id overlap between train and val splits
        VAL_TOPIC_PURITY  — val_topic contains only held_out_topic chunks
        VAL_SOURCE_PURITY — val_source contains only the configured held-out source
        NON_EMPTY         — each split has at least min_cell_size chunks
    """
    sc              = config_sampling["sanity_checks"]
    viable_topics   = config_sampling["viable_topics"]
    held_out_topic  = config_sampling["held_out_topic"]
    held_out_sources = config_sampling["held_out_sources"]
    min_cell_size   = config_sampling["min_cell_size"]

    all_results: list[dict] = []
    all_results.extend(_check_source_cap(splits, sc["max_source_pct_train"]))
    all_results.extend(_check_topic_kl(splits, viable_topics, sc["max_topic_kl_divergence"]))
    all_results.extend(_check_length_ratio(splits, sc["max_length_ratio"]))
    all_results.extend(_check_conf_ratio(splits, sc["max_conf_margin_ratio"]))
    all_results.extend(_check_no_doc_leakage(splits))
    all_results.extend(_check_val_topic_purity(splits, held_out_topic))
    all_results.extend(_check_val_source_purity(splits, held_out_sources))
    all_results.extend(_check_non_empty(splits, min_cell_size))

    all_passed = all(r["passed"] for r in all_results)
    for r in all_results:
        lvl = logging.INFO if r["passed"] else logging.WARNING
        log.log(lvl, f"Check {r['check_name']}: {'PASS' if r['passed'] else 'FAIL'} | {r.get('details', '')}")

    return all_passed, all_results


# === STAGE 9: WRITE OUTPUTS AND REPORT ===

def write_quadrant_outputs(
    quadrant: str,
    split: QuadrantSplit,
    output_dir: Path,
) -> None:
    """
    Writes the four JSONL files for one quadrant under <output_dir>/<quadrant>/.
    """
    q_dir = output_dir / quadrant
    write_jsonl(split.train,      q_dir / "train.jsonl")
    write_jsonl(split.val_indist, q_dir / "val_indist.jsonl")
    write_jsonl(split.val_source, q_dir / "val_source.jsonl")
    write_jsonl(split.val_topic,  q_dir / "val_topic.jsonl")
    log.info(
        f"Wrote {quadrant}: train={len(split.train):,}, "
        f"val_indist={len(split.val_indist):,}, "
        f"val_source={len(split.val_source):,}, "
        f"val_topic={len(split.val_topic):,}"
    )


# === MAIN ===

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build balanced expert train/val datasets from retained quadrant pools."
    )
    p.add_argument(
        "--retained-dir", required=True, type=Path,
        help="Root of quadrant-pools: <dir>/<source>/<quadrant>/retained.jsonl",
    )
    p.add_argument(
        "--output-dir", required=True, type=Path,
        help="Where to write <quadrant>/{train,val_indist,val_source,val_topic}.jsonl",
    )
    p.add_argument(
        "--config", default=CONFIG_PATH, type=Path,
        help=f"Path to config.yaml (default: {CONFIG_PATH})",
    )
    p.add_argument(
        "--report-path", type=Path, default=None,
        help="Where to write sampling_report.json (default: <output-dir>/sampling_report.json)",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Compute everything and print summary but write no dataset files.",
    )
    p.add_argument(
        "--seed", type=int, default=None,
        help="Override random seed from config.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    cfg = load_config(args.config)
    s   = cfg["validate_expert_datasets"]

    rng_seed = args.seed if args.seed is not None else s["random_seed"]
    rng      = random.Random(rng_seed)

    report_path = args.report_path or (args.output_dir / "sampling_report.json")

    report: dict[str, Any] = {
        "config_snapshot": s,
        "random_seed":     rng_seed,
        "timestamp":       datetime.now(timezone.utc).isoformat(),
    }

    # === Stage 0: Load ===
    log.info("Stage 0 | Loading retained pools …")
    lf = s["length_filter"]
    pools = {
        q: load_quadrant_pool(
            args.retained_dir, q,
            min_tokens=lf["min_tokens"],
            max_tokens=lf["max_tokens"],
        )
        for q in QUADRANTS
    }
    report["input_counts"] = {q: len(p) for q, p in pools.items()}

    # === Stage 1: Topic filter ===
    log.info("Stage 1 | Applying topic filter …")
    pools = apply_topic_filter(pools, s["viable_topics"], s["held_out_topic"])
    report["stage_1_post_topic_filter"] = {q: len(p) for q, p in pools.items()}

    # === Stage 2: Held-out carve-outs (topic first, then source) ===
    log.info("Stage 2 | Carving out held-out topic and source …")
    pools, val_topic_pools  = carve_held_out_topic(pools, s["held_out_topic"])
    pools, val_source_pools = carve_held_out_source(pools, s["held_out_sources"])
    report["stage_2_carve_outs"] = {
        "val_topic_counts":    {q: len(v) for q, v in val_topic_pools.items()},
        "val_source_counts":   {q: len(v) for q, v in val_source_pools.items()},
        "working_pool_counts": {q: len(p) for q, p in pools.items()},
    }

    # === Stage 4: Within-document dedupe (must run before stage 3 targets) ===
    if s["dedupe"]["enabled"]:
        log.info("Stage 4 | Applying within-document dedupe …")
        gap    = s["dedupe"]["min_chunk_index_gap"]
        before = {q: len(p) for q, p in pools.items()}
        pools  = {q: apply_dedupe(p, gap) for q, p in pools.items()}
        after  = {q: len(p) for q, p in pools.items()}
        report["stage_4_dedupe"] = {
            "before_per_quadrant": before, "after_per_quadrant": after,
        }
        for q in QUADRANTS:
            log.info(
                f"Stage 4 | {q}: {before[q]:,} → {after[q]:,} "
                f"({before[q] - after[q]:,} removed)"
            )
    else:
        report["stage_4_dedupe"] = {"enabled": False}

    # === Stage 4.5: Cell diagnostics ===
    log.info("Stage 4.5 | Cell diagnostics (topic × source counts after dedupe) …")
    cell_diag = print_cell_diagnostics(
        pools, s["viable_topics"], s["held_out_topic"], s["min_cell_size"]
    )
    report["stage_4_5_cell_diagnostics"] = cell_diag

    # === Stage 3: Cell target computation ===
    log.info("Stage 3 | Computing cell targets …")
    plan = compute_cell_targets(pools, s)
    report["stage_3_cell_targets"] = [
        {"quadrant": ct.quadrant, "topic": ct.topic,
         "source": ct.source, "target_count": ct.target_count}
        for ct in plan.cell_targets
    ]

    # === Stage 5: Sample template quadrant (left_auth) ===
    log.info(f"Stage 5 | Sampling template quadrant '{TEMPLATE_QUADRANT}' …")
    q_template_sampled, template = sample_q2_template(
        pools[TEMPLATE_QUADRANT], plan, s, rng
    )
    report["stage_5_q2_template"] = {
        "realized_cells":      template,
        "cells_below_min_size": [
            {"topic": ct.topic, "source": ct.source,
             "n_available": sum(
                 1 for c in pools[TEMPLATE_QUADRANT]
                 if c.topic_label == ct.topic and c.source == ct.source
             )}
            for ct in plan.cell_targets
            if ct.quadrant == TEMPLATE_QUADRANT and sum(
                1 for c in pools[TEMPLATE_QUADRANT]
                if c.topic_label == ct.topic and c.source == ct.source
            ) < s["min_cell_size"]
        ],
        "total_sampled": len(q_template_sampled),
    }
    plan.q2_realized = template

    # === Stage 6: Sample other quadrants to match template ===
    log.info("Stage 6 | Sampling other quadrants to match template …")
    other_sampled: dict[str, list[Chunk]] = {}
    stage6_report: dict[str, dict]        = {}
    for q in OTHER_QUADRANTS:
        sampled, devs = sample_quadrant_to_template(pools[q], q, template, s, rng)
        other_sampled[q] = sampled
        stage6_report[q] = {
            "total_sampled":          len(sampled),
            "deviations_from_template": devs,
        }
    report["stage_6_other_quadrants"] = stage6_report

    # === Stage 7: Train / val_indist document-level split ===
    log.info("Stage 7 | Splitting train / val_indist …")
    all_sampled = {**other_sampled, TEMPLATE_QUADRANT: q_template_sampled}
    splits: dict[str, QuadrantSplit] = {}
    stage7_report: dict[str, dict]   = {}

    for q in QUADRANTS:
        train, val_in = document_level_split(all_sampled[q], s["val_pct"], rng)
        splits[q] = QuadrantSplit(
            quadrant   = q,
            train      = train,
            val_indist = val_in,
            val_source = val_source_pools[q],
            val_topic  = val_topic_pools[q],
        )
        train_docs = {c.document_id for c in train}
        val_docs   = {c.document_id for c in val_in}
        stage7_report[q] = {
            "train_docs":    len(train_docs),
            "val_docs":      len(val_docs),
            "train_chunks":  len(train),
            "val_chunks":    len(val_in),
        }
        log.info(
            f"Stage 7 | {q}: train={len(train):,} ({len(train_docs)} docs), "
            f"val_indist={len(val_in):,} ({len(val_docs)} docs)"
        )
    report["stage_7_split"] = stage7_report

    # === Stage 8: Sanity checks ===
    log.info("Stage 8 | Running sanity checks …")
    all_passed, check_results = run_sanity_checks(splits, s)
    report["stage_8_sanity"] = {
        "all_passed": all_passed, "check_results": check_results,
    }

    # Final counts
    report["final_counts"] = {
        q: {
            "train":      len(splits[q].train),
            "val_indist": len(splits[q].val_indist),
            "val_source": len(splits[q].val_source),
            "val_topic":  len(splits[q].val_topic),
        }
        for q in QUADRANTS
    }

    # === Stage 9: Write ===
    if not all_passed and not args.dry_run:
        write_report(report, report_path)
        log.error(
            "SANITY CHECKS FAILED — dataset files were NOT written. "
            f"See report at {report_path} for details."
        )
        sys.exit(1)

    if not args.dry_run:
        log.info("Stage 9 | Writing output files …")
        for q, split in splits.items():
            write_quadrant_outputs(q, split, args.output_dir)

    write_report(report, report_path)
    log.info(f"Report written to {report_path}")

    if args.dry_run:
        log.info("Dry run — no dataset files written.")
    else:
        log.info("Done.")

    # Print final count summary
    print("\n=== Final counts ===")
    for q in QUADRANTS:
        fc = report["final_counts"][q]
        print(
            f"  {q:12s}  train={fc['train']:>5,}  "
            f"val_indist={fc['val_indist']:>4,}  "
            f"val_source={fc['val_source']:>5,}  "
            f"val_topic={fc['val_topic']:>5,}"
        )


if __name__ == "__main__":
    main()
