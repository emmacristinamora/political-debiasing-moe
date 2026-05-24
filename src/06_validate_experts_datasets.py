# src/06_validate_experts_datasets.py


# === IMPORTS ===

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import re
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


# === CONSTANTS ===

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH  = PROJECT_ROOT / "config" / "config.yaml"

QUADRANTS = ["right_auth", "left_auth", "left_lib", "right_lib"]

VALID_SOURCES = {
    "allsides",
    "ec_press",
    "ire_press",
    "uk_press",
    "hoc",
    "us_media",
    "us_speeches",
    "reddit_liberal",
    "reddit_conservative",
}

DEFAULT_SOURCE_GROUPS = {
    "reddit": [
        "reddit_liberal",
        "reddit_conservative",
    ],
    "press": [
        "allsides",
        "ec_press",
        "ire_press",
        "uk_press",
        "us_media",
    ],
    "speeches": [
        "hoc",
        "us_speeches",
    ],
}

DEFAULT_BOILERPLATE_PATTERNS = [
    r"\bAPA style\b",
    r"\breference page\b",
    r"\byour paper\b",
    r"\bassignment guidelines\b",
    r"\bwrite a short argument\b",
    r"\bcomplete the survey\b",
    r"\bsubmit your ideas\b",
    r"\bwe want to hear from you\b",
    r"\bdownload the submission form\b",
    r"\brelated policy papers\b",
    r"\bnewsletter\b",
    r"\bsubscribe\b",
]

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO,
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# === DATACLASSES ===

@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    document_id: str
    chunk_index: int
    text: str
    source: str
    topic_label: str
    s_econ: float
    s_soc: float
    conf_margin: float
    n_tokens: int
    quadrant: str


@dataclass(frozen=True)
class SamplingConfig:
    viable_topics: list[str]
    held_out_topic: str
    held_out_sources: dict[str, str]
    val_pct: float
    random_seed: int
    max_cell_size: int
    max_source_pct_train: float
    max_source_group_pct_train: float
    max_topic_pct_train: float
    min_tokens: int
    max_tokens: int
    dedupe_enabled: bool
    dedupe_min_chunk_index_gap: int
    source_groups: dict[str, list[str]]
    boilerplate_filter_enabled: bool
    boilerplate_patterns: list[str]
    sanity_checks: dict[str, Any]


@dataclass
class QuadrantSplit:
    quadrant: str
    train: list[Chunk]
    val_indist: list[Chunk]
    val_source: list[Chunk]
    val_topic: list[Chunk]


@dataclass
class SamplingState:
    selected: list[Chunk]
    selected_chunk_ids: set[str]
    source_counts: dict[str, int]
    source_group_counts: dict[str, int]
    topic_counts: dict[str, int]


# === CONFIG LOADING ===

_REQUIRED_KEYS = {
    "viable_topics",
    "held_out_topic",
    "held_out_sources",
    "val_pct",
    "random_seed",
    "dedupe",
    "sanity_checks",
    "length_filter",
}


def load_config(path: Path) -> dict:
    """
    Read and validate config.yaml.
    Args:
        path: path to config.yaml.
    Returns:
        Full parsed config dictionary.
    """
    with path.open(encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    section = cfg.get("validate_expert_datasets", {})
    missing = _REQUIRED_KEYS - set(section.keys())
    if missing:
        raise ValueError(
            f"config.yaml missing keys under validate_expert_datasets: {sorted(missing)}"
        )

    held_out_sources = section["held_out_sources"]
    missing_quadrants = set(QUADRANTS) - set(held_out_sources.keys())
    if missing_quadrants:
        raise ValueError(
            f"held_out_sources missing quadrants: {sorted(missing_quadrants)}"
        )

    unknown_sources = set(held_out_sources.values()) - VALID_SOURCES
    if unknown_sources:
        raise ValueError(
            f"held_out_sources contains unknown sources: {sorted(unknown_sources)}"
        )

    if section["held_out_topic"] in section["viable_topics"]:
        raise ValueError(
            f"held_out_topic '{section['held_out_topic']}' is also in viable_topics"
        )

    if not 0 < section["val_pct"] < 0.5:
        raise ValueError(f"val_pct must be in (0, 0.5), got {section['val_pct']}")

    return cfg


def build_sampling_config(raw_config: dict) -> SamplingConfig:
    section = raw_config["validate_expert_datasets"]
    length_filter = section["length_filter"]
    dedupe = section["dedupe"]
    sanity_checks = section["sanity_checks"]

    source_groups = section.get("source_groups", DEFAULT_SOURCE_GROUPS)

    max_source_pct_train = section.get(
        "max_source_pct_train",
        sanity_checks.get("max_source_pct_train", 0.70),
    )
    max_source_group_pct_train = section.get(
        "max_source_group_pct_train",
        sanity_checks.get("max_source_group_pct_train", 0.65),
    )
    max_topic_pct_train = section.get(
        "max_topic_pct_train",
        sanity_checks.get("max_topic_pct_train", 0.50),
    )
    max_cell_size = section.get("max_cell_size", 250)
    boilerplate_cfg = section.get("boilerplate_filter", {})

    validate_source_groups(source_groups)

    return SamplingConfig(
        viable_topics=section["viable_topics"],
        held_out_topic=section["held_out_topic"],
        held_out_sources=section["held_out_sources"],
        val_pct=section["val_pct"],
        random_seed=section["random_seed"],
        max_cell_size=max_cell_size,
        max_source_pct_train=max_source_pct_train,
        max_source_group_pct_train=max_source_group_pct_train,
        max_topic_pct_train=max_topic_pct_train,
        min_tokens=length_filter["min_tokens"],
        max_tokens=length_filter["max_tokens"],
        dedupe_enabled=dedupe.get("enabled", False),
        dedupe_min_chunk_index_gap=dedupe.get("min_chunk_index_gap", 2),
        source_groups=source_groups,
        boilerplate_filter_enabled=boilerplate_cfg.get("enabled", True),
        boilerplate_patterns=boilerplate_cfg.get(
            "patterns",
            DEFAULT_BOILERPLATE_PATTERNS,
        ),
        sanity_checks=sanity_checks,
    )


def validate_source_groups(source_groups: dict[str, list[str]]) -> None:
    seen_sources: set[str] = set()

    for group, sources in source_groups.items():
        if not sources:
            raise ValueError(f"source group '{group}' is empty")

        unknown = set(sources) - VALID_SOURCES
        if unknown:
            raise ValueError(
                f"source group '{group}' contains unknown sources: {sorted(unknown)}"
            )

        duplicated = seen_sources & set(sources)
        if duplicated:
            raise ValueError(
                f"sources assigned to multiple groups: {sorted(duplicated)}"
            )

        seen_sources.update(sources)


def build_source_to_group(source_groups: dict[str, list[str]]) -> dict[str, str]:
    return {
        source: group
        for group, sources in source_groups.items()
        for source in sources
    }


# === I/O HELPERS ===

def _parse_chunk_index(chunk_id: str) -> int:
    try:
        return int(chunk_id.rsplit("_chunk", 1)[1])
    except (IndexError, ValueError) as exc:
        raise ValueError(
            f"Cannot parse chunk_index from chunk_id '{chunk_id}'. "
            "Expected suffix '_chunk<digits>'."
        ) from exc


def load_quadrant_pool(
    retained_dir: Path,
    quadrant: str,
    min_tokens: int,
    max_tokens: int,
) -> list[Chunk]:
    """
    Load retained chunks for one quadrant.
    Args:
        retained_dir: root path with layout <source>/<quadrant>/retained.jsonl.
        quadrant: quadrant name.
        min_tokens: minimum token length.
        max_tokens: maximum token length.
    Returns:
        List of Chunk objects.
    """
    chunks: list[Chunk] = []
    dropped_length = 0

    for source_dir in sorted(retained_dir.iterdir()):
        if not source_dir.is_dir():
            continue

        source = source_dir.name
        if source not in VALID_SOURCES:
            log.warning("skipping unknown source directory: %s", source)
            continue

        path = source_dir / quadrant / "retained.jsonl"
        if not path.exists():
            continue

        with path.open(encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue

                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON at {path}:{lineno}: {exc}") from exc

                n_tokens = int(row["n_tokens"])
                if not min_tokens <= n_tokens <= max_tokens:
                    dropped_length += 1
                    continue

                chunk_id = row["chunk_id"]
                chunks.append(Chunk(
                    chunk_id=chunk_id,
                    document_id=row["document_id"],
                    chunk_index=_parse_chunk_index(chunk_id),
                    text=row["text"],
                    source=source,
                    topic_label=row["topic_primary"],
                    s_econ=float(row["score_econ"]),
                    s_soc=float(row["score_soc"]),
                    conf_margin=float(row["confidence_margin"]),
                    n_tokens=n_tokens,
                    quadrant=quadrant,
                ))

    log.info(
        "Loaded %s: %d chunks (%d dropped by length)",
        quadrant,
        len(chunks),
        dropped_length,
    )
    return chunks


def write_jsonl(chunks: list[Chunk], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for chunk in chunks:
            row = asdict(chunk)
            row.pop("quadrant")
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)


# === SOURCE GROUP HELPERS ===

def get_source_group(
    source: str,
    source_to_group: dict[str, str],
) -> str:
    try:
        return source_to_group[source]
    except KeyError as exc:
        raise ValueError(f"No source group defined for source: {source}") from exc


def compute_source_group_counts(
    chunks: list[Chunk],
    source_groups: dict[str, list[str]],
    source_to_group: dict[str, str],
) -> dict[str, int]:
    counts = {group: 0 for group in source_groups}

    for chunk in chunks:
        group = get_source_group(chunk.source, source_to_group)
        counts[group] += 1

    return counts


def compute_source_group_distribution(
    chunks: list[Chunk],
    source_groups: dict[str, list[str]],
    source_to_group: dict[str, str],
) -> dict[str, float]:
    counts = compute_source_group_counts(
        chunks,
        source_groups,
        source_to_group,
    )
    total = sum(counts.values()) or 1

    return {
        group: round(count / total, 6)
        for group, count in counts.items()
    }


def clip_and_renormalize_proportions(
    raw_props: dict[str, float],
    min_pct: float,
    max_pct: float,
) -> dict[str, float]:
    clipped = {
        label: min(max(prop, min_pct), max_pct)
        for label, prop in raw_props.items()
    }

    total = sum(clipped.values()) or 1

    return {
        label: clipped[label] / total
        for label in clipped
    }


def choose_reference_group_proportions(
    working_pools: dict[str, list[Chunk]],
    config: SamplingConfig,
    source_to_group: dict[str, str],
) -> tuple[str, dict[str, float], dict[str, Any]]:
    """
    Choose the weakest feasible quadrant and derive source-group proportions from it.

    The reference quadrant is selected using effective source-group capacity, not
    only raw count. This avoids choosing a quadrant that is large but almost
    entirely concentrated in one source family.
    """
    group_counts_by_quadrant = {
        quadrant: compute_source_group_counts(
            chunks,
            config.source_groups,
            source_to_group,
        )
        for quadrant, chunks in working_pools.items()
    }

    def effective_capacity(counts: dict[str, int]) -> int:
        group_cap = math.floor(
            config.target_n_per_quadrant * config.max_source_group_pct
        )
        return sum(min(count, group_cap) for count in counts.values())

    capacities = {
        quadrant: effective_capacity(counts)
        for quadrant, counts in group_counts_by_quadrant.items()
    }

    reference_quadrant = min(capacities, key=capacities.get)

    ref_counts = group_counts_by_quadrant[reference_quadrant]
    ref_total = sum(ref_counts.values()) or 1

    raw_props = {
        group: ref_counts[group] / ref_total
        for group in config.source_groups
    }

    matched_props = clip_and_renormalize_proportions(
        raw_props,
        config.min_source_group_pct,
        config.max_source_group_pct,
    )

    report = {
        "reference_quadrant": reference_quadrant,
        "group_counts_by_quadrant": group_counts_by_quadrant,
        "effective_capacities": capacities,
        "raw_reference_props": {
            group: round(prop, 6)
            for group, prop in raw_props.items()
        },
        "matched_reference_props": {
            group: round(prop, 6)
            for group, prop in matched_props.items()
        },
    }

    log.info(
        "Source-group reference quadrant=%s | props=%s",
        reference_quadrant,
        report["matched_reference_props"],
    )

    return reference_quadrant, matched_props, report


# === FILTERING ===

def apply_topic_filter(
    pools: dict[str, list[Chunk]],
    viable_topics: list[str],
    held_out_topic: str,
) -> dict[str, list[Chunk]]:
    keep_topics = set(viable_topics) | {held_out_topic}
    filtered_pools: dict[str, list[Chunk]] = {}

    for quadrant, chunks in pools.items():
        filtered = [chunk for chunk in chunks if chunk.topic_label in keep_topics]
        log.info(
            "Stage 1 | %s topic filter: %d -> %d",
            quadrant,
            len(chunks),
            len(filtered),
        )
        filtered_pools[quadrant] = filtered

    return filtered_pools


def compile_boilerplate_patterns(patterns: list[str]) -> list[re.Pattern]:
    return [re.compile(pattern, flags=re.IGNORECASE) for pattern in patterns]


def apply_boilerplate_filter(
    pools: dict[str, list[Chunk]],
    patterns: list[re.Pattern],
) -> tuple[dict[str, list[Chunk]], dict[str, dict[str, int]]]:
    filtered_pools: dict[str, list[Chunk]] = {}
    report: dict[str, dict[str, int]] = {}

    for quadrant, chunks in pools.items():
        kept: list[Chunk] = []
        dropped_by_pattern: dict[str, int] = defaultdict(int)

        for chunk in chunks:
            matched = None
            for pattern in patterns:
                if pattern.search(chunk.text):
                    matched = pattern.pattern
                    break

            if matched is None:
                kept.append(chunk)
            else:
                dropped_by_pattern[matched] += 1

        filtered_pools[quadrant] = kept
        report[quadrant] = dict(dropped_by_pattern)

        log.info(
            "Stage 1.5 | %s boilerplate filter: %d -> %d (%d dropped)",
            quadrant,
            len(chunks),
            len(kept),
            len(chunks) - len(kept),
        )

    return filtered_pools, report


# === HELD-OUT SPLITS ===

def carve_held_out_topic(
    pools: dict[str, list[Chunk]],
    held_out_topic: str,
) -> tuple[dict[str, list[Chunk]], dict[str, list[Chunk]]]:
    working_pools: dict[str, list[Chunk]] = {}
    val_topic_pools: dict[str, list[Chunk]] = {}

    for quadrant, chunks in pools.items():
        working_pools[quadrant] = [
            chunk for chunk in chunks if chunk.topic_label != held_out_topic
        ]
        val_topic_pools[quadrant] = [
            chunk for chunk in chunks if chunk.topic_label == held_out_topic
        ]

        log.info(
            "Stage 2a | %s held-out topic=%s: working=%d val_topic=%d",
            quadrant,
            held_out_topic,
            len(working_pools[quadrant]),
            len(val_topic_pools[quadrant]),
        )

    return working_pools, val_topic_pools


def carve_held_out_source(
    working_pools: dict[str, list[Chunk]],
    held_out_sources: dict[str, str],
) -> tuple[dict[str, list[Chunk]], dict[str, list[Chunk]]]:
    final_working: dict[str, list[Chunk]] = {}
    val_source_pools: dict[str, list[Chunk]] = {}

    for quadrant, chunks in working_pools.items():
        held_out_source = held_out_sources[quadrant]

        final_working[quadrant] = [
            chunk for chunk in chunks if chunk.source != held_out_source
        ]
        val_source_pools[quadrant] = [
            chunk for chunk in chunks if chunk.source == held_out_source
        ]

        log.info(
            "Stage 2b | %s held-out source=%s: working=%d val_source=%d",
            quadrant,
            held_out_source,
            len(final_working[quadrant]),
            len(val_source_pools[quadrant]),
        )

    return final_working, val_source_pools


# === DEDUPE ===

def select_non_overlapping_chunks(
    chunks: list[Chunk],
    min_index_gap: int,
) -> list[Chunk]:
    sorted_chunks = sorted(chunks, key=lambda chunk: chunk.chunk_index)
    selected: list[Chunk] = []
    last_selected = -min_index_gap

    for chunk in sorted_chunks:
        if chunk.chunk_index >= last_selected + min_index_gap:
            selected.append(chunk)
            last_selected = chunk.chunk_index

    return selected


def apply_dedupe(chunks: list[Chunk], min_index_gap: int) -> list[Chunk]:
    by_document: dict[str, list[Chunk]] = defaultdict(list)
    for chunk in chunks:
        by_document[chunk.document_id].append(chunk)

    selected: list[Chunk] = []
    for document_chunks in by_document.values():
        selected.extend(select_non_overlapping_chunks(document_chunks, min_index_gap))

    return selected


def apply_dedupe_to_pools(
    pools: dict[str, list[Chunk]],
    min_index_gap: int,
    label: str,
) -> tuple[dict[str, list[Chunk]], dict[str, dict[str, int]]]:
    deduped: dict[str, list[Chunk]] = {}
    report: dict[str, dict[str, int]] = {}

    for quadrant, chunks in pools.items():
        reduced = apply_dedupe(chunks, min_index_gap)
        deduped[quadrant] = reduced
        report[quadrant] = {
            "before": len(chunks),
            "after": len(reduced),
            "removed": len(chunks) - len(reduced),
        }

        log.info(
            "Stage 3 | %s dedupe %s: %d -> %d",
            quadrant,
            label,
            len(chunks),
            len(reduced),
        )

    return deduped, report


# === DIAGNOSTICS ===

def summarize_chunks(
    chunks: list[Chunk],
    source_to_group: dict[str, str] | None = None,
    source_groups: dict[str, list[str]] | None = None,
) -> dict[str, Any]:
    if not chunks:
        return {
            "n": 0,
            "mean_tokens": 0.0,
            "mean_conf_margin": 0.0,
            "mean_econ": 0.0,
            "mean_soc": 0.0,
            "sources": {},
            "source_groups": {},
            "topics": {},
        }

    summary = {
        "n": len(chunks),
        "mean_tokens": round(sum(chunk.n_tokens for chunk in chunks) / len(chunks), 3),
        "mean_conf_margin": round(
            sum(chunk.conf_margin for chunk in chunks) / len(chunks),
            6,
        ),
        "mean_econ": round(sum(chunk.s_econ for chunk in chunks) / len(chunks), 6),
        "mean_soc": round(sum(chunk.s_soc for chunk in chunks) / len(chunks), 6),
        "sources": dict(Counter(chunk.source for chunk in chunks)),
        "topics": dict(Counter(chunk.topic_label for chunk in chunks)),
    }

    if source_to_group is not None and source_groups is not None:
        summary["source_groups"] = compute_source_group_counts(
            chunks,
            source_groups,
            source_to_group,
        )
    else:
        summary["source_groups"] = {}

    return summary


def build_cell_diagnostics(
    pools: dict[str, list[Chunk]],
    viable_topics: list[str],
    source_to_group: dict[str, str],
) -> dict[str, Any]:
    diagnostics: dict[str, Any] = {}

    for quadrant, chunks in pools.items():
        topic_source_cells: dict[str, int] = defaultdict(int)
        topic_group_cells: dict[str, int] = defaultdict(int)

        for chunk in chunks:
            if chunk.topic_label not in viable_topics:
                continue

            group = get_source_group(chunk.source, source_to_group)

            topic_source_cells[f"{chunk.topic_label}|{chunk.source}"] += 1
            topic_group_cells[f"{chunk.topic_label}|{group}"] += 1

        diagnostics[quadrant] = {
            "summary": summarize_chunks(chunks),
            "topic_source_cells": dict(topic_source_cells),
            "topic_group_cells": dict(topic_group_cells),
        }

    return diagnostics


# === SAMPLING HELPERS ===

def confidence_rank_candidates(
    candidates: list[Chunk],
    top_frac: float,
    target_count: int,
    rng: random.Random,
) -> list[Chunk]:
    if len(candidates) <= target_count:
        ranked = list(candidates)
        rng.shuffle(ranked)
        return ranked

    ranked = sorted(candidates, key=lambda chunk: chunk.conf_margin, reverse=True)
    cutoff = max(target_count, math.ceil(len(ranked) * top_frac))
    eligible = ranked[:cutoff]
    rng.shuffle(eligible)
    return eligible


def allocate_integer_counts(
    total: int,
    labels: list[str],
) -> dict[str, int]:
    if not labels:
        return {}

    base = total // len(labels)
    remainder = total % len(labels)

    return {
        label: base + (1 if index < remainder else 0)
        for index, label in enumerate(labels)
    }


def allocate_from_proportions(
    total: int,
    proportions: dict[str, float],
) -> dict[str, int]:
    raw = {
        label: total * prop
        for label, prop in proportions.items()
    }

    floors = {
        label: math.floor(value)
        for label, value in raw.items()
    }

    remainder = total - sum(floors.values())

    ranked = sorted(
        raw,
        key=lambda label: raw[label] - floors[label],
        reverse=True,
    )

    for label in ranked[:remainder]:
        floors[label] += 1

    return floors


def source_cap(
    target_n: int,
    max_source_pct: float,
) -> int:
    return max(1, math.floor(target_n * max_source_pct))


def source_group_cap(
    target_n: int,
    max_source_group_pct: float,
) -> int:
    return max(1, math.floor(target_n * max_source_group_pct))


def topic_cap(
    target_n: int,
    max_topic_pct: float,
) -> int:
    return max(1, math.floor(target_n * max_topic_pct))


def can_add_chunk(
    chunk: Chunk,
    state: SamplingState,
    target_n: int,
    max_source_pct: float,
    max_source_group_pct: float,
    max_topic_pct: float,
    source_to_group: dict[str, str],
    enforce_group_cap: bool = True,
) -> bool:
    if chunk.chunk_id in state.selected_chunk_ids:
        return False

    group = get_source_group(chunk.source, source_to_group)

    if state.source_counts.get(chunk.source, 0) >= source_cap(
        target_n,
        max_source_pct,
    ):
        return False

    if enforce_group_cap and state.source_group_counts.get(group, 0) >= source_group_cap(
        target_n,
        max_source_group_pct,
    ):
        return False

    if state.topic_counts.get(chunk.topic_label, 0) >= topic_cap(
        target_n,
        max_topic_pct,
    ):
        return False

    return True


def add_chunk(
    chunk: Chunk,
    state: SamplingState,
    source_to_group: dict[str, str],
) -> None:
    group = get_source_group(chunk.source, source_to_group)

    state.selected.append(chunk)
    state.selected_chunk_ids.add(chunk.chunk_id)

    state.source_counts[chunk.source] = state.source_counts.get(chunk.source, 0) + 1
    state.source_group_counts[group] = state.source_group_counts.get(group, 0) + 1
    state.topic_counts[chunk.topic_label] = state.topic_counts.get(chunk.topic_label, 0) + 1


def group_chunks_by_source_group_topic(
    chunks: list[Chunk],
    viable_topics: list[str],
    source_groups: dict[str, list[str]],
    source_to_group: dict[str, str],
) -> dict[str, dict[str, list[Chunk]]]:
    viable_set = set(viable_topics)

    grouped: dict[str, dict[str, list[Chunk]]] = {
        group: {topic: [] for topic in viable_topics}
        for group in source_groups
    }

    for chunk in chunks:
        if chunk.topic_label not in viable_set:
            continue

        group = get_source_group(chunk.source, source_to_group)
        grouped[group][chunk.topic_label].append(chunk)

    return grouped


# === SAMPLING ===

def sample_source_group_matched_quadrant(
    chunks: list[Chunk],
    quadrant: str,
    config: SamplingConfig,
    source_group_props: dict[str, float],
    source_to_group: dict[str, str],
    rng: random.Random,
) -> tuple[list[Chunk], dict[str, Any]]:
    """
    Sample one quadrant using shared source-group proportions.

    Logic:
        1. Allocate the target size across source groups using reference proportions.
        2. Inside each source group, allocate roughly equal topic budgets.
        3. Sample high-confidence candidates for each group-topic cell.
        4. Top up within source group.
        5. Final top-up from any legal remaining candidate.
    """
    target_n = config.target_n_per_quadrant
    source_group_budget = allocate_from_proportions(target_n, source_group_props)

    grouped = group_chunks_by_source_group_topic(
        chunks,
        config.viable_topics,
        config.source_groups,
        source_to_group,
    )

    state = SamplingState(
        selected=[],
        selected_chunk_ids=set(),
        source_counts={},
        source_group_counts={},
        topic_counts={},
    )

    report: dict[str, Any] = {
        "quadrant": quadrant,
        "target_n": target_n,
        "source_group_props": {
            group: round(prop, 6)
            for group, prop in source_group_props.items()
        },
        "source_group_budget": source_group_budget,
        "group_stage": {},
        "final_top_up": {},
    }

    for group, group_target in source_group_budget.items():
        selected_before_group = len(state.selected)
        topic_budget = allocate_integer_counts(group_target, config.viable_topics)

        report["group_stage"][group] = {
            "target": group_target,
            "topic_budget": topic_budget,
            "topic_stage": {},
            "group_top_up": {},
        }

        for topic, topic_target in topic_budget.items():
            candidates = grouped.get(group, {}).get(topic, [])

            selected_before_topic = len(state.selected)

            if len(candidates) < config.min_cell_size:
                report["group_stage"][group]["topic_stage"][topic] = {
                    "target": topic_target,
                    "available": len(candidates),
                    "selected": 0,
                    "skipped_min_cell": True,
                }
                continue

            ranked = confidence_rank_candidates(
                candidates,
                config.confidence_top_frac,
                topic_target,
                rng,
            )

            for candidate in ranked:
                current_group_count = state.source_group_counts.get(group, 0)
                current_topic_group_added = len(state.selected) - selected_before_topic

                if current_group_count >= group_target:
                    break

                if current_topic_group_added >= topic_target:
                    break

                if can_add_chunk(
                    candidate,
                    state,
                    target_n,
                    config.max_source_pct_train,
                    config.max_source_group_pct_train,
                    config.max_topic_pct_train,
                    source_to_group,
                    enforce_group_cap=True,
                ):
                    add_chunk(candidate, state, source_to_group)

            report["group_stage"][group]["topic_stage"][topic] = {
                "target": topic_target,
                "available": len(candidates),
                "selected": len(state.selected) - selected_before_topic,
                "skipped_min_cell": False,
            }

        group_shortfall = group_target - state.source_group_counts.get(group, 0)

        if group_shortfall > 0:
            group_remaining = [
                chunk
                for topic_chunks in grouped.get(group, {}).values()
                for chunk in topic_chunks
                if chunk.chunk_id not in state.selected_chunk_ids
            ]

            group_remaining = confidence_rank_candidates(
                group_remaining,
                config.confidence_top_frac,
                group_shortfall,
                rng,
            )

            selected_before_top_up = len(state.selected)

            for candidate in group_remaining:
                if state.source_group_counts.get(group, 0) >= group_target:
                    break

                if can_add_chunk(
                    candidate,
                    state,
                    target_n,
                    config.max_source_pct_train,
                    config.max_source_group_pct_train,
                    config.max_topic_pct_train,
                    source_to_group,
                    enforce_group_cap=True,
                ):
                    add_chunk(candidate, state, source_to_group)

            report["group_stage"][group]["group_top_up"] = {
                "needed": group_shortfall,
                "selected": len(state.selected) - selected_before_top_up,
                "final_group_n": state.source_group_counts.get(group, 0),
            }

        report["group_stage"][group]["selected"] = (
            len(state.selected) - selected_before_group
        )

    if len(state.selected) < target_n:
        all_remaining = [
            chunk for chunk in chunks
            if chunk.topic_label in config.viable_topics
            and chunk.chunk_id not in state.selected_chunk_ids
        ]

        all_remaining = confidence_rank_candidates(
            all_remaining,
            config.confidence_top_frac,
            target_n - len(state.selected),
            rng,
        )

        selected_before_final_top_up = len(state.selected)

        for candidate in all_remaining:
            if len(state.selected) >= target_n:
                break

            if can_add_chunk(
                candidate,
                state,
                target_n,
                config.max_source_pct_train,
                config.max_source_group_pct_train,
                config.max_topic_pct_train,
                source_to_group,
                enforce_group_cap=False,
            ):
                add_chunk(candidate, state, source_to_group)

        report["final_top_up"] = {
            "needed": target_n - selected_before_final_top_up,
            "selected": len(state.selected) - selected_before_final_top_up,
            "final_n": len(state.selected),
        }

    report["final"] = summarize_chunks(
        state.selected,
        source_to_group=source_to_group,
        source_groups=config.source_groups,
    )

    if len(state.selected) < target_n:
        log.warning(
            "Stage 5 | %s sampled only %d/%d chunks under caps",
            quadrant,
            len(state.selected),
            target_n,
        )
    else:
        log.info(
            "Stage 5 | %s sampled %d/%d chunks",
            quadrant,
            len(state.selected),
            target_n,
        )

    return state.selected, report


# === CELL-CAP SAMPLING ===

def sample_source_topic_capped_quadrant(
    chunks: list[Chunk],
    quadrant: str,
    config: SamplingConfig,
    rng: random.Random,
) -> tuple[list[Chunk], dict[str, Any]]:
    """
    Sample one quadrant by capping each source × topic cell at max_cell_size.

    For every (source, topic) cell:
      - keep all chunks if n <= max_cell_size
      - randomly sample max_cell_size chunks if n > max_cell_size

    This prevents any single (source, topic) pair from dominating an expert
    while preserving all available diversity from small cells.
    """
    grouped: dict[tuple[str, str], list[Chunk]] = defaultdict(list)
    for chunk in chunks:
        if chunk.topic_label in config.viable_topics:
            grouped[(chunk.source, chunk.topic_label)].append(chunk)

    selected: list[Chunk] = []
    cell_report: dict[str, Any] = {}

    for (source, topic), candidates in sorted(grouped.items()):
        cell_key = f"{source}|{topic}"
        if len(candidates) <= config.max_cell_size:
            chosen = list(candidates)
            rng.shuffle(chosen)
        else:
            chosen = rng.sample(candidates, config.max_cell_size)
        selected.extend(chosen)
        cell_report[cell_key] = {
            "available": len(candidates),
            "selected": len(chosen),
            "capped": len(candidates) > config.max_cell_size,
        }

    rng.shuffle(selected)

    report = {
        "quadrant": quadrant,
        "max_cell_size": config.max_cell_size,
        "n_selected": len(selected),
        "n_cells": len(cell_report),
        "n_capped": sum(1 for v in cell_report.values() if v["capped"]),
        "cells": cell_report,
        "final": summarize_chunks(selected),
    }

    log.info(
        "Stage 5 | %s cell-cap sampled %d chunks from %d cells (%d capped)",
        quadrant,
        len(selected),
        len(cell_report),
        report["n_capped"],
    )

    return selected, report


# === SPLITTING ===

def document_level_split(
    chunks: list[Chunk],
    val_pct: float,
    rng: random.Random,
) -> tuple[list[Chunk], list[Chunk]]:
    document_ids = list({chunk.document_id for chunk in chunks})
    rng.shuffle(document_ids)

    n_val_docs = max(1, math.ceil(len(document_ids) * val_pct))
    val_document_ids = set(document_ids[:n_val_docs])

    train = [
        chunk for chunk in chunks
        if chunk.document_id not in val_document_ids
    ]
    val_indist = [
        chunk for chunk in chunks
        if chunk.document_id in val_document_ids
    ]

    return train, val_indist


def remove_train_doc_leakage(
    train: list[Chunk],
    val_chunks: list[Chunk],
) -> list[Chunk]:
    """Remove from val_chunks any chunk whose document_id appears in train."""
    train_docs = {chunk.document_id for chunk in train}
    return [chunk for chunk in val_chunks if chunk.document_id not in train_docs]


# === SANITY CHECKS ===

def _check_max_cell_size(
    splits: dict[str, QuadrantSplit],
    max_cell_size: int,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []

    for quadrant, split in splits.items():
        cell_counts: dict[tuple[str, str], int] = defaultdict(int)
        for chunk in split.train:
            cell_counts[(chunk.source, chunk.topic_label)] += 1

        worst_cell = max(cell_counts, key=cell_counts.get) if cell_counts else None
        worst_n = cell_counts[worst_cell] if worst_cell else 0
        passed = worst_n <= max_cell_size

        results.append({
            "check_name": "MAX_CELL_SIZE",
            "quadrant": quadrant,
            "passed": passed,
            "details": {
                "worst_cell": f"{worst_cell[0]}|{worst_cell[1]}" if worst_cell else "n/a",
                "n": worst_n,
                "limit": max_cell_size,
            },
        })

    return results


def _check_source_cap(
    splits: dict[str, QuadrantSplit],
    max_pct: float,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []

    for quadrant, split in splits.items():
        counts = Counter(chunk.source for chunk in split.train)
        total = len(split.train) or 1
        worst_source = max(counts, key=counts.get) if counts else "n/a"
        worst_pct = counts.get(worst_source, 0) / total
        passed = worst_pct <= max_pct

        results.append({
            "check_name": "SOURCE_CAP",
            "quadrant": quadrant,
            "passed": passed,
            "details": {
                "worst_source": worst_source,
                "pct": round(worst_pct, 4),
                "limit": max_pct,
            },
        })

    return results


def _check_source_group_cap(
    splits: dict[str, QuadrantSplit],
    source_groups: dict[str, list[str]],
    source_to_group: dict[str, str],
    max_pct: float,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []

    for quadrant, split in splits.items():
        counts = compute_source_group_counts(
            split.train,
            source_groups,
            source_to_group,
        )
        total = len(split.train) or 1
        worst_group = max(counts, key=counts.get) if counts else "n/a"
        worst_pct = counts.get(worst_group, 0) / total
        passed = worst_pct <= max_pct

        results.append({
            "check_name": "SOURCE_GROUP_CAP",
            "quadrant": quadrant,
            "passed": passed,
            "details": {
                "worst_group": worst_group,
                "pct": round(worst_pct, 4),
                "limit": max_pct,
                "counts": counts,
            },
        })

    return results


def _check_source_group_alignment(
    splits: dict[str, QuadrantSplit],
    source_groups: dict[str, list[str]],
    source_to_group: dict[str, str],
    max_abs_diff: float,
) -> list[dict[str, Any]]:
    distributions = {
        quadrant: compute_source_group_distribution(
            split.train,
            source_groups,
            source_to_group,
        )
        for quadrant, split in splits.items()
    }

    results: list[dict[str, Any]] = []

    for group in source_groups:
        values = {
            quadrant: dist.get(group, 0.0)
            for quadrant, dist in distributions.items()
        }

        max_diff = max(values.values()) - min(values.values()) if values else 0.0
        passed = max_diff <= max_abs_diff

        results.append({
            "check_name": "SOURCE_GROUP_ALIGNMENT",
            "source_group": group,
            "passed": passed,
            "details": {
                "proportions": values,
                "max_abs_diff": round(max_diff, 4),
                "limit": max_abs_diff,
            },
        })

    return results


def _check_topic_cap(
    splits: dict[str, QuadrantSplit],
    max_pct: float,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []

    for quadrant, split in splits.items():
        counts = Counter(chunk.topic_label for chunk in split.train)
        total = len(split.train) or 1
        worst_topic = max(counts, key=counts.get) if counts else "n/a"
        worst_pct = counts.get(worst_topic, 0) / total
        passed = worst_pct <= max_pct

        results.append({
            "check_name": "TOPIC_CAP",
            "quadrant": quadrant,
            "passed": passed,
            "details": {
                "worst_topic": worst_topic,
                "pct": round(worst_pct, 4),
                "limit": max_pct,
            },
        })

    return results


def _kl(p: list[float], q: list[float], eps: float = 1e-9) -> float:
    return sum(pi * math.log((pi + eps) / (qi + eps)) for pi, qi in zip(p, q))


def _check_topic_kl(
    splits: dict[str, QuadrantSplit],
    viable_topics: list[str],
    max_kl: float,
) -> list[dict[str, Any]]:
    distributions: dict[str, list[float]] = {}

    for quadrant, split in splits.items():
        counts = Counter(chunk.topic_label for chunk in split.train)
        total = sum(counts.get(topic, 0) for topic in viable_topics) or 1
        distributions[quadrant] = [
            counts.get(topic, 0) / total
            for topic in viable_topics
        ]

    results: list[dict[str, Any]] = []
    quadrants = list(splits.keys())

    for i, quadrant_a in enumerate(quadrants):
        for quadrant_b in quadrants[i + 1:]:
            divergence = _kl(distributions[quadrant_a], distributions[quadrant_b])
            results.append({
                "check_name": "TOPIC_KL",
                "passed": divergence <= max_kl,
                "details": {
                    "q_a": quadrant_a,
                    "q_b": quadrant_b,
                    "kl": round(divergence, 6),
                    "limit": max_kl,
                },
            })

    return results


def _check_length_ratio(
    splits: dict[str, QuadrantSplit],
    max_ratio: float,
) -> list[dict[str, Any]]:
    means = {
        quadrant: sum(chunk.n_tokens for chunk in split.train) / len(split.train)
        for quadrant, split in splits.items()
        if split.train
    }

    if not means:
        return [{"check_name": "LENGTH_RATIO", "passed": False, "details": "no train data"}]

    ratio = max(means.values()) / max(min(means.values()), 1e-9)

    return [{
        "check_name": "LENGTH_RATIO",
        "passed": ratio <= max_ratio,
        "details": {
            "means": {key: round(value, 2) for key, value in means.items()},
            "ratio": round(ratio, 4),
            "limit": max_ratio,
        },
    }]


def _check_conf_ratio(
    splits: dict[str, QuadrantSplit],
    max_ratio: float,
) -> list[dict[str, Any]]:
    means = {
        quadrant: sum(chunk.conf_margin for chunk in split.train) / len(split.train)
        for quadrant, split in splits.items()
        if split.train
    }

    if not means:
        return [{
            "check_name": "CONF_MARGIN_RATIO",
            "passed": False,
            "details": "no train data",
        }]

    ratio = max(means.values()) / max(min(means.values()), 1e-9)

    return [{
        "check_name": "CONF_MARGIN_RATIO",
        "passed": ratio <= max_ratio,
        "details": {
            "means": {key: round(value, 6) for key, value in means.items()},
            "ratio": round(ratio, 4),
            "limit": max_ratio,
        },
    }]


def _check_no_doc_leakage(
    splits: dict[str, QuadrantSplit],
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []

    for quadrant, split in splits.items():
        train_docs = {chunk.document_id for chunk in split.train}

        for split_name, val_chunks in [
            ("val_indist", split.val_indist),
            ("val_source", split.val_source),
            ("val_topic", split.val_topic),
        ]:
            val_docs = {chunk.document_id for chunk in val_chunks}
            overlap = train_docs & val_docs

            results.append({
                "check_name": "NO_DOC_LEAKAGE",
                "quadrant": quadrant,
                "val_split": split_name,
                "passed": len(overlap) == 0,
                "details": {"n_leaked_docs": len(overlap)},
            })

    return results


def _check_val_topic_purity(
    splits: dict[str, QuadrantSplit],
    held_out_topic: str,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []

    for quadrant, split in splits.items():
        impure = [
            chunk for chunk in split.val_topic
            if chunk.topic_label != held_out_topic
        ]

        results.append({
            "check_name": "VAL_TOPIC_PURITY",
            "quadrant": quadrant,
            "passed": len(impure) == 0,
            "details": {
                "n_impure": len(impure),
                "expected_topic": held_out_topic,
            },
        })

    return results


def _check_val_source_purity(
    splits: dict[str, QuadrantSplit],
    held_out_sources: dict[str, str],
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []

    for quadrant, split in splits.items():
        expected_source = held_out_sources[quadrant]
        impure = [
            chunk for chunk in split.val_source
            if chunk.source != expected_source
        ]

        results.append({
            "check_name": "VAL_SOURCE_PURITY",
            "quadrant": quadrant,
            "passed": len(impure) == 0,
            "details": {
                "n_impure": len(impure),
                "expected_source": expected_source,
            },
        })

    return results


def _check_non_empty(
    splits: dict[str, QuadrantSplit],
    min_size: int,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []

    for quadrant, split in splits.items():
        for split_name, chunks in [
            ("train", split.train),
            ("val_indist", split.val_indist),
            ("val_source", split.val_source),
            ("val_topic", split.val_topic),
        ]:
            results.append({
                "check_name": "NON_EMPTY",
                "quadrant": quadrant,
                "split": split_name,
                "passed": len(chunks) >= min_size,
                "details": {
                    "n": len(chunks),
                    "min": min_size,
                },
            })

    return results


def run_sanity_checks(
    splits: dict[str, QuadrantSplit],
    config: SamplingConfig,
    source_to_group: dict[str, str],
) -> tuple[bool, list[dict[str, Any]]]:
    checks = config.sanity_checks
    min_split_size = checks.get("min_split_size", 15)

    results: list[dict[str, Any]] = []

    results.extend(_check_max_cell_size(
        splits,
        checks.get("max_cell_size_train", config.max_cell_size),
    ))

    results.extend(_check_source_cap(
        splits,
        checks.get("max_source_pct_train", config.max_source_pct_train),
    ))

    if "max_source_group_pct_train" in checks:
        results.extend(_check_source_group_cap(
            splits,
            config.source_groups,
            source_to_group,
            checks["max_source_group_pct_train"],
        ))

    if "max_source_group_abs_diff" in checks:
        results.extend(_check_source_group_alignment(
            splits,
            config.source_groups,
            source_to_group,
            checks["max_source_group_abs_diff"],
        ))

    results.extend(_check_topic_cap(
        splits,
        checks.get("max_topic_pct_train", config.max_topic_pct_train),
    ))

    results.extend(_check_topic_kl(
        splits,
        config.viable_topics,
        checks["max_topic_kl_divergence"],
    ))

    results.extend(_check_length_ratio(
        splits,
        checks["max_length_ratio"],
    ))

    results.extend(_check_conf_ratio(
        splits,
        checks["max_conf_margin_ratio"],
    ))

    results.extend(_check_no_doc_leakage(splits))

    results.extend(_check_val_topic_purity(
        splits,
        config.held_out_topic,
    ))

    results.extend(_check_val_source_purity(
        splits,
        config.held_out_sources,
    ))

    results.extend(_check_non_empty(
        splits,
        min_split_size,
    ))

    all_passed = all(result["passed"] for result in results)

    for result in results:
        level = logging.INFO if result["passed"] else logging.WARNING
        log.log(
            level,
            "Check %s: %s | %s",
            result["check_name"],
            "PASS" if result["passed"] else "FAIL",
            result.get("details", ""),
        )

    return all_passed, results


# === OUTPUTS ===

def write_quadrant_outputs(
    quadrant: str,
    split: QuadrantSplit,
    output_dir: Path,
) -> None:
    quadrant_dir = output_dir / quadrant

    write_jsonl(split.train, quadrant_dir / "train.jsonl")
    write_jsonl(split.val_indist, quadrant_dir / "val_indist.jsonl")
    write_jsonl(split.val_source, quadrant_dir / "val_source.jsonl")
    write_jsonl(split.val_topic, quadrant_dir / "val_topic.jsonl")

    log.info(
        "Wrote %s: train=%d val_indist=%d val_source=%d val_topic=%d",
        quadrant,
        len(split.train),
        len(split.val_indist),
        len(split.val_source),
        len(split.val_topic),
    )


# === MAIN ===

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build balanced expert train/validation datasets from retained "
            "quadrant pools using source-group matched sampling."
        )
    )

    parser.add_argument(
        "--retained-dir",
        required=True,
        type=Path,
        help="Root of quadrant-pools: <dir>/<source>/<quadrant>/retained.jsonl",
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Where to write <quadrant>/{train,val_indist,val_source,val_topic}.jsonl",
    )

    parser.add_argument(
        "--config",
        default=CONFIG_PATH,
        type=Path,
        help=f"Path to config.yaml (default: {CONFIG_PATH})",
    )

    parser.add_argument(
        "--report-path",
        type=Path,
        default=None,
        help="Where to write sampling_report.json (default: <output-dir>/sampling_report.json)",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute datasets and report without writing JSONL files.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override random seed from config.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    raw_config = load_config(args.config)
    config = build_sampling_config(raw_config)

    source_to_group = build_source_to_group(config.source_groups)

    rng_seed = args.seed if args.seed is not None else config.random_seed
    rng = random.Random(rng_seed)

    report_path = args.report_path or (args.output_dir / "sampling_report.json")

    report: dict[str, Any] = {
        "config_snapshot": raw_config["validate_expert_datasets"],
        "resolved_sampling_config": asdict(config),
        "source_to_group": source_to_group,
        "random_seed": rng_seed,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    log.info("Stage 0 | Loading retained pools")

    pools = {
        quadrant: load_quadrant_pool(
            args.retained_dir,
            quadrant,
            config.min_tokens,
            config.max_tokens,
        )
        for quadrant in QUADRANTS
    }

    report["stage_0_input"] = {
        quadrant: summarize_chunks(
            chunks,
            source_to_group=source_to_group,
            source_groups=config.source_groups,
        )
        for quadrant, chunks in pools.items()
    }

    log.info("Stage 1 | Applying topic filter")

    pools = apply_topic_filter(
        pools,
        config.viable_topics,
        config.held_out_topic,
    )

    report["stage_1_topic_filter"] = {
        quadrant: summarize_chunks(
            chunks,
            source_to_group=source_to_group,
            source_groups=config.source_groups,
        )
        for quadrant, chunks in pools.items()
    }

    if config.boilerplate_filter_enabled:
        log.info("Stage 1.5 | Applying boilerplate filter")
        patterns = compile_boilerplate_patterns(config.boilerplate_patterns)

        pools, boilerplate_report = apply_boilerplate_filter(
            pools,
            patterns,
        )

        report["stage_1_5_boilerplate_filter"] = boilerplate_report
    else:
        report["stage_1_5_boilerplate_filter"] = {"enabled": False}

    log.info("Stage 2 | Carving held-out topic and source")

    working_pools, val_topic_pools = carve_held_out_topic(
        pools,
        config.held_out_topic,
    )

    working_pools, val_source_pools = carve_held_out_source(
        working_pools,
        config.held_out_sources,
    )

    report["stage_2_carve_outs"] = {
        "working": {
            quadrant: summarize_chunks(
                chunks,
                source_to_group=source_to_group,
                source_groups=config.source_groups,
            )
            for quadrant, chunks in working_pools.items()
        },
        "val_source": {
            quadrant: summarize_chunks(
                chunks,
                source_to_group=source_to_group,
                source_groups=config.source_groups,
            )
            for quadrant, chunks in val_source_pools.items()
        },
        "val_topic": {
            quadrant: summarize_chunks(
                chunks,
                source_to_group=source_to_group,
                source_groups=config.source_groups,
            )
            for quadrant, chunks in val_topic_pools.items()
        },
    }

    if config.dedupe_enabled:
        log.info("Stage 3 | Applying dedupe to working and validation pools")

        working_pools, working_dedupe = apply_dedupe_to_pools(
            working_pools,
            config.dedupe_min_chunk_index_gap,
            "working",
        )

        val_source_pools, val_source_dedupe = apply_dedupe_to_pools(
            val_source_pools,
            config.dedupe_min_chunk_index_gap,
            "val_source",
        )

        val_topic_pools, val_topic_dedupe = apply_dedupe_to_pools(
            val_topic_pools,
            config.dedupe_min_chunk_index_gap,
            "val_topic",
        )

        report["stage_3_dedupe"] = {
            "enabled": True,
            "working": working_dedupe,
            "val_source": val_source_dedupe,
            "val_topic": val_topic_dedupe,
        }
    else:
        log.info("Stage 3 | Dedupe disabled")
        report["stage_3_dedupe"] = {"enabled": False}

    log.info("Stage 4 | Building cell diagnostics")

    report["stage_4_cell_diagnostics"] = build_cell_diagnostics(
        working_pools,
        config.viable_topics,
        source_to_group,
    )

    log.info("Stage 5 | Sampling source-topic capped pools (max_cell_size=%d)", config.max_cell_size)

    sampled_by_quadrant: dict[str, list[Chunk]] = {}
    sampling_report: dict[str, Any] = {}

    for quadrant in QUADRANTS:
        sampled, quadrant_report = sample_source_topic_capped_quadrant(
            working_pools[quadrant],
            quadrant,
            config,
            rng,
        )

        sampled_by_quadrant[quadrant] = sampled
        sampling_report[quadrant] = quadrant_report

    report["stage_5_sampling"] = sampling_report

    log.info("Stage 6 | Splitting sampled chunks into train and val_indist")

    splits: dict[str, QuadrantSplit] = {}
    split_report: dict[str, Any] = {}

    for quadrant in QUADRANTS:
        train, val_indist = document_level_split(
            sampled_by_quadrant[quadrant],
            config.val_pct,
            rng,
        )

        clean_val_source = remove_train_doc_leakage(train, val_source_pools[quadrant])
        clean_val_topic  = remove_train_doc_leakage(train, val_topic_pools[quadrant])

        splits[quadrant] = QuadrantSplit(
            quadrant=quadrant,
            train=train,
            val_indist=val_indist,
            val_source=clean_val_source,
            val_topic=clean_val_topic,
        )

        split_report[quadrant] = {
            "train": summarize_chunks(
                train,
                source_to_group=source_to_group,
                source_groups=config.source_groups,
            ),
            "val_indist": summarize_chunks(
                val_indist,
                source_to_group=source_to_group,
                source_groups=config.source_groups,
            ),
            "val_source": summarize_chunks(
                clean_val_source,
                source_to_group=source_to_group,
                source_groups=config.source_groups,
            ),
            "val_topic": summarize_chunks(
                clean_val_topic,
                source_to_group=source_to_group,
                source_groups=config.source_groups,
            ),
            "train_docs": len({chunk.document_id for chunk in train}),
            "val_indist_docs": len({chunk.document_id for chunk in val_indist}),
        }

        log.info(
            "Stage 6 | %s train=%d val_indist=%d val_source=%d val_topic=%d",
            quadrant,
            len(train),
            len(val_indist),
            len(clean_val_source),
            len(clean_val_topic),
        )

    report["stage_6_split"] = split_report

    log.info("Stage 7 | Running sanity checks")

    all_passed, check_results = run_sanity_checks(
        splits,
        config,
        source_to_group,
    )

    report["stage_7_sanity"] = {
        "all_passed": all_passed,
        "check_results": check_results,
    }

    report["final_counts"] = {
        quadrant: {
            "train": len(split.train),
            "val_indist": len(split.val_indist),
            "val_source": len(split.val_source),
            "val_topic": len(split.val_topic),
        }
        for quadrant, split in splits.items()
    }

    report["final_source_group_distributions"] = {
        quadrant: {
            split_name: compute_source_group_distribution(
                chunks,
                config.source_groups,
                source_to_group,
            )
            for split_name, chunks in {
                "train": split.train,
                "val_indist": split.val_indist,
                "val_source": split.val_source,
                "val_topic": split.val_topic,
            }.items()
        }
        for quadrant, split in splits.items()
    }

    if not all_passed and not args.dry_run:
        write_json(report, report_path)
        log.error(
            "SANITY CHECKS FAILED — dataset files were not written. See %s",
            report_path,
        )
        sys.exit(1)

    if not args.dry_run:
        log.info("Stage 8 | Writing JSONL outputs")

        for quadrant, split in splits.items():
            write_quadrant_outputs(
                quadrant,
                split,
                args.output_dir,
            )
    else:
        log.info("Dry run — no JSONL dataset files written")

    write_json(report, report_path)
    log.info("Report written to %s", report_path)

    print("\n=== Sampling config ===")
    print(f"  sampling_mode:  source_topic_cell_cap")
    print(f"  max_cell_size:  {config.max_cell_size}")

    print("\n=== Final counts ===")
    for quadrant in QUADRANTS:
        counts = report["final_counts"][quadrant]
        sr = report["stage_5_sampling"][quadrant]
        print(
            f"  {quadrant:12s}  "
            f"train={counts['train']:>6,}  "
            f"val_indist={counts['val_indist']:>5,}  "
            f"val_source={counts['val_source']:>6,}  "
            f"val_topic={counts['val_topic']:>6,}  "
            f"(cells={sr['n_cells']}, capped={sr['n_capped']})"
        )

    print("\n=== Train source-group distributions ===")
    for quadrant in QUADRANTS:
        dist = report["final_source_group_distributions"][quadrant]["train"]
        pretty = "  ".join(
            f"{group}={dist.get(group, 0.0):.1%}"
            for group in config.source_groups
        )
        print(f"  {quadrant:12s}  {pretty}")


if __name__ == "__main__":
    main()