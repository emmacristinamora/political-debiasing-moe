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
    "reddit_liberal",
    "reddit_conservative",
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
    target_n_per_quadrant: int
    val_pct: float
    random_seed: int
    min_cell_size: int
    max_source_pct_train: float
    max_topic_pct_train: float
    confidence_top_frac: float
    min_tokens: int
    max_tokens: int
    dedupe_enabled: bool
    dedupe_min_chunk_index_gap: int
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
    topic_counts: dict[str, int]


# === CONFIG LOADING ===

_REQUIRED_KEYS = {
    "viable_topics",
    "held_out_topic",
    "target_n_per_quadrant",
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

    Logic:
        Reads the validate_expert_datasets block and validates required fields
        before any data is loaded.
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
        raise ValueError(f"held_out_sources contains unknown sources: {sorted(unknown_sources)}")

    if section["held_out_topic"] in section["viable_topics"]:
        raise ValueError(
            f"held_out_topic '{section['held_out_topic']}' is also in viable_topics"
        )

    if not 0 < section["val_pct"] < 0.5:
        raise ValueError(f"val_pct must be in (0, 0.5), got {section['val_pct']}")

    if section["target_n_per_quadrant"] <= 0:
        raise ValueError("target_n_per_quadrant must be positive")

    return cfg


def build_sampling_config(raw_config: dict) -> SamplingConfig:
    section = raw_config["validate_expert_datasets"]
    length_filter = section["length_filter"]
    dedupe = section["dedupe"]
    sanity_checks = section["sanity_checks"]

    max_source_pct_train = section.get(
        "max_source_pct_train",
        sanity_checks.get("max_source_pct_train", 0.35),
    )
    max_topic_pct_train = section.get(
        "max_topic_pct_train",
        sanity_checks.get("max_topic_pct_train", 0.30),
    )
    confidence_top_frac = section.get("confidence_top_frac", 0.70)
    boilerplate_cfg = section.get("boilerplate_filter", {})

    if not 0 < max_source_pct_train <= 1:
        raise ValueError(f"max_source_pct_train must be in (0, 1], got {max_source_pct_train}")
    if not 0 < max_topic_pct_train <= 1:
        raise ValueError(f"max_topic_pct_train must be in (0, 1], got {max_topic_pct_train}")
    if not 0 < confidence_top_frac <= 1:
        raise ValueError(f"confidence_top_frac must be in (0, 1], got {confidence_top_frac}")

    return SamplingConfig(
        viable_topics=section["viable_topics"],
        held_out_topic=section["held_out_topic"],
        held_out_sources=section["held_out_sources"],
        target_n_per_quadrant=section["target_n_per_quadrant"],
        val_pct=section["val_pct"],
        random_seed=section["random_seed"],
        min_cell_size=section["min_cell_size"],
        max_source_pct_train=max_source_pct_train,
        max_topic_pct_train=max_topic_pct_train,
        confidence_top_frac=confidence_top_frac,
        min_tokens=length_filter["min_tokens"],
        max_tokens=length_filter["max_tokens"],
        dedupe_enabled=dedupe["enabled"],
        dedupe_min_chunk_index_gap=dedupe["min_chunk_index_gap"],
        boilerplate_filter_enabled=boilerplate_cfg.get("enabled", True),
        boilerplate_patterns=boilerplate_cfg.get("patterns", DEFAULT_BOILERPLATE_PATTERNS),
        sanity_checks=sanity_checks,
    )


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

    Logic:
        Reads all source-specific retained files, maps raw fields into Chunk,
        parses chunk_index from chunk_id, and applies the length filter.
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

def summarize_chunks(chunks: list[Chunk]) -> dict[str, Any]:
    if not chunks:
        return {
            "n": 0,
            "mean_tokens": 0.0,
            "mean_conf_margin": 0.0,
            "mean_econ": 0.0,
            "mean_soc": 0.0,
            "sources": {},
            "topics": {},
        }

    return {
        "n": len(chunks),
        "mean_tokens": round(sum(chunk.n_tokens for chunk in chunks) / len(chunks), 3),
        "mean_conf_margin": round(sum(chunk.conf_margin for chunk in chunks) / len(chunks), 6),
        "mean_econ": round(sum(chunk.s_econ for chunk in chunks) / len(chunks), 6),
        "mean_soc": round(sum(chunk.s_soc for chunk in chunks) / len(chunks), 6),
        "sources": dict(Counter(chunk.source for chunk in chunks)),
        "topics": dict(Counter(chunk.topic_label for chunk in chunks)),
    }


def build_cell_diagnostics(
    pools: dict[str, list[Chunk]],
    viable_topics: list[str],
) -> dict[str, Any]:
    diagnostics: dict[str, Any] = {}

    for quadrant, chunks in pools.items():
        cells: dict[str, int] = defaultdict(int)
        for chunk in chunks:
            if chunk.topic_label in viable_topics:
                cells[f"{chunk.topic_label}|{chunk.source}"] += 1

        diagnostics[quadrant] = {
            "summary": summarize_chunks(chunks),
            "cells": dict(cells),
        }

    return diagnostics


# === SAMPLING HELPERS ===

def group_by_topic_source(
    chunks: list[Chunk],
    viable_topics: list[str],
) -> dict[str, dict[str, list[Chunk]]]:
    viable_set = set(viable_topics)
    grouped: dict[str, dict[str, list[Chunk]]] = defaultdict(lambda: defaultdict(list))

    for chunk in chunks:
        if chunk.topic_label in viable_set:
            grouped[chunk.topic_label][chunk.source].append(chunk)

    return grouped


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


def source_cap(
    target_n: int,
    max_source_pct: float,
) -> int:
    return max(1, math.floor(target_n * max_source_pct))


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
    max_topic_pct: float,
) -> bool:
    if chunk.chunk_id in state.selected_chunk_ids:
        return False

    if state.source_counts.get(chunk.source, 0) >= source_cap(target_n, max_source_pct):
        return False

    if state.topic_counts.get(chunk.topic_label, 0) >= topic_cap(target_n, max_topic_pct):
        return False

    return True


def add_chunk(chunk: Chunk, state: SamplingState) -> None:
    state.selected.append(chunk)
    state.selected_chunk_ids.add(chunk.chunk_id)
    state.source_counts[chunk.source] = state.source_counts.get(chunk.source, 0) + 1
    state.topic_counts[chunk.topic_label] = state.topic_counts.get(chunk.topic_label, 0) + 1


# === SAMPLING ===

def sample_topic_balanced_quadrant(
    chunks: list[Chunk],
    quadrant: str,
    config: SamplingConfig,
    rng: random.Random,
) -> tuple[list[Chunk], dict[str, Any]]:
    """
    Sample one quadrant with topic balance, source caps, and confidence ranking.

    Args:
        chunks: working chunks after held-out source/topic carve-outs.
        quadrant: quadrant name.
        config: sampling config.
        rng: random generator.

    Returns:
        Sampled chunks and a per-quadrant sampling report.

    Logic:
        First allocates an equal topic budget across viable topics. Within each
        topic, it samples from all available sources while enforcing a global
        source cap. Then it tops up from all remaining high-confidence candidates
        until target_n_per_quadrant is reached or no legal candidate remains.
    """
    target_n = config.target_n_per_quadrant
    topic_budget = allocate_integer_counts(target_n, config.viable_topics)
    grouped = group_by_topic_source(chunks, config.viable_topics)

    state = SamplingState(
        selected=[],
        selected_chunk_ids=set(),
        source_counts={},
        topic_counts={},
    )

    report: dict[str, Any] = {
        "quadrant": quadrant,
        "target_n": target_n,
        "topic_budget": topic_budget,
        "topic_stage": {},
        "top_up": {},
    }

    for topic in config.viable_topics:
        topic_target = topic_budget.get(topic, 0)
        topic_cells = grouped.get(topic, {})
        viable_sources = [
            source for source, candidates in topic_cells.items()
            if len(candidates) >= config.min_cell_size
        ]

        if not viable_sources:
            report["topic_stage"][topic] = {
                "target": topic_target,
                "selected": 0,
                "viable_sources": [],
                "skipped": True,
            }
            continue

        per_source_targets = allocate_integer_counts(topic_target, viable_sources)
        selected_before = len(state.selected)

        for source in viable_sources:
            candidates = confidence_rank_candidates(
                topic_cells[source],
                config.confidence_top_frac,
                per_source_targets[source],
                rng,
            )

            for candidate in candidates:
                if state.topic_counts.get(topic, 0) >= topic_target:
                    break
                if can_add_chunk(
                    candidate,
                    state,
                    target_n,
                    config.max_source_pct_train,
                    config.max_topic_pct_train,
                ):
                    add_chunk(candidate, state)

        topic_shortfall = topic_target - state.topic_counts.get(topic, 0)
        if topic_shortfall > 0:
            fallback_candidates: list[Chunk] = []
            for source_candidates in topic_cells.values():
                fallback_candidates.extend(source_candidates)

            fallback_candidates = confidence_rank_candidates(
                fallback_candidates,
                config.confidence_top_frac,
                topic_shortfall,
                rng,
            )

            for candidate in fallback_candidates:
                if state.topic_counts.get(topic, 0) >= topic_target:
                    break
                if can_add_chunk(
                    candidate,
                    state,
                    target_n,
                    config.max_source_pct_train,
                    config.max_topic_pct_train,
                ):
                    add_chunk(candidate, state)

        report["topic_stage"][topic] = {
            "target": topic_target,
            "selected": len(state.selected) - selected_before,
            "total_topic_selected": state.topic_counts.get(topic, 0),
            "viable_sources": viable_sources,
            "n_sources": len(viable_sources),
        }

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

        selected_before = len(state.selected)
        for candidate in all_remaining:
            if len(state.selected) >= target_n:
                break
            if can_add_chunk(
                candidate,
                state,
                target_n,
                config.max_source_pct_train,
                config.max_topic_pct_train,
            ):
                add_chunk(candidate, state)

        report["top_up"] = {
            "needed": target_n - selected_before,
            "selected": len(state.selected) - selected_before,
            "final_n": len(state.selected),
        }

    report["final"] = summarize_chunks(state.selected)

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


# === SANITY CHECKS ===

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
        return [{"check_name": "CONF_MARGIN_RATIO", "passed": False, "details": "no train data"}]

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
) -> tuple[bool, list[dict[str, Any]]]:
    checks = config.sanity_checks
    min_split_size = checks.get("min_split_size", config.min_cell_size)

    results: list[dict[str, Any]] = []
    results.extend(_check_source_cap(splits, checks.get("max_source_pct_train", config.max_source_pct_train)))
    results.extend(_check_topic_cap(splits, checks.get("max_topic_pct_train", config.max_topic_pct_train)))
    results.extend(_check_topic_kl(splits, config.viable_topics, checks["max_topic_kl_divergence"]))
    results.extend(_check_length_ratio(splits, checks["max_length_ratio"]))
    results.extend(_check_conf_ratio(splits, checks["max_conf_margin_ratio"]))
    results.extend(_check_no_doc_leakage(splits))
    results.extend(_check_val_topic_purity(splits, config.held_out_topic))
    results.extend(_check_val_source_purity(splits, config.held_out_sources))
    results.extend(_check_non_empty(splits, min_split_size))

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
        description="Build balanced expert train/validation datasets from retained quadrant pools."
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
    rng_seed = args.seed if args.seed is not None else config.random_seed
    rng = random.Random(rng_seed)

    report_path = args.report_path or (args.output_dir / "sampling_report.json")
    report: dict[str, Any] = {
        "config_snapshot": raw_config["validate_expert_datasets"],
        "resolved_sampling_config": asdict(config),
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
        quadrant: summarize_chunks(chunks)
        for quadrant, chunks in pools.items()
    }

    log.info("Stage 1 | Applying topic filter")
    pools = apply_topic_filter(pools, config.viable_topics, config.held_out_topic)
    report["stage_1_topic_filter"] = {
        quadrant: summarize_chunks(chunks)
        for quadrant, chunks in pools.items()
    }

    if config.boilerplate_filter_enabled:
        log.info("Stage 1.5 | Applying boilerplate filter")
        patterns = compile_boilerplate_patterns(config.boilerplate_patterns)
        pools, boilerplate_report = apply_boilerplate_filter(pools, patterns)
        report["stage_1_5_boilerplate_filter"] = boilerplate_report
    else:
        report["stage_1_5_boilerplate_filter"] = {"enabled": False}

    log.info("Stage 2 | Carving held-out topic and source")
    working_pools, val_topic_pools = carve_held_out_topic(pools, config.held_out_topic)
    working_pools, val_source_pools = carve_held_out_source(
        working_pools,
        config.held_out_sources,
    )
    report["stage_2_carve_outs"] = {
        "working": {
            quadrant: summarize_chunks(chunks)
            for quadrant, chunks in working_pools.items()
        },
        "val_source": {
            quadrant: summarize_chunks(chunks)
            for quadrant, chunks in val_source_pools.items()
        },
        "val_topic": {
            quadrant: summarize_chunks(chunks)
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
            "working": working_dedupe,
            "val_source": val_source_dedupe,
            "val_topic": val_topic_dedupe,
        }
    else:
        report["stage_3_dedupe"] = {"enabled": False}

    log.info("Stage 4 | Building cell diagnostics")
    report["stage_4_cell_diagnostics"] = build_cell_diagnostics(
        working_pools,
        config.viable_topics,
    )

    log.info("Stage 5 | Sampling train/val_indist candidate pools")
    sampled_by_quadrant: dict[str, list[Chunk]] = {}
    sampling_report: dict[str, Any] = {}

    for quadrant in QUADRANTS:
        sampled, quadrant_report = sample_topic_balanced_quadrant(
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
        splits[quadrant] = QuadrantSplit(
            quadrant=quadrant,
            train=train,
            val_indist=val_indist,
            val_source=val_source_pools[quadrant],
            val_topic=val_topic_pools[quadrant],
        )
        split_report[quadrant] = {
            "train": summarize_chunks(train),
            "val_indist": summarize_chunks(val_indist),
            "val_source": summarize_chunks(val_source_pools[quadrant]),
            "val_topic": summarize_chunks(val_topic_pools[quadrant]),
            "train_docs": len({chunk.document_id for chunk in train}),
            "val_indist_docs": len({chunk.document_id for chunk in val_indist}),
        }

        log.info(
            "Stage 6 | %s train=%d val_indist=%d val_source=%d val_topic=%d",
            quadrant,
            len(train),
            len(val_indist),
            len(val_source_pools[quadrant]),
            len(val_topic_pools[quadrant]),
        )

    report["stage_6_split"] = split_report

    log.info("Stage 7 | Running sanity checks")
    all_passed, check_results = run_sanity_checks(splits, config)
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
            write_quadrant_outputs(quadrant, split, args.output_dir)
    else:
        log.info("Dry run — no JSONL dataset files written")

    write_json(report, report_path)
    log.info("Report written to %s", report_path)

    print("\n=== Final counts ===")
    for quadrant in QUADRANTS:
        counts = report["final_counts"][quadrant]
        print(
            f"  {quadrant:12s}  "
            f"train={counts['train']:>6,}  "
            f"val_indist={counts['val_indist']:>5,}  "
            f"val_source={counts['val_source']:>6,}  "
            f"val_topic={counts['val_topic']:>6,}"
        )


if __name__ == "__main__":
    main()