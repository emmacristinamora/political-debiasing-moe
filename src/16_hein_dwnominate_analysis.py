# src/16_hein_dwnominate_analysis.py

# Tier 2 — external ground-truth validation of the steering vectors, step 3/3.
#
# Correlates the compass coordinates from step 2 against the legislators'
# DW-NOMINATE ideal points. The headline test is economic_coord vs
# nominate_dim1 (the economic left-right dimension); a strong, signed,
# permutation-significant correlation means the steering vectors track
# externally measured ideology, not just the project's own contrastive pairs.
#
# Also reported:
#   - the full coord x dimension correlation matrix (discriminant check)
#   - a split-half reliability ceiling from the per-window coordinates
#   - Democrat/Republican separability by economic_coord
#
#   step 1  src/14_hein_build_dataset.py     -> legislator_dataset.jsonl
#   step 2  src/15_hein_project_compass.py   -> compass_projections.jsonl
#   step 3  src/16_hein_dwnominate_analysis.py -> dwnominate_report.json


# === IMPORTS ===

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score


# === CONFIG ===

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_PROJECTIONS = PROJECT_ROOT / "data" / "external" / "hein_dwnominate" / "compass_projections.jsonl"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "external" / "hein_dwnominate" / "dwnominate_report.json"

COORDS = ("economic_coord", "social_coord")
DIMENSIONS = ("nominate_dim1", "nominate_dim2")


# === HELPERS: IO ===

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Tier 2 step 3 — correlate compass coordinates with DW-NOMINATE."
    )
    parser.add_argument("--projections", type=Path, default=DEFAULT_PROJECTIONS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--num-permutations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def load_projections(path: Path) -> list[dict[str, Any]]:
    """Load the per-legislator compass projections written by step 2."""
    if not path.is_file():
        raise FileNotFoundError(f"projections not found: {path}")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"{path}: no rows")
    return rows


def save_json(payload: dict[str, Any], path: Path) -> None:
    """Write a pretty-printed JSON report."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


# === HELPERS: STATISTICS ===

def correlation(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """Spearman and Pearson correlation between two arrays."""
    spearman = stats.spearmanr(x, y)
    pearson = stats.pearsonr(x, y)
    return {
        "spearman": float(spearman.statistic),
        "pearson": float(pearson.statistic),
        "r_squared": float(pearson.statistic ** 2),
    }


def permutation_p_value(x: np.ndarray, y: np.ndarray, num_permutations: int, seed: int) -> float:
    """
    Two-sided permutation p-value for the Spearman correlation.

    y is shuffled to break the pairing; the fraction of shuffles whose absolute
    correlation reaches the observed one estimates the chance probability.
    """
    rng = np.random.default_rng(seed)
    observed = abs(stats.spearmanr(x, y).statistic)
    shuffled = y.copy()
    hits = 0
    for _ in range(num_permutations):
        rng.shuffle(shuffled)
        if abs(stats.spearmanr(x, shuffled).statistic) >= observed:
            hits += 1
    return (hits + 1) / (num_permutations + 1)


def split_half_reliability(chunk_lists: list[list[float]], seed: int) -> dict[str, float]:
    """
    Split-half reliability of a per-legislator coordinate.

    Logic:
        Each legislator's window coordinates are split into two random halves
        and averaged; the Pearson correlation between the two half-means across
        legislators, Spearman-Brown corrected to full length, is the ceiling on
        any external correlation — a coordinate cannot correlate with anything
        better than it correlates with itself.
    """
    rng = random.Random(seed)
    first_half: list[float] = []
    second_half: list[float] = []
    for chunks in chunk_lists:
        order = chunks[:]
        rng.shuffle(order)
        mid = len(order) // 2
        first_half.append(float(np.mean(order[:mid])))
        second_half.append(float(np.mean(order[mid:])))
    half_corr = stats.pearsonr(np.array(first_half), np.array(second_half)).statistic
    corrected = (2 * half_corr) / (1 + half_corr) if half_corr > -1 else 0.0
    return {"half_split_pearson": float(half_corr), "spearman_brown_full": float(corrected)}


# === MAIN ===

def main() -> None:
    """Run the DW-NOMINATE correlation analysis and write a JSON report."""
    args = parse_args()

    rows = load_projections(args.projections)
    economic = np.array([r["economic_coord"] for r in rows])
    social = np.array([r["social_coord"] for r in rows])
    dim1 = np.array([r["nominate_dim1"] for r in rows])
    dim2 = np.array([r["nominate_dim2"] for r in rows])
    coords = {"economic_coord": economic, "social_coord": social}
    dims = {"nominate_dim1": dim1, "nominate_dim2": dim2}

    # full coordinate x dimension correlation matrix (discriminant check)
    matrix = {
        coord: {dim: correlation(coords[coord], dims[dim]) for dim in DIMENSIONS}
        for coord in COORDS
    }

    # headline test: economic_coord vs nominate_dim1
    headline_p = permutation_p_value(economic, dim1, args.num_permutations, args.seed)

    # measurement ceiling from the per-window coordinates
    economic_chunks = [r["economic_chunks"] for r in rows]
    social_chunks = [r["social_chunks"] for r in rows]
    reliability = {
        "economic_coord": split_half_reliability(economic_chunks, args.seed),
        "social_coord": split_half_reliability(social_chunks, args.seed),
    }

    # Democrat / Republican separability by economic_coord (R = positive class)
    labelled = [(r["economic_coord"], r["party"]) for r in rows if r["party"] in ("D", "R")]
    party_scores = np.array([score for score, _ in labelled])
    party_labels = np.array([1 if party == "R" else 0 for _, party in labelled])
    party_auc = float(roc_auc_score(party_labels, party_scores))

    report = {
        "n_legislators": len(rows),
        "config": {"num_permutations": args.num_permutations, "seed": args.seed},
        "headline": {
            "test": "economic_coord vs nominate_dim1",
            **matrix["economic_coord"]["nominate_dim1"],
            "permutation_p_value": headline_p,
        },
        "correlation_matrix": matrix,
        "split_half_reliability": reliability,
        "party_separation": {
            "test": "economic_coord separates Republicans from Democrats",
            "auc": party_auc,
            "n_democrat": int((party_labels == 0).sum()),
            "n_republican": int((party_labels == 1).sum()),
        },
        "interpretation": (
            "headline Spearman should be strongly positive and permutation-significant; "
            "economic_coord-dim1 should exceed the off-diagonal correlations; the "
            "headline cannot exceed the economic split-half reliability ceiling"
        ),
    }
    save_json(report, args.output)

    head = report["headline"]
    print(f"\n=== Tier 2 — DW-NOMINATE external validation ({len(rows)} legislators) ===")
    print(f"headline  economic_coord vs nominate_dim1: "
          f"Spearman={head['spearman']:+.3f}  Pearson={head['pearson']:+.3f}  "
          f"R^2={head['r_squared']:.3f}  perm_p={head['permutation_p_value']:.4f}")
    print("discriminant correlation matrix (Spearman):")
    for coord in COORDS:
        cells = "  ".join(f"{dim}={matrix[coord][dim]['spearman']:+.3f}" for dim in DIMENSIONS)
        print(f"  {coord:16s} {cells}")
    rel = reliability["economic_coord"]
    print(f"economic split-half reliability (ceiling): {rel['spearman_brown_full']:.3f}")
    print(f"party separation economic_coord -> R/D AUC: {party_auc:.3f}")
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
