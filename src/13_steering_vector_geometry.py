# src/13_steering_vector_geometry.py


# === IMPORTS ===

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold


# === CONFIG ===

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_VECTORS_DIR = PROJECT_ROOT / "data" / "steering-vectors" / "vectors"
DEFAULT_ACTIVATIONS_DIR = PROJECT_ROOT / "data" / "steering-vectors" / "activations"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "steering-vectors" / "reports" / "tier1_geometry_report.json"

AXES = ("economic", "social")
VALID_METHODS = ("mean_difference", "logistic_regression")


@dataclass
class AxisData:
    """Steering vectors and contrastive-pair activations for one compass axis."""

    name: str
    layers: list[int]
    weights: dict[int, float]                 # per-layer aggregation weights
    layer_vectors: dict[int, torch.Tensor]    # per-layer unit steering vectors
    final_vector: torch.Tensor                # quality-weighted aggregate direction
    pos_acts: dict[int, torch.Tensor]         # per-layer pooled activations, positive pole
    neg_acts: dict[int, torch.Tensor]         # per-layer pooled activations, negative pole


# === HELPERS: IO ===

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Tier 1 geometry checks for the political-compass steering vectors."
    )
    parser.add_argument("--vectors-dir", type=Path, default=DEFAULT_VECTORS_DIR)
    parser.add_argument("--activations-dir", type=Path, default=DEFAULT_ACTIVATIONS_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--method",
        choices=VALID_METHODS,
        default="mean_difference",
        help="Which steering-vector extraction method to evaluate.",
    )
    parser.add_argument("--num-permutations", type=int, default=500)
    parser.add_argument("--num-random", type=int, default=500)
    parser.add_argument("--kfolds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def save_json(payload: dict[str, Any], path: Path) -> None:
    """Write a JSON summary, creating the parent directory if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def load_axis_data(axis: str, vectors_dir: Path, activations_dir: Path, method: str) -> AxisData:
    """
    Load steering vectors and contrastive-pair activations for one axis.

    Args:
        axis: "economic" or "social".
        vectors_dir: directory holding {axis}_vectors.pt.
        activations_dir: directory holding {axis}_activations.pt.
        method: extraction method whose per-layer vectors are loaded.
    Returns:
        Populated AxisData object.
    """
    vectors_path = vectors_dir / f"{axis}_vectors.pt"
    activations_path = activations_dir / f"{axis}_activations.pt"
    if not vectors_path.exists():
        raise FileNotFoundError(f"Missing vectors file: {vectors_path}")
    if not activations_path.exists():
        raise FileNotFoundError(f"Missing activations file: {activations_path}")

    vectors = torch.load(vectors_path, map_location="cpu")
    activations = torch.load(activations_path, map_location="cpu")

    aggregation = vectors["aggregation"][method]
    layers = [int(layer) for layer in aggregation["layers"]]
    weights = dict(zip(layers, (float(w) for w in aggregation["normalized_weights"])))

    layer_vectors = {
        layer: vectors["per_layer"][layer][method]["vector"].to(torch.float32)
        for layer in layers
    }
    pos_acts = {layer: activations["activations"][layer]["pos"].to(torch.float32) for layer in layers}
    neg_acts = {layer: activations["activations"][layer]["neg"].to(torch.float32) for layer in layers}

    return AxisData(
        name=axis,
        layers=layers,
        weights=weights,
        layer_vectors=layer_vectors,
        final_vector=vectors["final_vectors"][method].to(torch.float32),
        pos_acts=pos_acts,
        neg_acts=neg_acts,
    )


# === HELPERS: GEOMETRY ===

def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    """Cosine similarity between two vectors."""
    return float(torch.dot(a, b) / (a.norm() * b.norm() + 1e-12))


def unit(vector: torch.Tensor) -> torch.Tensor:
    """Return the vector normalised to unit norm."""
    return vector / (vector.norm() + 1e-12)


def combined_projection(
    acts: dict[int, torch.Tensor],
    layer_vectors: dict[int, torch.Tensor],
    weights: dict[int, float],
) -> torch.Tensor:
    """
    Project per-layer activations onto per-layer directions, then combine layers.

    Logic:
        Each layer contributes (activation . unit_direction); contributions are
        summed with the same quality weights used to build the aggregate vector.
        This mirrors how 04_build_steering_vectors aggregates across layers.
    Returns:
        Tensor of shape [num_examples] with one compass coordinate per example.
    """
    score = None
    for layer, vector in layer_vectors.items():
        contribution = weights[layer] * (acts[layer] @ unit(vector))
        score = contribution if score is None else score + contribution
    return score


def standardized_separation(pos_scores: torch.Tensor, neg_scores: torch.Tensor) -> float:
    """
    Cohen's-d separation between the positive and negative pole scores.

    A large positive value means the projection cleanly splits the two poles;
    a value near zero means the direction does not distinguish them.
    """
    pos = pos_scores.numpy()
    neg = neg_scores.numpy()
    pooled_std = np.sqrt((pos.var(ddof=1) + neg.var(ddof=1)) / 2.0)
    return float((pos.mean() - neg.mean()) / (pooled_std + 1e-12))


def projection_auc(pos_scores: torch.Tensor, neg_scores: torch.Tensor) -> float:
    """ROC-AUC for ranking positive-pole examples above negative-pole examples."""
    labels = np.concatenate([np.ones(len(pos_scores)), np.zeros(len(neg_scores))])
    scores = np.concatenate([pos_scores.numpy(), neg_scores.numpy()])
    return float(roc_auc_score(labels, scores))


def mean_difference_vectors(
    pos_acts: dict[int, torch.Tensor],
    neg_acts: dict[int, torch.Tensor],
) -> dict[int, torch.Tensor]:
    """Rebuild per-layer unit mean-difference directions from a set of pairs."""
    return {
        layer: unit(pos_acts[layer].mean(dim=0) - neg_acts[layer].mean(dim=0))
        for layer in pos_acts
    }


def pooled_label_permutation(
    pos_acts: dict[int, torch.Tensor],
    neg_acts: dict[int, torch.Tensor],
    rng: np.random.Generator,
) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor]]:
    """
    Randomly re-partition pooled pos/neg activations into two balanced groups.

    Logic:
        A per-pair pos/neg swap is too weak a null: because all difference
        vectors share a common axis, a sign-shuffled combination still recovers
        it. Pooling all examples and drawing a fresh balanced split destroys the
        pos/neg structure properly. The same example indices are used for every
        layer so each example stays intact across layers.
    """
    layers = list(pos_acts.keys())
    num_pos = pos_acts[layers[0]].shape[0]
    perm = rng.permutation(2 * num_pos)
    idx_a = torch.from_numpy(perm[:num_pos].copy())
    idx_b = torch.from_numpy(perm[num_pos:].copy())
    group_a: dict[int, torch.Tensor] = {}
    group_b: dict[int, torch.Tensor] = {}
    for layer in layers:
        pooled = torch.cat([pos_acts[layer], neg_acts[layer]], dim=0)
        group_a[layer] = pooled[idx_a]
        group_b[layer] = pooled[idx_b]
    return group_a, group_b


# === HELPERS: METRICS ===

def check_orthogonality(economic: AxisData, social: AxisData) -> dict[str, Any]:
    """
    Measure how independent the economic and social axes are.

    A 2D compass requires |cos(economic, social)| near 0. A large value means
    the two axes collapse toward a single direction.
    """
    final_cos = cosine(economic.final_vector, social.final_vector)
    per_layer = {
        layer: cosine(economic.layer_vectors[layer], social.layer_vectors[layer])
        for layer in economic.layers
    }
    return {
        "final_vector_cosine": final_cos,
        "per_layer_cosine": per_layer,
        "max_abs_per_layer_cosine": max(abs(v) for v in per_layer.values()),
        "interpretation": "near 0 = independent axes; large |cos| = axes collapse to 1D",
    }


def cross_projection_matrix(economic: AxisData, social: AxisData) -> dict[str, Any]:
    """
    Project each axis's contrastive pairs onto both compass axes.

    Logic:
        The diagonal (statements scored on their own axis) should show large
        separation; the off-diagonal (statements scored on the other axis)
        should be near zero if the axes do not leak into each other.
    """
    axes = {"economic": economic, "social": social}
    matrix: dict[str, dict[str, float]] = {}
    for stmt_name, stmt in axes.items():
        matrix[stmt_name] = {}
        for proj_name, proj in axes.items():
            pos = combined_projection(stmt.pos_acts, proj.layer_vectors, proj.weights)
            neg = combined_projection(stmt.neg_acts, proj.layer_vectors, proj.weights)
            matrix[stmt_name][proj_name] = standardized_separation(pos, neg)

    off_diagonal = [matrix["economic"]["social"], matrix["social"]["economic"]]
    diagonal = [matrix["economic"]["economic"], matrix["social"]["social"]]
    return {
        "separation_matrix": matrix,
        "rows": "statement axis",
        "columns": "projection axis",
        "min_diagonal_separation": min(diagonal),
        "max_abs_off_diagonal_separation": max(abs(v) for v in off_diagonal),
        "interpretation": "large diagonal, near-zero off-diagonal = clean axis separation",
    }


def null_baselines(axis: AxisData, num_random: int, num_permutations: int, seed: int) -> dict[str, Any]:
    """
    Compare the real axis separation against random-direction and shuffled-label nulls.

    Logic:
        random null  - project onto random directions, keep the true labels;
                        shows separation achievable by an arbitrary direction.
        shuffled null - re-partition the pooled examples into two balanced
                        groups, rebuild the direction, re-measure; a permutation
                        test for the pos/neg signal. With 4096 dims and ~90 pairs
                        some spurious separation is expected, so the real value
                        must sit well above it.
    """
    rng = np.random.default_rng(seed)
    torch_gen = torch.Generator().manual_seed(seed)

    pos = combined_projection(axis.pos_acts, axis.layer_vectors, axis.weights)
    neg = combined_projection(axis.neg_acts, axis.layer_vectors, axis.weights)
    real_separation = standardized_separation(pos, neg)

    hidden_dim = next(iter(axis.layer_vectors.values())).shape[0]
    random_separations: list[float] = []
    for _ in range(num_random):
        random_vectors = {
            layer: unit(torch.randn(hidden_dim, generator=torch_gen)) for layer in axis.layers
        }
        rand_pos = combined_projection(axis.pos_acts, random_vectors, axis.weights)
        rand_neg = combined_projection(axis.neg_acts, random_vectors, axis.weights)
        random_separations.append(abs(standardized_separation(rand_pos, rand_neg)))

    shuffled_separations: list[float] = []
    for _ in range(num_permutations):
        group_a, group_b = pooled_label_permutation(axis.pos_acts, axis.neg_acts, rng)
        perm_vectors = mean_difference_vectors(group_a, group_b)
        perm_a = combined_projection(group_a, perm_vectors, axis.weights)
        perm_b = combined_projection(group_b, perm_vectors, axis.weights)
        shuffled_separations.append(standardized_separation(perm_a, perm_b))

    shuffled = np.array(shuffled_separations)
    random = np.array(random_separations)
    permutation_p = float((np.sum(shuffled >= real_separation) + 1) / (len(shuffled) + 1))
    return {
        "real_separation": real_separation,
        "random_direction_null": {
            "mean": float(random.mean()),
            "p95": float(np.percentile(random, 95)),
            "max": float(random.max()),
        },
        "shuffled_label_null": {
            "mean": float(shuffled.mean()),
            "p95": float(np.percentile(shuffled, 95)),
            "max": float(shuffled.max()),
            "permutation_p_value": permutation_p,
        },
        "interpretation": "real_separation must exceed both nulls; low p-value = real signal",
    }


def held_out_separability(axis: AxisData, kfolds: int, seed: int) -> dict[str, Any]:
    """
    Honest separability of the axis direction via k-fold cross-validation.

    Logic:
        The stored vector was fitted on all pairs, so in-sample AUC is optimistic.
        Each fold rebuilds the direction on the training pairs and scores the
        held-out pairs. A shuffled-label run gives the chance-level floor.
    """
    num_pairs = next(iter(axis.pos_acts.values())).shape[0]
    splitter = KFold(n_splits=kfolds, shuffle=True, random_state=seed)
    equal_weights = {layer: 1.0 / len(axis.layers) for layer in axis.layers}
    rng = np.random.default_rng(seed)

    def fold_scores(use_shuffled: bool) -> float:
        all_scores: list[float] = []
        all_labels: list[int] = []
        for train_idx, test_idx in splitter.split(np.arange(num_pairs)):
            train_pos = {layer: axis.pos_acts[layer][train_idx] for layer in axis.layers}
            train_neg = {layer: axis.neg_acts[layer][train_idx] for layer in axis.layers}
            if use_shuffled:
                train_pos, train_neg = pooled_label_permutation(train_pos, train_neg, rng)
            fold_vectors = mean_difference_vectors(train_pos, train_neg)
            test_pos = {layer: axis.pos_acts[layer][test_idx] for layer in axis.layers}
            test_neg = {layer: axis.neg_acts[layer][test_idx] for layer in axis.layers}
            pos_score = combined_projection(test_pos, fold_vectors, equal_weights)
            neg_score = combined_projection(test_neg, fold_vectors, equal_weights)
            all_scores.extend(pos_score.tolist() + neg_score.tolist())
            all_labels.extend([1] * len(test_idx) + [0] * len(test_idx))
        return float(roc_auc_score(all_labels, all_scores))

    in_sample_pos = combined_projection(axis.pos_acts, axis.layer_vectors, axis.weights)
    in_sample_neg = combined_projection(axis.neg_acts, axis.layer_vectors, axis.weights)
    return {
        "in_sample_auc": projection_auc(in_sample_pos, in_sample_neg),
        "cross_validated_auc": fold_scores(use_shuffled=False),
        "shuffled_label_auc": fold_scores(use_shuffled=True),
        "interpretation": "cross_validated_auc is the honest number; shuffled ~0.5 is the floor",
    }


def method_agreement(axis: AxisData, vectors_dir: Path) -> dict[str, Any]:
    """
    Internal-consistency check: do the two extraction methods agree?

    Logic:
        Reads back the per-layer cosine between the mean-difference vector and
        the logistic-regression vector. High values mean the steering direction
        is a property of the representation, not an artifact of one estimator.
    """
    vectors = torch.load(vectors_dir / f"{axis.name}_vectors.pt", map_location="cpu")
    method_cosines = {
        layer: float(vectors["per_layer"][layer]["method_cosine_similarity"])
        for layer in axis.layers
    }
    return {
        "method_cosine_per_layer": method_cosines,
        "min_method_cosine": min(method_cosines.values()),
        "interpretation": "high cosine = the two extraction methods agree on the direction",
    }


# === MAIN ===

def print_summary(report: dict[str, Any]) -> None:
    """Print a short human-readable summary of the Tier 1 report."""
    ortho = report["orthogonality"]
    cross = report["cross_projection"]
    print("\n=== Tier 1 geometry summary ===")
    print(f"economic vs social  : final cosine = {ortho['final_vector_cosine']:+.3f} "
          f"(max |per-layer| = {ortho['max_abs_per_layer_cosine']:.3f})")
    print(f"cross-projection    : min diagonal = {cross['min_diagonal_separation']:.2f}, "
          f"max |off-diagonal| = {cross['max_abs_off_diagonal_separation']:.2f}")
    for axis in AXES:
        nulls = report["axes"][axis]["null_baselines"]
        sep = report["axes"][axis]["held_out_separability"]
        print(f"{axis:9s}          : separation = {nulls['real_separation']:.2f} "
              f"(shuffled p95 = {nulls['shuffled_label_null']['p95']:.2f}, "
              f"perm p = {nulls['shuffled_label_null']['permutation_p_value']:.3f}) | "
              f"CV AUC = {sep['cross_validated_auc']:.3f} "
              f"(shuffled {sep['shuffled_label_auc']:.3f})")
    print()


def main() -> None:
    """Run all Tier 1 geometry checks and write a JSON report."""
    args = parse_args()

    economic = load_axis_data("economic", args.vectors_dir, args.activations_dir, args.method)
    social = load_axis_data("social", args.vectors_dir, args.activations_dir, args.method)

    report: dict[str, Any] = {
        "method": args.method,
        "config": {
            "num_permutations": args.num_permutations,
            "num_random": args.num_random,
            "kfolds": args.kfolds,
            "seed": args.seed,
        },
        "orthogonality": check_orthogonality(economic, social),
        "cross_projection": cross_projection_matrix(economic, social),
        "axes": {},
    }
    for axis in (economic, social):
        report["axes"][axis.name] = {
            "null_baselines": null_baselines(
                axis, args.num_random, args.num_permutations, args.seed
            ),
            "held_out_separability": held_out_separability(axis, args.kfolds, args.seed),
            "method_agreement": method_agreement(axis, args.vectors_dir),
        }

    save_json(report, args.output)
    print_summary(report)
    print(f"Report written to {args.output}")


if __name__ == "__main__":
    main()
