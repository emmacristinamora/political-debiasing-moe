# src/17_steering_vector_robustness.py


# === IMPORTS ===

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from sklearn.metrics import roc_auc_score


# === CONFIG ===

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_ACTIVATIONS_DIR = PROJECT_ROOT / "data" / "steering-vectors" / "activations"
DEFAULT_TOPIC_KEY = PROJECT_ROOT / "config" / "topic_holdout_key.yaml"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "steering-vectors" / "reports" / "tier3_robustness_report.json"

AXES = ("economic", "social")


# === HELPERS: IO ===

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Tier 3 robustness checks for the political-compass steering vectors."
    )
    parser.add_argument("--activations-dir", type=Path, default=DEFAULT_ACTIVATIONS_DIR)
    parser.add_argument("--topic-key", type=Path, default=DEFAULT_TOPIC_KEY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def save_json(payload: dict[str, Any], path: Path) -> None:
    """Write a JSON report, creating the parent directory if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def load_axis_activations(axis: str, activations_dir: Path) -> dict[str, Any]:
    """
    Load one axis's contrastive-pair activations.

    Returns the per-layer pos/neg tensors plus the statement_id and template_id
    of every pair, aligned row-for-row with the activation tensors.
    """
    path = activations_dir / f"{axis}_activations.pt"
    if not path.is_file():
        raise FileNotFoundError(f"activations file not found: {path}")
    data = torch.load(path, map_location="cpu")
    layers = sorted(int(layer) for layer in data["activations"])
    pos = {layer: data["activations"][layer]["pos"].to(torch.float32) for layer in layers}
    neg = {layer: data["activations"][layer]["neg"].to(torch.float32) for layer in layers}
    return {
        "layers": layers,
        "pos": pos,
        "neg": neg,
        "statement_ids": list(data["statement_ids"]),
        "template_ids": list(data["template_ids"]),
    }


def load_topic_key(path: Path) -> dict[str, dict[str, str]]:
    """
    Load the topic key as {axis: {statement_id: topic}}.

    Statements listed under `excluded` are omitted, so they fall through to a
    None topic and are never held out (they stay in every vector build).
    """
    if not path.is_file():
        raise FileNotFoundError(f"topic key not found: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    topic_of: dict[str, dict[str, str]] = {}
    for axis in AXES:
        topic_of[axis] = {}
        for topic, statement_ids in raw.get(axis, {}).items():
            for statement_id in statement_ids:
                topic_of[axis][statement_id] = topic
    return topic_of


# === HELPERS: GEOMETRY ===

def unit(vector: torch.Tensor) -> torch.Tensor:
    """Return the vector normalised to unit norm."""
    return vector / (vector.norm() + 1e-12)


def build_layer_vectors(
    pos: dict[int, torch.Tensor],
    neg: dict[int, torch.Tensor],
    layers: list[int],
    indices: list[int],
) -> dict[int, torch.Tensor]:
    """Build per-layer unit mean-difference directions from a subset of pairs."""
    rows = torch.tensor(indices, dtype=torch.long)
    return {
        layer: unit(pos[layer][rows].mean(dim=0) - neg[layer][rows].mean(dim=0))
        for layer in layers
    }


def project(
    acts: dict[int, torch.Tensor],
    layer_vectors: dict[int, torch.Tensor],
    layers: list[int],
    indices: list[int],
) -> np.ndarray:
    """
    Project the given pairs' activations onto the per-layer directions.

    Each layer contributes (activation . unit_direction); the per-layer scores
    are averaged with equal weight (the rebuilt vectors carry no quality
    weights, so the layers are weighted uniformly).
    """
    rows = torch.tensor(indices, dtype=torch.long)
    score = torch.zeros(len(indices))
    for layer in layers:
        score = score + acts[layer][rows] @ layer_vectors[layer]
    return (score / len(layers)).numpy()


def standardized_separation(pos_scores: np.ndarray, neg_scores: np.ndarray) -> float:
    """Cohen's-d separation between the positive and negative pole scores."""
    pooled_std = np.sqrt((pos_scores.var(ddof=1) + neg_scores.var(ddof=1)) / 2.0)
    return float((pos_scores.mean() - neg_scores.mean()) / (pooled_std + 1e-12))


def held_out_auc(pos_scores: np.ndarray, neg_scores: np.ndarray) -> float:
    """ROC-AUC for ranking the positive pole above the negative pole."""
    labels = np.concatenate([np.ones(len(pos_scores)), np.zeros(len(neg_scores))])
    scores = np.concatenate([pos_scores, neg_scores])
    return float(roc_auc_score(labels, scores))


# === ROBUSTNESS TESTS ===

def leave_one_group_out(axis_data: dict[str, Any], pair_groups: list[str | None]) -> dict[str, Any]:
    """
    Leave-one-group-out generalisation of the steering direction.

    Logic:
        Each pair carries a group label (a template or a topic; None = never
        held out). For every group, the vector is rebuilt from all pairs
        outside the group and scored on the held-out group's pairs. A direction
        that generalises keeps a high held-out AUC and separation.
    """
    layers, pos, neg = axis_data["layers"], axis_data["pos"], axis_data["neg"]
    groups = sorted({group for group in pair_groups if group is not None})
    results: dict[str, Any] = {}
    for group in groups:
        test_idx = [i for i, g in enumerate(pair_groups) if g == group]
        train_idx = [i for i, g in enumerate(pair_groups) if g != group]
        layer_vectors = build_layer_vectors(pos, neg, layers, train_idx)
        pos_scores = project(pos, layer_vectors, layers, test_idx)
        neg_scores = project(neg, layer_vectors, layers, test_idx)
        results[group] = {
            "n_test_pairs": len(test_idx),
            "held_out_auc": held_out_auc(pos_scores, neg_scores),
            "held_out_separation": standardized_separation(pos_scores, neg_scores),
        }
    aucs = [r["held_out_auc"] for r in results.values()]
    return {
        "per_group": results,
        "min_held_out_auc": min(aucs),
        "mean_held_out_auc": float(np.mean(aucs)),
    }


def template_invariance(axis_data: dict[str, Any]) -> dict[str, Any]:
    """
    Cross-template agreement of the per-statement steering signal.

    Logic:
        The 3 templates are paraphrases of the contrastive instruction. Using
        the vector built from all pairs, each pair gets a signed score
        (positive-pole minus negative-pole projection). If the templates are
        interchangeable, the per-statement scores rank statements the same way
        across templates, so the cross-template Pearson correlations are high.
    """
    layers, pos, neg = axis_data["layers"], axis_data["pos"], axis_data["neg"]
    statement_ids, template_ids = axis_data["statement_ids"], axis_data["template_ids"]
    all_idx = list(range(len(statement_ids)))

    layer_vectors = build_layer_vectors(pos, neg, layers, all_idx)
    pair_score = project(pos, layer_vectors, layers, all_idx) - project(neg, layer_vectors, layers, all_idx)

    by_template: dict[str, dict[str, float]] = {}
    for index, template in enumerate(template_ids):
        by_template.setdefault(template, {})[statement_ids[index]] = float(pair_score[index])

    templates = sorted(by_template)
    correlations: dict[str, float] = {}
    for i, first in enumerate(templates):
        for second in templates[i + 1:]:
            shared = sorted(set(by_template[first]) & set(by_template[second]))
            a = np.array([by_template[first][s] for s in shared])
            b = np.array([by_template[second][s] for s in shared])
            correlations[f"{first}-{second}"] = float(np.corrcoef(a, b)[0, 1])
    return {
        "cross_template_pearson": correlations,
        "min_cross_template_pearson": min(correlations.values()),
        "mean_cross_template_pearson": float(np.mean(list(correlations.values()))),
        "interpretation": "high correlation = templates rank statements alike (paraphrase-invariant)",
    }


# === MAIN ===

def print_summary(report: dict[str, Any]) -> None:
    """Print a short human-readable summary of the Tier 3 report."""
    print("\n=== Tier 3 robustness summary ===")
    for axis in AXES:
        axis_report = report["axes"][axis]
        template = axis_report["leave_one_template_out"]
        topic = axis_report["leave_one_topic_out"]
        invariance = axis_report["template_invariance"]
        print(f"\n{axis}:")
        print(f"  leave-one-template-out : min AUC {template['min_held_out_auc']:.3f}  "
              f"mean AUC {template['mean_held_out_auc']:.3f}")
        print(f"  template invariance    : min cross-template r "
              f"{invariance['min_cross_template_pearson']:.3f}")
        print(f"  leave-one-topic-out    : min AUC {topic['min_held_out_auc']:.3f}  "
              f"mean AUC {topic['mean_held_out_auc']:.3f}")
        worst = min(topic["per_group"].items(), key=lambda kv: kv[1]["held_out_auc"])
        print(f"  weakest held-out topic : {worst[0]} (AUC {worst[1]['held_out_auc']:.3f})")


def main() -> None:
    """Run the Tier 3 robustness checks and write a JSON report."""
    args = parse_args()
    topic_of = load_topic_key(args.topic_key)

    report: dict[str, Any] = {"axes": {}}
    for axis in AXES:
        axis_data = load_axis_activations(axis, args.activations_dir)
        topic_groups = [topic_of[axis].get(statement_id) for statement_id in axis_data["statement_ids"]]
        report["axes"][axis] = {
            "leave_one_template_out": leave_one_group_out(axis_data, axis_data["template_ids"]),
            "template_invariance": template_invariance(axis_data),
            "leave_one_topic_out": leave_one_group_out(axis_data, topic_groups),
        }

    save_json(report, args.output)
    print_summary(report)
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
