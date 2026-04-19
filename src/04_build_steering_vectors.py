# src/04_build_steering_vectors.py


# === IMPORTS ===

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression


# === CONFIG ===

PROJECT_ROOT = Path(__file__).resolve().parents[1]

HARDCODED_PATHS = {
    "economic": {
        "activations_file": PROJECT_ROOT / "data" / "steering-vectors" / "activations" / "economic_activations.pt",
        "vectors_file": PROJECT_ROOT / "data" / "steering-vectors" / "vectors" / "economic_vectors.pt",
        "report_file": PROJECT_ROOT / "data" / "steering-vectors" / "reports" / "economic_vectors_report.json",
    },
    "social": {
        "activations_file": PROJECT_ROOT / "data" / "steering-vectors" / "activations" / "social_activations.pt",
        "vectors_file": PROJECT_ROOT / "data" / "steering-vectors" / "vectors" / "social_vectors.pt",
        "report_file": PROJECT_ROOT / "data" / "steering-vectors" / "reports" / "social_vectors_report.json",
    },
}

VALID_AXES = {"economic", "social"}


@dataclass
class AxisConvention:
    axis: str
    positive_label: str
    negative_label: str


# === CLI ===

def parse_args() -> argparse.Namespace:
    """
    Parse runtime arguments.
    Returns:
        argparse.Namespace: Parsed CLI args.
    """
    parser = argparse.ArgumentParser(
        description="Build steering vectors from saved activations."
    )
    parser.add_argument(
        "--axis",
        type=str,
        required=True,
        choices=["economic", "social"],
        help="Axis to process.",
    )
    parser.add_argument(
        "--logistic-max-iter",
        type=int,
        default=1000,
        help="Maximum iterations for logistic regression.",
    )
    parser.add_argument(
        "--logistic-c",
        type=float,
        default=1.0,
        help="Inverse regularization strength for logistic regression.",
    )
    return parser.parse_args()


# === IO HELPERS ===

def load_pt(path: Path) -> dict[str, Any]:
    """
    Load a PyTorch artifact.
    Args:
        path: Input .pt path.
    Returns:
        dict[str, Any]: Loaded artifact.
    """
    if not path.exists():
        raise FileNotFoundError(f"Activation file not found: {path}")

    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected activation payload to be a dictionary: {path}")

    return payload


def save_pt(payload: dict[str, Any], path: Path) -> None:
    """
    Save a PyTorch artifact.
    Args:
        payload: Serializable payload.
        path: Output path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def save_json(payload: dict[str, Any], path: Path) -> None:
    """
    Save a JSON report.
    Args:
        payload: JSON-serializable payload.
        path: Output path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


# === CONFIG HELPERS ===

def get_hardcoded_paths(axis: str) -> tuple[Path, Path, Path]:
    """
    Resolve input and output paths for an axis.
    Args:
        axis: economic or social.
    Returns:
        tuple[Path, Path, Path]: activations, vectors, report paths.
    """
    if axis not in HARDCODED_PATHS:
        raise ValueError(f"Unsupported axis: {axis}")

    axis_paths = HARDCODED_PATHS[axis]
    return (
        axis_paths["activations_file"],
        axis_paths["vectors_file"],
        axis_paths["report_file"],
    )


def get_axis_convention(axis: str) -> AxisConvention:
    """
    Return the sign convention for an axis.
    Args:
        axis: economic or social.
    Returns:
        AxisConvention: Positive and negative labels.
    """
    if axis == "economic":
        return AxisConvention(
            axis="economic",
            positive_label="econ_right",
            negative_label="econ_left",
        )

    if axis == "social":
        return AxisConvention(
            axis="social",
            positive_label="authoritarian",
            negative_label="libertarian",
        )

    raise ValueError(f"Unsupported axis: {axis}")


# === VALIDATION HELPERS ===

def validate_activation_artifact(artifact: dict[str, Any], axis: str) -> None:
    """
    Validate the shape and metadata of the activation artifact.
    Args:
        artifact: Loaded activation payload.
        axis: Requested axis.
    """
    required_top_level = [
        "meta",
        "pair_ids",
        "statement_ids",
        "template_ids",
        "token_counts",
        "activations",
    ]

    for key in required_top_level:
        if key not in artifact:
            raise ValueError(f"Missing required key '{key}' in activation artifact")

    meta = artifact["meta"]
    if not isinstance(meta, dict):
        raise ValueError("Artifact 'meta' must be a dictionary")

    if meta.get("axis") != axis:
        raise ValueError(
            f"Axis mismatch in activation artifact: expected '{axis}', got '{meta.get('axis')}'"
        )

    activations = artifact["activations"]
    if not isinstance(activations, dict) or not activations:
        raise ValueError("Artifact 'activations' must be a non-empty dictionary")

    for layer_key, layer_payload in activations.items():
        if "pos" not in layer_payload or "neg" not in layer_payload:
            raise ValueError(f"Layer {layer_key} is missing 'pos' or 'neg' tensors")

        pos_tensor = layer_payload["pos"]
        neg_tensor = layer_payload["neg"]

        if not isinstance(pos_tensor, torch.Tensor) or not isinstance(neg_tensor, torch.Tensor):
            raise ValueError(f"Layer {layer_key} activations must be torch tensors")

        if pos_tensor.ndim != 2 or neg_tensor.ndim != 2:
            raise ValueError(
                f"Layer {layer_key} tensors must be 2D, got {tuple(pos_tensor.shape)} and {tuple(neg_tensor.shape)}"
            )

        if pos_tensor.shape != neg_tensor.shape:
            raise ValueError(
                f"Layer {layer_key} pos/neg shapes must match, got {tuple(pos_tensor.shape)} vs {tuple(neg_tensor.shape)}"
            )

        if pos_tensor.shape[0] == 0:
            raise ValueError(f"Layer {layer_key} has zero examples")


def canonicalize_layer_map(activations: dict[Any, Any]) -> dict[int, dict[str, torch.Tensor]]:
    """
    Convert layer keys to integers and sort them.
    Args:
        activations: Raw activations mapping.
    Returns:
        dict[int, dict[str, torch.Tensor]]: Canonicalized layer map.
    """
    canonicalized: dict[int, dict[str, torch.Tensor]] = {}

    for layer_key, payload in activations.items():
        layer_index = int(layer_key)
        canonicalized[layer_index] = payload

    return dict(sorted(canonicalized.items(), key=lambda item: item[0]))


# === VECTOR HELPERS ===

def normalize_vector(vector: torch.Tensor) -> tuple[torch.Tensor, float]:
    """
    Normalize a vector to unit norm.
    Args:
        vector: Input vector.
    Returns:
        tuple[torch.Tensor, float]: normalized vector and raw norm.
    """
    if vector.ndim != 1:
        raise ValueError(f"Expected 1D vector, got shape {tuple(vector.shape)}")

    raw_norm = float(torch.linalg.norm(vector).item())
    if raw_norm == 0.0:
        raise ValueError("Cannot normalize a zero-norm vector")

    normalized = vector / raw_norm
    return normalized, raw_norm


def compute_mean_difference_vector(
    pos_tensor: torch.Tensor,
    neg_tensor: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    Compute a mean-difference steering vector.
    Args:
        pos_tensor: Positive activations of shape [n, d].
        neg_tensor: Negative activations of shape [n, d].
    Returns:
        tuple[torch.Tensor, dict[str, float]]:
            normalized vector and method stats.
    """
    mu_pos = pos_tensor.mean(dim=0)
    mu_neg = neg_tensor.mean(dim=0)
    raw_vector = mu_pos - mu_neg
    normalized_vector, raw_norm = normalize_vector(raw_vector)

    stats = {
        "raw_norm": raw_norm,
    }
    return normalized_vector, stats


def compute_logistic_regression_vector(
    pos_tensor: torch.Tensor,
    neg_tensor: torch.Tensor,
    max_iter: int,
    c_value: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    Fit logistic regression and use the classifier weights as the steering direction.
    Args:
        pos_tensor: Positive activations of shape [n, d].
        neg_tensor: Negative activations of shape [n, d].
        max_iter: Maximum logistic iterations.
        c_value: Inverse regularization strength.
    Returns:
        tuple[torch.Tensor, dict[str, float]]:
            normalized vector and logistic stats.
    """
    x_pos = pos_tensor.numpy()
    x_neg = neg_tensor.numpy()

    x = np.concatenate([x_pos, x_neg], axis=0)
    y = np.concatenate(
        [
            np.ones(x_pos.shape[0], dtype=np.int64),
            np.zeros(x_neg.shape[0], dtype=np.int64),
        ],
        axis=0,
    )

    classifier = LogisticRegression(
        random_state=42,
        max_iter=max_iter,
        solver="liblinear",
        C=c_value,
    )
    classifier.fit(x, y)

    raw_vector = torch.tensor(classifier.coef_[0], dtype=torch.float32)
    normalized_vector, raw_norm = normalize_vector(raw_vector)

    train_accuracy = float(classifier.score(x, y))

    stats = {
        "raw_norm": raw_norm,
        "train_accuracy": train_accuracy,
    }
    return normalized_vector, stats


# === METRICS ===

def project_onto_vector(examples: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    """
    Project examples onto a direction vector.
    Args:
        examples: Tensor of shape [n, d].
        vector: Tensor of shape [d].
    Returns:
        torch.Tensor: Projections of shape [n].
    """
    if examples.ndim != 2:
        raise ValueError(f"Examples tensor must be 2D, got {tuple(examples.shape)}")

    if vector.ndim != 1:
        raise ValueError(f"Vector must be 1D, got {tuple(vector.shape)}")

    return examples @ vector


def compute_separation_stats(
    pos_projections: torch.Tensor,
    neg_projections: torch.Tensor,
) -> dict[str, float]:
    """
    Compute projection-based separation metrics.
    Args:
        pos_projections: Positive scalar projections.
        neg_projections: Negative scalar projections.
    Returns:
        dict[str, float]: Summary metrics.
    """
    mean_pos = float(pos_projections.mean().item())
    mean_neg = float(neg_projections.mean().item())

    std_pos = float(pos_projections.std(unbiased=False).item())
    std_neg = float(neg_projections.std(unbiased=False).item())

    pooled_variance = (std_pos ** 2 + std_neg ** 2) / 2.0
    pooled_std = float(np.sqrt(max(pooled_variance, 1e-12)))

    separation = abs(mean_pos - mean_neg) / pooled_std

    return {
        "mean_projection_pos": mean_pos,
        "mean_projection_neg": mean_neg,
        "std_projection_pos": std_pos,
        "std_projection_neg": std_neg,
        "pooled_std": pooled_std,
        "separation": separation,
    }


def enforce_sign_convention(
    vector: torch.Tensor,
    pos_tensor: torch.Tensor,
    neg_tensor: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Flip a vector if needed so positive examples project higher than negative ones.
    Args:
        vector: Candidate direction vector.
        pos_tensor: Positive activations.
        neg_tensor: Negative activations.
    Returns:
        tuple[torch.Tensor, dict[str, Any]]:
            sign-aligned vector and projection stats.
    """
    pos_projections = project_onto_vector(pos_tensor, vector)
    neg_projections = project_onto_vector(neg_tensor, vector)

    flipped = False
    if float(pos_projections.mean().item()) < float(neg_projections.mean().item()):
        vector = -vector
        flipped = True
        pos_projections = project_onto_vector(pos_tensor, vector)
        neg_projections = project_onto_vector(neg_tensor, vector)

    stats = compute_separation_stats(pos_projections, neg_projections)
    stats["flipped_for_sign"] = flipped
    return vector, stats


def cosine_similarity(vector_a: torch.Tensor, vector_b: torch.Tensor) -> float:
    """
    Compute cosine similarity between two normalized vectors.
    Args:
        vector_a: First vector.
        vector_b: Second vector.
    Returns:
        float: Cosine similarity.
    """
    return float(torch.dot(vector_a, vector_b).item())


def compute_quality_score(
    method_name: str,
    metric_stats: dict[str, Any],
) -> float:
    """
    Compute a paper-style layer quality score.
    Args:
        method_name: mean_difference or logistic_regression.
        metric_stats: Projection and accuracy stats.
    Returns:
        float: Quality score.
    """
    separation = float(metric_stats["separation"])

    if method_name == "logistic_regression":
        accuracy = float(metric_stats["train_accuracy"])
        return 0.6 * accuracy + 0.4 * min(separation / 2.0, 1.0)

    return max(separation, 1e-12)


# === AGGREGATION ===

def aggregate_vectors_weighted(
    vectors: list[torch.Tensor],
    quality_scores: list[float],
) -> tuple[torch.Tensor, list[float]]:
    """
    Aggregate normalized per-layer vectors with normalized quality weights.
    Args:
        vectors: Per-layer normalized vectors.
        quality_scores: Per-layer quality scores.
    Returns:
        tuple[torch.Tensor, list[float]]:
            final normalized vector and normalized weights.
    """
    if not vectors:
        raise ValueError("Cannot aggregate an empty vector list")

    if len(vectors) != len(quality_scores):
        raise ValueError("vectors and quality_scores must have the same length")

    weights = np.array(quality_scores, dtype=np.float64)
    weights = np.clip(weights, 1e-12, None)
    weights = weights / weights.sum()

    weighted_sum = torch.zeros_like(vectors[0], dtype=torch.float32)
    for weight, vector in zip(weights.tolist(), vectors):
        weighted_sum = weighted_sum + float(weight) * vector

    final_vector, _ = normalize_vector(weighted_sum)
    return final_vector, weights.tolist()


# === ORCHESTRATION HELPERS ===

def build_layer_result(
    layer_index: int,
    pos_tensor: torch.Tensor,
    neg_tensor: torch.Tensor,
    logistic_max_iter: int,
    logistic_c: float,
) -> dict[str, Any]:
    """
    Build both steering methods for one layer.
    Args:
        layer_index: Transformer layer index.
        pos_tensor: Positive activations [n, d].
        neg_tensor: Negative activations [n, d].
        logistic_max_iter: Logistic regression max_iter.
        logistic_c: Logistic regression C.
    Returns:
        dict[str, Any]: Per-layer results for both methods.
    """
    layer_result: dict[str, Any] = {
        "layer_index": layer_index,
        "num_examples": int(pos_tensor.shape[0]),
        "hidden_dim": int(pos_tensor.shape[1]),
    }

    mean_vector, mean_stats = compute_mean_difference_vector(pos_tensor, neg_tensor)
    mean_vector, mean_projection_stats = enforce_sign_convention(mean_vector, pos_tensor, neg_tensor)
    mean_metrics = {**mean_stats, **mean_projection_stats}
    mean_metrics["quality_score"] = compute_quality_score("mean_difference", mean_metrics)

    logistic_vector, logistic_stats = compute_logistic_regression_vector(
        pos_tensor=pos_tensor,
        neg_tensor=neg_tensor,
        max_iter=logistic_max_iter,
        c_value=logistic_c,
    )
    logistic_vector, logistic_projection_stats = enforce_sign_convention(
        logistic_vector,
        pos_tensor,
        neg_tensor,
    )
    logistic_metrics = {**logistic_stats, **logistic_projection_stats}
    logistic_metrics["quality_score"] = compute_quality_score("logistic_regression", logistic_metrics)

    layer_result["mean_difference"] = {
        "vector": mean_vector,
        "metrics": mean_metrics,
    }
    layer_result["logistic_regression"] = {
        "vector": logistic_vector,
        "metrics": logistic_metrics,
    }
    layer_result["method_cosine_similarity"] = cosine_similarity(mean_vector, logistic_vector)

    return layer_result


def build_all_layer_results(
    activation_artifact: dict[str, Any],
    logistic_max_iter: int,
    logistic_c: float,
) -> dict[int, dict[str, Any]]:
    """
    Build per-layer steering results from an activation artifact.
    Args:
        activation_artifact: Saved activations from stage 03.
        logistic_max_iter: Logistic regression max_iter.
        logistic_c: Logistic regression C.
    Returns:
        dict[int, dict[str, Any]]: Layer-indexed results.
    """
    canonical_layers = canonicalize_layer_map(activation_artifact["activations"])
    results: dict[int, dict[str, Any]] = {}

    for layer_index, payload in canonical_layers.items():
        pos_tensor = payload["pos"].to(torch.float32)
        neg_tensor = payload["neg"].to(torch.float32)

        results[layer_index] = build_layer_result(
            layer_index=layer_index,
            pos_tensor=pos_tensor,
            neg_tensor=neg_tensor,
            logistic_max_iter=logistic_max_iter,
            logistic_c=logistic_c,
        )

    return results


def build_final_method_vector(
    layer_results: dict[int, dict[str, Any]],
    method_name: str,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Aggregate per-layer vectors for one method.
    Args:
        layer_results: Per-layer results.
        method_name: mean_difference or logistic_regression.
    Returns:
        tuple[torch.Tensor, dict[str, Any]]: Final vector and aggregation metadata.
    """
    vectors: list[torch.Tensor] = []
    quality_scores: list[float] = []
    layer_order: list[int] = []

    for layer_index in sorted(layer_results):
        method_payload = layer_results[layer_index][method_name]
        vectors.append(method_payload["vector"])
        quality_scores.append(float(method_payload["metrics"]["quality_score"]))
        layer_order.append(layer_index)

    final_vector, normalized_weights = aggregate_vectors_weighted(vectors, quality_scores)

    metadata = {
        "layers": layer_order,
        "quality_scores": quality_scores,
        "normalized_weights": normalized_weights,
    }
    return final_vector, metadata


def build_vector_artifact(
    axis: str,
    convention: AxisConvention,
    activation_artifact: dict[str, Any],
    layer_results: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    """
    Build the final vector artifact for one axis.
    Args:
        axis: economic or social.
        convention: Axis sign convention.
        activation_artifact: Input activations.
        layer_results: Per-layer vector results.
    Returns:
        dict[str, Any]: Final vector payload.
    """
    mean_final_vector, mean_aggregation = build_final_method_vector(
        layer_results=layer_results,
        method_name="mean_difference",
    )
    logistic_final_vector, logistic_aggregation = build_final_method_vector(
        layer_results=layer_results,
        method_name="logistic_regression",
    )

    serializable_layer_results: dict[int, dict[str, Any]] = {}
    for layer_index, payload in layer_results.items():
        serializable_layer_results[layer_index] = {
            "layer_index": payload["layer_index"],
            "num_examples": payload["num_examples"],
            "hidden_dim": payload["hidden_dim"],
            "method_cosine_similarity": payload["method_cosine_similarity"],
            "mean_difference": {
                "vector": payload["mean_difference"]["vector"],
                "metrics": payload["mean_difference"]["metrics"],
            },
            "logistic_regression": {
                "vector": payload["logistic_regression"]["vector"],
                "metrics": payload["logistic_regression"]["metrics"],
            },
        }

    vector_artifact = {
        "meta": {
            "axis": axis,
            "source_model_name": activation_artifact["meta"]["model_name"],
            "source_layers": activation_artifact["meta"]["layers"],
            "num_pairs": activation_artifact["meta"]["num_pairs"],
            "pooling": activation_artifact["meta"]["pooling"],
            "positive_label": convention.positive_label,
            "negative_label": convention.negative_label,
            "sign_convention": f"positive={convention.positive_label}, negative={convention.negative_label}",
            "source_activation_meta": activation_artifact["meta"],
        },
        "pair_ids": activation_artifact["pair_ids"],
        "statement_ids": activation_artifact["statement_ids"],
        "template_ids": activation_artifact["template_ids"],
        "per_layer": serializable_layer_results,
        "final_vectors": {
            "mean_difference": mean_final_vector,
            "logistic_regression": logistic_final_vector,
            "positive_direction_mean": mean_final_vector,
            "negative_direction_mean": -mean_final_vector,
            "positive_direction_logistic": logistic_final_vector,
            "negative_direction_logistic": -logistic_final_vector,
        },
        "aggregation": {
            "mean_difference": mean_aggregation,
            "logistic_regression": logistic_aggregation,
        },
        "named_directions": {
            convention.positive_label: logistic_final_vector,
            convention.negative_label: -logistic_final_vector,
        },
    }

    return vector_artifact


def build_report(
    axis: str,
    convention: AxisConvention,
    activation_file: Path,
    vectors_file: Path,
    layer_results: dict[int, dict[str, Any]],
    vector_artifact: dict[str, Any],
) -> dict[str, Any]:
    """
    Build a human-readable report.
    Args:
        axis: economic or social.
        convention: Sign convention.
        activation_file: Source activation path.
        vectors_file: Output vector path.
        layer_results: Per-layer results.
        vector_artifact: Final vector artifact.
    Returns:
        dict[str, Any]: JSON-serializable report.
    """
    best_mean_layer = max(
        layer_results,
        key=lambda layer: layer_results[layer]["mean_difference"]["metrics"]["quality_score"],
    )
    best_logistic_layer = max(
        layer_results,
        key=lambda layer: layer_results[layer]["logistic_regression"]["metrics"]["quality_score"],
    )

    layer_summaries: dict[str, Any] = {}
    for layer_index in sorted(layer_results):
        payload = layer_results[layer_index]
        layer_summaries[str(layer_index)] = {
            "method_cosine_similarity": payload["method_cosine_similarity"],
            "mean_difference": payload["mean_difference"]["metrics"],
            "logistic_regression": payload["logistic_regression"]["metrics"],
        }

    final_mean = vector_artifact["final_vectors"]["mean_difference"]
    final_logistic = vector_artifact["final_vectors"]["logistic_regression"]

    report = {
        "axis": axis,
        "positive_label": convention.positive_label,
        "negative_label": convention.negative_label,
        "activation_input_file": str(activation_file),
        "vectors_output_file": str(vectors_file),
        "model_name": vector_artifact["meta"]["source_model_name"],
        "layers": vector_artifact["meta"]["source_layers"],
        "num_pairs": vector_artifact["meta"]["num_pairs"],
        "best_mean_difference_layer": int(best_mean_layer),
        "best_logistic_regression_layer": int(best_logistic_layer),
        "final_mean_logistic_cosine": cosine_similarity(final_mean, final_logistic),
        "aggregation": vector_artifact["aggregation"],
        "layer_summaries": layer_summaries,
    }
    return report


# === MAIN ===

def main() -> None:
    """
    Build steering vectors for one axis from saved activations.
    Logic:
        1. load saved activations
        2. compute per-layer mean-difference and logistic vectors
        3. enforce sign convention
        4. score layers
        5. aggregate layer vectors with quality weights
        6. save vector artifact and report
    """
    args = parse_args()
    axis = args.axis

    if axis not in VALID_AXES:
        raise ValueError(f"Unsupported axis: {axis}")

    print(f"\n{'='*60}")
    print(f"[start] Axis: {axis}")
    print(f"[config] logistic_max_iter={args.logistic_max_iter} | logistic_c={args.logistic_c}")
    print(f"{'='*60}\n")

    convention = get_axis_convention(axis)
    print(f"[convention] positive={convention.positive_label} | negative={convention.negative_label}")

    activations_file, vectors_file, report_file = get_hardcoded_paths(axis)
    print(f"[io] Input:  {activations_file}")
    print(f"[io] Output: {vectors_file}")
    print(f"[io] Report: {report_file}\n")

    print(f"[load] Loading activations...")
    activation_artifact = load_pt(activations_file)
    validate_activation_artifact(activation_artifact, axis)
    layers = activation_artifact["meta"]["layers"]
    num_pairs = activation_artifact["meta"]["num_pairs"]
    print(f"[load] {num_pairs} pairs | layers {layers}\n")

    print(f"[vectors] Building per-layer vectors...")
    layer_results = build_all_layer_results(
        activation_artifact=activation_artifact,
        logistic_max_iter=args.logistic_max_iter,
        logistic_c=args.logistic_c,
    )
    for layer_index in sorted(layer_results):
        md = layer_results[layer_index]["mean_difference"]["metrics"]
        lr = layer_results[layer_index]["logistic_regression"]["metrics"]
        cos = layer_results[layer_index]["method_cosine_similarity"]
        print(
            f"  layer {layer_index:2d} | "
            f"mean_diff sep={md['separation']:.3f} q={md['quality_score']:.3f} | "
            f"logistic acc={lr['train_accuracy']:.3f} q={lr['quality_score']:.3f} | "
            f"cosine={cos:.3f}"
        )

    vector_artifact = build_vector_artifact(
        axis=axis,
        convention=convention,
        activation_artifact=activation_artifact,
        layer_results=layer_results,
    )

    report = build_report(
        axis=axis,
        convention=convention,
        activation_file=activations_file,
        vectors_file=vectors_file,
        layer_results=layer_results,
        vector_artifact=vector_artifact,
    )

    save_pt(vector_artifact, vectors_file)
    save_json(report, report_file)

    print(f"\n{'='*60}")
    print(f"[done] Axis: {axis}")
    print(f"[done] Best mean_diff layer:  {report['best_mean_difference_layer']}")
    print(f"[done] Best logistic layer:   {report['best_logistic_regression_layer']}")
    print(f"[done] Final mean/logistic cosine: {report['final_mean_logistic_cosine']:.3f}")
    print(f"[done] Vectors saved to: {vectors_file}")
    print(f"[done] Report saved to:  {report_file}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()