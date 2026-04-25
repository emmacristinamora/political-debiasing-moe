# src/train_calibrated_router.py


# === IMPORTS ===

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any

import torch
from torch import nn


# === CONFIG ===

# canonical quadrant ordering. duplicated here intentionally instead of imported
# from src/06_moce_components.py because that filename starts with a digit and
# is awkward to import. this script must stay standalone; the order MUST match
# the runtime router exactly.
CANONICAL_QUADRANT_ORDER: tuple[str, ...] = (
    "left_lib",
    "left_auth",
    "right_lib",
    "right_auth",
)

REQUIRED_RECORD_FIELDS: tuple[str, ...] = (
    "example_id",
    "prompt_text",
    "quadrant_scores",
    "bias_magnitude",
    "target_policy",
    "hidden_representation_ref",
)

# floating-point tolerance for "target_policy sums to 1" check
DISTRIBUTION_SUM_TOLERANCE: float = 1e-6


# === HELPERS ===

def parse_args() -> argparse.Namespace:
    """
    Parse CLI arguments for the calibrated-router training script.
    Returns:
        argparse.Namespace: Parsed CLI args.
    """
    parser = argparse.ArgumentParser(
        description="Train the calibrated-router correction head.",
    )
    # paths
    parser.add_argument(
        "--records-path",
        type=Path,
        required=True,
        help="Path to JSONL records file.",
    )
    parser.add_argument(
        "--hidden-path",
        type=Path,
        required=True,
        help="Path to torch tensor artifact with hidden representations.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        required=True,
        help="Where to save the trained router checkpoint (.pt).",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        required=True,
        help="Where to save the JSON training report.",
    )
    # required hyper
    parser.add_argument(
        "--router-hidden-dim",
        type=int,
        required=True,
        help="Calibration head input dimension; must match runtime config.",
    )
    # training config
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--kl-weight", type=float, default=0.1)
    parser.add_argument("--entropy-weight", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Torch device, e.g. cpu, cuda, cuda:0.",
    )
    parser.add_argument(
        "--save-every-epoch",
        action="store_true",
        help="Save a checkpoint after every epoch in addition to the final one.",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Optional debug limiter; trims dataset to the first N records.",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """
    Load JSONL records into memory.
    Args:
        path: Input JSONL path.
    Returns:
        list[dict[str, Any]]: Parsed rows.
    Logic:
        Reads one JSON object per line. Raises loudly on a missing file, an
        empty file, or any malformed JSON line.
    """
    if not path.exists():
        raise FileNotFoundError(f"Records file not found: {path}")
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                records.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_number} in {path}: {exc}"
                ) from exc
    if not records:
        raise ValueError(f"No records loaded from {path}")
    return records


def load_hidden_tensor(path: Path) -> torch.Tensor:
    """
    Load the hidden-representation tensor artifact and validate its shape.
    Args:
        path: Input .pt path.
    Returns:
        torch.Tensor: 2D tensor of shape [num_examples, hidden_dim], float32.
    """
    if not path.exists():
        raise FileNotFoundError(f"Hidden tensor file not found: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, torch.Tensor):
        raise ValueError(
            f"Expected hidden tensor file to contain a torch.Tensor; "
            f"got {type(payload).__name__} at {path}"
        )
    if payload.dim() != 2:
        raise ValueError(
            f"Expected 2D hidden tensor [num_examples, hidden_dim]; "
            f"got shape {tuple(payload.shape)} at {path}"
        )
    return payload.to(dtype=torch.float32)


def parse_hidden_reference(ref: Any, expected_filename: str) -> int:
    """
    Resolve a 'hidden_representation_ref' string of the form '<filename>:<row_index>'.
    Args:
        ref: The reference string from a JSONL record.
        expected_filename: Filename portion that must match the loaded tensor file.
    Returns:
        int: Row index into the hidden tensor.
    """
    if not isinstance(ref, str):
        raise ValueError(
            f"hidden_representation_ref must be a string; got {type(ref).__name__}"
        )
    if ":" not in ref:
        raise ValueError(
            f"hidden_representation_ref must be '<filename>:<row_index>'; got {ref!r}"
        )
    filename, _, index_part = ref.rpartition(":")
    if filename != expected_filename:
        raise ValueError(
            f"hidden_representation_ref filename {filename!r} does not match "
            f"the loaded tensor file {expected_filename!r}"
        )
    try:
        row_index = int(index_part)
    except ValueError as exc:
        raise ValueError(
            f"hidden_representation_ref row index {index_part!r} is not a valid int; "
            f"ref={ref!r}"
        ) from exc
    if row_index < 0:
        raise ValueError(
            f"hidden_representation_ref row index must be >= 0; got {row_index}"
        )
    return row_index


def validate_score_dict(distribution: Any, field_name: str, example_id: str) -> None:
    """
    Validate a finite numeric dict over CANONICAL_QUADRANT_ORDER (no positivity / sum constraint).
    Logic:
        Checks dict-ness, exact canonical keys, and that every value is a
        finite int/float. Raises ValueError on any violation.
    """
    if not isinstance(distribution, dict):
        raise ValueError(
            f"[{example_id}] {field_name} must be a dict, "
            f"got {type(distribution).__name__}"
        )
    expected = set(CANONICAL_QUADRANT_ORDER)
    actual = set(distribution.keys())
    missing = expected - actual
    if missing:
        raise ValueError(
            f"[{example_id}] {field_name} missing required keys: {sorted(missing)}; "
            f"expected exactly {list(CANONICAL_QUADRANT_ORDER)}"
        )
    unexpected = actual - expected
    if unexpected:
        raise ValueError(
            f"[{example_id}] {field_name} has unexpected keys: {sorted(unexpected)}; "
            f"expected exactly {list(CANONICAL_QUADRANT_ORDER)}"
        )
    for key in CANONICAL_QUADRANT_ORDER:
        value = distribution[key]
        if not isinstance(value, (int, float)):
            raise ValueError(
                f"[{example_id}] {field_name}[{key!r}] must be int or float, "
                f"got {type(value).__name__}"
            )
        if math.isnan(value) or math.isinf(value):
            raise ValueError(f"[{example_id}] {field_name}[{key!r}] is not finite")


def validate_probability_dict(distribution: Any, field_name: str, example_id: str) -> None:
    """
    Validate a strictly-positive probability dict over CANONICAL_QUADRANT_ORDER summing to 1.
    Logic:
        Reuses validate_score_dict for shape/finiteness, then enforces
        strict positivity and sum-to-one within DISTRIBUTION_SUM_TOLERANCE.
    """
    validate_score_dict(distribution, field_name, example_id)
    for key in CANONICAL_QUADRANT_ORDER:
        if distribution[key] <= 0:
            raise ValueError(
                f"[{example_id}] {field_name}[{key!r}] must be strictly positive; "
                f"got {distribution[key]}"
            )
    total = sum(float(distribution[key]) for key in CANONICAL_QUADRANT_ORDER)
    if abs(total - 1.0) > DISTRIBUTION_SUM_TOLERANCE:
        raise ValueError(
            f"[{example_id}] {field_name} must sum to 1 within {DISTRIBUTION_SUM_TOLERANCE}; "
            f"got sum={total}"
        )


def validate_record(record: dict[str, Any]) -> None:
    """
    Validate the JSONL fields of a single training record.
    Logic:
        Checks required fields, type of example_id / prompt_text, finiteness
        of bias_magnitude, canonical quadrant_scores, and that target_policy
        is a valid probability distribution over CANONICAL_QUADRANT_ORDER.
        Hidden vector resolution is performed separately by the caller.
    """
    example_id = record.get("example_id", "<missing example_id>")
    for field in REQUIRED_RECORD_FIELDS:
        if field not in record:
            raise ValueError(
                f"[{example_id}] record missing required field {field!r}"
            )
    if not isinstance(record["example_id"], str):
        raise ValueError(
            f"record example_id must be str, got {type(record['example_id']).__name__}"
        )
    if not isinstance(record["prompt_text"], str):
        raise ValueError(f"[{example_id}] prompt_text must be str")

    bias_magnitude = record["bias_magnitude"]
    if not isinstance(bias_magnitude, (int, float)):
        raise ValueError(
            f"[{example_id}] bias_magnitude must be int or float, "
            f"got {type(bias_magnitude).__name__}"
        )
    if math.isnan(bias_magnitude) or math.isinf(bias_magnitude):
        raise ValueError(f"[{example_id}] bias_magnitude is not finite")

    validate_score_dict(record["quadrant_scores"], "quadrant_scores", example_id)
    validate_probability_dict(record["target_policy"], "target_policy", example_id)

    if "metadata" in record and not isinstance(record["metadata"], dict):
        raise ValueError(f"[{example_id}] metadata must be a dict if present")


def build_training_examples(
    records: list[dict[str, Any]],
    hidden_tensor: torch.Tensor,
    expected_filename: str,
    expected_hidden_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str]]:
    """
    Validate records and assemble batched training tensors in canonical order.
    Args:
        records: JSONL records loaded from disk.
        hidden_tensor: 2D tensor [num_examples, hidden_dim] from load_hidden_tensor.
        expected_filename: Filename portion required in hidden_representation_ref.
        expected_hidden_dim: Calibration input dimension; must match the head.
    Returns:
        hidden_features: [N, hidden_dim] float32
        quadrant_scores: [N, 4] float32 (canonical order)
        target_policies: [N, 4] float32 (canonical order)
        example_ids:     list of N strings, aligned with the rows above
    """
    if hidden_tensor.shape[1] != expected_hidden_dim:
        raise ValueError(
            f"Hidden tensor dim {hidden_tensor.shape[1]} does not match "
            f"--router-hidden-dim {expected_hidden_dim}"
        )

    n = len(records)
    hidden_features = torch.empty((n, expected_hidden_dim), dtype=torch.float32)
    quadrant_scores = torch.empty((n, len(CANONICAL_QUADRANT_ORDER)), dtype=torch.float32)
    target_policies = torch.empty((n, len(CANONICAL_QUADRANT_ORDER)), dtype=torch.float32)
    example_ids: list[str] = []

    for i, record in enumerate(records):
        validate_record(record)
        example_id = record["example_id"]
        row_index = parse_hidden_reference(
            record["hidden_representation_ref"], expected_filename
        )
        if row_index >= hidden_tensor.shape[0]:
            raise ValueError(
                f"[{example_id}] hidden row index {row_index} out of range "
                f"for tensor with {hidden_tensor.shape[0]} rows"
            )
        vector = hidden_tensor[row_index]
        if vector.shape[0] != expected_hidden_dim:
            raise ValueError(
                f"[{example_id}] resolved hidden vector length {vector.shape[0]} "
                f"does not match --router-hidden-dim {expected_hidden_dim}"
            )
        if not torch.isfinite(vector).all().item():
            raise ValueError(
                f"[{example_id}] resolved hidden vector contains NaN or inf"
            )
        hidden_features[i] = vector
        for j, key in enumerate(CANONICAL_QUADRANT_ORDER):
            quadrant_scores[i, j] = float(record["quadrant_scores"][key])
            target_policies[i, j] = float(record["target_policy"][key])
        example_ids.append(example_id)

    return hidden_features, quadrant_scores, target_policies, example_ids


def build_heuristic_prior_tensor(
    quadrant_scores: torch.Tensor,
    beta: float,
    temperature: float,
) -> torch.Tensor:
    """
    Compute log(pi_0) over a batch using the runtime heuristic formula.
    Args:
        quadrant_scores: [B, 4] float32 in canonical order
        beta:            scalar
        temperature:     non-zero scalar
    Returns:
        log_prior: [B, 4] float32; exp(log_prior).sum(dim=1) == 1
    Logic:
        Mirrors Router.build_heuristic_prior's softmax(-beta * q / T) without
        the runtime center-fallback gate (training optimizes target_policy
        directly; the gate is a runtime safety only).
    """
    if temperature == 0:
        raise ValueError("--temperature must be non-zero for heuristic prior")
    logits = -beta * quadrant_scores / temperature
    return torch.log_softmax(logits, dim=1)


def train_one_epoch(
    head: nn.Linear,
    optimizer: torch.optim.Optimizer,
    hidden_features: torch.Tensor,
    log_priors: torch.Tensor,
    target_policies: torch.Tensor,
    batch_size: int,
    kl_weight: float,
    entropy_weight: float,
    generator: torch.Generator,
) -> dict[str, float]:
    """
    Run one epoch of AdamW training and return mean per-example metrics.
    Logic:
        For each minibatch:
            delta    = head(h_batch)
            log_pi   = log_softmax(log_pi_0 + delta)
            pi       = exp(log_pi)
            supervised  = mean over batch of sum_i target_i * (log target_i - log pi_i)
            kl_anchor   = mean over batch of sum_i pi_i     * (log pi_i    - log pi_0_i)
            entropy     = mean over batch of -sum_i pi_i * log pi_i
            total_loss  = supervised + kl_weight * kl_anchor - entropy_weight * entropy
        The minus sign on entropy is explicit: subtracting H(pi) encourages
        higher-entropy (non-collapsed) calibrated policies.
    Returns:
        Dict with keys total_loss, supervised_loss, kl_anchor, entropy --
        each averaged per example over the epoch.
    """
    head.train()
    n = hidden_features.shape[0]
    perm = torch.randperm(n, generator=generator)

    totals = {"total_loss": 0.0, "supervised_loss": 0.0, "kl_anchor": 0.0, "entropy": 0.0}
    seen = 0

    for start in range(0, n, batch_size):
        idx = perm[start:start + batch_size]
        h_batch = hidden_features[idx]
        log_prior_batch = log_priors[idx]
        target_batch = target_policies[idx]

        delta = head(h_batch)
        combined = log_prior_batch + delta
        log_pi = torch.log_softmax(combined, dim=1)
        pi = log_pi.exp()

        log_target = target_batch.log()  # safe: target_policy is validated > 0
        supervised = (target_batch * (log_target - log_pi)).sum(dim=1).mean()
        kl_anchor = (pi * (log_pi - log_prior_batch)).sum(dim=1).mean()
        entropy = -(pi * log_pi).sum(dim=1).mean()
        loss = supervised + kl_weight * kl_anchor - entropy_weight * entropy

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_n = h_batch.shape[0]
        seen += batch_n
        totals["total_loss"]      += loss.item() * batch_n
        totals["supervised_loss"] += supervised.item() * batch_n
        totals["kl_anchor"]       += kl_anchor.item() * batch_n
        totals["entropy"]         += entropy.item() * batch_n

    return {key: value / max(seen, 1) for key, value in totals.items()}


def save_checkpoint(
    head: nn.Linear,
    args: argparse.Namespace,
    output_path: Path,
) -> None:
    """
    Save the trained correction head plus minimal metadata.
    Logic:
        Stores head.state_dict() alongside the canonical quadrant order,
        calibration input dim, prior hyperparameters used during training,
        and the loss weights / epoch count, plus traceability paths.
    """
    payload = {
        "state_dict": head.state_dict(),
        "router_hidden_dim": args.router_hidden_dim,
        "canonical_quadrant_order": list(CANONICAL_QUADRANT_ORDER),
        "beta": args.beta,
        "temperature": args.temperature,
        "kl_weight": args.kl_weight,
        "entropy_weight": args.entropy_weight,
        "epochs": args.epochs,
        "records_path": str(args.records_path),
        "hidden_path": str(args.hidden_path),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)


def save_json(payload: dict[str, Any], path: Path) -> None:
    """
    Save a JSON report.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


# === MAIN ===

def main() -> None:
    args = parse_args()

    # seed RNGs deterministically
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device)

    # load and validate dataset
    records = load_jsonl(args.records_path)
    if args.max_examples is not None:
        if args.max_examples <= 0:
            raise ValueError(f"--max-examples must be > 0; got {args.max_examples}")
        records = records[: args.max_examples]
    hidden_tensor = load_hidden_tensor(args.hidden_path)

    hidden_features, quadrant_scores, target_policies, _example_ids = build_training_examples(
        records=records,
        hidden_tensor=hidden_tensor,
        expected_filename=args.hidden_path.name,
        expected_hidden_dim=args.router_hidden_dim,
    )
    dataset_size = hidden_features.shape[0]

    # log(pi_0) is parameter-free and constant w.r.t. the head; precompute once
    log_priors = build_heuristic_prior_tensor(
        quadrant_scores, args.beta, args.temperature
    )

    hidden_features = hidden_features.to(device)
    log_priors = log_priors.to(device)
    target_policies = target_policies.to(device)

    # nn.Linear shape matches Router.calibration_module exactly
    head = nn.Linear(args.router_hidden_dim, len(CANONICAL_QUADRANT_ORDER)).to(device)
    optimizer = torch.optim.AdamW(
        head.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    # epoch shuffling RNG, derived from --seed for reproducibility
    generator = torch.Generator()
    generator.manual_seed(args.seed)

    epoch_history: list[dict[str, float]] = []
    for epoch in range(1, args.epochs + 1):
        metrics = train_one_epoch(
            head=head,
            optimizer=optimizer,
            hidden_features=hidden_features,
            log_priors=log_priors,
            target_policies=target_policies,
            batch_size=args.batch_size,
            kl_weight=args.kl_weight,
            entropy_weight=args.entropy_weight,
            generator=generator,
        )
        epoch_record = {"epoch": epoch, **metrics}
        epoch_history.append(epoch_record)
        print(
            f"epoch {epoch:>3d} | total={metrics['total_loss']:.4f} | "
            f"supervised={metrics['supervised_loss']:.4f} | "
            f"kl={metrics['kl_anchor']:.4f} | entropy={metrics['entropy']:.4f}"
        )
        if args.save_every_epoch:
            per_epoch_path = args.output_path.with_name(
                f"{args.output_path.stem}.epoch{epoch:03d}"
                f"{args.output_path.suffix or '.pt'}"
            )
            save_checkpoint(head, args, per_epoch_path)

    save_checkpoint(head, args, args.output_path)

    report = {
        "input_paths": {
            "records_path": str(args.records_path),
            "hidden_path": str(args.hidden_path),
        },
        "output_path": str(args.output_path),
        "dataset_size": dataset_size,
        "hidden_dim": args.router_hidden_dim,
        "hyperparameters": {
            "beta": args.beta,
            "temperature": args.temperature,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "kl_weight": args.kl_weight,
            "entropy_weight": args.entropy_weight,
            "seed": args.seed,
            "device": args.device,
        },
        "final_metrics": epoch_history[-1] if epoch_history else {},
        "epoch_history": epoch_history,
    }
    save_json(report, args.report_path)


if __name__ == "__main__":
    main()
