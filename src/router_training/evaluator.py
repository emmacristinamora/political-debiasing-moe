# src/evaluate_router_checkpoint.py


# === IMPORTS ===

from __future__ import annotations

import argparse
import json
import logging
import math
import statistics
import sys
from pathlib import Path
from typing import Any

import torch
from torch import nn


# router_training.config and router_training.validator are torch-free.
_SRC_DIR = Path(__file__).resolve().parents[1]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from router_training.config import (  # noqa: E402
    CANONICAL_QUADRANT_ORDER,
    load_router_calibration_config,
)
from router_training.validator import (  # noqa: E402
    load_records_jsonl,
    parse_hidden_ref,
    validate_router_dataset,
)


# === CONSTANTS ===

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# fallback locations (relative to PROJECT_ROOT) used only when the config does
# not provide the corresponding directory. exact strings match the spec.
FALLBACK_CHECKPOINT_PATH: Path = Path("data/router/checkpoints/calibrated_router.pt")
FALLBACK_VAL_RECORDS_PATH: Path = Path("data/router/val/records.jsonl")
FALLBACK_TEST_RECORDS_PATH: Path = Path("data/router/test/records.jsonl")
FALLBACK_REPORT_PATH: Path = Path("data/router/reports/router_checkpoint_eval.json")
FALLBACK_HIDDEN_PATH: Path = Path("data/router/hidden.pt")

DEFAULT_CHECKPOINT_FILENAME: str = "calibrated_router.pt"
DEFAULT_REPORT_FILENAME: str = "router_checkpoint_eval.json"
RECORDS_FILENAME: str = "records.jsonl"

DEFAULT_NUM_REPORT_EXAMPLES: int = 20

# floor used so log(0) never appears when an entry of a probability dict is
# numerically zero. matches the trainer's "target_policy strictly > 0" rule.
LOG_FLOOR: float = 1e-12


logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO,
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# === ARG PARSING ===

def parse_args() -> argparse.Namespace:
    """
    Parse CLI args for the offline router-checkpoint evaluator.
    Returns:
        argparse.Namespace: parsed args.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Offline evaluator for a calibrated-router checkpoint: compares "
            "heuristic prior, calibrated policy, and target policy on val/test "
            "records and writes a JSON report."
        ),
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint-path",   type=Path, default=None)
    parser.add_argument("--hidden-path",       type=Path, default=None)
    parser.add_argument("--val-records-path",  type=Path, default=None)
    parser.add_argument("--test-records-path", type=Path, default=None)
    parser.add_argument("--output-path",       type=Path, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument(
        "--num-report-examples",
        type=int,
        default=DEFAULT_NUM_REPORT_EXAMPLES,
    )
    return parser.parse_args()


# === NUMERIC HELPERS ===

def stable_log_softmax(values: list[float]) -> list[float]:
    """
    Numerically-stable log_softmax over a short list of floats. Pure-python
    so it stays usable in tests that build canonical-order lists by hand.
    """
    if not values:
        raise ValueError("stable_log_softmax requires non-empty input")
    m = max(values)
    shifted = [v - m for v in values]
    exps = [math.exp(s) for s in shifted]
    z = sum(exps)
    if z <= 0:
        raise ValueError("stable_log_softmax: zero normalising constant")
    log_z = math.log(z)
    return [s - log_z for s in shifted]


def heuristic_prior_from_scores(
    quadrant_scores: dict[str, float],
    beta: float,
    temperature: float,
) -> dict[str, float]:
    """
    Compute the heuristic prior over CANONICAL_QUADRANT_ORDER:
        log_prior = log_softmax(-beta * quadrant_scores / temperature)
    Mirrors src/train_calibrated_router.build_heuristic_prior_tensor without
    the runtime center-fallback gate.
    """
    if temperature == 0:
        raise ValueError("temperature must be non-zero for heuristic prior")
    logits = [
        -float(beta) * float(quadrant_scores[k]) / float(temperature)
        for k in CANONICAL_QUADRANT_ORDER
    ]
    log_p = stable_log_softmax(logits)
    return {k: math.exp(lp) for k, lp in zip(CANONICAL_QUADRANT_ORDER, log_p)}


def policy_from_logits(logits: Any) -> dict[str, float]:
    """
    Convert a list/tuple/dict in canonical order to a probability dict by
    stable softmax. Dict inputs must use the canonical quadrant keys.
    """
    if isinstance(logits, dict):
        missing = [k for k in CANONICAL_QUADRANT_ORDER if k not in logits]
        if missing:
            raise ValueError(f"policy_from_logits missing keys: {missing}")
        values = [float(logits[k]) for k in CANONICAL_QUADRANT_ORDER]
    else:
        values = [float(x) for x in logits]
        if len(values) != len(CANONICAL_QUADRANT_ORDER):
            raise ValueError(
                f"policy_from_logits expects {len(CANONICAL_QUADRANT_ORDER)} "
                f"values, got {len(values)}"
            )
    log_p = stable_log_softmax(values)
    return {k: math.exp(lp) for k, lp in zip(CANONICAL_QUADRANT_ORDER, log_p)}


def kl_policy(p: dict[str, float], q: dict[str, float]) -> float:
    """
    KL(p || q) over CANONICAL_QUADRANT_ORDER. Returns +inf if q is zero on
    a quadrant where p is positive; entries with p == 0 contribute zero
    (0 * log(0/q) := 0 by convention).
    """
    total = 0.0
    for k in CANONICAL_QUADRANT_ORDER:
        pk = float(p[k])
        qk = float(q[k])
        if pk <= 0:
            continue
        if qk <= 0:
            return float("inf")
        total += pk * (math.log(pk) - math.log(qk))
    return total


def entropy(policy: dict[str, float]) -> float:
    """Shannon entropy in nats; entries == 0 contribute zero."""
    total = 0.0
    for k in CANONICAL_QUADRANT_ORDER:
        v = float(policy[k])
        if v > 0:
            total -= v * math.log(v)
    return total


def l1_distance(p: dict[str, float], q: dict[str, float]) -> float:
    """L1 distance between two distributions over CANONICAL_QUADRANT_ORDER."""
    return sum(abs(float(p[k]) - float(q[k])) for k in CANONICAL_QUADRANT_ORDER)


def _argmax_quadrant(policy: dict[str, float]) -> str:
    """
    Argmax key over CANONICAL_QUADRANT_ORDER, with ties broken by canonical
    order (max() returns the first key on ties because we iterate the tuple
    in canonical order).
    """
    return max(CANONICAL_QUADRANT_ORDER, key=lambda k: float(policy[k]))


# === CHECKPOINT LOADER ===

def load_checkpoint_head(
    checkpoint_path: Path,
    expected_dim: int,
    device: Any = "cpu",
) -> tuple[nn.Linear, dict[str, Any]]:
    """
    Load the calibrated-router correction head from a trainer checkpoint.
    Args:
        checkpoint_path: path to the .pt file written by train_calibrated_router.py.
        expected_dim:    calibration input dim demanded by the runtime config.
        device:          torch device string or torch.device for the loaded head.
    Returns:
        (nn.Linear, metadata) — head loaded in eval() mode on `device`, plus
        the checkpoint payload sans state_dict.
    Logic:
        Validates that the checkpoint exposes either calibration_input_dim or
        the legacy router_hidden_dim alias, that this matches expected_dim,
        that canonical_quadrant_order (if present) equals the canonical order,
        and that the linear-head state_dict has weight [4, hidden_dim] and
        bias [4].
    """
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint file not found: {checkpoint_path}")

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(
            f"checkpoint must be a dict payload, got {type(payload).__name__}"
        )
    if "state_dict" not in payload:
        raise ValueError("checkpoint missing required 'state_dict' field")
    state_dict = payload["state_dict"]
    if not isinstance(state_dict, dict):
        raise ValueError(
            f"checkpoint['state_dict'] must be a dict, got {type(state_dict).__name__}"
        )

    ckpt_dim_raw = payload.get("calibration_input_dim")
    if ckpt_dim_raw is None:
        ckpt_dim_raw = payload.get("router_hidden_dim")
    if ckpt_dim_raw is None:
        raise ValueError(
            "checkpoint missing both 'calibration_input_dim' and legacy "
            "'router_hidden_dim'"
        )
    ckpt_dim = int(ckpt_dim_raw)
    if ckpt_dim != int(expected_dim):
        raise ValueError(
            f"checkpoint hidden dim {ckpt_dim} does not match expected "
            f"{int(expected_dim)} (from config / runtime)"
        )

    canonical = payload.get("canonical_quadrant_order")
    if canonical is not None and list(canonical) != list(CANONICAL_QUADRANT_ORDER):
        raise ValueError(
            f"checkpoint canonical_quadrant_order {list(canonical)} does not "
            f"match canonical {list(CANONICAL_QUADRANT_ORDER)}"
        )

    weight = state_dict.get("weight")
    bias = state_dict.get("bias")
    if weight is None or bias is None:
        raise ValueError("checkpoint state_dict missing 'weight' or 'bias'")
    expected_w = (len(CANONICAL_QUADRANT_ORDER), ckpt_dim)
    if tuple(weight.shape) != expected_w:
        raise ValueError(
            f"checkpoint weight shape {tuple(weight.shape)} does not match "
            f"expected {expected_w}"
        )
    if tuple(bias.shape) != (len(CANONICAL_QUADRANT_ORDER),):
        raise ValueError(
            f"checkpoint bias shape {tuple(bias.shape)} does not match "
            f"expected ({len(CANONICAL_QUADRANT_ORDER)},)"
        )

    head = nn.Linear(ckpt_dim, len(CANONICAL_QUADRANT_ORDER))
    head.load_state_dict(state_dict)
    head.eval()
    head.to(device)

    metadata = {k: v for k, v in payload.items() if k != "state_dict"}
    return head, metadata


# === HIDDEN-TENSOR LOADER ===

def load_hidden_tensor(path: Path) -> torch.Tensor:
    """
    Load and shape-check the hidden-representation tensor.
    Returns a 2D float32 tensor [num_examples, hidden_dim].
    """
    if not path.is_file():
        raise FileNotFoundError(f"hidden tensor file not found: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, torch.Tensor):
        raise ValueError(
            f"hidden tensor file must contain a torch.Tensor, got "
            f"{type(payload).__name__}"
        )
    if payload.dim() != 2:
        raise ValueError(
            f"hidden tensor must be 2D [N, hidden_dim], got shape "
            f"{tuple(payload.shape)}"
        )
    return payload.to(dtype=torch.float32)


# === PER-RECORD EVALUATION ===

def evaluate_record(
    record: dict[str, Any],
    hidden_tensor: torch.Tensor,
    head: nn.Linear,
    beta: float,
    temperature: float,
    hidden_filename: str,
) -> dict[str, Any]:
    """
    Score one record under heuristic, calibrated, and target policies.
    Returns a dict with the three policies and every per-example metric.
    """
    eid = record["example_id"]
    ref_filename, row_index = parse_hidden_ref(record["hidden_representation_ref"])
    if ref_filename != hidden_filename:
        raise ValueError(
            f"[{eid}] hidden_representation_ref filename {ref_filename!r} does "
            f"not match expected {hidden_filename!r}"
        )
    if row_index >= hidden_tensor.shape[0]:
        raise ValueError(
            f"[{eid}] hidden row index {row_index} out of range for tensor "
            f"with {hidden_tensor.shape[0]} rows"
        )

    target = {k: float(record["target_policy"][k]) for k in CANONICAL_QUADRANT_ORDER}
    quadrant_scores = record["quadrant_scores"]

    heuristic = heuristic_prior_from_scores(quadrant_scores, beta, temperature)

    head_device = head.weight.device
    h = hidden_tensor[row_index].to(dtype=torch.float32, device=head_device)
    log_prior = torch.tensor(
        [math.log(max(heuristic[k], LOG_FLOOR)) for k in CANONICAL_QUADRANT_ORDER],
        dtype=torch.float32,
        device=head_device,
    )
    with torch.no_grad():
        delta = head(h.unsqueeze(0)).squeeze(0).to(dtype=torch.float32)
    combined = log_prior + delta
    log_calibrated = torch.log_softmax(combined, dim=0)
    calibrated_values = log_calibrated.exp().tolist()
    calibrated = {k: float(v) for k, v in zip(CANONICAL_QUADRANT_ORDER, calibrated_values)}

    kl_t_h = kl_policy(target, heuristic)
    kl_t_c = kl_policy(target, calibrated)
    kl_c_h = kl_policy(calibrated, heuristic)

    top1_t = _argmax_quadrant(target)
    top1_h = _argmax_quadrant(heuristic)
    top1_c = _argmax_quadrant(calibrated)

    return {
        "example_id": eid,
        "prompt_text": record.get("prompt_text", ""),
        "target_policy": target,
        "heuristic_prior": heuristic,
        "calibrated_policy": calibrated,
        "metrics": {
            "kl_target_to_heuristic": kl_t_h,
            "kl_target_to_calibrated": kl_t_c,
            "kl_calibrated_to_heuristic": kl_c_h,
            "improvement_kl": kl_t_h - kl_t_c,
            "entropy_heuristic": entropy(heuristic),
            "entropy_calibrated": entropy(calibrated),
            "entropy_target": entropy(target),
            "l1_target_heuristic": l1_distance(target, heuristic),
            "l1_target_calibrated": l1_distance(target, calibrated),
            "l1_calibrated_heuristic": l1_distance(calibrated, heuristic),
            "top1_target": top1_t,
            "top1_heuristic": top1_h,
            "top1_calibrated": top1_c,
            "heuristic_top1_matches_target": bool(top1_h == top1_t),
            "calibrated_top1_matches_target": bool(top1_c == top1_t),
        },
    }


# === SUMMARY ===

_SUMMARY_KEYS: tuple[str, ...] = (
    "mean_kl_target_to_heuristic",
    "mean_kl_target_to_calibrated",
    "mean_improvement_kl",
    "median_improvement_kl",
    "frac_improved_kl",
    "mean_kl_calibrated_to_heuristic",
    "mean_entropy_heuristic",
    "mean_entropy_calibrated",
    "mean_entropy_target",
    "mean_l1_target_heuristic",
    "mean_l1_target_calibrated",
    "heuristic_top1_accuracy",
    "calibrated_top1_accuracy",
)


def summarize_examples(example_metrics: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Aggregate per-record metric dicts into the summary block. Returns None
    for every key when example_metrics is empty so the schema stays stable.
    """
    if not example_metrics:
        return {key: None for key in _SUMMARY_KEYS}

    def _mean(metric_key: str) -> float:
        return statistics.fmean(
            float(m["metrics"][metric_key]) for m in example_metrics
        )

    def _mean_bool(metric_key: str) -> float:
        return statistics.fmean(
            1.0 if m["metrics"][metric_key] else 0.0 for m in example_metrics
        )

    improvements = [float(m["metrics"]["improvement_kl"]) for m in example_metrics]

    return {
        "mean_kl_target_to_heuristic":   _mean("kl_target_to_heuristic"),
        "mean_kl_target_to_calibrated":  _mean("kl_target_to_calibrated"),
        "mean_improvement_kl":           _mean("improvement_kl"),
        "median_improvement_kl":         statistics.median(improvements),
        "frac_improved_kl":              statistics.fmean(
                                            1.0 if v > 0 else 0.0 for v in improvements
                                         ),
        "mean_kl_calibrated_to_heuristic": _mean("kl_calibrated_to_heuristic"),
        "mean_entropy_heuristic":        _mean("entropy_heuristic"),
        "mean_entropy_calibrated":       _mean("entropy_calibrated"),
        "mean_entropy_target":           _mean("entropy_target"),
        "mean_l1_target_heuristic":      _mean("l1_target_heuristic"),
        "mean_l1_target_calibrated":     _mean("l1_target_calibrated"),
        "heuristic_top1_accuracy":       _mean_bool("heuristic_top1_matches_target"),
        "calibrated_top1_accuracy":      _mean_bool("calibrated_top1_matches_target"),
    }


# === SPLIT EVALUATION ===

def evaluate_split(
    records: list[dict[str, Any]],
    hidden_tensor: torch.Tensor,
    head: nn.Linear,
    beta: float,
    temperature: float,
    hidden_filename: str,
    max_examples: int | None = None,
    num_report_examples: int = DEFAULT_NUM_REPORT_EXAMPLES,
) -> dict[str, Any]:
    """
    Evaluate every record in `records` and assemble the per-split block.
    `max_examples` truncates the dataset before evaluation; `num_report_examples`
    truncates only the included `examples` list (the summary always reflects
    every evaluated record).
    """
    if max_examples is not None:
        if max_examples <= 0:
            raise ValueError(f"max_examples must be > 0; got {max_examples}")
        records = records[:max_examples]
    if num_report_examples < 0:
        raise ValueError(
            f"num_report_examples must be >= 0; got {num_report_examples}"
        )

    per_record = [
        evaluate_record(r, hidden_tensor, head, beta, temperature, hidden_filename)
        for r in records
    ]
    summary = summarize_examples(per_record)
    examples = per_record[:num_report_examples]
    return {
        "num_records": len(per_record),
        "summary": summary,
        "examples": examples,
    }


# === REPORTING ===

def _json_safe(obj: Any) -> Any:
    """
    Coerce a value to something json.dumps can serialise. Tensors and unknown
    types are stringified via repr; primitives, dicts, and lists pass through
    structurally.
    """
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return repr(obj)


def write_report(report: dict[str, Any], path: Path) -> None:
    """Write the evaluation report as JSON, creating parent dirs as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    safe = _json_safe(report)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(safe, fh, ensure_ascii=False, indent=2)


# === PATH RESOLUTION ===

def resolve_paths(cfg: Any, args: argparse.Namespace) -> dict[str, Path]:
    """
    Resolve checkpoint/hidden/records/output paths with CLI overrides winning.
    Defaults follow the layout produced by Steps 9 and 10.
    """
    paths_cfg = getattr(cfg, "paths", None)

    output_dir = getattr(paths_cfg, "output_dir", None) if paths_cfg is not None else None
    base_dir = Path(output_dir) if output_dir is not None else PROJECT_ROOT / "data" / "router"

    checkpoints_dir = (
        getattr(paths_cfg, "checkpoints_dir", None) if paths_cfg is not None else None
    )
    reports_dir = getattr(paths_cfg, "reports_dir", None) if paths_cfg is not None else None
    hidden_path_cfg = (
        getattr(paths_cfg, "hidden_path", None) if paths_cfg is not None else None
    )

    if args.checkpoint_path is not None:
        checkpoint_path = args.checkpoint_path
    elif checkpoints_dir is not None:
        checkpoint_path = Path(checkpoints_dir) / DEFAULT_CHECKPOINT_FILENAME
    else:
        checkpoint_path = PROJECT_ROOT / FALLBACK_CHECKPOINT_PATH

    if args.hidden_path is not None:
        hidden_path = args.hidden_path
    elif hidden_path_cfg is not None:
        hidden_path = Path(hidden_path_cfg)
    else:
        hidden_path = PROJECT_ROOT / FALLBACK_HIDDEN_PATH

    val_records = (
        args.val_records_path if args.val_records_path is not None
        else base_dir / "val" / RECORDS_FILENAME
    )
    test_records = (
        args.test_records_path if args.test_records_path is not None
        else base_dir / "test" / RECORDS_FILENAME
    )

    if args.output_path is not None:
        output_path = args.output_path
    elif reports_dir is not None:
        output_path = Path(reports_dir) / DEFAULT_REPORT_FILENAME
    else:
        output_path = PROJECT_ROOT / FALLBACK_REPORT_PATH

    return {
        "checkpoint":   Path(checkpoint_path),
        "hidden":       Path(hidden_path),
        "val_records":  Path(val_records),
        "test_records": Path(test_records),
        "output":       Path(output_path),
    }


# === ORCHESTRATION ===

def _resolve_hparam(metadata: dict[str, Any], cfg: Any, name: str) -> float:
    """Prefer the value the checkpoint was trained with; fall back to config."""
    if name in metadata and metadata[name] is not None:
        return float(metadata[name])
    training = getattr(cfg, "training", None)
    if training is not None and hasattr(training, name):
        return float(getattr(training, name))
    raise ValueError(f"could not resolve hparam {name!r} from checkpoint or config")


def _resolve_expected_dim(cfg: Any) -> int:
    """Pull calibration_input_dim from training, else input_transformer."""
    training = getattr(cfg, "training", None)
    dim = getattr(training, "calibration_input_dim", None) if training is not None else None
    if dim is None:
        input_transformer = getattr(cfg, "input_transformer", None)
        dim = (
            getattr(input_transformer, "calibration_input_dim", None)
            if input_transformer is not None else None
        )
    if dim is None:
        raise ValueError(
            "config missing calibration_input_dim under 'training' or "
            "'input_transformer'"
        )
    return int(dim)


def run_evaluation(args: argparse.Namespace) -> dict[str, Any]:
    """
    End-to-end orchestration: load config + checkpoint + hidden tensor,
    validate and evaluate every existing split, write the report, and
    return it. Raises FileNotFoundError if neither val nor test records
    are present.
    """
    cfg = load_router_calibration_config(args.config)
    paths = resolve_paths(cfg, args)
    expected_dim = _resolve_expected_dim(cfg)

    head, metadata = load_checkpoint_head(
        paths["checkpoint"], expected_dim, device=args.device
    )
    hidden_tensor = load_hidden_tensor(paths["hidden"]).to(args.device)
    hidden_filename = paths["hidden"].name

    beta = _resolve_hparam(metadata, cfg, "beta")
    temperature = _resolve_hparam(metadata, cfg, "temperature")

    splits: dict[str, Any] = {}
    warnings: list[str] = []

    for split_name, key in (("val", "val_records"), ("test", "test_records")):
        split_path: Path = paths[key]
        if not split_path.is_file():
            warnings.append(f"{split_name} records not found: {split_path}")
            continue
        records = load_records_jsonl(split_path)
        validate_router_dataset(
            records,
            hidden_tensor,
            expected_hidden_dim=expected_dim,
            hidden_filename=hidden_filename,
        )
        block = evaluate_split(
            records=records,
            hidden_tensor=hidden_tensor,
            head=head,
            beta=beta,
            temperature=temperature,
            hidden_filename=hidden_filename,
            max_examples=args.max_examples,
            num_report_examples=args.num_report_examples,
        )
        block["records_path"] = str(split_path)
        splits[split_name] = block

    if not splits:
        raise FileNotFoundError(
            "neither val nor test records files exist; nothing to evaluate "
            f"(searched: {paths['val_records']}, {paths['test_records']})"
        )

    report = {
        "config_path": str(args.config),
        "checkpoint_path": str(paths["checkpoint"]),
        "hidden_path": str(paths["hidden"]),
        "splits": splits,
        "checkpoint_metadata": _json_safe(metadata),
        "hyperparameters": {
            "beta": beta,
            "temperature": temperature,
            "calibration_input_dim": expected_dim,
        },
        "warnings": warnings,
    }
    write_report(report, paths["output"])
    _log_summary(report)
    return report


def _log_summary(report: dict[str, Any]) -> None:
    """Print a concise per-split summary so the CLI is useful on its own."""
    for split_name, block in report["splits"].items():
        s = block["summary"]
        log.info(
            "split=%s n=%d kl_t->h=%.4f kl_t->c=%.4f improvement_kl=%.4f "
            "frac_improved=%.3f top1: heur=%.3f calib=%.3f",
            split_name,
            block["num_records"],
            s["mean_kl_target_to_heuristic"],
            s["mean_kl_target_to_calibrated"],
            s["mean_improvement_kl"],
            s["frac_improved_kl"],
            s["heuristic_top1_accuracy"],
            s["calibrated_top1_accuracy"],
        )


# === MAIN ===

def main() -> None:
    args = parse_args()
    run_evaluation(args)


if __name__ == "__main__":
    main()
