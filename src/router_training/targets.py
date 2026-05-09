# src/build_router_targets.py


# === IMPORTS ===

from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


# router_training.config and router_training.utils are both torch-free.
_SRC_DIR = Path(__file__).resolve().parents[1]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from router_training.config import (  # noqa: E402
    CANONICAL_QUADRANT_ORDER,
    RouterCalibrationConfig,
    load_router_calibration_config,
)
from router_training.utils import apply_min_probability  # noqa: E402


# === CONSTANTS ===

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# step 1's RouterPaths does not have a dedicated candidate_scores_path or
# target_report_path; default both relative to paths.output_dir to avoid
# changing config.yaml. CLI overrides take precedence over both defaults.
DEFAULT_CANDIDATE_SCORES_FILENAME: str = "candidate_scores.jsonl"
DEFAULT_TARGET_REPORT_FILENAME:    str = "target_report.json"

# heuristic priors across scored rows for the same example must agree to
# this tolerance; mismatch is loud (likely a data-pipeline regression).
HEURISTIC_PRIOR_MISMATCH_TOLERANCE: float = 1e-6

# tolerance for "policy sums to 1" check on inputs and outputs
DISTRIBUTION_SUM_TOLERANCE: float = 1e-6

# defensive floor inside KL log; never silently rescues a malformed input
KL_EPSILON: float = 1e-12

REQUIRED_FEATURE_FIELDS: tuple[str, ...] = (
    "example_id",
    "prompt_text",
    "quadrant_scores",
    "bias_magnitude",
    "hidden_representation_ref",
)
REQUIRED_SCORED_FIELDS: tuple[str, ...] = (
    "example_id",
    "prompt_text",
    "candidate_policy",
    "heuristic_prior",
    "metrics",
)

TARGET_POLICY_SOURCE_TAG: str = "offline_forced_policy_search_v1"

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO,
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# === DATACLASS ===

@dataclass
class TargetBuildConfig:
    """
    Effective build-time configuration after merging YAML defaults with CLI
    overrides. score_temperature and min_probability are pulled from the
    yaml-loaded RouterCalibrationConfig; paths are CLI-overridable.
    """
    score_temperature: float
    min_probability:   float
    features_path:     Path
    candidate_scores_path: Path
    records_path:      Path
    target_report_path: Path


# === HELPERS — IO ===

def load_jsonl(path: Path) -> list[dict]:
    """
    Read a JSONL file. Raises FileNotFoundError if missing, ValueError if
    empty or any line is not valid JSON.
    """
    if not path.is_file():
        raise FileNotFoundError(f"file not found: {path}")
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"file is empty: {path}")
    out: list[dict] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON at {path}:{lineno}") from exc
        out.append(record)
    return out


# === HELPERS — VALIDATION ===

def _is_finite_number(v: Any) -> bool:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return False
    return math.isfinite(float(v))


def validate_policy(policy: Any, field_name: str) -> dict[str, float]:
    """
    Validate a policy as a strictly-positive distribution over canonical
    quadrants summing to 1 within DISTRIBUTION_SUM_TOLERANCE. Returns a
    fresh dict in canonical key order.
    """
    if not isinstance(policy, dict):
        raise ValueError(
            f"{field_name}: must be a dict, got {type(policy).__name__}"
        )
    expected = set(CANONICAL_QUADRANT_ORDER)
    actual = set(policy.keys())
    if actual != expected:
        missing = sorted(expected - actual)
        extra   = sorted(actual - expected)
        raise ValueError(
            f"{field_name}: keys must equal canonical "
            f"{list(CANONICAL_QUADRANT_ORDER)}; missing={missing} extra={extra}"
        )
    out: dict[str, float] = {}
    total = 0.0
    for key in CANONICAL_QUADRANT_ORDER:
        v = policy[key]
        if not _is_finite_number(v):
            raise ValueError(f"{field_name}[{key!r}] = {v!r} is not finite")
        v = float(v)
        if v <= 0:
            raise ValueError(f"{field_name}[{key!r}] = {v} must be > 0")
        out[key] = v
        total += v
    if abs(total - 1.0) > DISTRIBUTION_SUM_TOLERANCE:
        raise ValueError(
            f"{field_name}: sum = {total} (must equal 1 within "
            f"{DISTRIBUTION_SUM_TOLERANCE})"
        )
    return out


def _validate_feature_row(row: Any, where: str) -> dict:
    if not isinstance(row, dict):
        raise ValueError(f"{where}: feature row must be a dict")
    missing = [k for k in REQUIRED_FEATURE_FIELDS if k not in row]
    if missing:
        raise ValueError(
            f"{where}: feature row missing required fields: {missing}"
        )
    eid = row["example_id"]
    if not isinstance(eid, str) or not eid.strip():
        raise ValueError(f"{where}: feature.example_id must be a non-empty string")
    prompt = row["prompt_text"]
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError(f"{where}: feature.prompt_text must be a non-empty string")
    if not _is_finite_number(row["bias_magnitude"]):
        raise ValueError(f"{where}: feature.bias_magnitude must be a finite number")
    href = row["hidden_representation_ref"]
    if not isinstance(href, str) or not href:
        raise ValueError(f"{where}: feature.hidden_representation_ref must be a non-empty string")

    qs = row["quadrant_scores"]
    if not isinstance(qs, dict) or set(qs.keys()) != set(CANONICAL_QUADRANT_ORDER):
        raise ValueError(
            f"{where}: feature.quadrant_scores keys must equal canonical "
            f"{list(CANONICAL_QUADRANT_ORDER)}"
        )
    for key in CANONICAL_QUADRANT_ORDER:
        if not _is_finite_number(qs[key]):
            raise ValueError(
                f"{where}: feature.quadrant_scores[{key!r}] must be finite"
            )
    return row


def _validate_scored_row(row: Any, where: str) -> dict:
    if not isinstance(row, dict):
        raise ValueError(f"{where}: scored row must be a dict")
    missing = [k for k in REQUIRED_SCORED_FIELDS if k not in row]
    if missing:
        raise ValueError(
            f"{where}: scored row missing required fields: {missing}"
        )
    metrics = row["metrics"]
    if not isinstance(metrics, dict) or "final_candidate_score" not in metrics:
        raise ValueError(
            f"{where}: scored.metrics must include 'final_candidate_score'"
        )
    score = metrics["final_candidate_score"]
    if not _is_finite_number(score):
        raise ValueError(
            f"{where}: final_candidate_score = {score!r} is not finite"
        )
    # validate candidate / heuristic policies (raises if malformed)
    validate_policy(row["candidate_policy"], f"{where}.candidate_policy")
    validate_policy(row["heuristic_prior"],  f"{where}.heuristic_prior")
    return row


# === HELPERS — MATH ===

def stable_softmax(values: list[float], temperature: float) -> list[float]:
    """
    Numerically-stable softmax over a list of logits scaled by 1/temperature.
    Subtracts the max scaled value before exponentiation so very large logits
    don't overflow.
    """
    if not isinstance(values, list) or len(values) == 0:
        raise ValueError("stable_softmax: values must be a non-empty list")
    if not _is_finite_number(temperature) or temperature <= 0:
        raise ValueError(
            f"stable_softmax: temperature must be a finite positive number, "
            f"got {temperature!r}"
        )

    scaled: list[float] = []
    for i, v in enumerate(values):
        if not _is_finite_number(v):
            raise ValueError(f"stable_softmax: values[{i}] = {v!r} not finite")
        scaled.append(float(v) / float(temperature))

    max_scaled = max(scaled)
    exps = [math.exp(s - max_scaled) for s in scaled]
    z = sum(exps)
    if z <= 0 or not math.isfinite(z):
        raise ValueError("stable_softmax: sum of exponentials is non-positive")
    return [e / z for e in exps]


def mix_policies(
    policies: list[dict[str, float]],
    weights: list[float],
) -> dict[str, float]:
    """
    Convex combination of policies under the supplied weights. Output keys
    are written in CANONICAL_QUADRANT_ORDER. Weights must be finite and >= 0.
    Caller is responsible for ensuring weights sum to (approximately) 1.
    """
    if not isinstance(policies, list) or not policies:
        raise ValueError("mix_policies: policies must be a non-empty list")
    if not isinstance(weights, list) or len(weights) != len(policies):
        raise ValueError(
            f"mix_policies: weights length {len(weights) if isinstance(weights, list) else '?'} "
            f"!= policies length {len(policies)}"
        )

    out = {key: 0.0 for key in CANONICAL_QUADRANT_ORDER}
    for j, (policy, w) in enumerate(zip(policies, weights)):
        if not _is_finite_number(w) or w < 0:
            raise ValueError(
                f"mix_policies: weights[{j}] = {w!r} must be a finite >= 0"
            )
        if not isinstance(policy, dict) or set(policy.keys()) != set(CANONICAL_QUADRANT_ORDER):
            raise ValueError(
                f"mix_policies: policies[{j}] must use canonical keys"
            )
        for key in CANONICAL_QUADRANT_ORDER:
            v = policy[key]
            if not _is_finite_number(v):
                raise ValueError(
                    f"mix_policies: policies[{j}][{key!r}] = {v!r} not finite"
                )
            out[key] += float(w) * float(v)
    return out


def entropy(policy: dict[str, float]) -> float:
    """Shannon entropy in nats. Input must be a valid distribution."""
    p = validate_policy(policy, "entropy.policy")
    return -sum(p[k] * math.log(p[k]) for k in CANONICAL_QUADRANT_ORDER)


def kl_policy(p: dict[str, float], q: dict[str, float]) -> float:
    """
    KL(p || q) over CANONICAL_QUADRANT_ORDER. Both must be valid
    distributions; KL_EPSILON is a defensive floor only.
    """
    p_v = validate_policy(p, "kl_policy.p")
    q_v = validate_policy(q, "kl_policy.q")
    total = 0.0
    for key in CANONICAL_QUADRANT_ORDER:
        pi = max(p_v[key], KL_EPSILON)
        qi = max(q_v[key], KL_EPSILON)
        total += pi * math.log(pi / qi)
    if -1e-9 < total < 0:
        total = 0.0
    return total


# === CONFIG RESOLUTION ===

def build_target_config(
    cfg: RouterCalibrationConfig,
    *,
    features_path: Path | None = None,
    candidate_scores_path: Path | None = None,
    records_path: Path | None = None,
    target_report_path: Path | None = None,
) -> TargetBuildConfig:
    """
    Resolve effective build-time config. CLI-supplied paths win over yaml
    defaults; missing yaml fields (candidate_scores_path, target_report_path)
    fall back to sensible siblings of paths.output_dir.
    """
    score_temperature = float(cfg.scoring.score_temperature)
    if score_temperature <= 0:
        raise ValueError(
            f"scoring.score_temperature must be > 0, got {score_temperature}"
        )
    min_probability = float(cfg.candidate_policies.min_probability)
    if not (0 < min_probability < 0.25):
        raise ValueError(
            f"candidate_policies.min_probability must be in (0, 0.25), "
            f"got {min_probability}"
        )

    fp = features_path if features_path is not None else cfg.paths.features_path

    csp_default = getattr(cfg.paths, "candidate_scores_path", None)
    if csp_default is None:
        csp_default = cfg.paths.output_dir / DEFAULT_CANDIDATE_SCORES_FILENAME
    csp = candidate_scores_path if candidate_scores_path is not None else csp_default

    rp = records_path if records_path is not None else cfg.paths.records_path

    trp_default = getattr(cfg.paths, "target_report_path", None)
    if trp_default is None:
        trp_default = cfg.paths.output_dir / DEFAULT_TARGET_REPORT_FILENAME
    trp = target_report_path if target_report_path is not None else trp_default

    return TargetBuildConfig(
        score_temperature=score_temperature,
        min_probability=min_probability,
        features_path=fp,
        candidate_scores_path=csp,
        records_path=rp,
        target_report_path=trp,
    )


# === BUILD: per-example ===

def _check_priors_match(
    scored_rows: list[dict],
    example_id: str,
) -> dict[str, float]:
    """
    Validate every scored row's heuristic_prior agrees with the first one
    within HEURISTIC_PRIOR_MISMATCH_TOLERANCE; raise ValueError on drift.
    Returns the canonical prior (taken from the first row).
    """
    priors = [
        validate_policy(
            row["heuristic_prior"],
            f"example_id={example_id} scored row[{i}].heuristic_prior",
        )
        for i, row in enumerate(scored_rows)
    ]
    ref = priors[0]
    for i, p in enumerate(priors[1:], start=1):
        for key in CANONICAL_QUADRANT_ORDER:
            if abs(p[key] - ref[key]) > HEURISTIC_PRIOR_MISMATCH_TOLERANCE:
                raise ValueError(
                    f"example_id={example_id}: heuristic_prior mismatch at "
                    f"key {key!r} between scored rows 0 and {i}: "
                    f"{ref[key]} vs {p[key]} "
                    f"(tolerance {HEURISTIC_PRIOR_MISMATCH_TOLERANCE})"
                )
    return ref


def build_target_for_example(
    feature: dict,
    scored_rows: list[dict],
    cfg: TargetBuildConfig,
) -> tuple[dict, dict]:
    """
    Build a single records.jsonl row for one example_id.

    Returns:
        (record, diagnostics). record matches the train_calibrated_router
        schema. diagnostics carries the per-row stats used for the report's
        summary block.

    Raises:
        ValueError on any malformed feature/scored input or a heuristic
        prior mismatch between scored rows for the same example.
    """
    if not isinstance(scored_rows, list) or not scored_rows:
        raise ValueError(
            f"example_id={feature.get('example_id')!r}: scored_rows must be a "
            "non-empty list"
        )
    example_id = feature["example_id"]

    # validate every scored row loudly — malformed candidate rows raise (spec v1)
    for i, row in enumerate(scored_rows):
        _validate_scored_row(row, f"example_id={example_id} row[{i}]")
        if row["example_id"] != example_id:
            raise ValueError(
                f"example_id={example_id}: scored row[{i}].example_id "
                f"= {row['example_id']!r} does not match"
            )
        if row["prompt_text"] != feature["prompt_text"]:
            raise ValueError(
                f"example_id={example_id}: scored row[{i}] prompt_text "
                "does not match feature.prompt_text"
            )

    heuristic_prior = _check_priors_match(scored_rows, example_id)

    scores: list[float] = [
        float(row["metrics"]["final_candidate_score"]) for row in scored_rows
    ]
    weights = stable_softmax(scores, cfg.score_temperature)

    policies: list[dict[str, float]] = [
        validate_policy(
            row["candidate_policy"],
            f"example_id={example_id} row[{i}].candidate_policy",
        )
        for i, row in enumerate(scored_rows)
    ]

    mixed = mix_policies(policies, weights)
    target_policy = apply_min_probability(mixed, cfg.min_probability)

    best_idx = max(range(len(scores)), key=lambda i: scores[i])
    best_score = scores[best_idx]
    best_policy = {
        key: float(policies[best_idx][key]) for key in CANONICAL_QUADRANT_ORDER
    }

    target_entropy = entropy(target_policy)
    kl_to_heuristic = kl_policy(target_policy, heuristic_prior)

    feature_metadata = feature.get("metadata", {})
    if not isinstance(feature_metadata, dict):
        feature_metadata = {}

    record_metadata: dict[str, Any] = {
        **copy.deepcopy(feature_metadata),
        "target_policy_source": TARGET_POLICY_SOURCE_TAG,
        "num_candidates":         len(scored_rows),
        "score_temperature":      cfg.score_temperature,
        "min_probability":        cfg.min_probability,
        "best_candidate_score":   best_score,
        "best_candidate_policy":  best_policy,
        "target_entropy":         target_entropy,
        "kl_target_to_heuristic": kl_to_heuristic,
    }

    record: dict[str, Any] = {
        "example_id":  example_id,
        "prompt_text": feature["prompt_text"],
        "quadrant_scores": {
            key: float(feature["quadrant_scores"][key])
            for key in CANONICAL_QUADRANT_ORDER
        },
        "bias_magnitude":  float(feature["bias_magnitude"]),
        "target_policy":   copy.deepcopy(target_policy),
        "hidden_representation_ref": feature["hidden_representation_ref"],
        "metadata":        record_metadata,
    }

    diagnostics = {
        "example_id":            example_id,
        "num_candidates":        len(scored_rows),
        "best_score":            best_score,
        "target_entropy":        target_entropy,
        "kl_target_to_heuristic": kl_to_heuristic,
    }
    return record, diagnostics


# === BUILD: full corpus ===

def build_all_targets(
    features: list[dict],
    scored_rows: list[dict],
    cfg: TargetBuildConfig,
    *,
    limit: int | None = None,
) -> tuple[list[dict], dict]:
    """
    Group scored_rows by example_id, build one record per feature row that
    has at least one matching scored candidate. Records preserve feature
    row order (deterministic). Mismatched-prompt and missing-pairing
    examples are skipped and reported; malformed rows raise (spec v1).
    """
    # validate features once and detect duplicate ids early
    feature_ids: set[str] = set()
    for i, f in enumerate(features):
        _validate_feature_row(f, f"features[{i}]")
        eid = f["example_id"]
        if eid in feature_ids:
            raise ValueError(f"duplicate feature example_id: {eid!r}")
        feature_ids.add(eid)

    # group scored rows by example_id (preserve list order within group)
    scored_by_id: dict[str, list[dict]] = {}
    for i, row in enumerate(scored_rows):
        if not isinstance(row, dict) or "example_id" not in row:
            raise ValueError(
                f"scored_rows[{i}] missing example_id or not a dict"
            )
        scored_by_id.setdefault(row["example_id"], []).append(row)

    selected = features if limit is None else features[: int(limit)]

    records: list[dict] = []
    skipped: list[dict] = []
    diagnostics_list: list[dict] = []

    for feature in selected:
        eid = feature["example_id"]
        rows_for_eid = scored_by_id.get(eid)
        if not rows_for_eid:
            skipped.append({
                "example_id": eid,
                "reason":     "no_scored_candidates",
            })
            log.info("skip example_id=%s — no scored candidates", eid)
            continue

        # prompt_text drift between feature and scored rows is a pairing
        # issue (skip + report), not a malformed-row issue. malformed
        # rows are caught later in build_target_for_example with a raise.
        mismatched = [
            i for i, r in enumerate(rows_for_eid)
            if isinstance(r, dict) and r.get("prompt_text") != feature["prompt_text"]
        ]
        if mismatched:
            skipped.append({
                "example_id":          eid,
                "reason":              "prompt_text_mismatch",
                "mismatched_indices":  mismatched,
            })
            log.info(
                "skip example_id=%s — prompt_text mismatch in scored rows %s",
                eid, mismatched,
            )
            continue

        record, diagnostics = build_target_for_example(feature, rows_for_eid, cfg)
        records.append(record)
        diagnostics_list.append(diagnostics)

    # report scored rows whose example_id matches no feature at all
    for sid in scored_by_id:
        if sid not in feature_ids:
            skipped.append({
                "example_id": sid,
                "reason":     "no_feature_row",
            })

    if diagnostics_list:
        n = len(diagnostics_list)
        mean_num     = sum(d["num_candidates"]        for d in diagnostics_list) / n
        mean_best    = sum(d["best_score"]            for d in diagnostics_list) / n
        mean_entropy = sum(d["target_entropy"]        for d in diagnostics_list) / n
        mean_kl      = sum(d["kl_target_to_heuristic"] for d in diagnostics_list) / n
    else:
        mean_num = mean_best = mean_entropy = mean_kl = None

    report: dict[str, Any] = {
        "num_feature_rows":    len(features),
        "num_scored_rows":     len(scored_rows),
        "num_records_written": len(records),
        "skipped_examples":    skipped,
        "score_temperature":   cfg.score_temperature,
        "min_probability":     cfg.min_probability,
        "summary": {
            "mean_num_candidates_per_record": mean_num,
            "mean_best_score":                mean_best,
            "mean_target_entropy":            mean_entropy,
            "mean_kl_target_to_heuristic":    mean_kl,
        },
    }
    return records, report


# === IO ===

def write_records_jsonl(records: list[dict], path: Path) -> None:
    """Write records to JSONL via .tmp + atomic rename. Parents auto-created."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    tmp.replace(path)


def write_report_json(report: dict, path: Path) -> None:
    """Write the report dict via .tmp + atomic rename. Parents auto-created."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    tmp.replace(path)


# === CLI ===

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "build router-calibration target policies from features.jsonl + "
            "candidate_scores.jsonl (write records.jsonl + target_report.json)"
        ),
    )
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--features-path",         type=Path, default=None)
    p.add_argument("--candidate-scores-path", type=Path, default=None)
    p.add_argument("--records-path",          type=Path, default=None)
    p.add_argument("--report-path",           type=Path, default=None)
    p.add_argument("--limit",                 type=int,  default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_router_calibration_config(args.config)
    log.info("config loaded from %s", args.config)

    target_cfg = build_target_config(
        cfg,
        features_path=args.features_path,
        candidate_scores_path=args.candidate_scores_path,
        records_path=args.records_path,
        target_report_path=args.report_path,
    )

    log.info("loading features from %s", target_cfg.features_path)
    features = load_jsonl(target_cfg.features_path)
    log.info("loaded %d feature rows", len(features))

    log.info("loading scored candidates from %s", target_cfg.candidate_scores_path)
    scored = load_jsonl(target_cfg.candidate_scores_path)
    log.info("loaded %d scored rows", len(scored))

    records, report = build_all_targets(
        features, scored, target_cfg, limit=args.limit,
    )
    log.info(
        "built %d records (%d skipped)",
        len(records), len(report["skipped_examples"]),
    )

    write_records_jsonl(records, target_cfg.records_path)
    log.info("wrote records → %s", target_cfg.records_path)
    write_report_json(report, target_cfg.target_report_path)
    log.info("wrote report  → %s", target_cfg.target_report_path)


if __name__ == "__main__":
    main()
