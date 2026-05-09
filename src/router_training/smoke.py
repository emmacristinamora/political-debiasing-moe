# src/run_router_pipeline_smoke.py


# === IMPORTS ===

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import sys
from pathlib import Path
from typing import Any


# every module imported below is torch-free.
_SRC_DIR = Path(__file__).resolve().parents[1]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from router_training.config import CANONICAL_QUADRANT_ORDER  # noqa: E402
from router_training.utils import (  # noqa: E402
    CandidatePolicyConfig,
    generate_candidate_policies,
)
from router_training.targets import (  # noqa: E402
    TargetBuildConfig,
    build_target_for_example,
)
from router_training.validator import validate_router_dataset  # noqa: E402
from router_training.splitter import (  # noqa: E402
    SplitBuildConfig,
    split_records,
)
from router_training.train_pipeline import build_training_command  # noqa: E402


# === CONSTANTS ===

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_PATH: Path = (
    PROJECT_ROOT / "data" / "router" / "smoke_pipeline_report.json"
)

DEFAULT_NUM_PROMPTS: int = 3
DEFAULT_HIDDEN_DIM: int = 8
DEFAULT_SEED: int = 42
DEFAULT_HIDDEN_FILENAME: str = "hidden.pt"

# heuristic-prior hparams used for synthetic supervision; chosen to be
# representative of the trainer's defaults so command construction below
# produces a realistic-looking argv.
SMOKE_BETA: float = 1.0
SMOKE_TEMPERATURE: float = 1.0
SMOKE_SCORE_TEMPERATURE: float = 1.0
SMOKE_MIN_PROBABILITY: float = 0.05

# split fractions chosen so n=3 splits cleanly into (1, 1, 1) via
# split_router_dataset.allocate_counts (every bucket gets at least 1 for n >= 3).
SMOKE_TRAIN_FRACTION: float = 0.34
SMOKE_VAL_FRACTION:   float = 0.33
SMOKE_TEST_FRACTION:  float = 0.33


logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO,
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# === SYNTHETIC DATA HELPERS ===

def _fake_prompts(num: int = DEFAULT_NUM_PROMPTS) -> list[dict[str, Any]]:
    """Build a tiny deterministic prompt set; no model, no IO."""
    if num <= 0:
        raise ValueError(f"num must be > 0, got {num}")
    return [
        {
            "example_id": f"smoke_prompt_{i:03d}",
            "prompt_text": f"Synthetic smoke prompt #{i}.",
        }
        for i in range(num)
    ]


def _fake_quadrant_scores(idx: int) -> dict[str, float]:
    """
    Deterministic per-prompt quadrant scores. Rotating the same base list
    produces feature variety without any randomness, so two runs with the
    same prompt count yield identical quadrant scores.
    """
    base = [0.5, -0.3, 0.2, -0.4]
    rot = idx % len(base)
    rotated = base[rot:] + base[:rot]
    return {key: float(rotated[i]) for i, key in enumerate(CANONICAL_QUADRANT_ORDER)}


def _fake_features(
    prompts: list[dict[str, Any]],
    *,
    hidden_filename: str = DEFAULT_HIDDEN_FILENAME,
) -> list[dict[str, Any]]:
    """
    Build feature rows with the schema expected by build_router_targets:
    example_id, prompt_text, quadrant_scores, bias_magnitude, hidden_representation_ref.
    """
    out: list[dict[str, Any]] = []
    for i, prompt in enumerate(prompts):
        scores = _fake_quadrant_scores(i)
        bias_mag = math.sqrt(sum(v * v for v in scores.values()))
        out.append({
            "example_id": prompt["example_id"],
            "prompt_text": prompt["prompt_text"],
            "quadrant_scores": scores,
            "bias_magnitude": bias_mag,
            "hidden_representation_ref": f"{hidden_filename}:{i}",
            "metadata": {"source": "smoke", "axis": "synthetic"},
        })
    return out


def _heuristic_prior_from_scores(
    quadrant_scores: dict[str, float],
    *,
    beta: float = SMOKE_BETA,
    temperature: float = SMOKE_TEMPERATURE,
) -> dict[str, float]:
    """Mirror the trainer: softmax(-beta * q / T) over CANONICAL_QUADRANT_ORDER."""
    if temperature == 0:
        raise ValueError("temperature must be non-zero")
    logits = [
        -beta * float(quadrant_scores[k]) / temperature for k in CANONICAL_QUADRANT_ORDER
    ]
    m = max(logits)
    exps = [math.exp(x - m) for x in logits]
    z = sum(exps)
    return {k: e / z for k, e in zip(CANONICAL_QUADRANT_ORDER, exps)}


def _fake_candidate_scores(
    feature: dict[str, Any],
    candidates: list[dict[str, float]],
    heuristic_prior: dict[str, float],
) -> list[dict[str, Any]]:
    """
    Synthesize the scored candidate-rows the real scorer would have produced.
    We never call the real projector or judge — score is derived purely from
    the candidate's L1 distance to the heuristic prior so the output is
    deterministic given the candidate list.
    """
    rows: list[dict[str, Any]] = []
    for c in candidates:
        l1 = sum(abs(c[k] - heuristic_prior[k]) for k in CANONICAL_QUADRANT_ORDER)
        # bounded score; small bias toward "moderately different" candidates
        score = float(1.0 - 0.5 * l1 + 0.1 * (1.0 - l1) * l1)
        rows.append({
            "example_id":      feature["example_id"],
            "prompt_text":     feature["prompt_text"],
            "candidate_policy": dict(c),
            "heuristic_prior":  dict(heuristic_prior),
            "metrics": {"final_candidate_score": score},
        })
    return rows


# === STAGE CONFIGS ===

def _candidate_cfg() -> CandidatePolicyConfig:
    """Synthetic candidate-generator config; deterministic given the rng seed."""
    return CandidatePolicyConfig(
        num_dirichlet_samples=2,
        dirichlet_alpha=1.0,
        include_uniform=True,
        include_sharpened=True,
        include_softened=True,
        include_opposite=True,
        include_adjacent=True,
        min_probability=SMOKE_MIN_PROBABILITY,
    )


def _target_cfg(tmp_dir: Path) -> TargetBuildConfig:
    """
    Target-build config with placeholder paths — the smoke runner builds
    targets in-memory, so the *_path fields are never written through.
    """
    return TargetBuildConfig(
        score_temperature=SMOKE_SCORE_TEMPERATURE,
        min_probability=SMOKE_MIN_PROBABILITY,
        features_path=tmp_dir / "features.jsonl",
        candidate_scores_path=tmp_dir / "candidate_scores.jsonl",
        records_path=tmp_dir / "records.jsonl",
        target_report_path=tmp_dir / "target_report.json",
    )


def _split_cfg(seed: int, tmp_dir: Path) -> SplitBuildConfig:
    """SplitBuildConfig with synthetic paths; in-memory split_records does not write."""
    return SplitBuildConfig(
        train_fraction=SMOKE_TRAIN_FRACTION,
        val_fraction=SMOKE_VAL_FRACTION,
        test_fraction=SMOKE_TEST_FRACTION,
        seed=int(seed),
        stratify_by="none",
        records_path=tmp_dir / "records.jsonl",
        output_dir=tmp_dir / "splits",
        report_path=tmp_dir / "split_report.json",
    )


def _smoke_training_paths(tmp_dir: Path) -> dict[str, Path]:
    """Synthetic paths fed to build_training_command; no files are touched."""
    return {
        "train_records_path": tmp_dir / "train" / "records.jsonl",
        "hidden_path":        tmp_dir / DEFAULT_HIDDEN_FILENAME,
        "router_checkpoint":  tmp_dir / "checkpoints" / "calibrated_router.pt",
        "trainer_report":     tmp_dir / "reports" / "train_report.json",
    }


def _smoke_training_hparams(hidden_dim: int, seed: int) -> dict[str, Any]:
    """Hyperparameters echoed into the synthetic training command."""
    return {
        "calibration_input_dim": int(hidden_dim),
        "beta":             SMOKE_BETA,
        "temperature":      SMOKE_TEMPERATURE,
        "learning_rate":    1e-3,
        "weight_decay":     1e-4,
        "batch_size":       4,
        "epochs":           1,
        "kl_weight":        0.1,
        "entropy_weight":   0.01,
        "seed":             int(seed),
        "device":           "cpu",
        "save_every_epoch": False,
    }


# === RUNNER ===

def run_smoke(
    *,
    num_prompts: int = DEFAULT_NUM_PROMPTS,
    hidden_dim: int = DEFAULT_HIDDEN_DIM,
    seed: int = DEFAULT_SEED,
    output_path: Path | None = None,
) -> dict[str, Any]:
    """
    Run the full router-calibration pipeline against synthetic data without
    touching any real model, the InputTransformer, or torch. Stages exercised:
        prompts -> features -> candidates -> scores -> targets ->
        records -> validation -> split -> training-command.
    Returns:
        A JSON-safe report dict. If output_path is provided, the same report
        is also persisted as JSON (parent dirs created as needed).
    """
    if num_prompts <= 0:
        raise ValueError(f"num_prompts must be > 0, got {num_prompts}")
    if hidden_dim <= 0:
        raise ValueError(f"hidden_dim must be > 0, got {hidden_dim}")

    prompts = _fake_prompts(num_prompts)
    features = _fake_features(prompts, hidden_filename=DEFAULT_HIDDEN_FILENAME)

    candidate_cfg = _candidate_cfg()
    rng = random.Random(int(seed))

    candidate_scores: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    target_diagnostics: list[dict[str, Any]] = []

    # in-memory dir reference used only to populate the *_path fields of the
    # TargetBuildConfig / SplitBuildConfig dataclasses; no IO goes through it.
    synthetic_dir = PROJECT_ROOT / "data" / "router" / "_smoke"
    target_cfg = _target_cfg(synthetic_dir)

    for feature in features:
        prior = _heuristic_prior_from_scores(feature["quadrant_scores"])
        candidates = generate_candidate_policies(
            quadrant_scores=feature["quadrant_scores"],
            heuristic_prior=prior,
            config=candidate_cfg,
            rng=rng,
        )
        scored_rows = _fake_candidate_scores(feature, candidates, prior)
        candidate_scores.extend(scored_rows)
        record, diagnostics = build_target_for_example(feature, scored_rows, target_cfg)
        records.append(record)
        target_diagnostics.append(diagnostics)

    # validate the in-memory dataset against the same rules the trainer applies.
    # passing hidden_tensor=None skips tensor-internals checks but still runs
    # every record-level check + the hidden_filename match.
    validate_router_dataset(
        records,
        None,
        expected_hidden_dim=hidden_dim,
        hidden_filename=DEFAULT_HIDDEN_FILENAME,
    )

    split_cfg = _split_cfg(seed=seed, tmp_dir=synthetic_dir)
    splits, split_report = split_records(records, split_cfg)

    paths = _smoke_training_paths(synthetic_dir)
    hparams = _smoke_training_hparams(hidden_dim=hidden_dim, seed=seed)
    command = build_training_command(paths, hparams, max_examples=num_prompts)

    report: dict[str, Any] = {
        "num_prompts": len(prompts),
        "num_records": len(records),
        "splits": {name: len(splits[name]) for name in ("train", "val", "test")},
        "command": list(command),
        "prompts": prompts,
        "features": features,
        "candidate_scores": candidate_scores,
        "records": records,
        "target_diagnostics": target_diagnostics,
        "split_report": split_report,
        "warnings": list(split_report.get("warnings", [])),
        "config": {
            "num_prompts": num_prompts,
            "hidden_dim": hidden_dim,
            "seed": seed,
            "beta": SMOKE_BETA,
            "temperature": SMOKE_TEMPERATURE,
            "score_temperature": SMOKE_SCORE_TEMPERATURE,
            "min_probability": SMOKE_MIN_PROBABILITY,
            "fractions": {
                "train": SMOKE_TRAIN_FRACTION,
                "val":   SMOKE_VAL_FRACTION,
                "test":  SMOKE_TEST_FRACTION,
            },
            "stratify_by": "none",
        },
    }

    if output_path is not None:
        write_report(report, output_path)

    return report


def write_report(report: dict[str, Any], path: Path) -> None:
    """Persist the smoke report as JSON, creating parent directories on demand."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2, sort_keys=True)


# === CLI ===

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "End-to-end smoke runner for the router-calibration pipeline. "
            "Exercises every torch-free stage on synthetic data and writes a "
            "JSON report. No models, no transformers, no MoCE."
        ),
    )
    p.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    p.add_argument("--num-prompts", type=int, default=DEFAULT_NUM_PROMPTS)
    p.add_argument("--hidden-dim",  type=int, default=DEFAULT_HIDDEN_DIM)
    p.add_argument("--seed",        type=int, default=DEFAULT_SEED)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    report = run_smoke(
        num_prompts=args.num_prompts,
        hidden_dim=args.hidden_dim,
        seed=args.seed,
        output_path=args.output_path,
    )
    log.info(
        "smoke pipeline ok — prompts=%d records=%d splits=%s output=%s",
        report["num_prompts"], report["num_records"], report["splits"],
        args.output_path,
    )


if __name__ == "__main__":
    main()
