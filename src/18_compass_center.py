# src/18_compass_center.py

# Computes the calibrated political-compass center from a tagged dataset of
# politically neutral prompts, then validates it against four acceptance
# criteria (R6).  The output center.json is consumed by src/19_repartition_chunks.py
# and config.yaml → quadrant_dataset.compass_center_path.
#
#   input   data/neutral_prompts.jsonl
#               schema: {id, category, subtype, topic, text}
#               category ∈ {apolitical, generic_task}
#           data/steering-vectors/validated_pairs/{economic,social}_pairs_validated.jsonl
#               used for the midpoint test (R6.2)
#
#   output  data/compass_center/center.json          centroid + per-prompt scores
#           data/compass_center/validation_report.json   R6 check results
#
# === R6 acceptance criteria ===
#   R6.1  subcategory agreement  — apolitical centroid ≈ generic_task centroid
#   R6.2  midpoint test          — centroid sits midway between PCT +/- polarity clouds
#   R6.3  no outliers            — no prompt beyond OUTLIER_Z std on either axis
#   R6.4  bootstrap stability    — centroid SE is small relative to charged spread


# === IMPORTS ===

from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# === CONFIG ===

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_DATASET      = PROJECT_ROOT / "data" / "neutral_prompts.jsonl"
DEFAULT_VECTORS_DIR  = PROJECT_ROOT / "data" / "steering-vectors" / "vectors"
DEFAULT_PAIRS_DIR    = PROJECT_ROOT / "data" / "steering-vectors" / "validated_pairs"
DEFAULT_OUTPUT       = PROJECT_ROOT / "data" / "compass_center" / "center.json"
DEFAULT_REPORT       = PROJECT_ROOT / "data" / "compass_center" / "validation_report.json"

DEFAULT_MODEL    = "mistralai/Mistral-7B-v0.1"
AXES             = ("economic", "social")
LAYER            = 20
VALID_DTYPES     = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
VALID_METHODS    = ("mean_difference", "logistic_regression")
VALID_CATEGORIES = {"apolitical", "generic_task"}

# R6 default tolerances
DEFAULT_OUTLIER_Z        = 2.5    # std units; prompts beyond this are flagged
DEFAULT_SUBCATEGORY_TOL  = 0.5    # max |centroid_A − centroid_B| in units of neutral std
DEFAULT_MIDPOINT_TOL     = 0.15   # max |center − midpoint| / charged_spread
DEFAULT_SE_TOL           = 0.05   # max bootstrap SE / charged_spread
DEFAULT_N_BOOTSTRAP      = 1000


# === HELPERS: IO ===

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute the calibrated compass center and run R6 validation checks."
    )
    parser.add_argument("--dataset",       type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--vectors-dir",   type=Path, default=DEFAULT_VECTORS_DIR)
    parser.add_argument("--pairs-dir",     type=Path, default=DEFAULT_PAIRS_DIR)
    parser.add_argument("--output",        type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report",        type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--model-name",    default=DEFAULT_MODEL)
    parser.add_argument("--dtype",         choices=sorted(VALID_DTYPES), default="float16")
    parser.add_argument("--device",        default="cuda")
    parser.add_argument("--method",        choices=VALID_METHODS, default="mean_difference")
    parser.add_argument("--batch-size",    type=int, default=16)
    parser.add_argument("--max-tokens",    type=int, default=256)
    parser.add_argument("--limit",         type=int, default=None,
                        help="Cap the number of neutral prompts (debugging).")
    parser.add_argument("--outlier-z",     type=float, default=DEFAULT_OUTLIER_Z)
    parser.add_argument("--subcategory-tol", type=float, default=DEFAULT_SUBCATEGORY_TOL)
    parser.add_argument("--midpoint-tol",  type=float, default=DEFAULT_MIDPOINT_TOL)
    parser.add_argument("--se-tol",        type=float, default=DEFAULT_SE_TOL)
    parser.add_argument("--n-bootstrap",   type=int, default=DEFAULT_N_BOOTSTRAP)
    parser.add_argument("--seed",          type=int, default=42)
    parser.add_argument("--keep-outliers", action="store_true",
                        help="Report outliers but do not remove them from the centroid.")
    return parser.parse_args()


def load_prompts(path: Path) -> list[dict[str, Any]]:
    """Load and validate the neutral-prompt dataset (R5 schema)."""
    if not path.is_file():
        raise FileNotFoundError(f"dataset not found: {path}")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"{path}: no rows")
    required = {"id", "category", "subtype", "topic", "text"}
    missing = required - set(rows[0].keys())
    if missing:
        raise ValueError(
            f"neutral_prompts.jsonl is missing fields: {sorted(missing)}. "
            f"Required schema: {{id, category, subtype, topic, text}}"
        )
    bad_cats = {r["category"] for r in rows} - VALID_CATEGORIES
    if bad_cats:
        raise ValueError(f"Unknown category values: {bad_cats}. Must be one of {VALID_CATEGORIES}")
    return rows


def load_pct_pairs(pairs_dir: Path, axis: str) -> list[dict[str, Any]]:
    """Load validated PCT contrastive pairs for one axis."""
    path = pairs_dir / f"{axis}_pairs_validated.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"validated pairs not found: {path}")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def save_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


# === HELPERS: MODEL & VECTORS ===

def load_model_and_tokenizer(model_name: str, dtype: torch.dtype, device: str) -> tuple[Any, Any]:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
    model = model.to(device)
    model.eval()
    return model, tokenizer


def load_final_vector(vectors_dir: Path, axis: str, method: str, device: str) -> torch.Tensor:
    """Load the final quality-weighted aggregate steering direction for one axis."""
    path = vectors_dir / f"{axis}_vectors.pt"
    if not path.is_file():
        raise FileNotFoundError(f"vectors file not found: {path}")
    data = torch.load(path, map_location="cpu")
    vector = data["final_vectors"][method].to(torch.float32)
    unit = vector / (vector.norm() + 1e-12)
    return unit.to(device)


# === HELPERS: PROJECTION ===

def mean_pool(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Mean-pool token representations using the attention mask (mirrors step 03)."""
    mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
    token_counts = mask.sum(dim=1).clamp(min=1.0)
    return (hidden_states * mask).sum(dim=1) / token_counts


def project_texts(
    texts: list[str],
    model: Any,
    tokenizer: Any,
    final_vectors: dict[str, torch.Tensor],
    max_tokens: int,
    batch_size: int,
    device: str,
    label: str = "",
) -> dict[str, list[float]]:
    """Project a list of texts onto both axes; returns per-axis score lists."""
    all_scores: dict[str, list[float]] = {axis: [] for axis in AXES}
    total = len(texts)

    for start in range(0, total, batch_size):
        batch = texts[start : start + batch_size]
        encoding = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_tokens,
            add_special_tokens=True,
        )
        input_ids      = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )

        # hidden_states[LAYER + 1] is the output of transformer block LAYER.
        layer_out = outputs.hidden_states[LAYER + 1].float()
        pooled    = mean_pool(layer_out, attention_mask)     # [batch, hidden_dim]

        for axis, unit_vec in final_vectors.items():
            all_scores[axis].extend((pooled @ unit_vec).detach().cpu().tolist())

        done = min(start + batch_size, total)
        if label and (done % 64 == 0 or done == total):
            print(f"  [{label}] {done}/{total}")

    return all_scores


# === HELPERS: STATISTICS ===

def mean_of(values: list[float]) -> float:
    return sum(values) / len(values)


def std_of(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mu = mean_of(values)
    return math.sqrt(sum((v - mu) ** 2 for v in values) / (len(values) - 1))


def z_scores(values: list[float]) -> list[float]:
    mu  = mean_of(values)
    std = std_of(values)
    if std == 0.0:
        return [0.0] * len(values)
    return [(v - mu) / std for v in values]


# === R6 CHECKS ===

def check_outliers(
    rows: list[dict[str, Any]],
    scores: dict[str, list[float]],
    outlier_z: float,
) -> dict[str, Any]:
    """R6.3 — flag prompts whose |z-score| > outlier_z on either axis."""
    econ_z = z_scores(scores["economic"])
    soc_z  = z_scores(scores["social"])

    outlier_ids: list[str] = []
    for i, row in enumerate(rows):
        if abs(econ_z[i]) > outlier_z or abs(soc_z[i]) > outlier_z:
            outlier_ids.append(row["id"])

    passed = len(outlier_ids) == 0
    return {
        "passed": passed,
        "n_outliers": len(outlier_ids),
        "outlier_z_threshold": outlier_z,
        "outlier_ids": outlier_ids,
        "interpretation": f"{'no outliers detected' if passed else f'{len(outlier_ids)} prompts exceed ±{outlier_z}σ — inspect and remove'}",
    }


def check_subcategory_agreement(
    rows: list[dict[str, Any]],
    scores: dict[str, list[float]],
    tol_std_units: float,
) -> dict[str, Any]:
    """R6.1 — apolitical and generic_task centroids must agree within tolerance."""
    by_cat: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for i, row in enumerate(rows):
        cat = row["category"]
        for axis in AXES:
            by_cat[cat][axis].append(scores[axis][i])

    centroids: dict[str, dict[str, float]] = {}
    for cat, ax_scores in by_cat.items():
        centroids[cat] = {axis: mean_of(ax_scores[axis]) for axis in AXES}

    # tolerance is expressed in units of the global per-axis std
    global_std = {axis: std_of(scores[axis]) for axis in AXES}

    diffs: dict[str, float] = {}
    passed_per_axis: dict[str, bool] = {}
    cats = list(centroids.keys())
    if len(cats) >= 2:
        c0, c1 = centroids[cats[0]], centroids[cats[1]]
        for axis in AXES:
            diff = abs(c0[axis] - c1[axis])
            diffs[axis] = diff
            threshold = tol_std_units * global_std[axis]
            passed_per_axis[axis] = diff <= threshold
    else:
        # only one category present — trivially passes
        for axis in AXES:
            diffs[axis] = 0.0
            passed_per_axis[axis] = True

    passed = all(passed_per_axis.values())
    return {
        "passed": passed,
        "centroids_by_category": {
            cat: {axis: centroids[cat][axis] for axis in AXES}
            for cat in centroids
        },
        "counts_by_category": {cat: len(list(by_cat[cat].values())[0]) for cat in by_cat},
        "centroid_diff": diffs,
        "tolerance_std_units": tol_std_units,
        "global_std": global_std,
        "passed_per_axis": passed_per_axis,
        "interpretation": (
            "centroids agree — subcategories are uncontaminated"
            if passed else
            "centroids diverge — one subcategory may be politically contaminated"
        ),
    }


def check_midpoint(
    center: dict[str, float],
    pct_scores: dict[str, dict[str, list[float]]],
    midpoint_tol: float,
) -> dict[str, Any]:
    """
    R6.2 — neutral centroid must sit ≈ midway between + and − polarity clouds.

    pct_scores[axis] = {"pos": [...], "neg": [...]}
    The midpoint is (mean_pos + mean_neg) / 2.
    The relative offset is |center - midpoint| / spread where spread = mean_pos - mean_neg.
    """
    results: dict[str, Any] = {}
    passed_per_axis: dict[str, bool] = {}

    for axis in AXES:
        pos_mean = mean_of(pct_scores[axis]["pos"])
        neg_mean = mean_of(pct_scores[axis]["neg"])
        midpoint = (pos_mean + neg_mean) / 2.0
        spread   = pos_mean - neg_mean          # positive when sign convention is correct
        offset   = center[axis] - midpoint
        relative = abs(offset) / (abs(spread) + 1e-12)
        ok = relative <= midpoint_tol
        passed_per_axis[axis] = ok
        results[axis] = {
            "pos_mean": pos_mean,
            "neg_mean": neg_mean,
            "midpoint": midpoint,
            "spread": spread,
            "center": center[axis],
            "offset": offset,
            "relative_offset": relative,
            "tolerance": midpoint_tol,
            "passed": ok,
        }

    passed = all(passed_per_axis.values())
    return {
        "passed": passed,
        "per_axis": results,
        "interpretation": (
            "centroid sits near the midpoint of the polarity clouds — not contaminated"
            if passed else
            "centroid is off-center relative to the polarity clouds — dataset may be biased"
        ),
    }


def check_bootstrap_stability(
    scores: dict[str, list[float]],
    pct_scores: dict[str, dict[str, list[float]]],
    se_tol: float,
    n_bootstrap: int,
    seed: int,
) -> dict[str, Any]:
    """
    R6.4 — bootstrap the centroid; SE must be small relative to the charged spread.

    SE is measured relative to the PCT spread (mean_pos - mean_neg) on each axis.
    """
    rng = random.Random(seed)
    n = len(scores["economic"])
    bootstrap_means: dict[str, list[float]] = {axis: [] for axis in AXES}

    for _ in range(n_bootstrap):
        sample = rng.choices(range(n), k=n)
        for axis in AXES:
            bootstrap_means[axis].append(mean_of([scores[axis][i] for i in sample]))

    results: dict[str, Any] = {}
    passed_per_axis: dict[str, bool] = {}

    for axis in AXES:
        se       = std_of(bootstrap_means[axis])
        spread   = abs(mean_of(pct_scores[axis]["pos"]) - mean_of(pct_scores[axis]["neg"]))
        se_ratio = se / (spread + 1e-12)
        ok = se_ratio <= se_tol
        passed_per_axis[axis] = ok
        results[axis] = {
            "bootstrap_se": se,
            "charged_spread": spread,
            "se_ratio": se_ratio,
            "tolerance": se_tol,
            "passed": ok,
        }

    passed = all(passed_per_axis.values())
    return {
        "passed": passed,
        "n_bootstrap": n_bootstrap,
        "per_axis": results,
        "interpretation": (
            "centroid is stable — SE is small relative to the charged spread"
            if passed else
            "centroid is unstable — consider a larger or more homogeneous neutral set"
        ),
    }


# === MAIN ===

def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    # ── load data ──────────────────────────────────────────────────────────────

    rows = load_prompts(args.dataset)
    if args.limit is not None:
        rows = rows[: args.limit]
    print(f"loaded {len(rows)} prompts  "
          f"({sum(1 for r in rows if r['category']=='apolitical')} apolitical, "
          f"{sum(1 for r in rows if r['category']=='generic_task')} generic_task)")

    pct_pairs = {axis: load_pct_pairs(args.pairs_dir, axis) for axis in AXES}
    print(f"loaded PCT pairs: "
          f"{len(pct_pairs['economic'])} economic, {len(pct_pairs['social'])} social")

    # ── load model & vectors ───────────────────────────────────────────────────

    model, tokenizer = load_model_and_tokenizer(
        args.model_name, VALID_DTYPES[args.dtype], args.device
    )
    final_vectors = {
        axis: load_final_vector(args.vectors_dir, axis, args.method, args.device)
        for axis in AXES
    }

    # ── project neutral prompts ────────────────────────────────────────────────

    texts = [r["text"] for r in rows]
    n_raw = len(texts)
    print(f"projecting {n_raw} neutral prompts (layer {LAYER}, method={args.method})")
    scores = project_texts(
        texts, model, tokenizer, final_vectors, args.max_tokens, args.batch_size, args.device,
        label="neutral",
    )

    # ── R6.3 outlier detection ─────────────────────────────────────────────────

    outlier_result = check_outliers(rows, scores, args.outlier_z)
    print(f"\nR6.3 outliers: {outlier_result['n_outliers']} flagged  "
          f"({'PASS' if outlier_result['passed'] else 'FAIL — inspect outlier_ids in report'})")

    if outlier_result["n_outliers"] > 0 and not args.keep_outliers:
        outlier_set = set(outlier_result["outlier_ids"])
        keep = [i for i, r in enumerate(rows) if r["id"] not in outlier_set]
        rows   = [rows[i] for i in keep]
        scores = {axis: [scores[axis][i] for i in keep] for axis in AXES}
        print(f"  removed {outlier_result['n_outliers']} outliers → {len(rows)} prompts remain")

    # ── compute center ─────────────────────────────────────────────────────────

    center = {axis: mean_of(scores[axis]) for axis in AXES}

    # ── project PCT pairs for R6.1 / R6.2 / R6.4 ──────────────────────────────

    print(f"\nprojecting PCT contrastive pairs for validation checks...")
    pct_scores: dict[str, dict[str, list[float]]] = {}
    for axis in AXES:
        pos_texts = [p["pos"] for p in pct_pairs[axis]]
        neg_texts = [p["neg"] for p in pct_pairs[axis]]
        pos_proj  = project_texts(pos_texts, model, tokenizer, final_vectors,
                                  args.max_tokens, args.batch_size, args.device,
                                  label=f"{axis}/pos")
        neg_proj  = project_texts(neg_texts, model, tokenizer, final_vectors,
                                  args.max_tokens, args.batch_size, args.device,
                                  label=f"{axis}/neg")
        pct_scores[axis] = {
            "pos": pos_proj[axis],
            "neg": neg_proj[axis],
        }

    # ── R6.1 subcategory agreement ─────────────────────────────────────────────

    subcat_result = check_subcategory_agreement(rows, scores, args.subcategory_tol)
    print(f"R6.1 subcategory agreement: {'PASS' if subcat_result['passed'] else 'FAIL'}")
    for axis in AXES:
        print(f"  {axis}: diff={subcat_result['centroid_diff'][axis]:.4f}  "
              f"threshold={args.subcategory_tol:.1f}×std={args.subcategory_tol * subcat_result['global_std'][axis]:.4f}")

    # ── R6.2 midpoint test ─────────────────────────────────────────────────────

    midpoint_result = check_midpoint(center, pct_scores, args.midpoint_tol)
    print(f"R6.2 midpoint test: {'PASS' if midpoint_result['passed'] else 'FAIL'}")
    for axis in AXES:
        r = midpoint_result["per_axis"][axis]
        print(f"  {axis}: center={r['center']:+.4f}  midpoint={r['midpoint']:+.4f}  "
              f"spread={r['spread']:+.4f}  rel_offset={r['relative_offset']:.3f}  "
              f"({'PASS' if r['passed'] else 'FAIL'})")

    # ── R6.4 bootstrap stability ───────────────────────────────────────────────

    bootstrap_result = check_bootstrap_stability(
        scores, pct_scores, args.se_tol, args.n_bootstrap, args.seed
    )
    print(f"R6.4 bootstrap stability: {'PASS' if bootstrap_result['passed'] else 'FAIL'}")
    for axis in AXES:
        r = bootstrap_result["per_axis"][axis]
        print(f"  {axis}: SE={r['bootstrap_se']:.5f}  spread={r['charged_spread']:.4f}  "
              f"SE/spread={r['se_ratio']:.4f}  ({'PASS' if r['passed'] else 'FAIL'})")

    # ── assemble outputs ───────────────────────────────────────────────────────

    all_passed = (
        outlier_result["passed"]
        and subcat_result["passed"]
        and midpoint_result["passed"]
        and bootstrap_result["passed"]
    )

    center_payload: dict[str, Any] = {
        "n_prompts_raw":   n_raw,
        "n_prompts_clean": len(rows),
        "n_outliers":      outlier_result["n_outliers"],
        "layer":           LAYER,
        "method":          args.method,
        "center":          center,
        "per_axis": {
            axis: {
                "mean": center[axis],
                "std":  std_of(scores[axis]),
                "scores": scores[axis],
            }
            for axis in AXES
        },
        "by_category": subcat_result["centroids_by_category"],
        "validation_passed": all_passed,
    }

    validation_payload: dict[str, Any] = {
        "all_passed": all_passed,
        "tolerances": {
            "outlier_z":        args.outlier_z,
            "subcategory_tol":  args.subcategory_tol,
            "midpoint_tol":     args.midpoint_tol,
            "se_tol":           args.se_tol,
            "n_bootstrap":      args.n_bootstrap,
        },
        "r6_1_subcategory_agreement": subcat_result,
        "r6_2_midpoint_test":         midpoint_result,
        "r6_3_outliers":              outlier_result,
        "r6_4_bootstrap_stability":   bootstrap_result,
    }

    save_json(center_payload, args.output)
    save_json(validation_payload, args.report)

    # ── final summary ──────────────────────────────────────────────────────────

    print(f"\n=== compass center ===")
    print(f"  economic : {center['economic']:+.6f}")
    print(f"  social   : {center['social']:+.6f}")
    print(f"\n=== R6 summary ===")
    print(f"  R6.1 subcategory agreement : {'PASS' if subcat_result['passed'] else 'FAIL'}")
    print(f"  R6.2 midpoint test         : {'PASS' if midpoint_result['passed'] else 'FAIL'}")
    print(f"  R6.3 no outliers           : {'PASS' if outlier_result['passed'] else 'FAIL'} ({outlier_result['n_outliers']} removed)")
    print(f"  R6.4 bootstrap stability   : {'PASS' if bootstrap_result['passed'] else 'FAIL'}")
    print(f"\n  overall: {'ALL PASS — center is valid' if all_passed else 'SOME CHECKS FAILED — see ' + str(args.report)}")
    print(f"\nwrote {args.output}")
    print(f"wrote {args.report}")


if __name__ == "__main__":
    main()
