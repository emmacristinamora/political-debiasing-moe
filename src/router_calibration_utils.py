# src/router_calibration_utils.py


# === IMPORTS ===

from __future__ import annotations

import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# router_calibration_config sits next to this module in src/ and owns the
# canonical quadrant order plus the YAML-loaded CandidatePoliciesConfig dataclass.
# add src/ to sys.path so the import works regardless of cwd, mirroring the
# pattern in src/build_router_prompt_set.py and src/build_router_features.py.
_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from router_calibration_config import (  # noqa: E402
    CANONICAL_QUADRANT_ORDER,
    CandidatePoliciesConfig,
)


# === CONSTANTS ===

# τ values for the structured power-scaled candidates. spec for step 4 fixes
# both: τ=0.5 sharpens the prior, τ=2.0 softens it.
DEFAULT_SHARPEN_TAU: float = 0.5
DEFAULT_SOFTEN_TAU: float = 2.0

# diagonal opposite mapping: each quadrant pairs with the one that flips both axes.
_OPPOSITE_QUADRANT: dict[str, str] = {
    "left_lib":   "right_auth",
    "left_auth":  "right_lib",
    "right_lib":  "left_auth",
    "right_auth": "left_lib",
}

# per-element tolerance for the "are these two policies the same" test.
DUPLICATE_TOLERANCE: float = 1e-6

# tolerance for "policy sums to 1" check on outputs and inputs.
SUM_TOLERANCE: float = 1e-6

# upper bound on min_probability — beyond 0.25, no valid four-quadrant
# distribution can satisfy "every entry >= min_p AND sum to 1".
_MIN_PROBABILITY_UPPER_BOUND: float = 0.25


# === DATACLASS ===

@dataclass
class CandidatePolicyConfig:
    """
    Flat, generator-shaped view of the candidate-policy knobs. Step 1's
    CandidatePoliciesConfig (loaded from config.yaml) carries history fields
    (sharpen_temperatures lists, seed) that the generator does not consult
    directly; from_router_calibration() adapts that richer config into this
    flatter shape.
    """
    num_dirichlet_samples: int
    dirichlet_alpha: float
    include_uniform: bool
    include_sharpened: bool
    include_softened: bool
    include_opposite: bool
    include_adjacent: bool
    min_probability: float


def from_router_calibration(cfg: CandidatePoliciesConfig) -> CandidatePolicyConfig:
    """
    Adapter from the YAML-loaded CandidatePoliciesConfig (step 1) to the
    generator-shaped CandidatePolicyConfig used here.

    Mapping:
        dirichlet_samples       -> num_dirichlet_samples
        dirichlet_concentration -> dirichlet_alpha
        sharpen_temperatures    -> include_sharpened (bool: list non-empty)
        soften_temperatures     -> include_softened  (bool: list non-empty)
        include_opposite_heavy  -> include_opposite
        include_adjacent_heavy  -> include_adjacent
        include_uniform         -> include_uniform
        min_probability         -> min_probability

    Note: actual τ values inside sharpen_temperatures/soften_temperatures are
    ignored; step 4 fixes τ to DEFAULT_SHARPEN_TAU / DEFAULT_SOFTEN_TAU. The
    list-of-temperatures shape is treated as an on/off toggle here.
    """
    return CandidatePolicyConfig(
        num_dirichlet_samples=cfg.dirichlet_samples,
        dirichlet_alpha=cfg.dirichlet_concentration,
        include_uniform=cfg.include_uniform,
        include_sharpened=bool(cfg.sharpen_temperatures),
        include_softened=bool(cfg.soften_temperatures),
        include_opposite=cfg.include_opposite_heavy,
        include_adjacent=cfg.include_adjacent_heavy,
        min_probability=cfg.min_probability,
    )


# === VALIDATION ===

def _is_finite_number(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return math.isfinite(float(value))


def _validate_keys(policy: Any, where: str) -> None:
    if not isinstance(policy, dict):
        raise ValueError(
            f"{where}: policy must be a dict, got {type(policy).__name__}"
        )
    expected = set(CANONICAL_QUADRANT_ORDER)
    actual = set(policy.keys())
    if actual != expected:
        missing = sorted(expected - actual)
        extra   = sorted(actual - expected)
        raise ValueError(
            f"{where}: policy keys must equal canonical "
            f"{list(CANONICAL_QUADRANT_ORDER)}; "
            f"missing={missing} extra={extra}"
        )


def _validate_distribution(
    policy: dict[str, float],
    where: str,
    *,
    sum_tol: float = SUM_TOLERANCE,
    require_sum_to_one: bool = True,
) -> None:
    """
    Check that a policy is a strictly-positive numeric distribution over the
    canonical quadrants. When require_sum_to_one is True, also verifies sum is
    1 within sum_tol.
    """
    _validate_keys(policy, where)
    total = 0.0
    for key in CANONICAL_QUADRANT_ORDER:
        v = policy[key]
        if not _is_finite_number(v):
            raise ValueError(
                f"{where}: policy[{key!r}] = {v!r} must be a finite number"
            )
        v = float(v)
        if v <= 0:
            raise ValueError(
                f"{where}: policy[{key!r}] = {v} must be strictly > 0"
            )
        total += v
    if require_sum_to_one and abs(total - 1.0) > sum_tol:
        raise ValueError(
            f"{where}: policy sum = {total} (expected 1 within {sum_tol})"
        )


def _validate_quadrant_scores(scores: Any, where: str) -> None:
    _validate_keys(scores, where)
    for key in CANONICAL_QUADRANT_ORDER:
        v = scores[key]
        if not _is_finite_number(v):
            raise ValueError(
                f"{where}: scores[{key!r}] = {v!r} must be a finite number"
            )


def _validate_config(cfg: Any) -> None:
    if not isinstance(cfg, CandidatePolicyConfig):
        raise ValueError(
            f"config must be CandidatePolicyConfig, got {type(cfg).__name__}"
        )

    if isinstance(cfg.num_dirichlet_samples, bool) or not isinstance(
        cfg.num_dirichlet_samples, int,
    ):
        raise ValueError(
            f"num_dirichlet_samples must be int, "
            f"got {type(cfg.num_dirichlet_samples).__name__}"
        )
    if cfg.num_dirichlet_samples < 0:
        raise ValueError(
            f"num_dirichlet_samples must be >= 0, got {cfg.num_dirichlet_samples}"
        )

    if not _is_finite_number(cfg.dirichlet_alpha) or cfg.dirichlet_alpha <= 0:
        raise ValueError(
            f"dirichlet_alpha must be a finite positive number, "
            f"got {cfg.dirichlet_alpha!r}"
        )

    for name in (
        "include_uniform", "include_sharpened", "include_softened",
        "include_opposite", "include_adjacent",
    ):
        v = getattr(cfg, name)
        if not isinstance(v, bool):
            raise ValueError(
                f"{name} must be bool, got {type(v).__name__}"
            )

    if not _is_finite_number(cfg.min_probability):
        raise ValueError(
            f"min_probability must be a finite number, got {cfg.min_probability!r}"
        )
    if cfg.min_probability <= 0 or cfg.min_probability >= _MIN_PROBABILITY_UPPER_BOUND:
        raise ValueError(
            f"min_probability must lie in (0, {_MIN_PROBABILITY_UPPER_BOUND}), "
            f"got {cfg.min_probability}"
        )


# === HELPERS ===

def normalize_distribution(policy: dict[str, float]) -> dict[str, float]:
    """
    Renormalize a non-negative policy so the entries sum to 1. Preserves
    canonical key order in the output dict.
    """
    _validate_keys(policy, "normalize_distribution")
    values: list[float] = []
    for key in CANONICAL_QUADRANT_ORDER:
        v = policy[key]
        if not _is_finite_number(v):
            raise ValueError(
                f"normalize_distribution: policy[{key!r}] = {v!r} is not finite"
            )
        v = float(v)
        if v < 0:
            raise ValueError(
                f"normalize_distribution: policy[{key!r}] = {v} must be >= 0"
            )
        values.append(v)
    total = sum(values)
    if total == 0:
        raise ValueError("normalize_distribution: sum is zero")
    return {key: v / total for key, v in zip(CANONICAL_QUADRANT_ORDER, values)}


def apply_min_probability(
    policy: dict[str, float],
    min_p: float,
) -> dict[str, float]:
    """
    Redistribute mass so every output entry is >= min_p while preserving
    sum-to-1 and the relative ordering of "excess above the floor".

    Algorithm (k = 4 quadrants):
        fixed_mass     = k * min_p
        remaining_mass = 1 - fixed_mass
        excess_i       = max(p_i - min_p, 0)
        if sum(excess) > 0:
            out_i = min_p + remaining_mass * excess_i / sum(excess)
        else:
            out_i = 1 / k                  # uniform fallback (defensive only)

    Properties:
      * out_i >= min_p for every i (modulo FP rounding within SUM_TOLERANCE).
      * sum(out_i) == 1 exactly in exact arithmetic; within SUM_TOLERANCE in FP.
      * inputs that already satisfy the floor with sum=1 pass through unchanged.

    The uniform fallback is unreachable for any valid sum-to-1 input with
    min_p in (0, 0.25): such an input must contain at least one entry > min_p,
    so sum(excess) > 0. The branch is retained only as a defensive guard.

    Args:
        policy: distribution over CANONICAL_QUADRANT_ORDER. Must be finite,
                non-negative, and sum to 1 within SUM_TOLERANCE. Zero entries
                are allowed (they get pinned to the floor).
        min_p:  per-entry floor. Must lie in (0, 0.25); outside that range,
                no four-quadrant distribution can satisfy the floor and
                sum-to-1 jointly.
    """
    # 1) validate min_p — must be a finite number strictly inside (0, 0.25)
    if not _is_finite_number(min_p):
        raise ValueError(f"min_p must be a finite number, got {min_p!r}")
    if min_p <= 0 or min_p >= _MIN_PROBABILITY_UPPER_BOUND:
        raise ValueError(
            f"min_p must lie in (0, {_MIN_PROBABILITY_UPPER_BOUND}), got {min_p}"
        )

    # 2) validate input distribution: keys, finite, non-negative, sum to 1
    _validate_keys(policy, "apply_min_probability")
    values: list[float] = []
    total = 0.0
    for key in CANONICAL_QUADRANT_ORDER:
        v = policy[key]
        if not _is_finite_number(v):
            raise ValueError(
                f"apply_min_probability: policy[{key!r}] = {v!r} is not finite"
            )
        v = float(v)
        if v < 0:
            raise ValueError(
                f"apply_min_probability: policy[{key!r}] = {v} must be >= 0"
            )
        values.append(v)
        total += v
    if abs(total - 1.0) > SUM_TOLERANCE:
        raise ValueError(
            f"apply_min_probability: input policy must sum to 1 within "
            f"{SUM_TOLERANCE}, got {total}"
        )

    # 3) compute floor budget
    floor = float(min_p)
    k = len(CANONICAL_QUADRANT_ORDER)  # always 4
    fixed_mass = k * floor
    remaining_mass = 1.0 - fixed_mass

    # 4) distribute remaining_mass proportional to original excess above floor
    excesses = [max(v - floor, 0.0) for v in values]
    excess_sum = sum(excesses)
    if excess_sum > 0:
        new_values = [
            floor + remaining_mass * (e / excess_sum)
            for e in excesses
        ]
    else:
        # unreachable for valid sum-to-1 inputs with min_p < 0.25
        new_values = [1.0 / k] * k

    out = {key: nv for key, nv in zip(CANONICAL_QUADRANT_ORDER, new_values)}

    # 5) validate output (defensive: catches FP drift / contract violations)
    _validate_distribution(
        out, "apply_min_probability output", require_sum_to_one=True,
    )
    return out


def dirichlet_sample(
    alpha_vector: list[float],
    rng: random.Random,
) -> list[float]:
    """
    Draw a single Dirichlet(α₁, …, αₖ) sample using the standard
    gamma-normalize trick. All αᵢ must be strictly positive and finite; the
    rng must be a random.Random instance so callers retain full control over
    determinism.
    """
    if not isinstance(rng, random.Random):
        raise ValueError(
            f"rng must be random.Random, got {type(rng).__name__}"
        )
    if not isinstance(alpha_vector, list) or len(alpha_vector) == 0:
        raise ValueError("alpha_vector must be a non-empty list")

    samples: list[float] = []
    for i, a in enumerate(alpha_vector):
        if not _is_finite_number(a) or float(a) <= 0:
            raise ValueError(
                f"alpha_vector[{i}] = {a!r} must be a finite positive number"
            )
        samples.append(rng.gammavariate(float(a), 1.0))
    total = sum(samples)
    if total <= 0 or not math.isfinite(total):
        raise ValueError("dirichlet_sample: gamma draws produced non-positive sum")
    return [x / total for x in samples]


def are_policies_equal(
    p1: dict[str, float],
    p2: dict[str, float],
    tol: float = DUPLICATE_TOLERANCE,
) -> bool:
    """
    Two policies are 'equal' if every canonical-quadrant entry differs by
    less than tol. Returns False on any structural mismatch (missing/extra
    keys, non-numeric values).
    """
    expected = set(CANONICAL_QUADRANT_ORDER)
    if not isinstance(p1, dict) or set(p1.keys()) != expected:
        return False
    if not isinstance(p2, dict) or set(p2.keys()) != expected:
        return False
    for key in CANONICAL_QUADRANT_ORDER:
        v1, v2 = p1[key], p2[key]
        if not (_is_finite_number(v1) and _is_finite_number(v2)):
            return False
        if abs(float(v1) - float(v2)) >= tol:
            return False
    return True


# === STRUCTURED BUILDERS ===

def _uniform_policy() -> dict[str, float]:
    return {key: 0.25 for key in CANONICAL_QUADRANT_ORDER}


def _power_policy(prior: dict[str, float], tau: float) -> dict[str, float]:
    """π_i ∝ π0_i^(1/τ) — sharpens for τ<1, softens for τ>1."""
    if not _is_finite_number(tau) or tau <= 0:
        raise ValueError(f"tau must be a finite positive number, got {tau!r}")
    exponent = 1.0 / float(tau)
    raw: dict[str, float] = {}
    for key in CANONICAL_QUADRANT_ORDER:
        v = float(prior[key])
        # prior is validated > 0 upstream so v**exponent is always finite
        raw[key] = v ** exponent
    return normalize_distribution(raw)


def _argmax_quadrant(scores: dict[str, float]) -> str:
    """Return the canonical-order argmax. Ties resolve to the first key."""
    best_key: str | None = None
    best_val: float = -math.inf
    for key in CANONICAL_QUADRANT_ORDER:
        v = float(scores[key])
        if v > best_val:
            best_val = v
            best_key = key
    # canonical order is non-empty so best_key is set
    assert best_key is not None
    return best_key


def _opposite_heavy_policy(scores: dict[str, float]) -> dict[str, float]:
    """
    Build the opposite-heavy policy:
        start uniform, opposite += 2, both adjacents += 1, aligned += 0.
    """
    q_max = _argmax_quadrant(scores)
    q_opp = _OPPOSITE_QUADRANT[q_max]
    raw = {key: 0.25 for key in CANONICAL_QUADRANT_ORDER}
    raw[q_opp] += 2.0
    for key in CANONICAL_QUADRANT_ORDER:
        if key not in (q_max, q_opp):
            raw[key] += 1.0
    return normalize_distribution(raw)


def _adjacent_variants(scores: dict[str, float]) -> list[dict[str, float]]:
    """
    Two adjacent-heavy variants — one per adjacent direction. The chosen
    adjacent gets a +2.0 boost; the other adjacent gets +0.5; aligned and
    opposite stay at 0.25 (uniform baseline).
    """
    q_max = _argmax_quadrant(scores)
    q_opp = _OPPOSITE_QUADRANT[q_max]
    adjacents = [k for k in CANONICAL_QUADRANT_ORDER if k not in (q_max, q_opp)]
    # by construction there are exactly two adjacents
    if len(adjacents) != 2:
        raise ValueError(
            f"expected exactly 2 adjacent quadrants for q_max={q_max!r}; "
            f"got {adjacents}"
        )
    out: list[dict[str, float]] = []
    for chosen in adjacents:
        raw = {key: 0.25 for key in CANONICAL_QUADRANT_ORDER}
        raw[chosen] += 2.0
        for other in adjacents:
            if other != chosen:
                raw[other] += 0.5
        out.append(normalize_distribution(raw))
    return out


def _dirichlet_policies(
    prior: dict[str, float],
    alpha: float,
    n: int,
    rng: random.Random,
) -> list[dict[str, float]]:
    if n == 0:
        return []
    alphas = [alpha * float(prior[key]) for key in CANONICAL_QUADRANT_ORDER]
    out: list[dict[str, float]] = []
    for _ in range(n):
        sample = dirichlet_sample(alphas, rng)
        out.append({key: sample[i] for i, key in enumerate(CANONICAL_QUADRANT_ORDER)})
    return out


# === DEDUPE ===

def _dedupe_policies(
    policies: list[dict[str, float]],
    tol: float = DUPLICATE_TOLERANCE,
) -> list[dict[str, float]]:
    out: list[dict[str, float]] = []
    for policy in policies:
        if any(are_policies_equal(policy, kept, tol=tol) for kept in out):
            continue
        out.append(policy)
    return out


# === MAIN ENTRY POINT ===

def generate_candidate_policies(
    quadrant_scores: dict[str, float],
    heuristic_prior: dict[str, float],
    config: CandidatePolicyConfig,
    rng: random.Random,
) -> list[dict[str, float]]:
    """
    Build the deduplicated, validated candidate-policy list for one prompt.

    Order of returned candidates:
        1) heuristic prior (always first, always present)
        2) uniform                       — if config.include_uniform
        3) sharpened                     — if config.include_sharpened
        4) softened                      — if config.include_softened
        5) opposite-heavy                — if config.include_opposite
        6) adjacent-heavy variants (×2)  — if config.include_adjacent
        7) Dirichlet samples             — config.num_dirichlet_samples

    Every candidate is post-processed with apply_min_probability — which
    redistributes mass so every entry satisfies p_i >= config.min_probability
    while preserving sum-to-1 — then validated as a strictly-positive
    distribution summing to 1. Near-duplicates
    (max-abs-diff < DUPLICATE_TOLERANCE) are collapsed to a single entry,
    keeping the first occurrence — which means heuristic always survives
    even when a structured candidate happens to match it exactly.
    """
    _validate_config(config)
    if not isinstance(rng, random.Random):
        raise ValueError(
            f"rng must be random.Random, got {type(rng).__name__}"
        )

    _validate_quadrant_scores(quadrant_scores, "quadrant_scores")
    _validate_distribution(
        heuristic_prior,
        "heuristic_prior",
        sum_tol=SUM_TOLERANCE,
        require_sum_to_one=True,
    )

    raw: list[dict[str, float]] = []

    # 1) heuristic — always first
    raw.append({key: float(heuristic_prior[key]) for key in CANONICAL_QUADRANT_ORDER})

    # 2) uniform
    if config.include_uniform:
        raw.append(_uniform_policy())

    # 3) sharpened
    if config.include_sharpened:
        raw.append(_power_policy(heuristic_prior, DEFAULT_SHARPEN_TAU))

    # 4) softened
    if config.include_softened:
        raw.append(_power_policy(heuristic_prior, DEFAULT_SOFTEN_TAU))

    # 5) opposite-heavy
    if config.include_opposite:
        raw.append(_opposite_heavy_policy(quadrant_scores))

    # 6) adjacent-heavy (two variants)
    if config.include_adjacent:
        raw.extend(_adjacent_variants(quadrant_scores))

    # 7) dirichlet samples
    raw.extend(_dirichlet_policies(
        heuristic_prior, config.dirichlet_alpha, config.num_dirichlet_samples, rng,
    ))

    # post-process: enforce min_probability, renormalize, then validate
    processed: list[dict[str, float]] = []
    for i, policy in enumerate(raw):
        cleaned = apply_min_probability(policy, config.min_probability)
        _validate_distribution(cleaned, where=f"candidate[{i}]")
        processed.append(cleaned)

    return _dedupe_policies(processed)
