# tests/test_router_candidate_policies.py


# === IMPORTS ===

from __future__ import annotations

import math
import random
import sys
import unittest
from pathlib import Path
from typing import Any


# === MODULE LOADING ===

# router_calibration_utils lives in src/. add it to sys.path so it can be
# imported by name without making src a package.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_DIR = _REPO_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import router_calibration_utils as rcu  # noqa: E402
from router_calibration_config import CandidatePoliciesConfig  # noqa: E402

CANONICAL = ("left_lib", "left_auth", "right_lib", "right_auth")


# === HELPERS ===

def _make_config(**overrides: Any) -> rcu.CandidatePolicyConfig:
    base: dict[str, Any] = dict(
        num_dirichlet_samples=8,
        dirichlet_alpha=10.0,
        include_uniform=True,
        include_sharpened=True,
        include_softened=True,
        include_opposite=True,
        include_adjacent=True,
        min_probability=1e-6,
    )
    base.update(overrides)
    return rcu.CandidatePolicyConfig(**base)


def _scores(
    left_lib: float = 0.4,
    left_auth: float = 0.1,
    right_lib: float = 0.3,
    right_auth: float = -0.2,
) -> dict[str, float]:
    return {
        "left_lib":   left_lib,
        "left_auth":  left_auth,
        "right_lib":  right_lib,
        "right_auth": right_auth,
    }


def _prior() -> dict[str, float]:
    return {"left_lib": 0.5, "left_auth": 0.2, "right_lib": 0.2, "right_auth": 0.1}


def _is_valid_distribution(p: dict[str, float], tol: float = 1e-6) -> bool:
    if set(p.keys()) != set(CANONICAL):
        return False
    s = 0.0
    for key in CANONICAL:
        v = p[key]
        if not isinstance(v, float) or not math.isfinite(v) or v <= 0:
            return False
        s += v
    return abs(s - 1.0) <= tol


# === TESTS — HELPERS ===

class HelperTests(unittest.TestCase):

    def test_normalize_distribution_basic(self) -> None:
        out = rcu.normalize_distribution(
            {"left_lib": 1, "left_auth": 1, "right_lib": 1, "right_auth": 1},
        )
        for v in out.values():
            self.assertAlmostEqual(v, 0.25, places=9)
        self.assertEqual(tuple(out.keys()), CANONICAL)

    def test_normalize_distribution_zero_sum_raises(self) -> None:
        with self.assertRaises(ValueError):
            rcu.normalize_distribution({k: 0.0 for k in CANONICAL})

    def test_normalize_distribution_negative_raises(self) -> None:
        bad = {"left_lib": -0.1, "left_auth": 0.5, "right_lib": 0.3, "right_auth": 0.3}
        with self.assertRaises(ValueError):
            rcu.normalize_distribution(bad)

    def test_apply_min_probability_floor_holds_after_redistribution(self) -> None:
        # input has two zero-mass entries and one below the floor; the new
        # redistribution algorithm pins every entry below the floor at
        # exactly min_p and routes the entire remaining mass to the single
        # entry whose original excess > 0.
        out = rcu.apply_min_probability(
            {"left_lib": 0.97, "left_auth": 0.03, "right_lib": 0.0, "right_auth": 0.0},
            min_p=0.05,
        )
        self.assertAlmostEqual(sum(out.values()), 1.0, places=6)
        for v in out.values():
            self.assertGreaterEqual(v, 0.05 - 1e-9)
        # left_auth (0.03) was below the floor → 0 excess → exactly floor
        self.assertAlmostEqual(out["left_auth"],  0.05, places=9)
        self.assertAlmostEqual(out["right_lib"],  0.05, places=9)
        self.assertAlmostEqual(out["right_auth"], 0.05, places=9)
        # left_lib gets all remaining_mass: 0.05 + (1 - 4*0.05) = 0.85
        self.assertAlmostEqual(out["left_lib"],   0.85, places=9)

    def test_apply_min_probability_input_already_above_floor_unchanged(self) -> None:
        # when every input entry already meets the floor, the redistribution
        # algorithm is a structural identity: out == input (modulo FP).
        inp = {"left_lib": 0.4, "left_auth": 0.3, "right_lib": 0.2, "right_auth": 0.1}
        out = rcu.apply_min_probability(inp, min_p=0.05)
        self.assertAlmostEqual(sum(out.values()), 1.0, places=6)
        for key in CANONICAL:
            self.assertAlmostEqual(out[key], inp[key], places=9)

    def test_apply_min_probability_multiple_entries_below_floor(self) -> None:
        # three entries below floor; all three pin to floor exactly and the
        # remaining mass goes to the lone entry with positive excess.
        out = rcu.apply_min_probability(
            {"left_lib": 0.85, "left_auth": 0.10, "right_lib": 0.03, "right_auth": 0.02},
            min_p=0.15,
        )
        self.assertAlmostEqual(sum(out.values()), 1.0, places=6)
        for v in out.values():
            self.assertGreaterEqual(v, 0.15 - 1e-9)
        # excess = [0.70, 0, 0, 0]; remaining_mass = 1 - 0.6 = 0.4
        # left_lib  = 0.15 + 0.4 * 0.70 / 0.70 = 0.55
        self.assertAlmostEqual(out["left_lib"],   0.55, places=9)
        self.assertAlmostEqual(out["left_auth"],  0.15, places=9)
        self.assertAlmostEqual(out["right_lib"],  0.15, places=9)
        self.assertAlmostEqual(out["right_auth"], 0.15, places=9)

    def test_apply_min_probability_two_entries_split_remaining_mass(self) -> None:
        # two entries above floor share remaining_mass in proportion to their
        # excess; two entries below floor pin to floor.
        out = rcu.apply_min_probability(
            {"left_lib": 0.55, "left_auth": 0.30, "right_lib": 0.10, "right_auth": 0.05},
            min_p=0.15,
        )
        self.assertAlmostEqual(sum(out.values()), 1.0, places=6)
        for v in out.values():
            self.assertGreaterEqual(v, 0.15 - 1e-9)
        # excess = [0.40, 0.15, 0, 0]; sum = 0.55; remaining_mass = 0.4
        self.assertAlmostEqual(out["left_lib"],   0.15 + 0.4 * 0.40 / 0.55, places=9)
        self.assertAlmostEqual(out["left_auth"],  0.15 + 0.4 * 0.15 / 0.55, places=9)
        self.assertAlmostEqual(out["right_lib"],  0.15, places=9)
        self.assertAlmostEqual(out["right_auth"], 0.15, places=9)
        # ordering of the above-floor entries is preserved
        self.assertGreater(out["left_lib"], out["left_auth"])

    def test_apply_min_probability_input_not_summing_to_one_raises(self) -> None:
        bad = {"left_lib": 0.5, "left_auth": 0.5, "right_lib": 0.5, "right_auth": 0.5}
        with self.assertRaises(ValueError) as ctx:
            rcu.apply_min_probability(bad, min_p=0.05)
        self.assertIn("sum to 1", str(ctx.exception))

    def test_apply_min_probability_invalid_min_p_raises(self) -> None:
        with self.assertRaises(ValueError):
            rcu.apply_min_probability(
                {k: 0.25 for k in CANONICAL}, min_p=0.0,
            )
        with self.assertRaises(ValueError):
            rcu.apply_min_probability(
                {k: 0.25 for k in CANONICAL}, min_p=0.25,
            )

    def test_dirichlet_sample_sums_to_one(self) -> None:
        rng = random.Random(42)
        out = rcu.dirichlet_sample([1.0, 1.0, 1.0, 1.0], rng)
        self.assertEqual(len(out), 4)
        self.assertAlmostEqual(sum(out), 1.0, places=6)
        for v in out:
            self.assertGreater(v, 0.0)

    def test_dirichlet_sample_invalid_alpha_raises(self) -> None:
        with self.assertRaises(ValueError):
            rcu.dirichlet_sample([1.0, 0.0, 1.0, 1.0], random.Random(1))
        with self.assertRaises(ValueError):
            rcu.dirichlet_sample([1.0, -1.0, 1.0, 1.0], random.Random(1))

    def test_dirichlet_sample_non_random_rng_raises(self) -> None:
        with self.assertRaises(ValueError):
            rcu.dirichlet_sample([1.0, 1.0, 1.0, 1.0], rng="not-a-random-instance")  # type: ignore[arg-type]

    def test_are_policies_equal_within_tolerance(self) -> None:
        p1 = {k: 0.25 for k in CANONICAL}
        p2 = {
            "left_lib":   0.25 + 1e-7,
            "left_auth":  0.25,
            "right_lib":  0.25,
            "right_auth": 0.25 - 1e-7,
        }
        self.assertTrue(rcu.are_policies_equal(p1, p2))
        p3 = {**p1, "left_lib": 0.25 + 1e-3}
        self.assertFalse(rcu.are_policies_equal(p1, p3))

    def test_are_policies_equal_structural_mismatch_returns_false(self) -> None:
        self.assertFalse(rcu.are_policies_equal({"left_lib": 1.0}, {k: 0.25 for k in CANONICAL}))


# === TESTS — generate_candidate_policies ===

class GenerateCandidatePoliciesTests(unittest.TestCase):

    def test_outputs_are_valid_distributions(self) -> None:
        rng = random.Random(42)
        out = rcu.generate_candidate_policies(_scores(), _prior(), _make_config(), rng)
        self.assertGreater(len(out), 0)
        for i, policy in enumerate(out):
            self.assertTrue(
                _is_valid_distribution(policy),
                msg=f"policy[{i}] = {policy} is not a valid distribution",
            )

    def test_canonical_keys_preserved(self) -> None:
        rng = random.Random(42)
        out = rcu.generate_candidate_policies(_scores(), _prior(), _make_config(), rng)
        for policy in out:
            self.assertEqual(tuple(policy.keys()), CANONICAL)

    def test_heuristic_policy_is_first(self) -> None:
        rng = random.Random(42)
        prior = _prior()
        cfg = _make_config(
            num_dirichlet_samples=0,
            include_uniform=False,
            include_sharpened=False,
            include_softened=False,
            include_opposite=False,
            include_adjacent=False,
        )
        out = rcu.generate_candidate_policies(_scores(), prior, cfg, rng)
        self.assertEqual(len(out), 1)
        self.assertTrue(rcu.are_policies_equal(out[0], prior))

    def test_no_duplicates_after_dedupe(self) -> None:
        # heuristic == uniform and τ-power forms of uniform == uniform; without
        # dedupe we'd see 4 identical entries, so collapse must produce 1
        rng = random.Random(42)
        uniform = {k: 0.25 for k in CANONICAL}
        cfg = _make_config(num_dirichlet_samples=0)
        out = rcu.generate_candidate_policies(_scores(), uniform, cfg, rng)
        for i in range(len(out)):
            for j in range(i + 1, len(out)):
                self.assertFalse(
                    rcu.are_policies_equal(out[i], out[j]),
                    msg=f"duplicate at {i} and {j}: {out[i]} vs {out[j]}",
                )
        # heuristic survives as the first entry even though uniform/sharpened/softened
        # collapsed onto it
        self.assertTrue(rcu.are_policies_equal(out[0], uniform))

    def test_min_probability_enforced(self) -> None:
        # new contract: every output entry across every candidate is at or
        # above min_probability — no slack from a renormalize step.
        rng = random.Random(42)
        cfg = _make_config(min_probability=0.05)
        out = rcu.generate_candidate_policies(_scores(), _prior(), cfg, rng)
        for i, policy in enumerate(out):
            for key, value in policy.items():
                self.assertGreaterEqual(
                    value, 0.05 - 1e-9,
                    msg=f"policy[{i}][{key}] = {value} below min_probability=0.05",
                )

    def test_dirichlet_sample_count(self) -> None:
        # disable structured candidates so we count only heuristic + dirichlet
        rng = random.Random(42)
        cfg = _make_config(
            num_dirichlet_samples=5,
            include_uniform=False,
            include_sharpened=False,
            include_softened=False,
            include_opposite=False,
            include_adjacent=False,
        )
        out = rcu.generate_candidate_policies(_scores(), _prior(), cfg, rng)
        # heuristic + 5 dirichlet draws; with continuous Dirichlet samples the
        # chance of accidental near-equality with the heuristic at 1e-6 is
        # negligible, so the count is exactly 6
        self.assertEqual(len(out), 6)

    def test_zero_dirichlet_samples_allowed(self) -> None:
        rng = random.Random(0)
        cfg = _make_config(num_dirichlet_samples=0)
        out = rcu.generate_candidate_policies(_scores(), _prior(), cfg, rng)
        # heuristic + uniform + sharpened + softened + opposite + 2 adjacent = 7
        self.assertEqual(len(out), 7)

    def test_determinism_with_fixed_seed(self) -> None:
        cfg = _make_config()
        a = rcu.generate_candidate_policies(_scores(), _prior(), cfg, random.Random(42))
        b = rcu.generate_candidate_policies(_scores(), _prior(), cfg, random.Random(42))
        self.assertEqual(len(a), len(b))
        for pa, pb in zip(a, b):
            self.assertTrue(rcu.are_policies_equal(pa, pb))

    def test_different_seeds_produce_different_dirichlet_tail(self) -> None:
        cfg = _make_config(num_dirichlet_samples=4)
        a = rcu.generate_candidate_policies(_scores(), _prior(), cfg, random.Random(1))
        b = rcu.generate_candidate_policies(_scores(), _prior(), cfg, random.Random(2))
        # the structured prefix is identical but the dirichlet tail must differ
        self.assertEqual(len(a), len(b))
        # at least one position past the structured candidates must differ
        differs = False
        for pa, pb in zip(a, b):
            if not rcu.are_policies_equal(pa, pb):
                differs = True
                break
        self.assertTrue(differs)

    def test_opposite_mapping_picks_diagonal(self) -> None:
        # left_lib argmax ⇒ opposite is right_auth
        scores = _scores(left_lib=1.0, left_auth=-1.0, right_lib=-1.0, right_auth=-1.0)
        out = rcu._opposite_heavy_policy(scores)
        max_key = max(out, key=lambda k: out[k])
        self.assertEqual(max_key, "right_auth")
        # aligned (left_lib) gets the smallest mass
        min_key = min(out, key=lambda k: out[k])
        self.assertEqual(min_key, "left_lib")
        self.assertAlmostEqual(sum(out.values()), 1.0, places=6)

    def test_opposite_mapping_for_each_quadrant(self) -> None:
        for q_max, q_opp in (
            ("left_lib",   "right_auth"),
            ("left_auth",  "right_lib"),
            ("right_lib",  "left_auth"),
            ("right_auth", "left_lib"),
        ):
            scores = {k: -1.0 for k in CANONICAL}
            scores[q_max] = 1.0
            policy = rcu._opposite_heavy_policy(scores)
            self.assertEqual(max(policy, key=lambda k: policy[k]), q_opp)

    def test_adjacent_variants_differ(self) -> None:
        scores = _scores(left_lib=1.0, left_auth=-1.0, right_lib=-1.0, right_auth=-1.0)
        variants = rcu._adjacent_variants(scores)
        self.assertEqual(len(variants), 2)
        self.assertFalse(rcu.are_policies_equal(variants[0], variants[1]))
        # adjacents for left_lib are left_auth and right_lib
        for variant in variants:
            top = max(variant, key=lambda k: variant[k])
            self.assertIn(top, {"left_auth", "right_lib"})
        # the two top quadrants together cover the full adjacency set
        tops = {max(v, key=lambda k: v[k]) for v in variants}
        self.assertEqual(tops, {"left_auth", "right_lib"})

    def test_invalid_config_alpha_raises(self) -> None:
        with self.assertRaises(ValueError):
            rcu.generate_candidate_policies(
                _scores(), _prior(),
                _make_config(dirichlet_alpha=0.0),
                random.Random(1),
            )
        with self.assertRaises(ValueError):
            rcu.generate_candidate_policies(
                _scores(), _prior(),
                _make_config(dirichlet_alpha=-1.0),
                random.Random(1),
            )

    def test_invalid_config_negative_samples_raises(self) -> None:
        with self.assertRaises(ValueError):
            rcu.generate_candidate_policies(
                _scores(), _prior(),
                _make_config(num_dirichlet_samples=-1),
                random.Random(1),
            )

    def test_invalid_config_min_probability_raises(self) -> None:
        with self.assertRaises(ValueError):
            rcu.generate_candidate_policies(
                _scores(), _prior(),
                _make_config(min_probability=0.0),
                random.Random(1),
            )
        with self.assertRaises(ValueError):
            rcu.generate_candidate_policies(
                _scores(), _prior(),
                _make_config(min_probability=0.5),
                random.Random(1),
            )

    def test_invalid_config_non_bool_toggle_raises(self) -> None:
        with self.assertRaises(ValueError):
            rcu.generate_candidate_policies(
                _scores(), _prior(),
                _make_config(include_uniform="yes"),  # type: ignore[arg-type]
                random.Random(1),
            )

    def test_invalid_rng_raises(self) -> None:
        with self.assertRaises(ValueError):
            rcu.generate_candidate_policies(
                _scores(), _prior(),
                _make_config(),
                rng="not-a-random-instance",  # type: ignore[arg-type]
            )

    def test_missing_keys_in_prior_raises(self) -> None:
        bad = {"left_lib": 0.5, "left_auth": 0.5}
        with self.assertRaises(ValueError):
            rcu.generate_candidate_policies(
                _scores(), bad, _make_config(), random.Random(1),
            )

    def test_extra_keys_in_prior_raises(self) -> None:
        bad = {**_prior(), "extra": 0.0}
        with self.assertRaises(ValueError):
            rcu.generate_candidate_policies(
                _scores(), bad, _make_config(), random.Random(1),
            )

    def test_prior_not_summing_to_one_raises(self) -> None:
        bad = {"left_lib": 0.4, "left_auth": 0.4, "right_lib": 0.4, "right_auth": 0.4}
        with self.assertRaises(ValueError):
            rcu.generate_candidate_policies(
                _scores(), bad, _make_config(), random.Random(1),
            )

    def test_negative_value_in_prior_raises(self) -> None:
        bad = {"left_lib": -0.1, "left_auth": 0.4, "right_lib": 0.4, "right_auth": 0.3}
        with self.assertRaises(ValueError):
            rcu.generate_candidate_policies(
                _scores(), bad, _make_config(), random.Random(1),
            )

    def test_non_finite_quadrant_score_raises(self) -> None:
        bad = {**_scores(), "left_lib": float("nan")}
        with self.assertRaises(ValueError):
            rcu.generate_candidate_policies(
                bad, _prior(), _make_config(), random.Random(1),
            )

    def test_inf_quadrant_score_raises(self) -> None:
        bad = {**_scores(), "left_lib": float("inf")}
        with self.assertRaises(ValueError):
            rcu.generate_candidate_policies(
                bad, _prior(), _make_config(), random.Random(1),
            )


# === TESTS — adapter ===

class FromRouterCalibrationAdapterTests(unittest.TestCase):

    def test_maps_all_fields(self) -> None:
        existing = CandidatePoliciesConfig(
            include_heuristic_prior=True,
            include_uniform=True,
            sharpen_temperatures=[0.5],
            soften_temperatures=[2.0],
            include_opposite_heavy=True,
            include_adjacent_heavy=True,
            dirichlet_samples=16,
            dirichlet_concentration=64.0,
            min_probability=1e-6,
            seed=42,
        )
        new = rcu.from_router_calibration(existing)
        self.assertEqual(new.num_dirichlet_samples, 16)
        self.assertEqual(new.dirichlet_alpha, 64.0)
        self.assertTrue(new.include_uniform)
        self.assertTrue(new.include_sharpened)
        self.assertTrue(new.include_softened)
        self.assertTrue(new.include_opposite)
        self.assertTrue(new.include_adjacent)
        self.assertEqual(new.min_probability, 1e-6)

    def test_empty_temperature_lists_disable_toggles(self) -> None:
        existing = CandidatePoliciesConfig(
            include_heuristic_prior=True,
            include_uniform=True,
            sharpen_temperatures=[],
            soften_temperatures=[],
            include_opposite_heavy=True,
            include_adjacent_heavy=True,
            dirichlet_samples=4,
            dirichlet_concentration=10.0,
            min_probability=1e-6,
            seed=42,
        )
        new = rcu.from_router_calibration(existing)
        self.assertFalse(new.include_sharpened)
        self.assertFalse(new.include_softened)


# === MAIN ===

def main() -> None:
    unittest.main()


if __name__ == "__main__":
    main()
