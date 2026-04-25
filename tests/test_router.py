# tests/test_router.py


# === IMPORTS ===

import importlib.util
import math
import sys
import unittest
from pathlib import Path
from typing import Any


# === MODULE LOADING ===

# src/06_moce_components.py starts with a digit, so it cannot be imported via
# normal "import" syntax. load it explicitly by absolute path with importlib.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_MODULE_PATH = _REPO_ROOT / "src" / "06_moce_components.py"

_spec = importlib.util.spec_from_file_location("moce_components", _MODULE_PATH)
moce_components = importlib.util.module_from_spec(_spec)
sys.modules["moce_components"] = moce_components
_spec.loader.exec_module(moce_components)

Router = moce_components.Router
RouterConfig = moce_components.RouterConfig
PromptState = moce_components.PromptState
RouterState = moce_components.RouterState
CANONICAL_QUADRANT_ORDER = moce_components.CANONICAL_QUADRANT_ORDER


# === HELPERS ===

def _make_prompt_state(
    quadrant_scores: dict[str, float] | None = None,
    bias_magnitude: Any = 0.5,
    economic_score: float = 0.0,
    social_score: float = 0.0,
) -> PromptState:
    if quadrant_scores is None:
        quadrant_scores = {key: 0.0 for key in CANONICAL_QUADRANT_ORDER}
    return PromptState(
        prompt_text="test prompt",
        hidden_representation=None,
        economic_score=economic_score,
        social_score=social_score,
        quadrant_scores=dict(quadrant_scores),
        bias_magnitude=bias_magnitude,
        metadata={},
    )


# === TESTS ===

class ValidationFailureTests(unittest.TestCase):

    def setUp(self) -> None:
        self.router = Router(RouterConfig())

    def test_missing_quadrant_key_raises(self) -> None:
        scores = {key: 0.1 for key in CANONICAL_QUADRANT_ORDER}
        del scores["right_auth"]
        prompt_state = _make_prompt_state(quadrant_scores=scores)
        with self.assertRaises(ValueError):
            self.router.route(prompt_state)

    def test_extra_quadrant_key_raises(self) -> None:
        scores = {key: 0.1 for key in CANONICAL_QUADRANT_ORDER}
        scores["centrist"] = 0.0
        prompt_state = _make_prompt_state(quadrant_scores=scores)
        with self.assertRaises(ValueError):
            self.router.route(prompt_state)

    def test_non_numeric_quadrant_score_raises(self) -> None:
        scores = {key: 0.1 for key in CANONICAL_QUADRANT_ORDER}
        scores["left_lib"] = "not a number"
        prompt_state = _make_prompt_state(quadrant_scores=scores)
        with self.assertRaises(ValueError):
            self.router.route(prompt_state)

    def test_nan_quadrant_score_raises(self) -> None:
        scores = {key: 0.1 for key in CANONICAL_QUADRANT_ORDER}
        scores["left_lib"] = float("nan")
        prompt_state = _make_prompt_state(quadrant_scores=scores)
        with self.assertRaises(ValueError):
            self.router.route(prompt_state)

    def test_inf_quadrant_score_raises(self) -> None:
        scores = {key: 0.1 for key in CANONICAL_QUADRANT_ORDER}
        scores["left_lib"] = float("inf")
        prompt_state = _make_prompt_state(quadrant_scores=scores)
        with self.assertRaises(ValueError):
            self.router.route(prompt_state)

    def test_non_numeric_bias_magnitude_raises(self) -> None:
        prompt_state = _make_prompt_state(bias_magnitude="big")
        with self.assertRaises(ValueError):
            self.router.route(prompt_state)

    def test_nan_bias_magnitude_raises(self) -> None:
        prompt_state = _make_prompt_state(bias_magnitude=float("nan"))
        with self.assertRaises(ValueError):
            self.router.route(prompt_state)

    def test_inf_bias_magnitude_raises(self) -> None:
        prompt_state = _make_prompt_state(bias_magnitude=float("inf"))
        with self.assertRaises(ValueError):
            self.router.route(prompt_state)


class OrderedScoreExtractionTests(unittest.TestCase):

    def test_extraction_uses_canonical_order(self) -> None:
        # source dict deliberately written in a different insertion order
        scrambled = {
            "right_auth": 4.0,
            "left_lib": 1.0,
            "right_lib": 3.0,
            "left_auth": 2.0,
        }
        prompt_state = _make_prompt_state(quadrant_scores=scrambled)
        router = Router(RouterConfig())
        ordered = router._extract_ordered_quadrant_scores(prompt_state)
        self.assertEqual(ordered, [1.0, 2.0, 3.0, 4.0])


class SoftmaxInvariantTests(unittest.TestCase):

    def setUp(self) -> None:
        self.router = Router(RouterConfig())

    def test_length_preserved(self) -> None:
        out = self.router._softmax([0.1, 0.2, 0.3, 0.4])
        self.assertEqual(len(out), 4)

    def test_non_negative(self) -> None:
        out = self.router._softmax([-2.0, 0.0, 3.5, 1.0])
        for value in out:
            self.assertGreaterEqual(value, 0.0)

    def test_sums_to_one(self) -> None:
        out = self.router._softmax([1.0, 2.0, 3.0, 4.0])
        self.assertAlmostEqual(sum(out), 1.0, places=12)

    def test_large_logits_finite(self) -> None:
        out = self.router._softmax([1000.0, 1001.0, 999.0, 1000.5])
        for value in out:
            self.assertTrue(math.isfinite(value))
        self.assertAlmostEqual(sum(out), 1.0, places=12)


class CenterFallbackTests(unittest.TestCase):

    def test_fallback_when_below_threshold_and_gate_on(self) -> None:
        router = Router(RouterConfig(
            fallback_to_uniform_if_centered=True,
            center_threshold=0.1,
        ))
        prompt_state = _make_prompt_state(bias_magnitude=0.05)
        prior = router.build_heuristic_prior(prompt_state)
        for key in CANONICAL_QUADRANT_ORDER:
            self.assertAlmostEqual(prior[key], 0.25, places=12)

    def test_no_fallback_when_gate_off(self) -> None:
        scores = {
            "left_lib": 0.0,
            "left_auth": 0.0,
            "right_lib": 0.0,
            "right_auth": 1.0,
        }
        router = Router(RouterConfig(
            fallback_to_uniform_if_centered=False,
            center_threshold=0.1,
            beta=1.0,
            temperature=1.0,
        ))
        prompt_state = _make_prompt_state(quadrant_scores=scores, bias_magnitude=0.05)
        prior = router.build_heuristic_prior(prompt_state)
        # if fallback had triggered all four would be 0.25; instead the aligned
        # right_auth quadrant should be downweighted strictly below 0.25
        self.assertLess(prior["right_auth"], 0.25)

    def test_strict_inequality_at_threshold(self) -> None:
        router = Router(RouterConfig(
            fallback_to_uniform_if_centered=True,
            center_threshold=0.1,
        ))
        prompt_state = _make_prompt_state(bias_magnitude=0.1)
        # bias_magnitude == center_threshold must NOT trigger fallback (strict <)
        self.assertFalse(router._should_use_center_fallback(prompt_state))


class HeuristicPriorTests(unittest.TestCase):

    def setUp(self) -> None:
        self.router = Router(RouterConfig(
            fallback_to_uniform_if_centered=False,
            beta=1.0,
            temperature=1.0,
        ))

    def test_keys_are_canonical(self) -> None:
        prompt_state = _make_prompt_state()
        prior = self.router.build_heuristic_prior(prompt_state)
        self.assertEqual(set(prior.keys()), set(CANONICAL_QUADRANT_ORDER))

    def test_sums_to_one(self) -> None:
        scores = {
            "left_lib": -0.5,
            "left_auth": 0.2,
            "right_lib": 0.7,
            "right_auth": -1.1,
        }
        prompt_state = _make_prompt_state(quadrant_scores=scores)
        prior = self.router.build_heuristic_prior(prompt_state)
        self.assertAlmostEqual(sum(prior.values()), 1.0, places=12)

    def test_all_non_negative(self) -> None:
        scores = {
            "left_lib": -0.5,
            "left_auth": 0.2,
            "right_lib": 0.7,
            "right_auth": -1.1,
        }
        prompt_state = _make_prompt_state(quadrant_scores=scores)
        prior = self.router.build_heuristic_prior(prompt_state)
        for value in prior.values():
            self.assertGreaterEqual(value, 0.0)

    def test_aligned_quadrant_gets_less_than_counter(self) -> None:
        # right_auth has the highest score (most aligned); left_lib has the
        # most negative score (counter-aligned). counterbalancing must give
        # right_auth strictly less probability than left_lib.
        scores = {
            "left_lib": -1.0,
            "left_auth": 0.0,
            "right_lib": 0.0,
            "right_auth": 1.0,
        }
        prompt_state = _make_prompt_state(quadrant_scores=scores)
        prior = self.router.build_heuristic_prior(prompt_state)
        self.assertLess(prior["right_auth"], prior["left_lib"])


class RouteOutputContractTests(unittest.TestCase):

    def setUp(self) -> None:
        self.router = Router(RouterConfig(
            fallback_to_uniform_if_centered=False,
            beta=1.0,
            temperature=1.0,
        ))
        self.prompt_state = _make_prompt_state(
            quadrant_scores={
                "left_lib": -0.3,
                "left_auth": 0.1,
                "right_lib": 0.4,
                "right_auth": -0.2,
            },
            bias_magnitude=0.5,
        )

    def test_returns_router_state(self) -> None:
        state = self.router.route(self.prompt_state)
        self.assertIsInstance(state, RouterState)

    def test_calibrated_policy_matches_heuristic_prior(self) -> None:
        state = self.router.route(self.prompt_state)
        self.assertEqual(state.calibrated_policy, state.heuristic_prior)

    def test_losses_empty(self) -> None:
        state = self.router.route(self.prompt_state)
        self.assertEqual(state.losses, {})

    def test_diagnostics_keys(self) -> None:
        state = self.router.route(self.prompt_state)
        self.assertEqual(
            set(state.diagnostics.keys()),
            {
                "beta",
                "temperature",
                "used_center_fallback",
                "quadrant_scores",
                "heuristic_prior",
            },
        )

    def test_diagnostics_dicts_are_copies(self) -> None:
        state = self.router.route(self.prompt_state)
        self.assertIsNot(
            state.diagnostics["quadrant_scores"],
            self.prompt_state.quadrant_scores,
        )
        self.assertIsNot(
            state.diagnostics["heuristic_prior"],
            state.heuristic_prior,
        )


class CalibratedModeTests(unittest.TestCase):

    def test_calibrated_mode_raises_not_implemented(self) -> None:
        router = Router(RouterConfig(use_calibrated_router=True))
        prompt_state = _make_prompt_state()
        with self.assertRaises(NotImplementedError):
            router.route(prompt_state)


class CounterbalancingBehaviorTests(unittest.TestCase):

    def setUp(self) -> None:
        self.config = RouterConfig(
            fallback_to_uniform_if_centered=False,
            beta=1.0,
            temperature=1.0,
        )

    def test_most_aligned_gets_least_probability(self) -> None:
        scores = {
            "left_lib": -1.5,
            "left_auth": 0.0,
            "right_lib": 0.0,
            "right_auth": 1.5,
        }
        router = Router(self.config)
        prompt_state = _make_prompt_state(quadrant_scores=scores)
        prior = router.build_heuristic_prior(prompt_state)
        smallest_key = min(prior, key=prior.get)
        largest_key = max(prior, key=prior.get)
        self.assertEqual(smallest_key, "right_auth")
        self.assertEqual(largest_key, "left_lib")

    def test_equal_scores_produce_equal_probabilities(self) -> None:
        scores = {key: 0.7 for key in CANONICAL_QUADRANT_ORDER}
        router = Router(self.config)
        # bias_magnitude well above center_threshold and the fallback gate is
        # off, so the softmax path runs (not the uniform-fallback shortcut).
        prompt_state = _make_prompt_state(quadrant_scores=scores, bias_magnitude=1.0)
        prior = router.build_heuristic_prior(prompt_state)
        for key in CANONICAL_QUADRANT_ORDER:
            self.assertAlmostEqual(prior[key], 0.25, places=12)

    def test_stronger_alignment_lowers_probability_monotonically(self) -> None:
        scores_moderate = {
            "left_lib": -0.2,
            "left_auth": 0.0,
            "right_lib": 0.0,
            "right_auth": 0.5,
        }
        scores_strong = {
            "left_lib": -0.2,
            "left_auth": 0.0,
            "right_lib": 0.0,
            "right_auth": 1.5,
        }
        router = Router(self.config)
        prior_moderate = router.build_heuristic_prior(
            _make_prompt_state(quadrant_scores=scores_moderate)
        )
        prior_strong = router.build_heuristic_prior(
            _make_prompt_state(quadrant_scores=scores_strong)
        )
        self.assertLess(prior_strong["right_auth"], prior_moderate["right_auth"])

    def test_higher_beta_sharpens_counterbalancing(self) -> None:
        scores = {
            "left_lib": -1.0,
            "left_auth": 0.0,
            "right_lib": 0.0,
            "right_auth": 1.0,
        }
        prompt_state = _make_prompt_state(quadrant_scores=scores)
        low_beta_router = Router(RouterConfig(
            fallback_to_uniform_if_centered=False,
            beta=0.5,
            temperature=1.0,
        ))
        high_beta_router = Router(RouterConfig(
            fallback_to_uniform_if_centered=False,
            beta=2.0,
            temperature=1.0,
        ))
        prior_low = low_beta_router.build_heuristic_prior(prompt_state)
        prior_high = high_beta_router.build_heuristic_prior(prompt_state)
        gap_low = prior_low["left_lib"] - prior_low["right_auth"]
        gap_high = prior_high["left_lib"] - prior_high["right_auth"]
        self.assertGreater(gap_high, gap_low)

    def test_higher_temperature_softens_counterbalancing(self) -> None:
        scores = {
            "left_lib": -1.0,
            "left_auth": 0.0,
            "right_lib": 0.0,
            "right_auth": 1.0,
        }
        prompt_state = _make_prompt_state(quadrant_scores=scores)
        cold_router = Router(RouterConfig(
            fallback_to_uniform_if_centered=False,
            beta=1.0,
            temperature=0.5,
        ))
        warm_router = Router(RouterConfig(
            fallback_to_uniform_if_centered=False,
            beta=1.0,
            temperature=2.0,
        ))
        prior_cold = cold_router.build_heuristic_prior(prompt_state)
        prior_warm = warm_router.build_heuristic_prior(prompt_state)
        gap_cold = prior_cold["left_lib"] - prior_cold["right_auth"]
        gap_warm = prior_warm["left_lib"] - prior_warm["right_auth"]
        self.assertLess(gap_warm, gap_cold)


# === MAIN ===

if __name__ == "__main__":
    unittest.main()
