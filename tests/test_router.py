# tests/test_router.py


# === IMPORTS ===

import importlib.util
import math
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

import torch


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
    hidden_representation: Any = None,
) -> PromptState:
    if quadrant_scores is None:
        quadrant_scores = {key: 0.0 for key in CANONICAL_QUADRANT_ORDER}
    return PromptState(
        prompt_text="test prompt",
        hidden_representation=hidden_representation,
        economic_score=economic_score,
        social_score=social_score,
        quadrant_scores=dict(quadrant_scores),
        bias_magnitude=bias_magnitude,
        metadata={},
    )


def _calibrated_router(hidden_dim: int = 4) -> Router:
    return Router(RouterConfig(
        fallback_to_uniform_if_centered=False,
        beta=1.0,
        temperature=1.0,
        use_calibrated_router=True,
        router_hidden_dim=hidden_dim,
    ))


def _zero_calibration(router: Router) -> None:
    # zero out the linear correction head so delta(h) = 0 deterministically
    with torch.no_grad():
        router.calibration_module.weight.zero_()
        router.calibration_module.bias.zero_()


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


class CalibratedRouteTests(unittest.TestCase):

    def setUp(self) -> None:
        self.hidden_dim = 4
        self.router = _calibrated_router(hidden_dim=self.hidden_dim)
        _zero_calibration(self.router)
        self.skewed_scores = {
            "left_lib": -0.3,
            "left_auth": 0.1,
            "right_lib": 0.4,
            "right_auth": -0.2,
        }

    # --- A. missing hidden_representation -------------------------------------

    def test_missing_hidden_representation_raises(self) -> None:
        prompt_state = _make_prompt_state(
            quadrant_scores=self.skewed_scores,
            bias_magnitude=0.5,
            hidden_representation=None,
        )
        with self.assertRaisesRegex(ValueError, "hidden_representation"):
            self.router.route(prompt_state)

    # --- B. invalid hidden_representation -------------------------------------

    def test_rank_two_tensor_raises(self) -> None:
        prompt_state = _make_prompt_state(
            quadrant_scores=self.skewed_scores,
            hidden_representation=torch.zeros((2, self.hidden_dim)),
        )
        with self.assertRaises(ValueError):
            self.router.route(prompt_state)

    def test_nested_list_raises(self) -> None:
        prompt_state = _make_prompt_state(
            quadrant_scores=self.skewed_scores,
            hidden_representation=[[0.0] * self.hidden_dim],
        )
        with self.assertRaises(ValueError):
            self.router.route(prompt_state)

    def test_wrong_length_raises(self) -> None:
        prompt_state = _make_prompt_state(
            quadrant_scores=self.skewed_scores,
            hidden_representation=[0.0] * (self.hidden_dim - 1),
        )
        with self.assertRaises(ValueError):
            self.router.route(prompt_state)

    def test_non_finite_value_raises(self) -> None:
        prompt_state = _make_prompt_state(
            quadrant_scores=self.skewed_scores,
            hidden_representation=[0.0, float("nan"), 0.0, 0.0],
        )
        with self.assertRaises(ValueError):
            self.router.route(prompt_state)

    # --- C. zero correction reproduces heuristic prior ------------------------

    def test_zero_correction_reproduces_heuristic_prior(self) -> None:
        prompt_state = _make_prompt_state(
            quadrant_scores=self.skewed_scores,
            bias_magnitude=0.5,
            hidden_representation=[0.7, -0.3, 1.2, 0.0],
        )
        state = self.router.route(prompt_state)
        for key in CANONICAL_QUADRANT_ORDER:
            self.assertAlmostEqual(
                state.calibrated_policy[key],
                state.heuristic_prior[key],
                places=10,
            )
        self.assertAlmostEqual(state.losses["kl"], 0.0, places=12)
        self.assertIn("entropy", state.losses)
        self.assertTrue(math.isfinite(state.losses["entropy"]))

    # --- D. non-zero correction shifts the calibrated policy ------------------

    def test_nonzero_bias_increases_favored_quadrant(self) -> None:
        # zero weights, then push bias on the left_lib output (canonical index 0)
        with torch.no_grad():
            self.router.calibration_module.bias[0] = 5.0
        prompt_state = _make_prompt_state(
            quadrant_scores=self.skewed_scores,
            bias_magnitude=0.5,
            hidden_representation=[0.0] * self.hidden_dim,
        )
        state = self.router.route(prompt_state)
        self.assertNotEqual(state.calibrated_policy, state.heuristic_prior)
        self.assertGreater(
            state.calibrated_policy["left_lib"],
            state.heuristic_prior["left_lib"],
        )

    # --- E. diagnostics keys + copy semantics ---------------------------------

    def test_diagnostics_keys_and_copies(self) -> None:
        prompt_state = _make_prompt_state(
            quadrant_scores=self.skewed_scores,
            bias_magnitude=0.5,
            hidden_representation=[0.1, 0.2, 0.3, 0.4],
        )
        state = self.router.route(prompt_state)
        self.assertEqual(
            set(state.diagnostics.keys()),
            {
                "beta",
                "temperature",
                "used_center_fallback",
                "quadrant_scores",
                "heuristic_prior",
                "correction_logits",
                "calibrated_policy",
            },
        )
        # value equality
        self.assertEqual(state.diagnostics["quadrant_scores"], prompt_state.quadrant_scores)
        self.assertEqual(state.diagnostics["heuristic_prior"], state.heuristic_prior)
        self.assertEqual(state.diagnostics["calibrated_policy"], state.calibrated_policy)
        # not aliasing the source / returned dicts
        self.assertIsNot(state.diagnostics["quadrant_scores"], prompt_state.quadrant_scores)
        self.assertIsNot(state.diagnostics["heuristic_prior"], state.heuristic_prior)
        self.assertIsNot(state.diagnostics["calibrated_policy"], state.calibrated_policy)
        # correction_logits is a fresh dict with canonical keys
        self.assertIsInstance(state.diagnostics["correction_logits"], dict)
        self.assertEqual(
            set(state.diagnostics["correction_logits"].keys()),
            set(CANONICAL_QUADRANT_ORDER),
        )

    # --- F. losses present with correct keys ----------------------------------

    def test_losses_have_kl_and_entropy(self) -> None:
        prompt_state = _make_prompt_state(
            quadrant_scores=self.skewed_scores,
            bias_magnitude=0.5,
            hidden_representation=[0.1, 0.2, 0.3, 0.4],
        )
        state = self.router.route(prompt_state)
        self.assertEqual(set(state.losses.keys()), {"kl", "entropy"})
        self.assertIsInstance(state.losses["kl"], float)
        self.assertIsInstance(state.losses["entropy"], float)
        self.assertTrue(math.isfinite(state.losses["kl"]))
        self.assertTrue(math.isfinite(state.losses["entropy"]))


class ComputeRouterCorrectionTests(unittest.TestCase):

    def setUp(self) -> None:
        self.hidden_dim = 4
        self.router = _calibrated_router(hidden_dim=self.hidden_dim)
        _zero_calibration(self.router)

    def test_accepts_torch_tensor(self) -> None:
        prompt_state = _make_prompt_state(
            hidden_representation=torch.tensor([0.5, -0.5, 1.0, 0.0]),
        )
        out = self.router.compute_router_correction(prompt_state)
        self.assertEqual(list(out.keys()), list(CANONICAL_QUADRANT_ORDER))

    def test_accepts_python_list(self) -> None:
        prompt_state = _make_prompt_state(
            hidden_representation=[0.5, -0.5, 1.0, 0.0],
        )
        out = self.router.compute_router_correction(prompt_state)
        self.assertEqual(list(out.keys()), list(CANONICAL_QUADRANT_ORDER))

    def test_accepts_python_tuple(self) -> None:
        prompt_state = _make_prompt_state(
            hidden_representation=(0.5, -0.5, 1.0, 0.0),
        )
        out = self.router.compute_router_correction(prompt_state)
        self.assertEqual(list(out.keys()), list(CANONICAL_QUADRANT_ORDER))

    def test_zeroed_layer_returns_zero_logits(self) -> None:
        prompt_state = _make_prompt_state(
            hidden_representation=[0.7, -0.3, 1.2, 0.0],
        )
        out = self.router.compute_router_correction(prompt_state)
        for key in CANONICAL_QUADRANT_ORDER:
            self.assertAlmostEqual(out[key], 0.0, places=10)

    def test_bias_only_returns_bias_values(self) -> None:
        # zero weights but distinct biases per output dim; canonical order is
        # (left_lib, left_auth, right_lib, right_auth) -> bias indices 0..3
        with torch.no_grad():
            self.router.calibration_module.bias.copy_(
                torch.tensor([1.0, -2.0, 0.5, 3.0])
            )
        prompt_state = _make_prompt_state(
            # hidden values are arbitrary; weights are zero so they do not matter
            hidden_representation=[10.0, -5.0, 7.0, 2.0],
        )
        out = self.router.compute_router_correction(prompt_state)
        expected = {
            "left_lib": 1.0,
            "left_auth": -2.0,
            "right_lib": 0.5,
            "right_auth": 3.0,
        }
        for key in CANONICAL_QUADRANT_ORDER:
            self.assertAlmostEqual(out[key], expected[key], places=10)


class CombinePriorAndCorrectionTests(unittest.TestCase):

    def setUp(self) -> None:
        # combine_prior_and_correction is pure-python and works on any router
        self.router = Router(RouterConfig())

    def test_zero_correction_returns_prior(self) -> None:
        prior = {
            "left_lib": 0.4,
            "left_auth": 0.1,
            "right_lib": 0.3,
            "right_auth": 0.2,
        }
        zero_correction = {key: 0.0 for key in CANONICAL_QUADRANT_ORDER}
        out = self.router.combine_prior_and_correction(prior, zero_correction)
        for key in CANONICAL_QUADRANT_ORDER:
            self.assertAlmostEqual(out[key], prior[key], places=12)

    def test_increasing_one_logit_increases_that_quadrant(self) -> None:
        prior = {key: 0.25 for key in CANONICAL_QUADRANT_ORDER}
        zero_correction = {key: 0.0 for key in CANONICAL_QUADRANT_ORDER}
        boosted_correction = {key: 0.0 for key in CANONICAL_QUADRANT_ORDER}
        boosted_correction["left_lib"] = 2.0

        out_zero = self.router.combine_prior_and_correction(prior, zero_correction)
        out_boosted = self.router.combine_prior_and_correction(prior, boosted_correction)

        self.assertGreater(out_boosted["left_lib"], out_zero["left_lib"])
        self.assertAlmostEqual(sum(out_boosted.values()), 1.0, places=12)


class ComputeRouterLossesTests(unittest.TestCase):

    def setUp(self) -> None:
        self.router = Router(RouterConfig())

    def test_identical_distributions_give_zero_kl(self) -> None:
        distribution = {key: 0.25 for key in CANONICAL_QUADRANT_ORDER}
        out = self.router.compute_router_losses(distribution, distribution)
        self.assertAlmostEqual(out["kl"], 0.0, places=12)

    def test_entropy_finite_and_positive_for_non_degenerate_policy(self) -> None:
        prior = {key: 0.25 for key in CANONICAL_QUADRANT_ORDER}
        policy = {
            "left_lib": 0.4,
            "left_auth": 0.1,
            "right_lib": 0.3,
            "right_auth": 0.2,
        }
        out = self.router.compute_router_losses(prior, policy)
        self.assertTrue(math.isfinite(out["entropy"]))
        self.assertGreater(out["entropy"], 0.0)


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


class CalibrationCheckpointTests(unittest.TestCase):

    def setUp(self) -> None:
        self.hidden_dim = 4
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _save_checkpoint(self, payload: dict[str, Any], name: str = "ckpt.pt") -> Path:
        path = self.tmp_path / name
        torch.save(payload, path)
        return path

    def _state_dict_filled(
        self,
        hidden_dim: int,
        weight_value: float,
        bias_value: float,
    ) -> dict[str, Any]:
        # build a deterministic donor layer of the right shape, then return
        # its state_dict so we can save it into a synthetic checkpoint
        layer = torch.nn.Linear(hidden_dim, len(CANONICAL_QUADRANT_ORDER))
        with torch.no_grad():
            layer.weight.fill_(weight_value)
            layer.bias.fill_(bias_value)
        return layer.state_dict()

    def test_successful_load(self) -> None:
        router = _calibrated_router(hidden_dim=self.hidden_dim)
        state_dict = self._state_dict_filled(self.hidden_dim, 0.5, 0.25)
        path = self._save_checkpoint({
            "state_dict": state_dict,
            "router_hidden_dim": self.hidden_dim,
            "canonical_quadrant_order": list(CANONICAL_QUADRANT_ORDER),
            "beta": 1.5,
            "temperature": 0.75,
        })

        # precondition: pre-load metadata is None
        self.assertIsNone(router.calibration_checkpoint_metadata)

        router.load_calibration_checkpoint(path)

        expected_w = torch.full_like(router.calibration_module.weight, 0.5)
        expected_b = torch.full_like(router.calibration_module.bias, 0.25)
        self.assertTrue(torch.allclose(router.calibration_module.weight, expected_w))
        self.assertTrue(torch.allclose(router.calibration_module.bias, expected_b))

        meta = router.calibration_checkpoint_metadata
        self.assertIsNotNone(meta)
        self.assertEqual(meta["checkpoint_path"], str(path))
        self.assertEqual(meta["router_hidden_dim"], self.hidden_dim)
        self.assertEqual(meta["canonical_quadrant_order"], list(CANONICAL_QUADRANT_ORDER))
        self.assertEqual(meta["beta"], 1.5)
        self.assertEqual(meta["temperature"], 0.75)

    def test_heuristic_router_rejects_load(self) -> None:
        router = Router(RouterConfig())  # use_calibrated_router=False
        path = self._save_checkpoint({
            "state_dict": self._state_dict_filled(self.hidden_dim, 0.0, 0.0),
            "router_hidden_dim": self.hidden_dim,
            "canonical_quadrant_order": list(CANONICAL_QUADRANT_ORDER),
        })
        with self.assertRaisesRegex(ValueError, "calibration"):
            router.load_calibration_checkpoint(path)

    def test_hidden_dim_mismatch_raises(self) -> None:
        router = _calibrated_router(hidden_dim=self.hidden_dim)
        bad_dim = self.hidden_dim + 1
        path = self._save_checkpoint({
            "state_dict": self._state_dict_filled(bad_dim, 0.0, 0.0),
            "router_hidden_dim": bad_dim,
            "canonical_quadrant_order": list(CANONICAL_QUADRANT_ORDER),
        })
        with self.assertRaisesRegex(ValueError, "router_hidden_dim"):
            router.load_calibration_checkpoint(path)
        # failed load must not populate metadata
        self.assertIsNone(router.calibration_checkpoint_metadata)

    def test_canonical_order_mismatch_raises(self) -> None:
        router = _calibrated_router(hidden_dim=self.hidden_dim)
        scrambled = ["right_auth", "right_lib", "left_auth", "left_lib"]
        path = self._save_checkpoint({
            "state_dict": self._state_dict_filled(self.hidden_dim, 0.0, 0.0),
            "router_hidden_dim": self.hidden_dim,
            "canonical_quadrant_order": scrambled,
        })
        with self.assertRaisesRegex(ValueError, "canonical_quadrant_order"):
            router.load_calibration_checkpoint(path)
        self.assertIsNone(router.calibration_checkpoint_metadata)

    def test_missing_required_key_raises(self) -> None:
        router = _calibrated_router(hidden_dim=self.hidden_dim)
        # omit state_dict
        path = self._save_checkpoint({
            "router_hidden_dim": self.hidden_dim,
            "canonical_quadrant_order": list(CANONICAL_QUADRANT_ORDER),
        })
        with self.assertRaisesRegex(ValueError, "state_dict"):
            router.load_calibration_checkpoint(path)
        self.assertIsNone(router.calibration_checkpoint_metadata)

    def test_missing_file_raises(self) -> None:
        router = _calibrated_router(hidden_dim=self.hidden_dim)
        with self.assertRaises(FileNotFoundError):
            router.load_calibration_checkpoint(self.tmp_path / "does_not_exist.pt")


# === MAIN ===

if __name__ == "__main__":
    unittest.main()
