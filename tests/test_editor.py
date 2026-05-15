# tests/test_editor.py


# === IMPORTS ===

import importlib.util
import sys
import unittest
from pathlib import Path
from typing import Any

import torch


# === MODULE LOADING ===

# src/09_moce_components.py starts with a digit, so it cannot be imported via
# normal "import" syntax. load it explicitly by absolute path with importlib.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_MODULE_PATH = _REPO_ROOT / "src" / "09_moce_components.py"

_spec = importlib.util.spec_from_file_location("moce_components", _MODULE_PATH)
moce_components = importlib.util.module_from_spec(_spec)
sys.modules["moce_components"] = moce_components
_spec.loader.exec_module(moce_components)

Editor = moce_components.Editor
EditorConfig = moce_components.EditorConfig
EditorStepTrace = moce_components.EditorStepTrace
EditorResult = moce_components.EditorResult
GenerationConfig = moce_components.GenerationConfig
PromptState = moce_components.PromptState
RouterState = moce_components.RouterState
ExpertOutput = moce_components.ExpertOutput
CANONICAL_QUADRANT_ORDER = moce_components.CANONICAL_QUADRANT_ORDER


# === FAKES ===

class _FakeInputTransformer:
    """
    Minimal stand-in for InputTransformer covering exactly the methods
    Editor.score_current_mixture(...) calls. Every method returns
    deterministic values; tests can override any field by passing it
    to the constructor.
    """

    def __init__(
        self,
        axis_scores: dict[str, float] | None = None,
        quadrant_scores: dict[str, float] | None = None,
        bias_magnitude: float | None = None,
    ) -> None:
        self._axis_scores = axis_scores
        self._quadrant_scores = quadrant_scores
        self._bias_magnitude = bias_magnitude

    def maybe_center_representation(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def compute_axis_scores(self, x: torch.Tensor) -> dict[str, float]:
        if self._axis_scores is not None:
            return dict(self._axis_scores)
        flat = x.flatten()
        return {
            "economic_score": float(flat[0]),
            "social_score": float(flat[1]),
        }

    def compute_quadrant_scores(self, x: torch.Tensor) -> dict[str, float]:
        if self._quadrant_scores is not None:
            return dict(self._quadrant_scores)
        flat = x.flatten()
        return {
            key: float(flat[index])
            for index, key in enumerate(CANONICAL_QUADRANT_ORDER)
        }

    def compute_bias_magnitude(
        self, economic_score: float, social_score: float
    ) -> float:
        if self._bias_magnitude is not None:
            return self._bias_magnitude
        return float((economic_score ** 2 + social_score ** 2) ** 0.5)


# === EDITOR CONSTRUCTION ===

class _TestEditor(Editor):
    """
    Bypass Editor.__init__ (still NotImplementedError in production) and
    attach exactly the attributes the implemented methods touch. The
    real __init__ also takes model/tokenizer/generation_config, none of
    which are exercised by the methods under test.
    """

    def __init__(
        self,
        input_transformer: Any,
        config: EditorConfig,
    ) -> None:
        self.input_transformer = input_transformer
        self.config = config


# === HELPERS ===

def _make_router_state(
    calibrated_policy: dict[str, float] | None = None,
) -> RouterState:
    if calibrated_policy is None:
        calibrated_policy = {key: 0.25 for key in CANONICAL_QUADRANT_ORDER}
    return RouterState(
        heuristic_prior=dict(calibrated_policy),
        calibrated_policy=dict(calibrated_policy),
        diagnostics={},
        losses={},
    )


def _make_prompt_state(
    quadrant_scores: dict[str, float] | None = None,
) -> PromptState:
    if quadrant_scores is None:
        quadrant_scores = {key: 0.0 for key in CANONICAL_QUADRANT_ORDER}
    return PromptState(
        prompt_text="test prompt",
        hidden_representation=None,
        economic_score=0.0,
        social_score=0.0,
        quadrant_scores=dict(quadrant_scores),
        bias_magnitude=0.0,
        metadata={},
    )


def _make_expert_outputs(
    tensors_by_quadrant: dict[str, torch.Tensor] | None = None,
    shape: tuple[int, ...] = (3, 4),
    fill: float = 1.0,
) -> dict[str, ExpertOutput]:
    # default hidden_output is rank-2 [seq_len, hidden_dim], matching the
    # ExpertManager contract _validate_expert_outputs enforces. fill is
    # non-zero so the mixed hidden state has a positive L2 norm and
    # score_current_mixture can normalize it.
    if tensors_by_quadrant is None:
        tensors_by_quadrant = {
            key: torch.full(shape, fill) for key in CANONICAL_QUADRANT_ORDER
        }
    return {
        key: ExpertOutput(
            expert_name=key,
            hidden_output=tensors_by_quadrant[key],
        )
        for key in CANONICAL_QUADRANT_ORDER
    }


def _uniform_alpha() -> dict[str, float]:
    n = len(CANONICAL_QUADRANT_ORDER)
    return {key: 1.0 / n for key in CANONICAL_QUADRANT_ORDER}


def _zero_alignment() -> dict[str, float]:
    return {key: 0.0 for key in CANONICAL_QUADRANT_ORDER}


def _default_editor(
    initialization_mode: str = "router_policy",
    use_recursive_editing: bool = True,
    max_edit_steps: int = 1,
    correction_beta: float = 1.0,
    convergence_threshold: float = 1e-3,
    keep_edit_trace: bool = True,
    fake: _FakeInputTransformer | None = None,
) -> Editor:
    config = EditorConfig(
        max_edit_steps=max_edit_steps,
        use_recursive_editing=use_recursive_editing,
        correction_beta=correction_beta,
        convergence_threshold=convergence_threshold,
        keep_edit_trace=keep_edit_trace,
        initialization_mode=initialization_mode,
    )
    transformer = fake if fake is not None else _FakeInputTransformer()
    return _TestEditor(input_transformer=transformer, config=config)


# === TESTS ===

class ValidationFailureTests(unittest.TestCase):

    def test_invalid_initialization_mode_raises(self) -> None:
        editor = _default_editor(initialization_mode="not_a_mode")
        with self.assertRaises(ValueError):
            editor.initialize_editor_weights(_make_router_state())

    def test_malformed_quadrant_scores_in_run_editing_loop_raises(self) -> None:
        editor = _default_editor()
        bad_quad = {key: 0.0 for key in CANONICAL_QUADRANT_ORDER}
        bad_quad["left_lib"] = float("nan")
        with self.assertRaises(ValueError):
            editor.run_editing_loop(
                "p",
                _make_prompt_state(quadrant_scores=bad_quad),
                _make_router_state(),
                _make_expert_outputs(),
            )

    def test_malformed_calibrated_policy_in_run_editing_loop_raises(self) -> None:
        editor = _default_editor()
        bad_policy = {key: 0.25 for key in CANONICAL_QUADRANT_ORDER}
        bad_policy["right_auth"] = -0.25  # not strictly positive
        with self.assertRaises(ValueError):
            editor.run_editing_loop(
                "p",
                _make_prompt_state(),
                _make_router_state(calibrated_policy=bad_policy),
                _make_expert_outputs(),
            )

    def test_expert_outputs_missing_quadrant_raises(self) -> None:
        editor = _default_editor()
        expert_outputs = _make_expert_outputs()
        del expert_outputs["right_auth"]
        with self.assertRaises(ValueError):
            editor.run_editing_loop(
                "p",
                _make_prompt_state(),
                _make_router_state(),
                expert_outputs,
            )

    def test_expert_hidden_output_none_raises(self) -> None:
        editor = _default_editor()
        expert_outputs = _make_expert_outputs()
        expert_outputs["left_lib"] = ExpertOutput(
            expert_name="left_lib", hidden_output=None
        )
        with self.assertRaises(ValueError):
            editor.run_editing_loop(
                "p",
                _make_prompt_state(),
                _make_router_state(),
                expert_outputs,
            )

    def test_expert_tensors_shape_mismatch_raises(self) -> None:
        editor = _default_editor()
        tensors = {key: torch.zeros(3, 4) for key in CANONICAL_QUADRANT_ORDER}
        tensors["right_auth"] = torch.zeros(3, 8)
        with self.assertRaises(ValueError):
            editor.run_editing_loop(
                "p",
                _make_prompt_state(),
                _make_router_state(),
                _make_expert_outputs(tensors_by_quadrant=tensors),
            )


class InitializationTests(unittest.TestCase):

    def test_router_policy_returns_calibrated_distribution(self) -> None:
        editor = _default_editor(initialization_mode="router_policy")
        policy = {
            "left_lib": 0.4,
            "left_auth": 0.3,
            "right_lib": 0.2,
            "right_auth": 0.1,
        }
        alpha = editor.initialize_editor_weights(
            _make_router_state(calibrated_policy=policy)
        )
        for key in CANONICAL_QUADRANT_ORDER:
            self.assertAlmostEqual(alpha[key], policy[key], places=12)

    def test_uniform_returns_equal_weights(self) -> None:
        editor = _default_editor(initialization_mode="uniform")
        alpha = editor.initialize_editor_weights(_make_router_state())
        for key in CANONICAL_QUADRANT_ORDER:
            self.assertAlmostEqual(alpha[key], 0.25, places=12)

    def test_router_policy_returns_fresh_copy(self) -> None:
        editor = _default_editor(initialization_mode="router_policy")
        router_state = _make_router_state()
        alpha = editor.initialize_editor_weights(router_state)
        self.assertIsNot(alpha, router_state.calibrated_policy)
        # mutating the returned dict must not bleed into the source
        alpha["left_lib"] = 0.99
        self.assertAlmostEqual(
            router_state.calibrated_policy["left_lib"], 0.25, places=12
        )


class DeltaUpdateTests(unittest.TestCase):

    def test_delta_negates_alignment_scaled_by_beta(self) -> None:
        editor = _default_editor(correction_beta=2.0)
        alignment = {
            "left_lib": 0.5,
            "left_auth": -0.25,
            "right_lib": 1.5,
            "right_auth": 0.0,
        }
        delta = editor._compute_delta_from_alignment(alignment)
        for key in CANONICAL_QUADRANT_ORDER:
            self.assertAlmostEqual(delta[key], -2.0 * alignment[key], places=12)

    def test_zero_delta_preserves_alpha(self) -> None:
        editor = _default_editor()
        alpha = {
            "left_lib": 0.4,
            "left_auth": 0.3,
            "right_lib": 0.2,
            "right_auth": 0.1,
        }
        zero_delta = {key: 0.0 for key in CANONICAL_QUADRANT_ORDER}
        next_alpha = editor._update_alpha(alpha, zero_delta)
        for key in CANONICAL_QUADRANT_ORDER:
            self.assertAlmostEqual(next_alpha[key], alpha[key], places=12)

    def test_increasing_one_delta_increases_that_alpha(self) -> None:
        editor = _default_editor()
        alpha = _uniform_alpha()
        zero_delta = {key: 0.0 for key in CANONICAL_QUADRANT_ORDER}
        boosted_delta = dict(zero_delta)
        boosted_delta["right_auth"] = 1.0
        baseline = editor._update_alpha(alpha, zero_delta)
        boosted = editor._update_alpha(alpha, boosted_delta)
        self.assertGreater(boosted["right_auth"], baseline["right_auth"])
        # softmax conservation: holding alpha and the other deltas fixed,
        # the remaining quadrants must give up mass
        for key in ("left_lib", "left_auth", "right_lib"):
            self.assertLess(boosted[key], baseline[key])


class HiddenStateFusionTests(unittest.TestCase):

    def test_weighted_sum_matches_expected(self) -> None:
        editor = _default_editor()
        tensors = {
            "left_lib":   torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
            "left_auth":  torch.tensor([[0.0, 2.0, 0.0, 0.0]]),
            "right_lib":  torch.tensor([[0.0, 0.0, 3.0, 0.0]]),
            "right_auth": torch.tensor([[0.0, 0.0, 0.0, 4.0]]),
        }
        alpha = {
            "left_lib": 0.4,
            "left_auth": 0.3,
            "right_lib": 0.2,
            "right_auth": 0.1,
        }
        mixed = editor._mix_hidden_states(
            _make_expert_outputs(tensors_by_quadrant=tensors), alpha
        )
        expected = torch.tensor(
            [[0.4 * 1.0, 0.3 * 2.0, 0.2 * 3.0, 0.1 * 4.0]]
        )
        self.assertTrue(torch.allclose(mixed, expected, atol=1e-7))

    def test_output_shape_matches_inputs(self) -> None:
        editor = _default_editor()
        mixed = editor._mix_hidden_states(
            _make_expert_outputs(shape=(2, 3)), _uniform_alpha()
        )
        self.assertEqual(tuple(mixed.shape), (2, 3))

    def test_non_finite_expert_tensor_rejected(self) -> None:
        editor = _default_editor()
        tensors = {key: torch.zeros(1, 4) for key in CANONICAL_QUADRANT_ORDER}
        tensors["left_lib"] = torch.tensor([[0.0, float("nan"), 0.0, 0.0]])
        with self.assertRaises(ValueError):
            editor._mix_hidden_states(
                _make_expert_outputs(tensors_by_quadrant=tensors),
                _uniform_alpha(),
            )


class ScoreCurrentMixtureTests(unittest.TestCase):

    def test_returns_full_diagnostics(self) -> None:
        fake = _FakeInputTransformer(
            axis_scores={"economic_score": 0.4, "social_score": -0.3},
            quadrant_scores={
                "left_lib": 0.1,
                "left_auth": 0.2,
                "right_lib": 0.3,
                "right_auth": 0.4,
            },
            bias_magnitude=0.5,
        )
        editor = _default_editor(fake=fake)
        scores = editor.score_current_mixture(torch.tensor([1.0, 2.0, 3.0, 4.0]))
        self.assertEqual(
            set(scores.keys()),
            {"economic_score", "social_score", "quadrant_scores", "bias_magnitude"},
        )
        self.assertAlmostEqual(scores["economic_score"], 0.4, places=12)
        self.assertAlmostEqual(scores["social_score"], -0.3, places=12)
        self.assertAlmostEqual(scores["bias_magnitude"], 0.5, places=12)
        self.assertEqual(
            set(scores["quadrant_scores"].keys()),
            set(CANONICAL_QUADRANT_ORDER),
        )
        self.assertIsInstance(scores["economic_score"], float)
        self.assertIsInstance(scores["social_score"], float)
        self.assertIsInstance(scores["bias_magnitude"], float)

    def test_quadrant_scores_rebuilt_in_canonical_order(self) -> None:
        fake = _FakeInputTransformer(
            quadrant_scores={
                # deliberately scrambled insertion order
                "right_auth": 4.0,
                "left_lib": 1.0,
                "right_lib": 3.0,
                "left_auth": 2.0,
            },
        )
        editor = _default_editor(fake=fake)
        scores = editor.score_current_mixture(torch.tensor([1.0, 2.0, 3.0, 4.0]))
        self.assertEqual(
            list(scores["quadrant_scores"].keys()),
            list(CANONICAL_QUADRANT_ORDER),
        )

    def test_missing_economic_score_raises(self) -> None:
        fake = _FakeInputTransformer(
            axis_scores={"social_score": 0.0},  # missing economic_score
        )
        editor = _default_editor(fake=fake)
        with self.assertRaises(ValueError):
            editor.score_current_mixture(torch.tensor([1.0, 2.0, 3.0, 4.0]))

    def test_negative_bias_magnitude_raises(self) -> None:
        fake = _FakeInputTransformer(bias_magnitude=-0.1)
        editor = _default_editor(fake=fake)
        with self.assertRaises(ValueError):
            editor.score_current_mixture(torch.tensor([1.0, 2.0, 3.0, 4.0]))

    def test_non_tensor_input_raises(self) -> None:
        editor = _default_editor()
        with self.assertRaises(ValueError):
            editor.score_current_mixture([0.0, 0.0, 0.0, 0.0])  # type: ignore[arg-type]

    def test_zero_norm_input_raises(self) -> None:
        # a zero mixed hidden state has L2 norm 0 and cannot be normalized
        # onto the unit-norm scale PromptState scoring uses
        editor = _default_editor()
        with self.assertRaises(ValueError):
            editor.score_current_mixture(torch.zeros(4))


class EditLoopTests(unittest.TestCase):

    def _converging_editor(
        self,
        max_edit_steps: int = 5,
        keep_edit_trace: bool = True,
    ) -> Editor:
        # the fake returns alignment -> [0,0,0,0] always and prompt
        # alignment is zero; alpha and alignment both freeze on step 1.
        fake = _FakeInputTransformer(
            quadrant_scores={key: 0.0 for key in CANONICAL_QUADRANT_ORDER},
        )
        return _default_editor(
            initialization_mode="uniform",
            use_recursive_editing=True,
            max_edit_steps=max_edit_steps,
            convergence_threshold=1e-9,
            keep_edit_trace=keep_edit_trace,
            fake=fake,
        )

    def _non_converging_editor(
        self,
        max_edit_steps: int = 3,
        keep_edit_trace: bool = True,
    ) -> Editor:
        # constant non-zero alignment from the fake. Per-step alpha keeps
        # evolving toward a fixed point but does not reach 1e-9 in the
        # configured budget.
        fake = _FakeInputTransformer(
            quadrant_scores={
                "left_lib":   0.5,
                "left_auth":  0.0,
                "right_lib":  0.0,
                "right_auth": 0.0,
            },
        )
        return _default_editor(
            initialization_mode="uniform",
            use_recursive_editing=True,
            max_edit_steps=max_edit_steps,
            correction_beta=1.0,
            convergence_threshold=1e-9,
            keep_edit_trace=keep_edit_trace,
            fake=fake,
        )

    def test_non_recursive_runs_exactly_one_step(self) -> None:
        fake = _FakeInputTransformer(
            quadrant_scores={key: 0.0 for key in CANONICAL_QUADRANT_ORDER},
        )
        editor = _default_editor(
            initialization_mode="uniform",
            use_recursive_editing=False,
            max_edit_steps=5,
            convergence_threshold=1e-9,
            fake=fake,
        )
        result = editor._run_edit_loop(
            initial_alpha=_uniform_alpha(),
            initial_alignment=_zero_alignment(),
            expert_outputs=_make_expert_outputs(),
        )
        self.assertEqual(result.num_steps_run, 1)
        self.assertFalse(result.stopped_early)
        self.assertIsNone(result.stop_reason)

    def test_recursive_respects_max_edit_steps(self) -> None:
        editor = self._non_converging_editor(max_edit_steps=3)
        result = editor._run_edit_loop(
            initial_alpha=_uniform_alpha(),
            initial_alignment=_zero_alignment(),
            expert_outputs=_make_expert_outputs(),
        )
        self.assertEqual(result.num_steps_run, 3)
        self.assertFalse(result.stopped_early)
        self.assertIsNone(result.stop_reason)

    def test_early_stop_on_convergence(self) -> None:
        editor = self._converging_editor(max_edit_steps=5)
        result = editor._run_edit_loop(
            initial_alpha=_uniform_alpha(),
            initial_alignment=_zero_alignment(),
            expert_outputs=_make_expert_outputs(),
        )
        self.assertEqual(result.num_steps_run, 1)
        self.assertTrue(result.stopped_early)
        self.assertEqual(result.stop_reason, "converged")

    def test_editor_result_shape(self) -> None:
        editor = self._converging_editor()
        result = editor._run_edit_loop(
            initial_alpha=_uniform_alpha(),
            initial_alignment=_zero_alignment(),
            expert_outputs=_make_expert_outputs(),
        )
        self.assertIsInstance(result, EditorResult)
        self.assertIsInstance(result.final_mixed_hidden_state, torch.Tensor)
        self.assertEqual(
            set(result.final_alpha.keys()), set(CANONICAL_QUADRANT_ORDER)
        )
        self.assertEqual(
            set(result.final_alignment.keys()), set(CANONICAL_QUADRANT_ORDER)
        )
        self.assertIsInstance(result.num_steps_run, int)
        self.assertIsInstance(result.stopped_early, bool)


class TraceBehaviorTests(unittest.TestCase):

    def test_trace_kept_when_enabled(self) -> None:
        fake = _FakeInputTransformer(
            quadrant_scores={
                "left_lib":   0.5,
                "left_auth":  0.0,
                "right_lib":  0.0,
                "right_auth": 0.0,
            },
        )
        editor = _default_editor(
            initialization_mode="uniform",
            use_recursive_editing=True,
            max_edit_steps=2,
            convergence_threshold=1e-9,
            keep_edit_trace=True,
            fake=fake,
        )
        result = editor._run_edit_loop(
            initial_alpha=_uniform_alpha(),
            initial_alignment=_zero_alignment(),
            expert_outputs=_make_expert_outputs(),
        )
        self.assertEqual(len(result.step_traces), result.num_steps_run)
        for trace in result.step_traces:
            self.assertIsInstance(trace, EditorStepTrace)

    def test_trace_empty_when_disabled(self) -> None:
        fake = _FakeInputTransformer(
            quadrant_scores={key: 0.0 for key in CANONICAL_QUADRANT_ORDER},
        )
        editor = _default_editor(
            initialization_mode="uniform",
            use_recursive_editing=True,
            max_edit_steps=2,
            convergence_threshold=1e-9,
            keep_edit_trace=False,
            fake=fake,
        )
        result = editor._run_edit_loop(
            initial_alpha=_uniform_alpha(),
            initial_alignment=_zero_alignment(),
            expert_outputs=_make_expert_outputs(),
        )
        self.assertEqual(result.step_traces, [])

    def test_trace_dicts_are_copies_not_aliases(self) -> None:
        fake = _FakeInputTransformer(
            quadrant_scores={
                "left_lib":   0.5,
                "left_auth":  0.0,
                "right_lib":  0.0,
                "right_auth": 0.0,
            },
        )
        editor = _default_editor(
            initialization_mode="uniform",
            use_recursive_editing=True,
            max_edit_steps=1,
            convergence_threshold=1e-9,
            keep_edit_trace=True,
            fake=fake,
        )
        result = editor._run_edit_loop(
            initial_alpha=_uniform_alpha(),
            initial_alignment=_zero_alignment(),
            expert_outputs=_make_expert_outputs(),
        )
        snapshot = dict(result.step_traces[0].alpha_after)
        # mutate the returned final dicts; the trace must be unaffected
        for key in result.final_alpha:
            result.final_alpha[key] = -999.0
        for key in result.final_alignment:
            result.final_alignment[key] = -999.0
        for key in CANONICAL_QUADRANT_ORDER:
            self.assertAlmostEqual(
                result.step_traces[0].alpha_after[key],
                snapshot[key],
                places=12,
            )


class RunEditingLoopOrchestrationTests(unittest.TestCase):

    def test_returns_editor_result(self) -> None:
        fake = _FakeInputTransformer(
            quadrant_scores={key: 0.0 for key in CANONICAL_QUADRANT_ORDER},
        )
        editor = _default_editor(
            initialization_mode="uniform",
            use_recursive_editing=True,
            max_edit_steps=1,
            convergence_threshold=1e-9,
            fake=fake,
        )
        result = editor.run_editing_loop(
            "prompt text",
            _make_prompt_state(),
            _make_router_state(),
            _make_expert_outputs(),
        )
        self.assertIsInstance(result, EditorResult)

    def test_initial_alignment_seeded_from_prompt_state_quadrant_scores(self) -> None:
        captured: dict[str, Any] = {}

        class _SpyEditor(_TestEditor):
            def _run_edit_loop(self, initial_alpha, initial_alignment, expert_outputs):
                captured["initial_alpha"] = dict(initial_alpha)
                captured["initial_alignment"] = dict(initial_alignment)
                return EditorResult(
                    final_mixed_hidden_state=torch.zeros(4),
                    final_alpha=dict(initial_alpha),
                    final_alignment=dict(initial_alignment),
                    step_traces=[],
                    num_steps_run=0,
                    stopped_early=False,
                    stop_reason=None,
                )

        editor = _SpyEditor(
            input_transformer=_FakeInputTransformer(),
            config=EditorConfig(initialization_mode="uniform"),
        )
        prompt_quadrants = {
            "left_lib": 0.4,
            "left_auth": -0.2,
            "right_lib": 0.7,
            "right_auth": -0.1,
        }
        editor.run_editing_loop(
            "p",
            _make_prompt_state(quadrant_scores=prompt_quadrants),
            _make_router_state(),
            _make_expert_outputs(),
        )
        for key in CANONICAL_QUADRANT_ORDER:
            self.assertAlmostEqual(
                captured["initial_alignment"][key],
                prompt_quadrants[key],
                places=12,
            )

    def test_uses_initialization_mode_from_config(self) -> None:
        captured: dict[str, Any] = {}

        class _SpyEditor(_TestEditor):
            def _run_edit_loop(self, initial_alpha, initial_alignment, expert_outputs):
                captured["initial_alpha"] = dict(initial_alpha)
                return EditorResult(
                    final_mixed_hidden_state=torch.zeros(4),
                    final_alpha=dict(initial_alpha),
                    final_alignment=dict(initial_alignment),
                    step_traces=[],
                    num_steps_run=0,
                    stopped_early=False,
                    stop_reason=None,
                )

        policy = {
            "left_lib": 0.4,
            "left_auth": 0.3,
            "right_lib": 0.2,
            "right_auth": 0.1,
        }

        editor_router = _SpyEditor(
            input_transformer=_FakeInputTransformer(),
            config=EditorConfig(initialization_mode="router_policy"),
        )
        editor_router.run_editing_loop(
            "p",
            _make_prompt_state(),
            _make_router_state(calibrated_policy=policy),
            _make_expert_outputs(),
        )
        for key in CANONICAL_QUADRANT_ORDER:
            self.assertAlmostEqual(captured["initial_alpha"][key], policy[key], places=12)

        editor_uniform = _SpyEditor(
            input_transformer=_FakeInputTransformer(),
            config=EditorConfig(initialization_mode="uniform"),
        )
        editor_uniform.run_editing_loop(
            "p",
            _make_prompt_state(),
            _make_router_state(calibrated_policy=policy),
            _make_expert_outputs(),
        )
        for key in CANONICAL_QUADRANT_ORDER:
            self.assertAlmostEqual(captured["initial_alpha"][key], 0.25, places=12)


_UNSET = object()


class EditorConstructionTests(unittest.TestCase):

    def _build(
        self,
        *,
        model: Any = _UNSET,
        tokenizer: Any = _UNSET,
        input_transformer: Any = _UNSET,
        config: Any = _UNSET,
        generation_config: Any = _UNSET,
    ) -> Editor:
        return Editor(
            model=object() if model is _UNSET else model,
            tokenizer=object() if tokenizer is _UNSET else tokenizer,
            input_transformer=(
                _FakeInputTransformer()
                if input_transformer is _UNSET
                else input_transformer
            ),
            config=EditorConfig() if config is _UNSET else config,
            generation_config=(
                GenerationConfig()
                if generation_config is _UNSET
                else generation_config
            ),
        )

    def test_constructs_and_stores_attributes(self) -> None:
        model = object()
        tokenizer = object()
        transformer = _FakeInputTransformer()
        config = EditorConfig()
        generation_config = GenerationConfig()
        editor = Editor(
            model=model,
            tokenizer=tokenizer,
            input_transformer=transformer,
            config=config,
            generation_config=generation_config,
        )
        self.assertIs(editor.model, model)
        self.assertIs(editor.tokenizer, tokenizer)
        self.assertIs(editor.input_transformer, transformer)
        self.assertIs(editor.config, config)
        self.assertIs(editor.generation_config, generation_config)

    def test_invalid_config_type_raises(self) -> None:
        with self.assertRaises(ValueError):
            self._build(config="not a config")

    def test_invalid_generation_config_type_raises(self) -> None:
        with self.assertRaises(ValueError):
            self._build(generation_config="not a generation config")

    def test_input_transformer_none_raises(self) -> None:
        with self.assertRaises(ValueError):
            self._build(input_transformer=None)  # type: ignore[arg-type]

    def test_invalid_initialization_mode_raises(self) -> None:
        with self.assertRaises(ValueError):
            self._build(config=EditorConfig(initialization_mode="not_a_mode"))

    def test_max_edit_steps_zero_raises(self) -> None:
        with self.assertRaises(ValueError):
            self._build(config=EditorConfig(max_edit_steps=0))

    def test_convergence_threshold_negative_raises(self) -> None:
        with self.assertRaises(ValueError):
            self._build(config=EditorConfig(convergence_threshold=-1))

    def test_correction_beta_nan_raises(self) -> None:
        with self.assertRaises(ValueError):
            self._build(config=EditorConfig(correction_beta=float("nan")))


if __name__ == "__main__":
    unittest.main()
