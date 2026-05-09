# tests/test_router_forced_policy_runner.py


# === IMPORTS ===

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any


# === MODULE LOADING ===

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_DIR = _REPO_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from router_training import forced_policy_runner as rfp  # noqa: E402

CANONICAL = ("left_lib", "left_auth", "right_lib", "right_auth")


# === FAKES ===

class _FakeInputTransformer:
    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def transform(self, prompt_text: str) -> Any:
        self.calls.append(("transform", prompt_text))
        return SimpleNamespace(
            prompt_text=prompt_text,
            hidden_representation=None,
            quadrant_scores={k: 0.25 for k in CANONICAL},
            bias_magnitude=0.0,
            economic_score=0.0,
            social_score=0.0,
            metadata={"encoding_layer": 20},
        )


class _FakeRouter:
    def __init__(self, prior: dict[str, float] | None = None) -> None:
        self.calls: list[tuple] = []
        self._prior = dict(prior) if prior is not None else {k: 0.25 for k in CANONICAL}

    def build_heuristic_prior(self, prompt_state: Any) -> dict[str, float]:
        self.calls.append(("build_heuristic_prior", prompt_state))
        return dict(self._prior)


class _FakeExpertManager:
    def __init__(self, outputs: dict[str, Any] | None = None) -> None:
        self.calls: list[tuple] = []
        self._outputs = outputs if outputs is not None else {
            k: SimpleNamespace(text=f"text_{k}", hidden_state=None) for k in CANONICAL
        }

    def run_all_experts(self, prompt_text: str, prompt_state: Any) -> dict[str, Any]:
        self.calls.append(("run_all_experts", prompt_text, prompt_state))
        return self._outputs


class _FakeEditor:
    def __init__(self, result: Any | None = None) -> None:
        self.calls: list[tuple] = []
        self._result = result if result is not None else SimpleNamespace(
            final_mixed_hidden_state=None,
            final_alpha={k: 0.25 for k in CANONICAL},
            final_alignment={k: 0.0 for k in CANONICAL},
            step_traces=[],
            num_steps_run=1,
            stopped_early=False,
            stop_reason=None,
        )

    def run_editing_loop(
        self,
        prompt_text: str,
        prompt_state: Any,
        router_state: Any,
        expert_outputs: dict[str, Any],
    ) -> Any:
        self.calls.append((
            "run_editing_loop", prompt_text, prompt_state, router_state, expert_outputs,
        ))
        return self._result


class _FakeEngine:
    """
    Composable fake engine. `decode_method` selects which of the three decode
    fallback names will be exposed; pass None to expose none of them and
    exercise the missing-decode-boundary error.
    """

    def __init__(
        self,
        *,
        prior: dict[str, float] | None = None,
        decode_method: str | None = "decode_editor_result",
        decode_text: str = "decoded text",
        editor_result: Any | None = None,
        expert_outputs: dict[str, Any] | None = None,
    ) -> None:
        self.input_transformer = _FakeInputTransformer()
        self.router = _FakeRouter(prior=prior)
        self.expert_manager = _FakeExpertManager(outputs=expert_outputs)
        self.editor = _FakeEditor(result=editor_result)
        self._decode_text = decode_text
        self.decode_calls: list[tuple] = []
        if decode_method is not None:
            setattr(self, decode_method, self._decode_impl)

    def _decode_impl(
        self,
        prompt_text: str,
        prompt_state: Any,
        router_state: Any,
        expert_outputs: dict[str, Any],
        editor_result: Any,
    ) -> str:
        self.decode_calls.append((
            "decode", prompt_text, prompt_state, router_state, expert_outputs, editor_result,
        ))
        return self._decode_text


def _candidate_policy() -> dict[str, float]:
    return {"left_lib": 0.4, "left_auth": 0.3, "right_lib": 0.2, "right_auth": 0.1}


def _uniform_prior() -> dict[str, float]:
    return {k: 0.25 for k in CANONICAL}


# === TESTS — happy path ===

class HappyPathTests(unittest.TestCase):

    def test_runs_and_returns_final_text(self) -> None:
        engine = _FakeEngine()
        runner = rfp.ForcedPolicyMoCERunner(engine)
        result = runner.run(
            example_id="ex1",
            prompt_text="hello",
            candidate_policy=_candidate_policy(),
        )
        self.assertEqual(result.final_text, "decoded text")
        self.assertEqual(result.example_id, "ex1")
        self.assertEqual(result.prompt_text, "hello")
        self.assertEqual(result.forced_policy, _candidate_policy())

    def test_candidate_policy_routed_to_calibrated(self) -> None:
        engine = _FakeEngine()
        runner = rfp.ForcedPolicyMoCERunner(engine)
        runner.run(
            example_id="ex1", prompt_text="p", candidate_policy=_candidate_policy(),
        )
        # editor sees a router_state whose calibrated_policy == the forced candidate
        self.assertEqual(len(engine.editor.calls), 1)
        passed_router_state = engine.editor.calls[0][3]
        self.assertEqual(passed_router_state.calibrated_policy, _candidate_policy())

    def test_diagnostics_marks_forced(self) -> None:
        engine = _FakeEngine()
        runner = rfp.ForcedPolicyMoCERunner(engine)
        result = runner.run(
            example_id="ex1", prompt_text="p", candidate_policy=_candidate_policy(),
        )
        diag = result.router_state.diagnostics
        self.assertTrue(diag.get("forced_policy"))
        self.assertEqual(diag["calibrated_policy"], _candidate_policy())

    def test_pipeline_call_order(self) -> None:
        # input_transformer.transform → router.build_heuristic_prior →
        # expert_manager.run_all_experts → editor.run_editing_loop → decode
        engine = _FakeEngine()
        runner = rfp.ForcedPolicyMoCERunner(engine)
        runner.run(
            example_id="ex", prompt_text="hi", candidate_policy=_candidate_policy(),
        )
        self.assertEqual(engine.input_transformer.calls[0][0], "transform")
        self.assertEqual(engine.router.calls[0][0], "build_heuristic_prior")
        self.assertEqual(engine.expert_manager.calls[0][0], "run_all_experts")
        self.assertEqual(engine.editor.calls[0][0], "run_editing_loop")
        self.assertEqual(engine.decode_calls[0][0], "decode")


# === TESTS — heuristic prior ===

class HeuristicPriorTests(unittest.TestCase):

    def test_provided_prior_skips_router_build(self) -> None:
        engine = _FakeEngine()
        runner = rfp.ForcedPolicyMoCERunner(engine)
        custom = {"left_lib": 0.7, "left_auth": 0.1, "right_lib": 0.1, "right_auth": 0.1}
        result = runner.run(
            example_id="ex", prompt_text="p",
            candidate_policy=_candidate_policy(),
            heuristic_prior=custom,
        )
        self.assertEqual(result.heuristic_prior, custom)
        self.assertEqual(engine.router.calls, [])

    def test_no_prior_calls_router(self) -> None:
        engine = _FakeEngine(prior=_uniform_prior())
        runner = rfp.ForcedPolicyMoCERunner(engine)
        result = runner.run(
            example_id="ex", prompt_text="p", candidate_policy=_candidate_policy(),
        )
        self.assertEqual(len(engine.router.calls), 1)
        self.assertEqual(result.heuristic_prior, _uniform_prior())


# === TESTS — input validation ===

class CandidatePolicyValidationTests(unittest.TestCase):

    def setUp(self) -> None:
        self.engine = _FakeEngine()
        self.runner = rfp.ForcedPolicyMoCERunner(self.engine)

    def test_missing_key_raises(self) -> None:
        bad = {"left_lib": 0.5, "left_auth": 0.5}
        with self.assertRaises(ValueError):
            self.runner.run(
                example_id="x", prompt_text="x", candidate_policy=bad,
            )

    def test_extra_key_raises(self) -> None:
        bad = {**_candidate_policy(), "extra": 0.0}
        with self.assertRaises(ValueError):
            self.runner.run(
                example_id="x", prompt_text="x", candidate_policy=bad,
            )

    def test_zero_value_raises(self) -> None:
        bad = dict(_candidate_policy())
        bad["right_auth"] = 0.0
        with self.assertRaises(ValueError):
            self.runner.run(
                example_id="x", prompt_text="x", candidate_policy=bad,
            )

    def test_negative_value_raises(self) -> None:
        bad = {"left_lib": -0.1, "left_auth": 0.4, "right_lib": 0.4, "right_auth": 0.3}
        with self.assertRaises(ValueError):
            self.runner.run(
                example_id="x", prompt_text="x", candidate_policy=bad,
            )

    def test_sum_not_one_raises(self) -> None:
        bad = {"left_lib": 0.5, "left_auth": 0.5, "right_lib": 0.5, "right_auth": 0.5}
        with self.assertRaises(ValueError):
            self.runner.run(
                example_id="x", prompt_text="x", candidate_policy=bad,
            )

    def test_bool_value_rejected_as_numeric(self) -> None:
        # True is an int subclass in Python; the validator must reject it.
        bad = {"left_lib": True, "left_auth": 0.3, "right_lib": 0.4, "right_auth": 0.3}
        with self.assertRaises(ValueError):
            self.runner.run(
                example_id="x", prompt_text="x", candidate_policy=bad,
            )

    def test_nan_value_rejected(self) -> None:
        bad = {**_candidate_policy(), "left_lib": float("nan")}
        with self.assertRaises(ValueError):
            self.runner.run(
                example_id="x", prompt_text="x", candidate_policy=bad,
            )

    def test_empty_example_id_raises(self) -> None:
        with self.assertRaises(ValueError):
            self.runner.run(
                example_id="   ", prompt_text="x", candidate_policy=_candidate_policy(),
            )

    def test_empty_prompt_text_raises(self) -> None:
        with self.assertRaises(ValueError):
            self.runner.run(
                example_id="x", prompt_text="", candidate_policy=_candidate_policy(),
            )

    def test_invalid_provided_heuristic_raises(self) -> None:
        bad_prior = {"left_lib": 1.5, "left_auth": -0.5, "right_lib": 0.4, "right_auth": -0.4}
        with self.assertRaises(ValueError):
            self.runner.run(
                example_id="x",
                prompt_text="x",
                candidate_policy=_candidate_policy(),
                heuristic_prior=bad_prior,
            )

    def test_invalid_router_built_prior_raises(self) -> None:
        # router returns a malformed prior; runner must catch via validation
        engine = _FakeEngine(prior={"left_lib": 0.5, "left_auth": 0.5})  # missing keys
        runner = rfp.ForcedPolicyMoCERunner(engine)
        with self.assertRaises(ValueError):
            runner.run(
                example_id="x", prompt_text="x", candidate_policy=_candidate_policy(),
            )


# === TESTS — engine component contracts ===

class EngineComponentTests(unittest.TestCase):

    def test_engine_none_raises(self) -> None:
        with self.assertRaises(ValueError):
            rfp.ForcedPolicyMoCERunner(None)

    def test_missing_input_transformer_raises(self) -> None:
        engine = _FakeEngine()
        del engine.input_transformer
        with self.assertRaises(ValueError) as ctx:
            rfp.ForcedPolicyMoCERunner(engine)
        self.assertIn("input_transformer", str(ctx.exception))

    def test_missing_router_raises(self) -> None:
        engine = _FakeEngine()
        del engine.router
        with self.assertRaises(ValueError) as ctx:
            rfp.ForcedPolicyMoCERunner(engine)
        self.assertIn("router", str(ctx.exception))

    def test_missing_expert_manager_raises(self) -> None:
        engine = _FakeEngine()
        del engine.expert_manager
        with self.assertRaises(ValueError) as ctx:
            rfp.ForcedPolicyMoCERunner(engine)
        self.assertIn("expert_manager", str(ctx.exception))

    def test_missing_editor_raises(self) -> None:
        engine = _FakeEngine()
        del engine.editor
        with self.assertRaises(ValueError) as ctx:
            rfp.ForcedPolicyMoCERunner(engine)
        self.assertIn("editor", str(ctx.exception))

    def test_component_missing_method_raises(self) -> None:
        engine = _FakeEngine()
        engine.editor = SimpleNamespace()  # has no run_editing_loop
        with self.assertRaises(ValueError) as ctx:
            rfp.ForcedPolicyMoCERunner(engine)
        self.assertIn("run_editing_loop", str(ctx.exception))

    def test_empty_expert_outputs_raises(self) -> None:
        engine = _FakeEngine(expert_outputs={})
        runner = rfp.ForcedPolicyMoCERunner(engine)
        with self.assertRaises(ValueError):
            runner.run(
                example_id="x", prompt_text="x", candidate_policy=_candidate_policy(),
            )

    def test_non_dict_expert_outputs_raises(self) -> None:
        engine = _FakeEngine()
        engine.expert_manager._outputs = ["wrong"]  # type: ignore[assignment]
        runner = rfp.ForcedPolicyMoCERunner(engine)
        with self.assertRaises(ValueError):
            runner.run(
                example_id="x", prompt_text="x", candidate_policy=_candidate_policy(),
            )

    def test_expert_outputs_missing_quadrant_key_raises(self) -> None:
        # only 3 of the 4 canonical quadrants returned — runner must surface
        # the missing key by name in the error message
        partial = {k: SimpleNamespace(text=f"t_{k}", hidden_state=None) for k in CANONICAL[:3]}
        engine = _FakeEngine(expert_outputs=partial)
        runner = rfp.ForcedPolicyMoCERunner(engine)
        with self.assertRaises(ValueError) as ctx:
            runner.run(
                example_id="x", prompt_text="x", candidate_policy=_candidate_policy(),
            )
        msg = str(ctx.exception)
        self.assertIn("missing=['right_auth']", msg)
        self.assertIn("extra=[]", msg)

    def test_expert_outputs_unexpected_key_raises(self) -> None:
        # all four canonical keys present plus an unexpected fifth — runner
        # must surface the extra key by name
        bad = {k: SimpleNamespace(text=f"t_{k}", hidden_state=None) for k in CANONICAL}
        bad["wrong_key"] = SimpleNamespace(text="oops", hidden_state=None)
        engine = _FakeEngine(expert_outputs=bad)
        runner = rfp.ForcedPolicyMoCERunner(engine)
        with self.assertRaises(ValueError) as ctx:
            runner.run(
                example_id="x", prompt_text="x", candidate_policy=_candidate_policy(),
            )
        msg = str(ctx.exception)
        self.assertIn("extra=['wrong_key']", msg)
        self.assertIn("missing=[]", msg)


# === TESTS — decode fallback ===

class DecodeFallbackTests(unittest.TestCase):

    def test_decode_editor_result_used(self) -> None:
        engine = _FakeEngine(decode_method="decode_editor_result", decode_text="alpha")
        runner = rfp.ForcedPolicyMoCERunner(engine)
        result = runner.run(
            example_id="x", prompt_text="x", candidate_policy=_candidate_policy(),
        )
        self.assertEqual(result.final_text, "alpha")
        self.assertEqual(result.metadata["decode_callable"], "decode_editor_result")

    def test_decode_final_text_used(self) -> None:
        engine = _FakeEngine(decode_method="decode_final_text", decode_text="beta")
        runner = rfp.ForcedPolicyMoCERunner(engine)
        result = runner.run(
            example_id="x", prompt_text="x", candidate_policy=_candidate_policy(),
        )
        self.assertEqual(result.final_text, "beta")
        self.assertEqual(result.metadata["decode_callable"], "decode_final_text")

    def test_underscore_decode_editor_result_used(self) -> None:
        engine = _FakeEngine(decode_method="_decode_editor_result", decode_text="gamma")
        runner = rfp.ForcedPolicyMoCERunner(engine)
        result = runner.run(
            example_id="x", prompt_text="x", candidate_policy=_candidate_policy(),
        )
        self.assertEqual(result.final_text, "gamma")
        self.assertEqual(result.metadata["decode_callable"], "_decode_editor_result")

    def test_priority_picks_first_available(self) -> None:
        # both decode_editor_result and decode_final_text exist; runner must
        # prefer the first listed in DECODE_FALLBACKS
        engine = _FakeEngine(decode_method="decode_editor_result", decode_text="primary")
        engine.decode_final_text = lambda *args, **kwargs: "secondary"
        runner = rfp.ForcedPolicyMoCERunner(engine)
        result = runner.run(
            example_id="x", prompt_text="x", candidate_policy=_candidate_policy(),
        )
        self.assertEqual(result.final_text, "primary")

    def test_no_decode_method_raises(self) -> None:
        engine = _FakeEngine(decode_method=None)
        with self.assertRaises(ValueError) as ctx:
            rfp.ForcedPolicyMoCERunner(engine)
        self.assertIn("decode boundary", str(ctx.exception))

    def test_decode_returns_empty_string_raises(self) -> None:
        engine = _FakeEngine(decode_text="")
        runner = rfp.ForcedPolicyMoCERunner(engine)
        with self.assertRaises(ValueError):
            runner.run(
                example_id="x", prompt_text="x", candidate_policy=_candidate_policy(),
            )

    def test_decode_returns_non_string_raises(self) -> None:
        engine = _FakeEngine()
        engine._decode_text = 42  # type: ignore[assignment]
        runner = rfp.ForcedPolicyMoCERunner(engine)
        with self.assertRaises(ValueError):
            runner.run(
                example_id="x", prompt_text="x", candidate_policy=_candidate_policy(),
            )


# === TESTS — serialization ===

class SerializationTests(unittest.TestCase):

    def _build_result(self, *, editor_result: Any | None = None) -> rfp.ForcedPolicyRunResult:
        engine = _FakeEngine(editor_result=editor_result)
        runner = rfp.ForcedPolicyMoCERunner(engine)
        return runner.run(
            example_id="ex1",
            prompt_text="hello",
            candidate_policy=_candidate_policy(),
        )

    def test_returns_json_safe_dict(self) -> None:
        out = rfp.serialize_forced_policy_result(self._build_result())
        encoded = json.dumps(out)
        decoded = json.loads(encoded)
        self.assertEqual(decoded["example_id"], "ex1")
        self.assertEqual(decoded["prompt_text"], "hello")
        self.assertEqual(decoded["forced_policy"], _candidate_policy())
        self.assertEqual(decoded["heuristic_prior"], _uniform_prior())
        self.assertEqual(decoded["final_text"], "decoded text")

    def test_excludes_object_fields(self) -> None:
        out = rfp.serialize_forced_policy_result(self._build_result())
        self.assertNotIn("prompt_state", out)
        self.assertNotIn("router_state", out)
        self.assertNotIn("editor_result", out)

    def test_includes_editor_metadata(self) -> None:
        out = rfp.serialize_forced_policy_result(self._build_result())
        md = out["metadata"]
        self.assertEqual(md["editor_final_alpha"], {k: 0.25 for k in CANONICAL})
        self.assertEqual(md["editor_final_alignment"], {k: 0.0 for k in CANONICAL})
        self.assertEqual(md["editor_num_steps_run"], 1)
        self.assertFalse(md["editor_stopped_early"])
        # default editor result has stop_reason=None → omitted from metadata
        self.assertNotIn("editor_stop_reason", md)

    def test_includes_router_diagnostics(self) -> None:
        out = rfp.serialize_forced_policy_result(self._build_result())
        diag = out["metadata"]["router_diagnostics"]
        self.assertTrue(diag["forced_policy"])
        self.assertEqual(diag["calibrated_policy"], _candidate_policy())

    def test_includes_stop_reason_when_set(self) -> None:
        editor_result = SimpleNamespace(
            final_mixed_hidden_state=None,
            final_alpha={k: 0.25 for k in CANONICAL},
            final_alignment={k: 0.1 for k in CANONICAL},
            step_traces=[],
            num_steps_run=3,
            stopped_early=True,
            stop_reason="converged",
        )
        out = rfp.serialize_forced_policy_result(
            self._build_result(editor_result=editor_result),
        )
        md = out["metadata"]
        self.assertEqual(md["editor_stop_reason"], "converged")
        self.assertTrue(md["editor_stopped_early"])
        self.assertEqual(md["editor_num_steps_run"], 3)

    def test_invalid_input_raises(self) -> None:
        with self.assertRaises(ValueError):
            rfp.serialize_forced_policy_result({"not": "a result"})  # type: ignore[arg-type]

    def test_handles_missing_editor_attrs_gracefully(self) -> None:
        # editor result without any of the expected attributes
        out = rfp.serialize_forced_policy_result(
            self._build_result(editor_result=SimpleNamespace()),
        )
        md = out["metadata"]
        for missing in (
            "editor_final_alpha", "editor_final_alignment",
            "editor_num_steps_run", "editor_stopped_early", "editor_stop_reason",
        ):
            self.assertNotIn(missing, md)


# === TESTS — defensive copies ===

class DefensiveCopyTests(unittest.TestCase):

    def test_input_candidate_policy_not_mutated(self) -> None:
        engine = _FakeEngine()
        runner = rfp.ForcedPolicyMoCERunner(engine)
        candidate = _candidate_policy()
        snapshot = dict(candidate)
        runner.run(
            example_id="x", prompt_text="x", candidate_policy=candidate,
        )
        self.assertEqual(candidate, snapshot)

    def test_input_heuristic_prior_not_mutated(self) -> None:
        engine = _FakeEngine()
        runner = rfp.ForcedPolicyMoCERunner(engine)
        prior = _uniform_prior()
        snapshot = dict(prior)
        runner.run(
            example_id="x", prompt_text="x",
            candidate_policy=_candidate_policy(),
            heuristic_prior=prior,
        )
        self.assertEqual(prior, snapshot)

    def test_result_forced_policy_independent_from_diagnostics(self) -> None:
        engine = _FakeEngine()
        runner = rfp.ForcedPolicyMoCERunner(engine)
        result = runner.run(
            example_id="x", prompt_text="x", candidate_policy=_candidate_policy(),
        )
        # mutating the result-level forced_policy must not bleed into diagnostics
        result.forced_policy["left_lib"] = 999.0
        diag_calibrated = result.router_state.diagnostics["calibrated_policy"]
        self.assertNotEqual(diag_calibrated["left_lib"], 999.0)

    def test_serialized_dict_is_independent(self) -> None:
        engine = _FakeEngine()
        runner = rfp.ForcedPolicyMoCERunner(engine)
        result = runner.run(
            example_id="x", prompt_text="x", candidate_policy=_candidate_policy(),
        )
        out = rfp.serialize_forced_policy_result(result)
        out["forced_policy"]["left_lib"] = 999.0
        self.assertNotEqual(result.forced_policy["left_lib"], 999.0)


# === MAIN ===

def main() -> None:
    unittest.main()


if __name__ == "__main__":
    main()
