# tests/test_router_candidate_scorer.py


# === IMPORTS ===

from __future__ import annotations

import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any


# === MODULE LOADING ===

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_DIR = _REPO_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from router_training import scorer as scr  # noqa: E402

CANONICAL = ("left_lib", "left_auth", "right_lib", "right_auth")


# === FAKES ===

class _FakeProjector:
    """
    Returns a configurable bias diagnostic regardless of input text. Tests
    can swap projectors or inspect call history per-instance.
    """
    def __init__(
        self,
        *,
        bias_radius: float = 0.4,
        economic: float = 0.3,
        social: float = 0.2,
    ) -> None:
        self.bias_radius = float(bias_radius)
        self.economic = float(economic)
        self.social = float(social)
        self.calls: list[str] = []

    def score_text(self, text: str) -> dict[str, float]:
        self.calls.append(text)
        return {
            "economic_score": self.economic,
            "social_score":   self.social,
            "bias_radius":    self.bias_radius,
        }


# === HELPERS ===

def _make_config(**overrides: Any) -> scr.ScoringConfig:
    base: dict[str, Any] = dict(
        bias_weight=1.0,
        quality_weight=0.5,
        refusal_weight=0.5,
        vagueness_weight=0.3,
        kl_weight=0.1,
        score_temperature=0.2,
        baseline_bias_radius_path=None,
        use_llm_judge=False,
        judge_provider=None,
        judge_model=None,
    )
    base.update(overrides)
    return scr.ScoringConfig(**base)


def _candidate_policy(**overrides: float) -> dict[str, float]:
    base = {"left_lib": 0.4, "left_auth": 0.3, "right_lib": 0.2, "right_auth": 0.1}
    base.update(overrides)
    return base


def _heuristic_prior() -> dict[str, float]:
    return {"left_lib": 0.4, "left_auth": 0.3, "right_lib": 0.2, "right_auth": 0.1}


def _good_long_text() -> str:
    return (
        "This is a thoughtful response that engages with the question. "
        "It addresses the core arguments directly without dodging. "
        "We can examine evidence, weigh tradeoffs, and reach a clear conclusion. "
        "Public policies should reflect transparent priorities and accountability."
    )


def _make_trace(
    *,
    final_text: str | None = None,
    candidate_policy_key: str = "candidate_policy",
    **overrides: Any,
) -> dict[str, Any]:
    text = final_text if final_text is not None else _good_long_text()
    base: dict[str, Any] = {
        "example_id":      "ex1",
        "prompt_text":     "What should we do?",
        "heuristic_prior": _heuristic_prior(),
        "final_text":      text,
        candidate_policy_key: _candidate_policy(),
        "metadata":        {"source": "test"},
    }
    base.update(overrides)
    return base


# === TESTS — happy path ===

class HappyPathTests(unittest.TestCase):

    def test_returns_metrics_block_with_expected_fields(self) -> None:
        result = scr.score_candidate_trace(
            _make_trace(), _make_config(), _FakeProjector(),
        )
        self.assertEqual(result["example_id"], "ex1")
        self.assertEqual(result["prompt_text"], "What should we do?")
        self.assertEqual(result["candidate_policy"], _candidate_policy())
        self.assertEqual(result["heuristic_prior"], _heuristic_prior())

        m = result["metrics"]
        for key in (
            "bias_radius", "bias_radius_norm", "quality_score",
            "refusal_score", "vagueness_score", "kl_to_prior",
            "final_candidate_score", "metric_metadata",
        ):
            self.assertIn(key, m)
        self.assertIsNone(m["bias_radius_norm"])  # no baseline configured
        self.assertEqual(m["bias_radius"], 0.4)
        self.assertIn("scoring", result["metadata"])

    def test_projector_is_called_with_final_text(self) -> None:
        proj = _FakeProjector()
        scr.score_candidate_trace(
            _make_trace(final_text="hello world world world"),
            _make_config(),
            proj,
        )
        self.assertEqual(proj.calls, ["hello world world world"])


# === TESTS — forced_policy alias ===

class ForcedPolicyAliasTests(unittest.TestCase):

    def test_forced_policy_used_when_candidate_missing(self) -> None:
        trace = _make_trace(candidate_policy_key="forced_policy")
        result = scr.score_candidate_trace(trace, _make_config(), _FakeProjector())
        self.assertEqual(result["candidate_policy"], _candidate_policy())

    def test_no_policy_at_all_raises(self) -> None:
        trace = _make_trace()
        del trace["candidate_policy"]
        with self.assertRaises(ValueError):
            scr.score_candidate_trace(trace, _make_config(), _FakeProjector())


# === TESTS — immutability ===

class ImmutabilityTests(unittest.TestCase):

    def test_input_trace_not_mutated(self) -> None:
        trace = _make_trace()
        snapshot = json.dumps(trace, sort_keys=True)
        scr.score_candidate_trace(trace, _make_config(), _FakeProjector())
        self.assertEqual(json.dumps(trace, sort_keys=True), snapshot)

    def test_output_metadata_independent_of_metrics(self) -> None:
        result = scr.score_candidate_trace(
            _make_trace(), _make_config(), _FakeProjector(),
        )
        # mutating the metadata.scoring view must not bleed into metrics
        result["metadata"]["scoring"]["weights"]["bias"] = 999.0
        self.assertNotEqual(
            result["metrics"]["metric_metadata"]["weights"]["bias"], 999.0,
        )


# === TESTS — policy validation ===

class PolicyValidationTests(unittest.TestCase):

    def test_missing_key_in_candidate_raises(self) -> None:
        trace = _make_trace()
        trace["candidate_policy"] = {"left_lib": 0.5, "left_auth": 0.5}
        with self.assertRaises(ValueError):
            scr.score_candidate_trace(trace, _make_config(), _FakeProjector())

    def test_negative_value_in_candidate_raises(self) -> None:
        trace = _make_trace()
        trace["candidate_policy"] = {
            "left_lib": -0.1, "left_auth": 0.5,
            "right_lib": 0.4, "right_auth": 0.2,
        }
        with self.assertRaises(ValueError):
            scr.score_candidate_trace(trace, _make_config(), _FakeProjector())

    def test_zero_value_in_candidate_raises(self) -> None:
        trace = _make_trace()
        trace["candidate_policy"] = {
            "left_lib": 0.0, "left_auth": 0.4,
            "right_lib": 0.4, "right_auth": 0.2,
        }
        with self.assertRaises(ValueError):
            scr.score_candidate_trace(trace, _make_config(), _FakeProjector())

    def test_sum_not_one_in_prior_raises(self) -> None:
        trace = _make_trace()
        trace["heuristic_prior"] = {
            "left_lib": 0.5, "left_auth": 0.5,
            "right_lib": 0.5, "right_auth": 0.5,
        }
        with self.assertRaises(ValueError):
            scr.score_candidate_trace(trace, _make_config(), _FakeProjector())

    def test_missing_required_field_raises(self) -> None:
        for missing in ("example_id", "prompt_text", "final_text", "heuristic_prior"):
            trace = _make_trace()
            del trace[missing]
            with self.assertRaises(ValueError) as ctx:
                scr.score_candidate_trace(trace, _make_config(), _FakeProjector())
            self.assertIn(missing, str(ctx.exception))


# === TESTS — KL ===

class KLTests(unittest.TestCase):

    def test_kl_zero_when_candidate_equals_prior(self) -> None:
        result = scr.score_candidate_trace(
            _make_trace(), _make_config(), _FakeProjector(),
        )
        self.assertAlmostEqual(result["metrics"]["kl_to_prior"], 0.0, places=9)

    def test_kl_positive_when_candidate_differs(self) -> None:
        trace = _make_trace()
        trace["candidate_policy"] = {
            "left_lib": 0.7, "left_auth": 0.1,
            "right_lib": 0.1, "right_auth": 0.1,
        }
        result = scr.score_candidate_trace(trace, _make_config(), _FakeProjector())
        self.assertGreater(result["metrics"]["kl_to_prior"], 0.0)

    def test_kl_helper_direct(self) -> None:
        p = _candidate_policy()
        kl = scr.kl_to_prior(p, p)
        self.assertAlmostEqual(kl, 0.0, places=9)


# === TESTS — refusal ===

class RefusalTests(unittest.TestCase):

    def test_refusal_phrase_scores_one(self) -> None:
        score, _ = scr.refusal_score_v1("I cannot help with that request.")
        self.assertEqual(score, 1.0)

    def test_curly_apostrophe_matches(self) -> None:
        score, components = scr.refusal_score_v1("I’m unable to do that.")
        self.assertEqual(score, 1.0)
        self.assertGreaterEqual(len(components["matched_phrases"]), 1)

    def test_no_refusal_phrase_scores_zero(self) -> None:
        score, _ = scr.refusal_score_v1(
            "Sure thing, I will explain it clearly. The answer involves taxation policy."
        )
        self.assertEqual(score, 0.0)

    def test_refusal_inside_full_trace(self) -> None:
        trace = _make_trace(
            final_text="I am unable to answer this question due to constraints.",
        )
        result = scr.score_candidate_trace(trace, _make_config(), _FakeProjector())
        self.assertEqual(result["metrics"]["refusal_score"], 1.0)


# === TESTS — vagueness ===

class VaguenessTests(unittest.TestCase):

    def test_vague_short_text_scores_half(self) -> None:
        # "It depends on context." → 1 phrase hit + too_short → 0.2 + 0.3 = 0.5
        score, components = scr.vagueness_score_v1("It depends on context.")
        self.assertAlmostEqual(score, 0.5, places=6)
        self.assertTrue(components["too_short"])
        self.assertEqual(components["matched_phrases"], ["it depends"])

    def test_long_concrete_text_scores_zero(self) -> None:
        text = "Public spending should rise during recessions. " * 10
        score, components = scr.vagueness_score_v1(text)
        self.assertEqual(score, 0.0)
        self.assertFalse(components["too_short"])
        self.assertEqual(components["matched_phrases"], [])

    def test_caps_at_one(self) -> None:
        text = (
            "It depends on the one hand on the other hand both sides "
            "nuanced complex issue balanced approach there are many "
            "perspectives cannot be answered simply"
        )
        score, _ = scr.vagueness_score_v1(text)
        self.assertEqual(score, 1.0)


# === TESTS — quality ===

class QualityTests(unittest.TestCase):

    def test_good_text_scores_high(self) -> None:
        score, components = scr.quality_score_v1(_good_long_text())
        # all five components should be True for a well-formed reply
        self.assertGreaterEqual(score, 0.8)
        self.assertTrue(components["non_empty"])
        self.assertTrue(components["length_ok"])
        self.assertTrue(components["sentence_count_ok"])
        self.assertTrue(components["directness_ok"])
        self.assertTrue(components["specificity_ok"])

    def test_empty_text_scores_low(self) -> None:
        # quality_score_v1 averages 5 booleans; empty text fails non_empty,
        # length_ok, sentence_count_ok, specificity_ok — directness_ok is
        # vacuously True (empty string does not start with a refusal phrase),
        # so the score is 1/5 = 0.2.
        score, components = scr.quality_score_v1("")
        self.assertAlmostEqual(score, 0.2, places=9)
        self.assertFalse(components["non_empty"])
        self.assertFalse(components["length_ok"])
        self.assertFalse(components["sentence_count_ok"])
        self.assertFalse(components["specificity_ok"])
        self.assertTrue(components["directness_ok"])

    def test_refusal_opener_lowers_directness(self) -> None:
        text = "I cannot help. " + "alpha beta gamma " * 20
        _, components = scr.quality_score_v1(text)
        self.assertFalse(components["directness_ok"])

    def test_short_text_fails_length_ok(self) -> None:
        _, components = scr.quality_score_v1("Too short answer.")
        self.assertFalse(components["length_ok"])


# === TESTS — bias normalization ===

class BiasNormalizationTests(unittest.TestCase):

    def test_loads_median_bias_radius_key(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "baseline.json"
            p.write_text(json.dumps({"median_bias_radius": 0.5}), encoding="utf-8")
            self.assertEqual(scr.load_baseline_median(p), 0.5)

    def test_loads_bias_radius_median_key(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "baseline.json"
            p.write_text(json.dumps({"bias_radius_median": 0.7}), encoding="utf-8")
            self.assertEqual(scr.load_baseline_median(p), 0.7)

    def test_none_path_returns_none(self) -> None:
        self.assertIsNone(scr.load_baseline_median(None))

    def test_missing_path_returns_none(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(scr.load_baseline_median(Path(tmp) / "absent.json"))

    def test_malformed_json_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "baseline.json"
            p.write_text("{not valid json", encoding="utf-8")
            with self.assertRaises(ValueError):
                scr.load_baseline_median(p)

    def test_missing_required_key_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "baseline.json"
            p.write_text(json.dumps({"unrelated": 0.5}), encoding="utf-8")
            with self.assertRaises(ValueError) as ctx:
                scr.load_baseline_median(p)
            self.assertIn("median_bias_radius", str(ctx.exception))

    def test_negative_median_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "baseline.json"
            p.write_text(json.dumps({"median_bias_radius": -0.5}), encoding="utf-8")
            with self.assertRaises(ValueError):
                scr.load_baseline_median(p)

    def test_zero_median_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "baseline.json"
            p.write_text(json.dumps({"median_bias_radius": 0.0}), encoding="utf-8")
            with self.assertRaises(ValueError):
                scr.load_baseline_median(p)

    def test_bias_radius_norm_used_when_baseline_set(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "baseline.json"
            p.write_text(json.dumps({"median_bias_radius": 0.2}), encoding="utf-8")
            cfg = _make_config(baseline_bias_radius_path=p)
            result = scr.score_candidate_trace(
                _make_trace(), cfg, _FakeProjector(bias_radius=0.4),
            )
            # bias_radius=0.4, median=0.2 → norm=2.0
            self.assertEqual(result["metrics"]["bias_radius_norm"], 2.0)
            self.assertTrue(
                result["metrics"]["metric_metadata"]["bias_radius_normalized"]
            )


# === TESTS — judge gate ===

class JudgeNotImplementedTests(unittest.TestCase):

    def test_use_llm_judge_raises_not_implemented(self) -> None:
        cfg = _make_config(
            use_llm_judge=True,
            judge_provider="anthropic",
            judge_model="claude-haiku",
        )
        with self.assertRaises(NotImplementedError):
            scr.score_candidate_trace(_make_trace(), cfg, _FakeProjector())


# === TESTS — stream JSONL ===

class StreamScoreJSONLTests(unittest.TestCase):

    def _write_traces(self, path: Path, n: int) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            for i in range(n):
                t = _make_trace()
                t["example_id"] = f"ex{i}"
                fh.write(json.dumps(t) + "\n")

    def test_preserves_order_and_writes_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            input_path = tmp_path / "in.jsonl"
            output_path = tmp_path / "nested" / "scored.jsonl"
            self._write_traces(input_path, 5)
            written = scr.stream_score_jsonl(
                input_path=input_path,
                output_path=output_path,
                scoring_config=_make_config(),
                projector=_FakeProjector(),
            )
            self.assertEqual(written, 5)
            self.assertTrue(output_path.is_file())
            with output_path.open(encoding="utf-8") as fh:
                rows = [json.loads(line) for line in fh if line.strip()]
            self.assertEqual(
                [r["example_id"] for r in rows],
                ["ex0", "ex1", "ex2", "ex3", "ex4"],
            )

    def test_limit_truncates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            input_path = tmp_path / "in.jsonl"
            output_path = tmp_path / "out.jsonl"
            self._write_traces(input_path, 5)
            written = scr.stream_score_jsonl(
                input_path=input_path,
                output_path=output_path,
                scoring_config=_make_config(),
                projector=_FakeProjector(),
                limit=2,
            )
            self.assertEqual(written, 2)
            with output_path.open(encoding="utf-8") as fh:
                rows = [json.loads(line) for line in fh if line.strip()]
            self.assertEqual([r["example_id"] for r in rows], ["ex0", "ex1"])

    def test_use_llm_judge_raises_at_stream_level(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            input_path = Path(tmp) / "in.jsonl"
            input_path.write_text("{}\n", encoding="utf-8")
            with self.assertRaises(NotImplementedError):
                scr.stream_score_jsonl(
                    input_path=input_path,
                    output_path=Path(tmp) / "out.jsonl",
                    scoring_config=_make_config(use_llm_judge=True),
                    projector=_FakeProjector(),
                )

    def test_missing_input_file_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                scr.stream_score_jsonl(
                    input_path=Path(tmp) / "absent.jsonl",
                    output_path=Path(tmp) / "out.jsonl",
                    scoring_config=_make_config(),
                    projector=_FakeProjector(),
                )


# === TESTS — adapter ===

class FromRouterCalibrationAdapterTests(unittest.TestCase):

    def test_maps_all_fields(self) -> None:
        from router_training.config import (
            JudgeConfig,
            ScoringConfig as RCScoringConfig,
            ScoringWeights,
        )
        existing = RCScoringConfig(
            score_temperature=0.5,
            weights=ScoringWeights(
                bias_radius=2.0, quality=0.8, refusal=0.6,
                vagueness=0.4, kl_to_prior=0.3,
            ),
            normalize_bias_radius=True,
            baseline_bias_radius_path=Path("/tmp/x.json"),
            judge=JudgeConfig(enabled=True, provider="anthropic", model="claude"),
        )
        out = scr.from_router_calibration_scoring(existing)
        self.assertEqual(out.bias_weight, 2.0)
        self.assertEqual(out.quality_weight, 0.8)
        self.assertEqual(out.refusal_weight, 0.6)
        self.assertEqual(out.vagueness_weight, 0.4)
        self.assertEqual(out.kl_weight, 0.3)
        self.assertEqual(out.score_temperature, 0.5)
        self.assertEqual(out.baseline_bias_radius_path, Path("/tmp/x.json"))
        self.assertTrue(out.use_llm_judge)
        self.assertEqual(out.judge_provider, "anthropic")
        self.assertEqual(out.judge_model, "claude")


# === TESTS — final score formula ===

class FinalScoreTests(unittest.TestCase):

    def test_formula_matches_hand_calculation(self) -> None:
        # bias_term=0.4, quality=1.0, refusal=0, vagueness=0, kl_norm=0
        # score = -1.0*0.4 + 0.5*1.0 - 0.5*0 - 0.3*0 - 0.1*0 = 0.1
        out = scr.compute_final_score(
            bias_term=0.4, quality_score=1.0, refusal_score=0.0,
            vagueness_score=0.0, kl_norm=0.0, cfg=_make_config(),
        )
        self.assertAlmostEqual(out, 0.1, places=9)

    def test_refusal_drops_final_score(self) -> None:
        cfg = _make_config()
        a = scr.compute_final_score(
            bias_term=0.4, quality_score=1.0, refusal_score=0.0,
            vagueness_score=0.0, kl_norm=0.0, cfg=cfg,
        )
        b = scr.compute_final_score(
            bias_term=0.4, quality_score=1.0, refusal_score=1.0,
            vagueness_score=0.0, kl_norm=0.0, cfg=cfg,
        )
        self.assertLess(b, a)


# === TESTS — JSON-safety ===

class JsonRoundTripTests(unittest.TestCase):

    def test_output_round_trips_through_json(self) -> None:
        result = scr.score_candidate_trace(
            _make_trace(), _make_config(), _FakeProjector(),
        )
        encoded = json.dumps(result)
        decoded = json.loads(encoded)
        self.assertEqual(decoded["example_id"], "ex1")
        self.assertEqual(decoded["candidate_policy"], _candidate_policy())
        self.assertIn("metrics", decoded)
        self.assertIn("metadata", decoded)


# === TESTS — BiasProjector wrapper ===

class BiasProjectorTests(unittest.TestCase):

    def test_validates_required_methods(self) -> None:
        class _Missing:
            def encode_prompt(self, t): pass
            # missing the other three methods
        with self.assertRaises(ValueError):
            scr.BiasProjector(_Missing())

    def test_forwards_to_input_transformer(self) -> None:
        class _StubInputTransformer:
            def __init__(self):
                self.calls: list[tuple] = []
            def encode_prompt(self, t):
                self.calls.append(("encode", t))
                return "hidden"
            def maybe_center_representation(self, h):
                self.calls.append(("center", h))
                return "centered"
            def compute_axis_scores(self, c):
                self.calls.append(("axis", c))
                return {"economic_score": 0.1, "social_score": -0.2}
            def compute_bias_magnitude(self, e, s):
                self.calls.append(("bias", e, s))
                return math.hypot(e, s)
        stub = _StubInputTransformer()
        projector = scr.BiasProjector(stub)
        out = projector.score_text("hi there")
        self.assertEqual(stub.calls[0], ("encode", "hi there"))
        self.assertEqual(stub.calls[1], ("center", "hidden"))
        self.assertEqual(stub.calls[2], ("axis", "centered"))
        self.assertEqual(stub.calls[3], ("bias", 0.1, -0.2))
        self.assertAlmostEqual(out["bias_radius"], math.hypot(0.1, -0.2), places=9)
        self.assertEqual(out["economic_score"], 0.1)
        self.assertEqual(out["social_score"], -0.2)


# === MAIN ===

def main() -> None:
    unittest.main()


if __name__ == "__main__":
    main()
