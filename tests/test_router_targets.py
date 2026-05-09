# tests/test_router_targets.py


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

from router_training import targets as brt  # noqa: E402

CANONICAL = ("left_lib", "left_auth", "right_lib", "right_auth")


# === HELPERS ===

def _policy(**overrides: float) -> dict[str, float]:
    base = {"left_lib": 0.4, "left_auth": 0.3, "right_lib": 0.2, "right_auth": 0.1}
    base.update(overrides)
    return base


def _quadrant_scores() -> dict[str, float]:
    # alignment scores can be negative; this is the SCORES field, not a policy
    return {"left_lib": 0.5, "left_auth": 0.0, "right_lib": -0.3, "right_auth": -0.1}


def _make_feature(
    eid: str = "ex1",
    *,
    prompt: str = "prompt",
    row_index: int = 0,
    metadata: dict | None = None,
) -> dict:
    return {
        "example_id": eid,
        "prompt_text": prompt,
        "source": "test",
        "quadrant_scores": _quadrant_scores(),
        "bias_magnitude": 0.7,
        "economic_score": 0.4,
        "social_score": 0.0,
        "hidden_representation_ref": f"hidden.pt:{row_index}",
        "metadata": metadata if metadata is not None else {"axis": "economic"},
    }


def _make_scored(
    *,
    eid: str = "ex1",
    score: float = 1.0,
    candidate: dict | None = None,
    prior: dict | None = None,
    prompt: str = "prompt",
) -> dict:
    return {
        "example_id":      eid,
        "prompt_text":     prompt,
        "candidate_policy": candidate if candidate is not None else _policy(),
        "heuristic_prior":  prior if prior is not None else _policy(),
        "final_text":      "some response",
        "metrics":         {"final_candidate_score": float(score)},
        "metadata":        {},
    }


def _build_cfg(**overrides: Any) -> brt.TargetBuildConfig:
    base: dict[str, Any] = dict(
        score_temperature=1.0,
        min_probability=1e-6,
        features_path=Path("/dev/null/features.jsonl"),
        candidate_scores_path=Path("/dev/null/scores.jsonl"),
        records_path=Path("/dev/null/records.jsonl"),
        target_report_path=Path("/dev/null/report.json"),
    )
    base.update(overrides)
    return brt.TargetBuildConfig(**base)


# === TESTS — load_jsonl ===

class LoadJsonlTests(unittest.TestCase):

    def test_happy_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "data.jsonl"
            path.write_text('{"a":1}\n{"a":2}\n', encoding="utf-8")
            self.assertEqual(brt.load_jsonl(path), [{"a": 1}, {"a": 2}])

    def test_missing_raises(self) -> None:
        with self.assertRaises(FileNotFoundError):
            brt.load_jsonl(Path("/nope/missing.jsonl"))

    def test_empty_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "x.jsonl"
            path.write_text("", encoding="utf-8")
            with self.assertRaises(ValueError):
                brt.load_jsonl(path)

    def test_malformed_json_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "x.jsonl"
            path.write_text('{"a":1}\n{not json}\n', encoding="utf-8")
            with self.assertRaises(ValueError):
                brt.load_jsonl(path)


# === TESTS — stable_softmax ===

class StableSoftmaxTests(unittest.TestCase):

    def test_equal_scores_uniform_weights(self) -> None:
        out = brt.stable_softmax([1.0, 1.0, 1.0, 1.0], temperature=0.5)
        for w in out:
            self.assertAlmostEqual(w, 0.25, places=9)

    def test_higher_score_higher_weight(self) -> None:
        out = brt.stable_softmax([0.0, 1.0, 2.0], temperature=1.0)
        self.assertLess(out[0], out[1])
        self.assertLess(out[1], out[2])
        self.assertAlmostEqual(sum(out), 1.0, places=9)

    def test_stable_for_large_scores(self) -> None:
        # without max-subtraction, exp(1000) overflows
        out = brt.stable_softmax([1000.0, 999.0, 998.0], temperature=1.0)
        self.assertAlmostEqual(sum(out), 1.0, places=6)
        self.assertGreater(out[0], out[1])
        self.assertGreater(out[1], out[2])

    def test_temperature_must_be_positive(self) -> None:
        with self.assertRaises(ValueError):
            brt.stable_softmax([1.0, 2.0], temperature=0.0)
        with self.assertRaises(ValueError):
            brt.stable_softmax([1.0, 2.0], temperature=-0.1)

    def test_empty_values_raises(self) -> None:
        with self.assertRaises(ValueError):
            brt.stable_softmax([], temperature=1.0)

    def test_temperature_sharpens(self) -> None:
        warm = brt.stable_softmax([0.0, 1.0], temperature=1.0)
        cool = brt.stable_softmax([0.0, 1.0], temperature=0.1)
        # lower temperature pushes mass toward the higher-score entry
        self.assertGreater(cool[1], warm[1])


# === TESTS — mix_policies ===

class MixPoliciesTests(unittest.TestCase):

    def test_weighted_average_correct(self) -> None:
        a = {"left_lib": 0.7, "left_auth": 0.1, "right_lib": 0.1, "right_auth": 0.1}
        b = {"left_lib": 0.1, "left_auth": 0.7, "right_lib": 0.1, "right_auth": 0.1}
        out = brt.mix_policies([a, b], [0.5, 0.5])
        self.assertAlmostEqual(out["left_lib"],  0.4, places=9)
        self.assertAlmostEqual(out["left_auth"], 0.4, places=9)
        self.assertAlmostEqual(out["right_lib"],  0.1, places=9)
        self.assertAlmostEqual(out["right_auth"], 0.1, places=9)

    def test_canonical_order_preserved(self) -> None:
        out = brt.mix_policies([_policy()], [1.0])
        self.assertEqual(tuple(out.keys()), CANONICAL)

    def test_length_mismatch_raises(self) -> None:
        with self.assertRaises(ValueError):
            brt.mix_policies([_policy()], [0.5, 0.5])

    def test_negative_weight_raises(self) -> None:
        with self.assertRaises(ValueError):
            brt.mix_policies([_policy()], [-0.1])


# === TESTS — entropy / kl_policy ===

class EntropyTests(unittest.TestCase):

    def test_uniform_max_entropy(self) -> None:
        e = brt.entropy({k: 0.25 for k in CANONICAL})
        self.assertAlmostEqual(e, math.log(4), places=9)

    def test_concentrated_low_entropy(self) -> None:
        p = {"left_lib": 0.97, "left_auth": 0.01, "right_lib": 0.01, "right_auth": 0.01}
        self.assertLess(brt.entropy(p), 0.5)


class KLPolicyTests(unittest.TestCase):

    def test_zero_when_equal(self) -> None:
        self.assertAlmostEqual(brt.kl_policy(_policy(), _policy()), 0.0, places=9)

    def test_positive_when_different(self) -> None:
        a = {"left_lib": 0.7, "left_auth": 0.1, "right_lib": 0.1, "right_auth": 0.1}
        b = {"left_lib": 0.1, "left_auth": 0.7, "right_lib": 0.1, "right_auth": 0.1}
        self.assertGreater(brt.kl_policy(a, b), 0.0)


# === TESTS — build_target_for_example ===

class BuildTargetForExampleTests(unittest.TestCase):

    def test_two_candidates_produce_non_argmax_mixture(self) -> None:
        feat = _make_feature()
        a = {"left_lib": 0.7, "left_auth": 0.1, "right_lib": 0.1, "right_auth": 0.1}
        b = {"left_lib": 0.1, "left_auth": 0.7, "right_lib": 0.1, "right_auth": 0.1}
        scored = [
            _make_scored(score=1.0, candidate=a),
            _make_scored(score=0.5, candidate=b),
        ]
        record, _ = brt.build_target_for_example(feat, scored, _build_cfg())
        target = record["target_policy"]
        # mixture is NOT pure-argmax: the lower-scoring candidate's mass
        # leaks through the softmax weighting
        self.assertGreater(target["left_auth"], 0.1)
        self.assertLess(target["left_lib"],     0.7)
        self.assertAlmostEqual(sum(target.values()), 1.0, places=6)
        for v in target.values():
            self.assertGreater(v, 0)
        self.assertEqual(tuple(target.keys()), CANONICAL)

    def test_metadata_has_required_fields(self) -> None:
        feat = _make_feature()
        record, _ = brt.build_target_for_example(
            feat, [_make_scored(score=1.0)], _build_cfg(),
        )
        md = record["metadata"]
        for key in (
            "target_policy_source", "num_candidates", "score_temperature",
            "min_probability", "best_candidate_score", "best_candidate_policy",
            "target_entropy", "kl_target_to_heuristic",
        ):
            self.assertIn(key, md)
        self.assertEqual(md["num_candidates"], 1)
        self.assertEqual(md["target_policy_source"], "offline_forced_policy_search_v1")
        # original feature metadata is preserved alongside scoring metadata
        self.assertEqual(md["axis"], "economic")
        # canonical key order in best_candidate_policy
        self.assertEqual(tuple(md["best_candidate_policy"].keys()), CANONICAL)

    def test_min_probability_floor_after_mix(self) -> None:
        feat = _make_feature()
        # peaked candidate (one entry near 1, others near 0)
        peaked = {
            "left_lib": 1.0 - 3e-6, "left_auth": 1e-6,
            "right_lib": 1e-6, "right_auth": 1e-6,
        }
        cfg = _build_cfg(min_probability=0.05)
        record, _ = brt.build_target_for_example(
            feat, [_make_scored(score=1.0, candidate=peaked)], cfg,
        )
        for v in record["target_policy"].values():
            self.assertGreaterEqual(v, 0.05 - 1e-9)

    def test_heuristic_prior_mismatch_raises(self) -> None:
        feat = _make_feature()
        a = _policy()
        b = {"left_lib": 0.5, "left_auth": 0.2, "right_lib": 0.2, "right_auth": 0.1}
        scored = [
            _make_scored(score=1.0, prior=a),
            _make_scored(score=0.5, prior=b),
        ]
        with self.assertRaises(ValueError) as ctx:
            brt.build_target_for_example(feat, scored, _build_cfg())
        self.assertIn("heuristic_prior mismatch", str(ctx.exception))

    def test_prior_mismatch_within_tolerance_ok(self) -> None:
        feat = _make_feature()
        a = _policy()
        # drift below 1e-6 must NOT trip the loud check
        b = {**_policy(), "left_lib": _policy()["left_lib"] + 1e-7,
             "right_auth": _policy()["right_auth"] - 1e-7}
        scored = [
            _make_scored(score=1.0, prior=a),
            _make_scored(score=0.5, prior=b),
        ]
        record, _ = brt.build_target_for_example(feat, scored, _build_cfg())
        self.assertIsNotNone(record)

    def test_malformed_candidate_policy_raises(self) -> None:
        feat = _make_feature()
        bad = _make_scored(score=1.0)
        bad["candidate_policy"] = {"left_lib": 0.5, "left_auth": 0.5}  # missing keys
        with self.assertRaises(ValueError):
            brt.build_target_for_example(feat, [bad], _build_cfg())

    def test_missing_score_raises(self) -> None:
        feat = _make_feature()
        bad = _make_scored(score=1.0)
        bad["metrics"] = {}  # final_candidate_score missing
        with self.assertRaises(ValueError):
            brt.build_target_for_example(feat, [bad], _build_cfg())

    def test_non_finite_score_raises(self) -> None:
        feat = _make_feature()
        bad = _make_scored(score=1.0)
        bad["metrics"]["final_candidate_score"] = float("nan")
        with self.assertRaises(ValueError):
            brt.build_target_for_example(feat, [bad], _build_cfg())

    def test_prompt_mismatch_raises_at_lower_level(self) -> None:
        # the lower-level helper still raises; build_all_targets handles the
        # softer skip + report path
        feat = _make_feature(prompt="A")
        with self.assertRaises(ValueError) as ctx:
            brt.build_target_for_example(
                feat, [_make_scored(prompt="B", score=1.0)], _build_cfg(),
            )
        self.assertIn("prompt_text", str(ctx.exception))


# === TESTS — build_all_targets ===

class BuildAllTargetsTests(unittest.TestCase):

    def test_records_in_feature_order(self) -> None:
        feats = [_make_feature(eid=f"ex{i}", row_index=i) for i in range(3)]
        scored = [_make_scored(eid=f"ex{i}", score=1.0) for i in range(3)]
        records, report = brt.build_all_targets(feats, scored, _build_cfg())
        self.assertEqual([r["example_id"] for r in records], ["ex0", "ex1", "ex2"])
        self.assertEqual(report["num_records_written"], 3)
        self.assertEqual(report["skipped_examples"], [])

    def test_missing_scored_skipped_with_reason(self) -> None:
        feats = [_make_feature(eid="ex1"), _make_feature(eid="ex2", row_index=1)]
        scored = [_make_scored(eid="ex1", score=1.0)]
        records, report = brt.build_all_targets(feats, scored, _build_cfg())
        self.assertEqual([r["example_id"] for r in records], ["ex1"])
        self.assertEqual(len(report["skipped_examples"]), 1)
        skip = report["skipped_examples"][0]
        self.assertEqual(skip["example_id"], "ex2")
        self.assertEqual(skip["reason"], "no_scored_candidates")

    def test_orphan_scored_reported(self) -> None:
        feats = [_make_feature(eid="ex1")]
        scored = [
            _make_scored(eid="ex1",    score=1.0),
            _make_scored(eid="orphan", score=0.5),
        ]
        records, report = brt.build_all_targets(feats, scored, _build_cfg())
        self.assertEqual(len(records), 1)
        skipped_pairs = {(s["example_id"], s["reason"]) for s in report["skipped_examples"]}
        self.assertIn(("orphan", "no_feature_row"), skipped_pairs)

    def test_prompt_mismatch_skipped_in_corpus_path(self) -> None:
        # build_all_targets converts prompt_text drift into a SKIP (not a raise).
        # malformed-row issues still raise at the lower function layer.
        feats = [_make_feature(eid="ex1", prompt="alpha")]
        scored = [_make_scored(eid="ex1", score=1.0, prompt="beta")]
        records, report = brt.build_all_targets(feats, scored, _build_cfg())
        self.assertEqual(records, [])
        self.assertEqual(len(report["skipped_examples"]), 1)
        self.assertEqual(report["skipped_examples"][0]["reason"], "prompt_text_mismatch")

    def test_duplicate_feature_id_raises(self) -> None:
        feats = [_make_feature(eid="ex1"), _make_feature(eid="ex1", row_index=1)]
        with self.assertRaises(ValueError):
            brt.build_all_targets(feats, [_make_scored(eid="ex1", score=1.0)], _build_cfg())

    def test_malformed_candidate_row_raises(self) -> None:
        # build_all_targets must raise (not skip) when an actual row is malformed
        feats = [_make_feature(eid="ex1")]
        bad = _make_scored(eid="ex1", score=1.0)
        bad["candidate_policy"] = {"left_lib": 0.5, "left_auth": 0.5}  # missing keys
        with self.assertRaises(ValueError):
            brt.build_all_targets(feats, [bad], _build_cfg())

    def test_summary_stats_present(self) -> None:
        feats = [_make_feature(eid="ex1"), _make_feature(eid="ex2", row_index=1)]
        scored = [
            _make_scored(eid="ex1", score=1.0),
            _make_scored(eid="ex2", score=0.5),
        ]
        records, report = brt.build_all_targets(feats, scored, _build_cfg())
        s = report["summary"]
        self.assertEqual(s["mean_num_candidates_per_record"], 1.0)
        self.assertEqual(s["mean_best_score"], 0.75)
        self.assertGreater(s["mean_target_entropy"], 0)
        self.assertGreaterEqual(s["mean_kl_target_to_heuristic"], 0)

    def test_limit_truncates_features(self) -> None:
        feats   = [_make_feature(eid=f"ex{i}", row_index=i) for i in range(5)]
        scored  = [_make_scored(eid=f"ex{i}", score=1.0)    for i in range(5)]
        records, report = brt.build_all_targets(feats, scored, _build_cfg(), limit=2)
        self.assertEqual([r["example_id"] for r in records], ["ex0", "ex1"])
        self.assertEqual(report["num_records_written"], 2)

    def test_empty_diagnostics_summary_is_none(self) -> None:
        # all features skipped → no diagnostics → summary fields are None
        feats = [_make_feature(eid="ex1")]
        scored: list[dict] = []
        records, report = brt.build_all_targets(feats, scored, _build_cfg())
        self.assertEqual(records, [])
        for v in report["summary"].values():
            self.assertIsNone(v)


# === TESTS — record schema (compat with train_calibrated_router.py) ===

class RecordSchemaTests(unittest.TestCase):

    def test_record_has_trainer_required_fields(self) -> None:
        feat = _make_feature()
        record, _ = brt.build_target_for_example(
            feat, [_make_scored(score=1.0)], _build_cfg(),
        )
        # mirrors REQUIRED_RECORD_FIELDS in src/train_calibrated_router.py
        for key in (
            "example_id", "prompt_text", "quadrant_scores",
            "bias_magnitude", "target_policy", "hidden_representation_ref",
        ):
            self.assertIn(key, record)
        self.assertEqual(tuple(record["target_policy"].keys()), CANONICAL)
        self.assertEqual(tuple(record["quadrant_scores"].keys()), CANONICAL)
        # JSON-roundtrips cleanly (no torch tensors leaked)
        encoded = json.dumps(record)
        decoded = json.loads(encoded)
        self.assertEqual(decoded["example_id"], "ex1")

    def test_no_hidden_tensor_or_other_heavy_fields(self) -> None:
        feat = _make_feature()
        record, _ = brt.build_target_for_example(
            feat, [_make_scored(score=1.0)], _build_cfg(),
        )
        # hidden_representation_ref is just the string pointer; no tensor
        self.assertIsInstance(record["hidden_representation_ref"], str)
        self.assertNotIn("hidden_representation", record)


# === TESTS — feature validation ===

class FeatureValidationTests(unittest.TestCase):

    def test_missing_required_field_raises(self) -> None:
        for missing in ("example_id", "prompt_text", "quadrant_scores",
                        "bias_magnitude", "hidden_representation_ref"):
            feat = _make_feature()
            del feat[missing]
            with self.assertRaises(ValueError) as ctx:
                brt.build_all_targets([feat], [_make_scored(score=1.0)], _build_cfg())
            self.assertIn(missing, str(ctx.exception))

    def test_wrong_quadrant_scores_keys_raises(self) -> None:
        feat = _make_feature()
        feat["quadrant_scores"] = {"left_lib": 0.0, "left_auth": 0.0, "right_lib": 0.0}
        with self.assertRaises(ValueError):
            brt.build_all_targets([feat], [_make_scored(score=1.0)], _build_cfg())


# === TESTS — IO + CLI round-trip ===

class IOTests(unittest.TestCase):

    def test_writes_records_and_report_in_feature_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            features_path = tmp_path / "features.jsonl"
            scores_path   = tmp_path / "scores.jsonl"
            records_path  = tmp_path / "out" / "records.jsonl"
            report_path   = tmp_path / "out" / "target_report.json"

            with features_path.open("w", encoding="utf-8") as fh:
                for i in range(3):
                    fh.write(json.dumps(_make_feature(f"ex{i}", row_index=i)) + "\n")
            with scores_path.open("w", encoding="utf-8") as fh:
                # write scored rows out-of-order for ex0/ex1/ex2 to confirm
                # records preserve FEATURE order, not scored order
                for i in (2, 0, 1):
                    fh.write(json.dumps(_make_scored(eid=f"ex{i}", score=1.0)) + "\n")

            features = brt.load_jsonl(features_path)
            scored   = brt.load_jsonl(scores_path)
            cfg = brt.TargetBuildConfig(
                score_temperature=1.0,
                min_probability=1e-6,
                features_path=features_path,
                candidate_scores_path=scores_path,
                records_path=records_path,
                target_report_path=report_path,
            )
            records, report = brt.build_all_targets(features, scored, cfg)
            brt.write_records_jsonl(records, records_path)
            brt.write_report_json(report, report_path)

            self.assertTrue(records_path.is_file())
            self.assertTrue(report_path.is_file())

            with records_path.open() as fh:
                rows = [json.loads(line) for line in fh if line.strip()]
            self.assertEqual([r["example_id"] for r in rows], ["ex0", "ex1", "ex2"])

            with report_path.open() as fh:
                report_data = json.load(fh)
            self.assertEqual(report_data["num_records_written"], 3)
            self.assertEqual(report_data["num_feature_rows"],   3)
            self.assertEqual(report_data["num_scored_rows"],    3)
            self.assertEqual(report_data["skipped_examples"],   [])

    def test_creates_parent_directories(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            records_path = tmp_path / "deep" / "nested" / "records.jsonl"
            report_path  = tmp_path / "deep" / "nested" / "report.json"
            brt.write_records_jsonl([], records_path)
            brt.write_report_json({"x": 1}, report_path)
            self.assertTrue(records_path.is_file())
            self.assertTrue(report_path.is_file())


# === MAIN ===

def main() -> None:
    unittest.main()


if __name__ == "__main__":
    main()
