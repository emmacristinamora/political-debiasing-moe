# tests/test_router_pipeline_smoke.py


# === IMPORTS ===

from __future__ import annotations

import json
import sys
import tempfile
import time
import unittest
from pathlib import Path


# === MODULE LOADING ===

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_DIR = _REPO_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import run_router_pipeline_smoke as smoke  # noqa: E402
from validate_router_dataset import validate_router_dataset  # noqa: E402


# === TESTS ===

class SmokeRunCompletionTests(unittest.TestCase):

    def test_run_smoke_completes_without_exception(self) -> None:
        # default arguments — must not raise on a fresh run
        report = smoke.run_smoke()
        self.assertIsInstance(report, dict)
        self.assertGreater(report["num_prompts"], 0)
        self.assertGreater(report["num_records"], 0)

    def test_run_smoke_is_fast(self) -> None:
        # smoke should remain trivial; if this ever blows past a second the
        # synthetic stages have grown beyond the spirit of "smoke".
        start = time.perf_counter()
        smoke.run_smoke()
        elapsed = time.perf_counter() - start
        self.assertLess(elapsed, 1.0, f"smoke run took {elapsed:.3f}s (>1s)")


class SmokeReportShapeTests(unittest.TestCase):

    def test_report_contains_required_top_level_keys(self) -> None:
        report = smoke.run_smoke()
        for key in ("num_prompts", "num_records", "splits", "command"):
            self.assertIn(key, report, f"missing required report key {key!r}")
        # split counts present and integer
        for split_name in ("train", "val", "test"):
            self.assertIn(split_name, report["splits"])
            self.assertIsInstance(report["splits"][split_name], int)
        # command is a non-empty list of strings
        self.assertIsInstance(report["command"], list)
        self.assertGreater(len(report["command"]), 0)
        for entry in report["command"]:
            self.assertIsInstance(entry, str)
        # command points at the trainer
        self.assertIn("src/train_calibrated_router.py", report["command"])

    def test_split_counts_sum_to_num_records(self) -> None:
        report = smoke.run_smoke()
        total = sum(report["splits"][k] for k in ("train", "val", "test"))
        self.assertEqual(total, report["num_records"])

    def test_output_file_written_when_path_provided(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            out = Path(raw_tmp) / "nested" / "smoke.json"
            report = smoke.run_smoke(output_path=out)
            self.assertTrue(out.is_file())
            roundtrip = json.loads(out.read_text(encoding="utf-8"))
        self.assertEqual(roundtrip["num_prompts"], report["num_prompts"])
        self.assertEqual(roundtrip["num_records"], report["num_records"])
        self.assertEqual(roundtrip["splits"], report["splits"])
        self.assertEqual(roundtrip["command"], report["command"])


class SmokeRecordsValidatorTests(unittest.TestCase):

    def test_generated_records_pass_step8_validator(self) -> None:
        report = smoke.run_smoke()
        # validator runs every record-level check the trainer applies; passing
        # without an exception confirms the synthetic dataset is schema-valid.
        validate_router_dataset(
            report["records"],
            None,
            expected_hidden_dim=smoke.DEFAULT_HIDDEN_DIM,
            hidden_filename=smoke.DEFAULT_HIDDEN_FILENAME,
        )


class SmokeDeterminismTests(unittest.TestCase):

    def test_two_runs_same_seed_produce_identical_report(self) -> None:
        first  = smoke.run_smoke(seed=123)
        second = smoke.run_smoke(seed=123)
        # JSON-roundtrip both so any non-JSON-safe drift would show up here
        self.assertEqual(
            json.dumps(first, sort_keys=True),
            json.dumps(second, sort_keys=True),
        )

    def test_changing_seed_changes_candidate_scores(self) -> None:
        first  = smoke.run_smoke(seed=123)
        second = smoke.run_smoke(seed=456)
        # records may coincide if Dirichlet samples happen to round to the same
        # min-prob-clipped distribution, but the candidate-score block should
        # differ since the candidate sets differ between seeds.
        self.assertNotEqual(
            json.dumps(first["candidate_scores"], sort_keys=True),
            json.dumps(second["candidate_scores"], sort_keys=True),
        )


# === ENTRY POINT ===

if __name__ == "__main__":
    unittest.main()
