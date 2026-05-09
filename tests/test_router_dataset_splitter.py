# tests/test_router_dataset_splitter.py


# === IMPORTS ===

from __future__ import annotations

import json
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

from router_training import splitter as srs  # noqa: E402


# === HELPERS ===

def _build_cfg(**overrides: Any) -> srs.SplitBuildConfig:
    base: dict[str, Any] = dict(
        train_fraction=0.6,
        val_fraction=0.2,
        test_fraction=0.2,
        seed=42,
        stratify_by="none",
        records_path=Path("/dev/null/records.jsonl"),
        output_dir=Path("/dev/null"),
        report_path=Path("/dev/null/report.json"),
    )
    base.update(overrides)
    return srs.SplitBuildConfig(**base)


def _record(
    eid: str,
    *,
    source: str = "method12",
    axis: str = "economic",
    row_index: int = 0,
) -> dict:
    return {
        "example_id": eid,
        "prompt_text": f"prompt for {eid}",
        "quadrant_scores": {"left_lib": 0.1, "left_auth": 0.2, "right_lib": 0.3, "right_auth": 0.4},
        "bias_magnitude": 0.5,
        "target_policy": {"left_lib": 0.4, "left_auth": 0.3, "right_lib": 0.2, "right_auth": 0.1},
        "hidden_representation_ref": f"hidden.pt:{row_index}",
        "metadata": {"source": source, "axis": axis},
    }


def _records_n(n: int, **kwargs: Any) -> list[dict]:
    return [_record(f"ex{i}", row_index=i, **kwargs) for i in range(n)]


# === TESTS — allocate_counts ===

class AllocateCountsTests(unittest.TestCase):

    def test_zero(self) -> None:
        self.assertEqual(srs.allocate_counts(0, (0.6, 0.2, 0.2)), (0, 0, 0))

    def test_one(self) -> None:
        self.assertEqual(srs.allocate_counts(1, (0.6, 0.2, 0.2)), (1, 0, 0))

    def test_two(self) -> None:
        self.assertEqual(srs.allocate_counts(2, (0.6, 0.2, 0.2)), (1, 1, 0))

    def test_three_balanced(self) -> None:
        self.assertEqual(srs.allocate_counts(3, (0.6, 0.2, 0.2)), (1, 1, 1))

    def test_three_skewed_still_each_gets_one(self) -> None:
        # extreme skew toward train; the at-least-one-each rule kicks in
        self.assertEqual(srs.allocate_counts(3, (0.95, 0.025, 0.025)), (1, 1, 1))

    def test_ten_clean(self) -> None:
        self.assertEqual(srs.allocate_counts(10, (0.6, 0.2, 0.2)), (6, 2, 2))

    def test_largest_remainder_resolves_ties(self) -> None:
        # n=10, fractions=(0.7, 0.15, 0.15); raw=(7.0, 1.5, 1.5), leftover=1
        # tie between val and test on remainder (0.5); train tie-breaker first index
        # -> (7, 2, 1)
        self.assertEqual(srs.allocate_counts(10, (0.7, 0.15, 0.15)), (7, 2, 1))

    def test_negative_n_raises(self) -> None:
        with self.assertRaises(ValueError):
            srs.allocate_counts(-1, (0.6, 0.2, 0.2))

    def test_bool_n_raises(self) -> None:
        with self.assertRaises(ValueError):
            srs.allocate_counts(True, (0.6, 0.2, 0.2))  # type: ignore[arg-type]


# === TESTS — validate_split_fractions ===

class ValidateSplitFractionsTests(unittest.TestCase):

    def test_valid_passes(self) -> None:
        srs.validate_split_fractions(0.6, 0.2, 0.2)

    def test_zero_raises(self) -> None:
        with self.assertRaises(ValueError):
            srs.validate_split_fractions(0.0, 0.5, 0.5)

    def test_negative_raises(self) -> None:
        with self.assertRaises(ValueError):
            srs.validate_split_fractions(-0.1, 0.55, 0.55)

    def test_sum_not_one_raises(self) -> None:
        with self.assertRaises(ValueError):
            srs.validate_split_fractions(0.5, 0.2, 0.2)


# === TESTS — get_stratum_key ===

class GetStratumKeyTests(unittest.TestCase):

    def test_none_returns_constant(self) -> None:
        self.assertEqual(srs.get_stratum_key(_record("a"), "none"), "all")

    def test_source_resolution_top_level(self) -> None:
        self.assertEqual(
            srs.get_stratum_key(_record("a", source="method12"), "source"),
            "method12",
        )

    def test_source_resolution_input_metadata(self) -> None:
        rec = {"example_id": "a", "metadata": {"input_metadata": {"source": "method3"}}}
        self.assertEqual(srs.get_stratum_key(rec, "source"), "method3")

    def test_source_resolution_original_source(self) -> None:
        rec = {"example_id": "a", "metadata": {"original_source": "legacy"}}
        self.assertEqual(srs.get_stratum_key(rec, "source"), "legacy")

    def test_source_unknown_when_missing(self) -> None:
        rec = {"example_id": "a", "metadata": {}}
        self.assertEqual(srs.get_stratum_key(rec, "source"), "unknown")

    def test_axis_resolution(self) -> None:
        self.assertEqual(
            srs.get_stratum_key(_record("a", axis="social"), "axis"),
            "social",
        )

    def test_axis_unknown_when_missing(self) -> None:
        rec = {"example_id": "a", "metadata": {}}
        self.assertEqual(srs.get_stratum_key(rec, "axis"), "unknown")

    def test_source_axis_combined(self) -> None:
        self.assertEqual(
            srs.get_stratum_key(_record("a", source="m12", axis="economic"), "source_axis"),
            "m12::economic",
        )

    def test_unsupported_raises(self) -> None:
        with self.assertRaises(ValueError):
            srs.get_stratum_key(_record("a"), "wrong_key")


# === TESTS — split_records ===

class SplitRecordsTests(unittest.TestCase):

    def test_valid_split_writes_train_val_test(self) -> None:
        records = _records_n(20)
        splits, report = srs.split_records(records, _build_cfg(stratify_by="none"))
        self.assertEqual(len(splits["train"]), 12)
        self.assertEqual(len(splits["val"]),    4)
        self.assertEqual(len(splits["test"]),   4)
        self.assertEqual(report["counts"]["train"], 12)
        self.assertEqual(report["counts"]["val"],    4)
        self.assertEqual(report["counts"]["test"],   4)
        self.assertEqual(report["num_records"], 20)
        self.assertIn("strata", report)

    def test_deterministic_with_fixed_seed(self) -> None:
        records = _records_n(50)
        splits_a, _ = srs.split_records(records, _build_cfg(seed=42))
        splits_b, _ = srs.split_records(records, _build_cfg(seed=42))
        self.assertEqual(
            [r["example_id"] for r in splits_a["train"]],
            [r["example_id"] for r in splits_b["train"]],
        )
        self.assertEqual(
            [r["example_id"] for r in splits_a["val"]],
            [r["example_id"] for r in splits_b["val"]],
        )

    def test_changing_seed_changes_membership(self) -> None:
        records = _records_n(50)
        splits_a, _ = srs.split_records(records, _build_cfg(seed=1))
        splits_b, _ = srs.split_records(records, _build_cfg(seed=999))
        # counts stay the same (deterministic allocation), but membership differs
        self.assertEqual(len(splits_a["train"]), len(splits_b["train"]))
        a_ids = [r["example_id"] for r in splits_a["train"]]
        b_ids = [r["example_id"] for r in splits_b["train"]]
        self.assertNotEqual(a_ids, b_ids)

    def test_input_order_does_not_change_output_with_fixed_seed(self) -> None:
        # records sorted by example_id within each stratum, then shuffled,
        # so swapping input order must not affect output
        records = _records_n(20)
        records_reversed = list(reversed(records))
        splits_a, _ = srs.split_records(records,          _build_cfg(seed=7))
        splits_b, _ = srs.split_records(records_reversed, _build_cfg(seed=7))
        self.assertEqual(
            sorted(r["example_id"] for r in splits_a["train"]),
            sorted(r["example_id"] for r in splits_b["train"]),
        )
        # the within-split ordering is also identical because we sort before shuffle
        self.assertEqual(
            [r["example_id"] for r in splits_a["train"]],
            [r["example_id"] for r in splits_b["train"]],
        )

    def test_duplicate_example_id_raises(self) -> None:
        records = [_record("ex1"), _record("ex1", row_index=1)]
        with self.assertRaises(ValueError) as ctx:
            srs.split_records(records, _build_cfg())
        self.assertIn("duplicate example_id", str(ctx.exception))

    def test_missing_example_id_raises(self) -> None:
        rec = _record("ex1")
        del rec["example_id"]
        with self.assertRaises(ValueError):
            srs.split_records([rec], _build_cfg())

    def test_missing_hidden_ref_raises(self) -> None:
        rec = _record("ex1")
        del rec["hidden_representation_ref"]
        with self.assertRaises(ValueError):
            srs.split_records([rec], _build_cfg())

    def test_invalid_fractions_raise(self) -> None:
        with self.assertRaises(ValueError):
            srs.split_records(
                _records_n(10),
                _build_cfg(train_fraction=0.5, val_fraction=0.3, test_fraction=0.3),
            )
        with self.assertRaises(ValueError):
            srs.split_records(
                _records_n(10),
                _build_cfg(train_fraction=0.0, val_fraction=0.5, test_fraction=0.5),
            )

    def test_unsupported_stratify_by_raises(self) -> None:
        with self.assertRaises(ValueError):
            srs.split_records(_records_n(10), _build_cfg(stratify_by="random_thing"))

    def test_empty_records_raises(self) -> None:
        with self.assertRaises(ValueError):
            srs.split_records([], _build_cfg())

    def test_source_stratification_preserves_proportions(self) -> None:
        # 70 of source A + 30 of source B; 60/20/20 fractions →
        # per stratum exact: A→(42,14,14), B→(18,6,6); train_A/train = 42/60 = 0.7
        a = [_record(f"a{i}", source="A", row_index=i)        for i in range(70)]
        b = [_record(f"b{i}", source="B", row_index=70 + i)   for i in range(30)]
        records = a + b
        splits, report = srs.split_records(records, _build_cfg(stratify_by="source"))

        train_a = sum(1 for r in splits["train"] if r["metadata"]["source"] == "A")
        self.assertEqual(train_a, 42)
        self.assertEqual(report["strata"]["A"]["train"], 42)
        self.assertEqual(report["strata"]["B"]["train"], 18)
        self.assertAlmostEqual(train_a / len(splits["train"]), 0.7, places=2)

    def test_source_axis_stratum_keys(self) -> None:
        records = [
            _record("a1", source="m12", axis="economic"),
            _record("a2", source="m12", axis="economic"),
            _record("a3", source="m12", axis="economic"),
            _record("b1", source="m12", axis="social"),
            _record("b2", source="m12", axis="social"),
            _record("b3", source="m12", axis="social"),
            _record("c1", source="m3",  axis="economic"),
            _record("c2", source="m3",  axis="economic"),
            _record("c3", source="m3",  axis="economic"),
        ]
        _, report = srs.split_records(records, _build_cfg(stratify_by="source_axis"))
        self.assertIn("m12::economic", report["strata"])
        self.assertIn("m12::social",   report["strata"])
        self.assertIn("m3::economic",  report["strata"])

    def test_tiny_dataset_assigns_at_least_one_train(self) -> None:
        for n in (1, 2, 3):
            splits, report = srs.split_records(
                _records_n(n), _build_cfg(stratify_by="none"),
            )
            self.assertGreaterEqual(report["counts"]["train"], 1)

    def test_singleton_stratum_warning(self) -> None:
        # one stratum with 1 example; warning surfaces
        records = [_record("a1", source="solo", row_index=0)] + _records_n(20)
        _, report = srs.split_records(records, _build_cfg(stratify_by="source"))
        self.assertTrue(any(
            "solo" in w and "1 example" in w for w in report["warnings"]
        ))

    def test_hidden_ref_preserved_unchanged(self) -> None:
        records = _records_n(10)
        original_refs = {r["example_id"]: r["hidden_representation_ref"] for r in records}
        splits, _ = srs.split_records(records, _build_cfg())
        for split in splits.values():
            for r in split:
                self.assertEqual(
                    r["hidden_representation_ref"],
                    original_refs[r["example_id"]],
                )

    def test_no_id_appears_in_two_splits(self) -> None:
        records = _records_n(50)
        splits, _ = srs.split_records(records, _build_cfg())
        train_ids = {r["example_id"] for r in splits["train"]}
        val_ids   = {r["example_id"] for r in splits["val"]}
        test_ids  = {r["example_id"] for r in splits["test"]}
        self.assertEqual(train_ids & val_ids,  set())
        self.assertEqual(train_ids & test_ids, set())
        self.assertEqual(val_ids   & test_ids, set())
        # total covers every record
        self.assertEqual(
            train_ids | val_ids | test_ids,
            {r["example_id"] for r in records},
        )

    def test_report_contains_required_keys(self) -> None:
        _, report = srs.split_records(_records_n(10), _build_cfg())
        for key in (
            "input_path", "output_paths", "num_records", "fractions",
            "seed", "stratify_by", "counts", "strata", "warnings",
        ):
            self.assertIn(key, report)
        for split in ("train", "val", "test"):
            self.assertIn(split, report["counts"])
            self.assertIn(split, report["output_paths"])

    def test_empty_train_branch_unreachable_for_n_ge_1(self) -> None:
        # the at-least-1-train rule means train is always non-empty when records
        # is non-empty; we exercise n=1 explicitly here
        splits, _ = srs.split_records(_records_n(1), _build_cfg())
        self.assertEqual(len(splits["train"]), 1)


# === TESTS — load_records ===

class LoadRecordsTests(unittest.TestCase):

    def test_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "records.jsonl"
            with path.open("w", encoding="utf-8") as fh:
                for r in _records_n(3):
                    fh.write(json.dumps(r) + "\n")
            out = srs.load_records(path)
            self.assertEqual([r["example_id"] for r in out], ["ex0", "ex1", "ex2"])

    def test_missing_file_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                srs.load_records(Path(tmp) / "absent.jsonl")

    def test_empty_file_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "x.jsonl"
            path.write_text("", encoding="utf-8")
            with self.assertRaises(ValueError):
                srs.load_records(path)

    def test_malformed_json_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "x.jsonl"
            path.write_text('{"a":1}\n{not_json\n', encoding="utf-8")
            with self.assertRaises(ValueError):
                srs.load_records(path)


# === TESTS — write_splits / full IO ===

class WriteSplitsTests(unittest.TestCase):

    def test_full_pipeline_writes_all_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            records_path = tmp_path / "records.jsonl"
            output_dir   = tmp_path / "out"
            report_path  = output_dir / "split_report.json"

            with records_path.open("w", encoding="utf-8") as fh:
                for r in _records_n(20):
                    fh.write(json.dumps(r) + "\n")

            cfg = _build_cfg(
                records_path=records_path,
                output_dir=output_dir,
                report_path=report_path,
            )
            records = srs.load_records(cfg.records_path)
            splits, report = srs.split_records(records, cfg)
            srs.write_splits(splits, report, output_dir, report_path)

            for name in ("train", "val", "test"):
                split_path = output_dir / name / "records.jsonl"
                self.assertTrue(split_path.is_file(), f"{name} missing")
                with split_path.open() as fh:
                    rows = [json.loads(line) for line in fh if line.strip()]
                self.assertEqual(len(rows), report["counts"][name])

            self.assertTrue(report_path.is_file())
            with report_path.open() as fh:
                report_data = json.load(fh)
            self.assertEqual(report_data["num_records"], 20)
            self.assertEqual(
                report_data["counts"]["train"] + report_data["counts"]["val"]
                + report_data["counts"]["test"],
                20,
            )

    def test_creates_parent_directories(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir  = Path(tmp) / "deep" / "nested" / "out"
            report_path = output_dir / "report.json"
            empty_splits = {name: [] for name in ("train", "val", "test")}
            empty_splits["train"] = [_record("ex1")]
            srs.write_splits(empty_splits, {"x": 1}, output_dir, report_path)
            self.assertTrue((output_dir / "train" / "records.jsonl").is_file())
            self.assertTrue((output_dir / "val"   / "records.jsonl").is_file())
            self.assertTrue((output_dir / "test"  / "records.jsonl").is_file())
            self.assertTrue(report_path.is_file())


# === MAIN ===

def main() -> None:
    unittest.main()


if __name__ == "__main__":
    main()
