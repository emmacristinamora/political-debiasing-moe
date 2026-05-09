# tests/test_router_dataset_validator.py


# === IMPORTS ===

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any


# === MODULE LOADING ===

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_DIR = _REPO_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from router_training import validator as vrd  # noqa: E402

CANONICAL = ("left_lib", "left_auth", "right_lib", "right_auth")


# === HELPERS ===

def _policy(**overrides: float) -> dict[str, float]:
    base = {"left_lib": 0.4, "left_auth": 0.3, "right_lib": 0.2, "right_auth": 0.1}
    base.update(overrides)
    return base


def _quadrant_scores(**overrides: float) -> dict[str, float]:
    base = {"left_lib": 0.5, "left_auth": 0.0, "right_lib": -0.3, "right_auth": -0.1}
    base.update(overrides)
    return base


def _make_record(
    eid: str = "ex1",
    *,
    row_index: int = 0,
    target: dict | None = None,
    quadrant: dict | None = None,
    bias_magnitude: float = 0.7,
    metadata: Any = None,
    hidden_filename: str = "hidden.pt",
) -> dict:
    record = {
        "example_id": eid,
        "prompt_text": f"prompt for {eid}",
        "quadrant_scores": quadrant if quadrant is not None else _quadrant_scores(),
        "bias_magnitude": bias_magnitude,
        "target_policy":  target if target is not None else _policy(),
        "hidden_representation_ref": f"{hidden_filename}:{row_index}",
    }
    if metadata is not None:
        record["metadata"] = metadata
    return record


def _fake_tensor(rows: int, cols: int) -> Any:
    """Duck-typed stand-in for a hidden tensor — just exposes .shape."""
    return SimpleNamespace(shape=(rows, cols))


# === TESTS — happy path ===

class HappyPathTests(unittest.TestCase):

    def test_valid_dataset_passes(self) -> None:
        records = [_make_record(f"ex{i}", row_index=i) for i in range(3)]
        # no exception expected
        vrd.validate_router_dataset(records, _fake_tensor(3, 128))

    def test_with_metadata_dict(self) -> None:
        records = [_make_record(metadata={"axis": "economic"})]
        vrd.validate_router_dataset(records, _fake_tensor(1, 128))

    def test_no_hidden_tensor_passes_with_structural_checks(self) -> None:
        records = [_make_record(f"ex{i}", row_index=i) for i in range(3)]
        # hidden_tensor=None → tensor checks skipped; structural checks run
        vrd.validate_router_dataset(records, hidden_tensor=None)

    def test_filename_match_succeeds(self) -> None:
        records = [_make_record(hidden_filename="custom.pt")]
        vrd.validate_router_dataset(
            records, _fake_tensor(1, 128), hidden_filename="custom.pt",
        )

    def test_unused_rows_allowed(self) -> None:
        records = [_make_record("ex1", row_index=0)]
        # 1 record but 5 rows in tensor — extra rows are fine
        vrd.validate_router_dataset(records, _fake_tensor(5, 128))


# === TESTS — record-level violations ===

class RecordValidationTests(unittest.TestCase):

    def test_empty_records_raises(self) -> None:
        with self.assertRaises(ValueError):
            vrd.validate_router_dataset([], None)

    def test_non_list_records_raises(self) -> None:
        with self.assertRaises(ValueError):
            vrd.validate_router_dataset({"x": 1}, None)  # type: ignore[arg-type]

    def test_duplicate_example_id_raises(self) -> None:
        records = [_make_record("ex1", row_index=0), _make_record("ex1", row_index=1)]
        with self.assertRaises(ValueError) as ctx:
            vrd.validate_router_dataset(records, _fake_tensor(2, 128))
        self.assertIn("duplicate example_id", str(ctx.exception))

    def test_missing_required_field_raises(self) -> None:
        for missing in (
            "example_id", "prompt_text", "quadrant_scores",
            "bias_magnitude", "target_policy", "hidden_representation_ref",
        ):
            record = _make_record()
            del record[missing]
            with self.assertRaises(ValueError) as ctx:
                vrd.validate_router_dataset([record], _fake_tensor(1, 128))
            self.assertIn(missing, str(ctx.exception))

    def test_non_string_example_id_raises(self) -> None:
        record = _make_record()
        record["example_id"] = 42
        with self.assertRaises(ValueError):
            vrd.validate_router_dataset([record], _fake_tensor(1, 128))

    def test_empty_prompt_text_raises(self) -> None:
        record = _make_record()
        record["prompt_text"] = "  "
        with self.assertRaises(ValueError):
            vrd.validate_router_dataset([record], _fake_tensor(1, 128))

    def test_non_finite_bias_magnitude_raises(self) -> None:
        record = _make_record()
        record["bias_magnitude"] = float("nan")
        with self.assertRaises(ValueError):
            vrd.validate_router_dataset([record], _fake_tensor(1, 128))

    def test_metadata_non_dict_raises(self) -> None:
        record = _make_record(metadata=["not", "a", "dict"])
        with self.assertRaises(ValueError) as ctx:
            vrd.validate_router_dataset([record], _fake_tensor(1, 128))
        self.assertIn("metadata", str(ctx.exception))


# === TESTS — quadrant_scores ===

class QuadrantScoresTests(unittest.TestCase):

    def test_missing_quadrant_key_raises(self) -> None:
        bad = _quadrant_scores()
        del bad["left_lib"]
        record = _make_record(quadrant=bad)
        with self.assertRaises(ValueError) as ctx:
            vrd.validate_router_dataset([record], _fake_tensor(1, 128))
        self.assertIn("left_lib", str(ctx.exception))

    def test_extra_quadrant_key_raises(self) -> None:
        bad = {**_quadrant_scores(), "extra": 0.0}
        record = _make_record(quadrant=bad)
        with self.assertRaises(ValueError) as ctx:
            vrd.validate_router_dataset([record], _fake_tensor(1, 128))
        self.assertIn("unexpected", str(ctx.exception).lower())

    def test_non_finite_quadrant_score_raises(self) -> None:
        bad = {**_quadrant_scores(), "left_lib": float("inf")}
        record = _make_record(quadrant=bad)
        with self.assertRaises(ValueError):
            vrd.validate_router_dataset([record], _fake_tensor(1, 128))


# === TESTS — target_policy ===

class TargetPolicyTests(unittest.TestCase):

    def test_zero_value_raises(self) -> None:
        bad = {**_policy(), "right_auth": 0.0}
        record = _make_record(target=bad)
        with self.assertRaises(ValueError):
            vrd.validate_router_dataset([record], _fake_tensor(1, 128))

    def test_negative_value_raises(self) -> None:
        bad = {"left_lib": -0.1, "left_auth": 0.5, "right_lib": 0.4, "right_auth": 0.2}
        record = _make_record(target=bad)
        with self.assertRaises(ValueError):
            vrd.validate_router_dataset([record], _fake_tensor(1, 128))

    def test_sum_not_one_raises(self) -> None:
        bad = {"left_lib": 0.5, "left_auth": 0.5, "right_lib": 0.5, "right_auth": 0.5}
        record = _make_record(target=bad)
        with self.assertRaises(ValueError) as ctx:
            vrd.validate_router_dataset([record], _fake_tensor(1, 128))
        self.assertIn("sum to 1", str(ctx.exception))

    def test_sum_within_tolerance_ok(self) -> None:
        ok = {"left_lib": 0.4 + 1e-7, "left_auth": 0.3, "right_lib": 0.2, "right_auth": 0.1 - 1e-7}
        record = _make_record(target=ok)
        vrd.validate_router_dataset([record], _fake_tensor(1, 128))


# === TESTS — hidden_representation_ref ===

class HiddenRefTests(unittest.TestCase):

    def test_missing_colon_raises(self) -> None:
        record = _make_record()
        record["hidden_representation_ref"] = "no_colon_here"
        with self.assertRaises(ValueError) as ctx:
            vrd.validate_router_dataset([record], _fake_tensor(1, 128))
        self.assertIn("must be '<filename>:<row_index>'", str(ctx.exception))

    def test_non_int_row_index_raises(self) -> None:
        record = _make_record()
        record["hidden_representation_ref"] = "hidden.pt:not_int"
        with self.assertRaises(ValueError):
            vrd.validate_router_dataset([record], _fake_tensor(1, 128))

    def test_negative_row_index_raises(self) -> None:
        record = _make_record()
        record["hidden_representation_ref"] = "hidden.pt:-1"
        with self.assertRaises(ValueError):
            vrd.validate_router_dataset([record], _fake_tensor(1, 128))

    def test_non_string_ref_raises(self) -> None:
        record = _make_record()
        record["hidden_representation_ref"] = 42
        with self.assertRaises(ValueError):
            vrd.validate_router_dataset([record], _fake_tensor(1, 128))

    def test_empty_string_ref_raises(self) -> None:
        record = _make_record()
        record["hidden_representation_ref"] = ""
        with self.assertRaises(ValueError):
            vrd.validate_router_dataset([record], _fake_tensor(1, 128))

    def test_filename_mismatch_raises(self) -> None:
        record = _make_record(hidden_filename="other.pt")
        with self.assertRaises(ValueError) as ctx:
            vrd.validate_router_dataset(
                [record], _fake_tensor(1, 128), hidden_filename="hidden.pt",
            )
        self.assertIn("does not match", str(ctx.exception))

    def test_out_of_range_row_index_raises(self) -> None:
        # tensor only has 5 rows; record references row 10
        record = _make_record(row_index=10)
        with self.assertRaises(ValueError) as ctx:
            vrd.validate_router_dataset([record], _fake_tensor(5, 128))
        msg = str(ctx.exception)
        self.assertIn("out of range", msg)
        self.assertIn("max=4", msg)

    def test_no_tensor_skips_index_check(self) -> None:
        # without a tensor, the index range check is skipped (just structural)
        record = _make_record(row_index=99999)
        vrd.validate_router_dataset([record], hidden_tensor=None)


# === TESTS — hidden tensor / cross-consistency ===

class HiddenTensorTests(unittest.TestCase):

    def test_hidden_dim_mismatch_raises(self) -> None:
        records = [_make_record()]
        # tensor has 64 cols but expected is 128
        with self.assertRaises(ValueError) as ctx:
            vrd.validate_router_dataset(
                records, _fake_tensor(1, 64), expected_hidden_dim=128,
            )
        self.assertIn("hidden_dim", str(ctx.exception))

    def test_hidden_dim_match_ok(self) -> None:
        records = [_make_record()]
        vrd.validate_router_dataset(
            records, _fake_tensor(1, 128), expected_hidden_dim=128,
        )

    def test_records_exceed_tensor_rows_raises(self) -> None:
        records = [_make_record(f"ex{i}", row_index=i) for i in range(3)]
        with self.assertRaises(ValueError) as ctx:
            vrd.validate_router_dataset(records, _fake_tensor(2, 128))
        # the row-index range check fires first for the third record
        # (row_index=2 >= max=1), which is also a valid pre-cross-check failure
        self.assertIn("out of range", str(ctx.exception))

    def test_tensor_without_shape_raises(self) -> None:
        with self.assertRaises(ValueError):
            vrd.validate_router_dataset(
                [_make_record()], hidden_tensor=object(),  # no .shape
            )


# === TESTS — helper functions ===

class HelperFunctionTests(unittest.TestCase):

    def test_parse_hidden_ref_happy(self) -> None:
        self.assertEqual(vrd.parse_hidden_ref("hidden.pt:42"), ("hidden.pt", 42))

    def test_parse_hidden_ref_path_with_slashes(self) -> None:
        # rpartition on the last colon — directory separators stay in filename
        self.assertEqual(
            vrd.parse_hidden_ref("data/router/hidden.pt:7"),
            ("data/router/hidden.pt", 7),
        )

    def test_validate_quadrant_dict_finite_score(self) -> None:
        vrd.validate_quadrant_dict(
            {"left_lib": -0.3, "left_auth": 0.0, "right_lib": 0.5, "right_auth": 0.1},
            "quadrant_scores", "ex",
        )  # negative scores are fine for the score variant

    def test_validate_probability_dict_zero_raises(self) -> None:
        with self.assertRaises(ValueError):
            vrd.validate_probability_dict(
                {"left_lib": 0.0, "left_auth": 0.5, "right_lib": 0.3, "right_auth": 0.2},
                "target_policy", "ex",
            )


# === TESTS — torch-availability behavior ===

class TorchAvailabilityTests(unittest.TestCase):

    def test_structural_validation_accepts_none_hidden_tensor(self) -> None:
        # structural checks must pass and hidden_tensor=None must be accepted
        # regardless of whether torch is installed.
        records = [_make_record(f"ex{i}", row_index=i) for i in range(2)]
        vrd.validate_router_dataset(records, hidden_tensor=None)

    def test_validate_hidden_tensor_without_torch_raises(self) -> None:
        # the strict tensor-internals helper requires torch; without it the
        # function refuses to silently no-op and raises a clear error
        if vrd._try_import_torch() is not None:
            self.skipTest("torch is available; this branch is only reachable without torch")
        with self.assertRaises(ValueError) as ctx:
            vrd.validate_hidden_tensor(_fake_tensor(1, 128), expected_hidden_dim=128)
        self.assertIn("torch", str(ctx.exception).lower())


# === TESTS — load_records_jsonl ===

class LoadRecordsJsonlTests(unittest.TestCase):

    def test_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "records.jsonl"
            with path.open("w", encoding="utf-8") as fh:
                for i in range(3):
                    fh.write(json.dumps(_make_record(f"ex{i}", row_index=i)) + "\n")
            records = vrd.load_records_jsonl(path)
            self.assertEqual(len(records), 3)
            self.assertEqual(records[0]["example_id"], "ex0")

    def test_missing_file_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                vrd.load_records_jsonl(Path(tmp) / "nope.jsonl")

    def test_empty_file_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "x.jsonl"
            path.write_text("", encoding="utf-8")
            with self.assertRaises(ValueError):
                vrd.load_records_jsonl(path)

    def test_malformed_json_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "x.jsonl"
            path.write_text('{"a":1}\n{not_json\n', encoding="utf-8")
            with self.assertRaises(ValueError):
                vrd.load_records_jsonl(path)


# === TESTS — full file-to-validator round trip ===

class FullPipelineTests(unittest.TestCase):

    def test_load_then_validate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "records.jsonl"
            with path.open("w", encoding="utf-8") as fh:
                for i in range(3):
                    fh.write(json.dumps(_make_record(f"ex{i}", row_index=i)) + "\n")
            records = vrd.load_records_jsonl(path)
            vrd.validate_router_dataset(
                records, _fake_tensor(3, 128), expected_hidden_dim=128,
                hidden_filename="hidden.pt",
            )


# === MAIN ===

def main() -> None:
    unittest.main()


if __name__ == "__main__":
    main()
