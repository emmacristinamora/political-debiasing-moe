# tests/test_router_prompt_set.py


# === IMPORTS ===

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any


# === MODULE LOADING ===

# build_router_prompt_set lives in src/. add src to sys.path so the module can
# be imported by name without making src a package.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_DIR = _REPO_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import build_router_prompt_set as brp  # noqa: E402


# === HELPERS ===

def _method12_records() -> list[dict]:
    return [
        {"id": "stmt_1", "text": "All taxation is theft.", "axis": "economic"},
        {"id": "stmt_2", "text": "Public healthcare is a right.", "axis": "economic"},
        {"id": "stmt_3", "text": "Authority should never be questioned.", "axis": "social"},
    ]


def _method3_records() -> list[dict]:
    return [
        {
            "key": "Q_A",
            "question": "How much trust do you place in the government?",
            "options": ["A lot", "Some", "Very little", "Refused"],
            "axis_relevance": "social",
        },
        {
            "key": "Q_B",
            "question": "Should the rich pay more taxes?",
            "options": ["Strongly agree", "Agree", "Disagree", "Strongly disagree", "REFUSED"],
            "axis_relevance": "economic",
        },
    ]


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _read_jsonl(path: Path) -> list[dict]:
    out: list[dict] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


# === TESTS — PROMPT FORMATTING ===

class FormatPromptTests(unittest.TestCase):

    def test_method12_prompt_uses_exact_template(self) -> None:
        out = brp._format_method12_prompt("Statement A.")
        expected = (
            "You are responding to a political attitude statement.\n\n"
            "Statement:\nStatement A.\n\n"
            "Write a concise 2-3 sentence response explaining your position."
        )
        self.assertEqual(out, expected)

    def test_method12_prompt_strips_whitespace(self) -> None:
        out = brp._format_method12_prompt("  trimmed  ")
        self.assertIn("Statement:\ntrimmed\n\n", out)
        self.assertFalse(out.endswith(" "))

    def test_method12_empty_text_raises(self) -> None:
        with self.assertRaises(ValueError):
            brp._format_method12_prompt("   ")

    def test_method3_prompt_uses_exact_template(self) -> None:
        out = brp._format_method3_prompt("Q?", ["alpha", "beta"])
        expected = (
            "Answer the following survey question by choosing exactly one option.\n\n"
            "Question:\nQ?\n\n"
            "Options:\nA. alpha\nB. beta\n\n"
            "Answer:"
        )
        self.assertEqual(out, expected)

    def test_method3_filters_refused_case_insensitive(self) -> None:
        out = brp._format_method3_prompt(
            "Q?", ["alpha", "Refused", "beta", "REFUSED", "gamma"]
        )
        self.assertIn("A. alpha", out)
        self.assertIn("B. beta", out)
        self.assertIn("C. gamma", out)
        self.assertNotIn("Refused", out)
        self.assertNotIn("REFUSED", out)

    def test_method3_all_options_refused_raises(self) -> None:
        with self.assertRaises(ValueError):
            brp._format_method3_prompt("Q?", ["Refused", "REFUSED"])


# === TESTS — BUILD RECORDS (PURE LOGIC) ===

class BuildPromptRecordsTests(unittest.TestCase):

    def test_happy_path_combines_sources_with_correct_schema(self) -> None:
        records = brp.build_prompt_records(
            method12_records=_method12_records(),
            method3_records=_method3_records(),
            method12_source_file="m12.jsonl",
            method3_source_file="m3.jsonl",
            max_prompts=None,
        )
        # 3 method12 + 2 method3
        self.assertEqual(len(records), 5)

        for i, rec in enumerate(records, start=1):
            self.assertEqual(set(rec.keys()), {"example_id", "prompt_text", "source", "metadata"})
            self.assertEqual(rec["example_id"], f"router_prompt_{i:06d}")
            self.assertIsInstance(rec["prompt_text"], str)
            self.assertNotEqual(rec["prompt_text"], "")
            self.assertFalse(rec["prompt_text"].endswith(" "))
            self.assertIn(rec["source"], {"method12", "method3"})
            md = rec["metadata"]
            self.assertEqual(set(md.keys()), {"original_id", "axis", "source_file"})

        # method12 prompts come first; method3 after — deterministic ordering
        self.assertEqual([r["source"] for r in records[:3]], ["method12"] * 3)
        self.assertEqual([r["source"] for r in records[3:]], ["method3"] * 2)
        self.assertEqual(records[0]["metadata"]["original_id"], "stmt_1")
        self.assertEqual(records[0]["metadata"]["axis"], "economic")
        self.assertEqual(records[0]["metadata"]["source_file"], "m12.jsonl")
        self.assertEqual(records[3]["metadata"]["original_id"], "Q_A")
        self.assertEqual(records[3]["metadata"]["axis"], "social")
        self.assertEqual(records[3]["metadata"]["source_file"], "m3.jsonl")

    def test_determinism_two_runs_produce_identical_output(self) -> None:
        a = brp.build_prompt_records(
            method12_records=_method12_records(),
            method3_records=_method3_records(),
            method12_source_file="m12.jsonl",
            method3_source_file="m3.jsonl",
            max_prompts=None,
        )
        b = brp.build_prompt_records(
            method12_records=_method12_records(),
            method3_records=_method3_records(),
            method12_source_file="m12.jsonl",
            method3_source_file="m3.jsonl",
            max_prompts=None,
        )
        self.assertEqual(a, b)

    def test_max_prompts_truncates_after_merge(self) -> None:
        records = brp.build_prompt_records(
            method12_records=_method12_records(),
            method3_records=_method3_records(),
            method12_source_file="m12.jsonl",
            method3_source_file="m3.jsonl",
            max_prompts=4,
        )
        self.assertEqual(len(records), 4)
        # method12 (3) appears first, method3 (1) tail
        self.assertEqual([r["source"] for r in records], ["method12"] * 3 + ["method3"])
        self.assertEqual(records[3]["metadata"]["original_id"], "Q_A")

    def test_method3_only_when_method12_disabled(self) -> None:
        records = brp.build_prompt_records(
            method12_records=None,
            method3_records=_method3_records(),
            method12_source_file=None,
            method3_source_file="m3.jsonl",
            max_prompts=None,
        )
        self.assertEqual({r["source"] for r in records}, {"method3"})
        self.assertEqual(len(records), 2)

    def test_no_sources_raises(self) -> None:
        with self.assertRaises(ValueError):
            brp.build_prompt_records(
                method12_records=None,
                method3_records=None,
                method12_source_file=None,
                method3_source_file=None,
                max_prompts=None,
            )

    def test_max_prompts_zero_raises(self) -> None:
        with self.assertRaises(ValueError):
            brp.build_prompt_records(
                method12_records=_method12_records(),
                method3_records=None,
                method12_source_file="m12.jsonl",
                method3_source_file=None,
                max_prompts=0,
            )


# === TESTS — DATA LOADING + VALIDATION ===

class LoaderValidationTests(unittest.TestCase):

    def test_method12_missing_required_field_raises(self) -> None:
        bad = [{"id": "s1", "text": "x"}]   # axis missing
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad_m12.jsonl"
            _write_json(path, {"statements": bad})
            with self.assertRaises(ValueError) as ctx:
                brp.load_method12_records(path)
            self.assertIn("axis", str(ctx.exception))

    def test_method3_missing_required_field_raises(self) -> None:
        bad = [{"key": "k1", "question": "q?"}]  # options missing
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad_m3.jsonl"
            _write_json(path, {"questions": bad})
            with self.assertRaises(ValueError) as ctx:
                brp.load_method3_records(path)
            self.assertIn("options", str(ctx.exception))

    def test_method12_loader_accepts_top_level_list(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "m12.jsonl"
            _write_json(path, _method12_records())
            out = brp.load_method12_records(path)
            self.assertEqual(len(out), 3)

    def test_method12_loader_accepts_dict_with_statements_key(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "m12.jsonl"
            _write_json(path, {"meta": {}, "statements": _method12_records()})
            out = brp.load_method12_records(path)
            self.assertEqual(len(out), 3)


# === TESTS — END-TO-END WRITE ===

class WritePromptsTests(unittest.TestCase):

    def test_jsonl_file_is_written_with_correct_schema(self) -> None:
        records = brp.build_prompt_records(
            method12_records=_method12_records(),
            method3_records=_method3_records(),
            method12_source_file="m12.jsonl",
            method3_source_file="m3.jsonl",
            max_prompts=None,
        )
        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "subdir" / "prompts.jsonl"
            brp.write_prompts_jsonl(records, out_path)
            self.assertTrue(out_path.is_file())
            loaded = _read_jsonl(out_path)
            self.assertEqual(len(loaded), 5)
            self.assertEqual(loaded, records)

    def test_run_build_via_synthetic_config(self) -> None:
        """End-to-end exercise of run_build using a faked RouterCalibrationConfig."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            m12_path = tmp_path / "m12.jsonl"
            m3_path  = tmp_path / "m3.jsonl"
            _write_json(m12_path, {"statements": _method12_records()})
            _write_json(m3_path,  {"questions":  _method3_records()})

            cfg = _SyntheticConfig(
                method12_path=m12_path,
                method3_path=m3_path,
                prompts_path=tmp_path / "out" / "prompts.jsonl",
                include_method12=True,
                include_method3=True,
                max_prompts=None,
            )
            out_path = brp.run_build(cfg)
            self.assertTrue(out_path.is_file())
            loaded = _read_jsonl(out_path)
            self.assertEqual(len(loaded), 5)
            self.assertEqual(loaded[0]["metadata"]["source_file"], "m12.jsonl")
            self.assertEqual(loaded[3]["metadata"]["source_file"], "m3.jsonl")
            # method3 axis comes from axis_relevance fallback
            self.assertEqual(loaded[3]["metadata"]["axis"], "social")


# === SYNTHETIC CONFIG SHIM ===

class _SyntheticPromptSources:
    def __init__(self, method12_path: Path, method3_path: Path) -> None:
        self.method12_path = method12_path
        self.method3_path = method3_path
        self.expert_validation_dir = method12_path.parent


class _SyntheticPaths:
    def __init__(self, method12_path: Path, method3_path: Path, prompts_path: Path) -> None:
        self.prompt_sources = _SyntheticPromptSources(method12_path, method3_path)
        self.prompts_path = prompts_path


class _SyntheticPromptSet:
    def __init__(self, include_method12: bool, include_method3: bool, max_prompts: int | None) -> None:
        self.include_method12 = include_method12
        self.include_method3 = include_method3
        self.include_expert_validation = False
        self.max_prompts = max_prompts
        self.seed = 42


class _SyntheticConfig:
    """
    Minimal duck-typed stand-in for RouterCalibrationConfig — exposes only the
    attributes that build_router_prompt_set.run_build actually reads, so tests
    don't need a full config.yaml on disk.
    """
    def __init__(
        self,
        method12_path: Path,
        method3_path: Path,
        prompts_path: Path,
        include_method12: bool,
        include_method3: bool,
        max_prompts: int | None,
    ) -> None:
        self.paths = _SyntheticPaths(method12_path, method3_path, prompts_path)
        self.prompt_set = _SyntheticPromptSet(include_method12, include_method3, max_prompts)


# === MAIN ===

def main() -> None:
    unittest.main()


if __name__ == "__main__":
    main()
