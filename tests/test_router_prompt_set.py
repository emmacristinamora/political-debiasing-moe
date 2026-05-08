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
import router_calibration_config as rcc  # noqa: E402


# === HELPERS ===

CANONICAL_QUADRANTS = ("left_lib", "left_auth", "right_lib", "right_auth")


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


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _read_jsonl(path: Path) -> list[dict]:
    out: list[dict] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _make_expert_val_chunk(
    chunk_id: str,
    document_id: str,
    text: str,
    *,
    source: str = "allsides",
    topic_label: str = "economy",
) -> dict:
    return {
        "chunk_id": chunk_id,
        "document_id": document_id,
        "chunk_index": 0,
        "text": text,
        "source": source,
        "topic_label": topic_label,
    }


def _materialise_expert_val_root(
    root: Path,
    *,
    splits: tuple[str, ...] = ("val_indist", "val_source", "val_topic"),
    rows_per_file: int = 1,
) -> None:
    """Lay down the canonical 4-quadrant × N-split JSONL tree under root."""
    for q_idx, quadrant in enumerate(CANONICAL_QUADRANTS):
        for s_idx, split in enumerate(splits):
            rows = [
                _make_expert_val_chunk(
                    chunk_id=f"{quadrant}_{split}_chunk{i:02d}",
                    document_id=f"{quadrant}_{split}_doc{i:02d}",
                    text=f"Body for {quadrant}/{split} #{i}.",
                    topic_label="economy" if (q_idx + s_idx + i) % 2 == 0 else "social",
                )
                for i in range(rows_per_file)
            ]
            _write_jsonl(root / quadrant / f"{split}.jsonl", rows)


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
        prompt, num_options = brp._format_method3_prompt("Q?", ["alpha", "beta"])
        expected = (
            "Answer the following survey question by choosing exactly one option.\n\n"
            "Question:\nQ?\n\n"
            "Options:\nA. alpha\nB. beta\n\n"
            "Answer:"
        )
        self.assertEqual(prompt, expected)
        self.assertEqual(num_options, 2)

    def test_method3_filters_refused_case_insensitive(self) -> None:
        prompt, num_options = brp._format_method3_prompt(
            "Q?", ["alpha", "Refused", "beta", "REFUSED", "gamma"]
        )
        self.assertIn("A. alpha", prompt)
        self.assertIn("B. beta", prompt)
        self.assertIn("C. gamma", prompt)
        self.assertNotIn("Refused", prompt)
        self.assertNotIn("REFUSED", prompt)
        self.assertEqual(num_options, 3)

    def test_method3_all_options_refused_raises(self) -> None:
        with self.assertRaises(ValueError):
            brp._format_method3_prompt("Q?", ["Refused", "REFUSED"])

    def test_method3_too_many_options_raises(self) -> None:
        too_many = [f"opt_{i}" for i in range(27)]
        with self.assertRaises(ValueError):
            brp._format_method3_prompt("Q?", too_many)

    def test_expert_validation_prompt_uses_exact_template(self) -> None:
        out = brp._format_expert_validation_prompt("Body text.")
        expected = (
            "Read the following political text excerpt and write a concise, "
            "balanced 2-3 sentence response that addresses its main claim "
            "without adopting a partisan stance.\n\n"
            "Excerpt:\nBody text.\n\n"
            "Response:"
        )
        self.assertEqual(out, expected)

    def test_expert_validation_empty_text_raises(self) -> None:
        with self.assertRaises(ValueError):
            brp._format_expert_validation_prompt("   ")


# === TESTS — BUILD RECORDS (PURE LOGIC) ===

class BuildPromptRecordsTests(unittest.TestCase):

    def test_happy_path_combines_method12_and_method3_with_correct_schema(self) -> None:
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

        # method12: exactly the 3 historic metadata keys
        for rec in records[:3]:
            self.assertEqual(
                set(rec["metadata"].keys()),
                {"original_id", "axis", "source_file"},
            )
        # method3: now also carries question_key + num_options per spec
        for rec in records[3:]:
            self.assertEqual(
                set(rec["metadata"].keys()),
                {"original_id", "axis", "source_file", "question_key", "num_options"},
            )

        # method12 prompts come first; method3 after — deterministic ordering
        self.assertEqual([r["source"] for r in records[:3]], ["method12"] * 3)
        self.assertEqual([r["source"] for r in records[3:]], ["method3"] * 2)
        self.assertEqual(records[0]["metadata"]["original_id"], "stmt_1")
        self.assertEqual(records[0]["metadata"]["axis"], "economic")
        self.assertEqual(records[0]["metadata"]["source_file"], "m12.jsonl")
        self.assertEqual(records[3]["metadata"]["original_id"], "Q_A")
        self.assertEqual(records[3]["metadata"]["axis"], "social")
        self.assertEqual(records[3]["metadata"]["source_file"], "m3.jsonl")
        self.assertEqual(records[3]["metadata"]["question_key"], "Q_A")
        self.assertEqual(records[3]["metadata"]["num_options"], 3)  # 4 - 1 refused
        self.assertEqual(records[4]["metadata"]["num_options"], 4)  # 5 - 1 REFUSED

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

    def test_max_prompts_truncates_after_merge_and_dedupe(self) -> None:
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
        # ids remain contiguous
        self.assertEqual(
            [r["example_id"] for r in records],
            [f"router_prompt_{i:06d}" for i in range(1, 5)],
        )

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

    def test_dedupe_keeps_first_occurrence_and_keeps_ids_contiguous(self) -> None:
        # the first method12 statement and a method3 prompt that happens to
        # produce identical text would be a contrived case; instead duplicate
        # one method12 statement to exercise the dedup branch deterministically.
        m12_with_dup = _method12_records() + [
            {
                "id": "stmt_1_dup",
                "text": "All taxation is theft.",  # same text → same prompt body
                "axis": "economic",
            },
        ]
        records = brp.build_prompt_records(
            method12_records=m12_with_dup,
            method3_records=None,
            method12_source_file="m12.jsonl",
            method3_source_file=None,
            max_prompts=None,
        )
        # 4 inputs but stmt_1 / stmt_1_dup share text → 3 outputs, first kept
        self.assertEqual(len(records), 3)
        self.assertEqual(records[0]["metadata"]["original_id"], "stmt_1")
        # ids 1..3 contiguous, no gaps from removed duplicate
        self.assertEqual(
            [r["example_id"] for r in records],
            [f"router_prompt_{i:06d}" for i in range(1, 4)],
        )


# === TESTS — METHOD 3 (SPEC TEST GROUP A) ===

class Method3BuilderSpecTests(unittest.TestCase):

    def test_method3_file_with_two_questions_one_with_refused(self) -> None:
        """Spec test group A: synthetic method3 file with refused option."""
        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp = Path(raw_tmp)
            m3_path = tmp / "method3_synth.jsonl"
            payload = {
                "questions": [
                    {
                        "key": "Q_REFUSED",
                        "question": "Pick one.",
                        "options": ["yes", "no", "Refused"],
                    },
                    {
                        "key": "Q_PLAIN",
                        "question": "Choose your favourite.",
                        "options": ["red", "blue", "green"],
                        "axis_relevance": "social",
                    },
                ],
            }
            _write_json(m3_path, payload)
            records = brp.load_method3_records(m3_path)
            prompts = brp._build_method3_prompts(records, m3_path.name)

        # all 'Refused' variants are excluded from the rendered prompt body
        for prompt in prompts:
            self.assertNotIn("Refused", prompt["prompt_text"])

        self.assertEqual(prompts[0]["source"], "method3")
        self.assertEqual(prompts[0]["metadata"]["question_key"], "Q_REFUSED")
        self.assertEqual(prompts[0]["metadata"]["num_options"], 2)
        self.assertEqual(prompts[1]["metadata"]["question_key"], "Q_PLAIN")
        self.assertEqual(prompts[1]["metadata"]["num_options"], 3)
        # axis fallback: missing 'axis' falls through to axis_relevance
        self.assertEqual(prompts[1]["metadata"]["axis"], "social")

    def test_method3_all_options_refused_raises_with_question_key_in_run_build(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp = Path(raw_tmp)
            m3_path = tmp / "method3.jsonl"
            _write_json(m3_path, {"questions": [
                {"key": "Q_DEAD", "question": "?", "options": ["Refused", "refused"]},
            ]})
            records = brp.load_method3_records(m3_path)
            with self.assertRaises(ValueError):
                brp._build_method3_prompts(records, m3_path.name)


# === TESTS — EXPERT VALIDATION (SPEC TEST GROUP B) ===

class ExpertValidationBuilderTests(unittest.TestCase):

    def test_synthetic_expert_validation_root_produces_prompts_per_split(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp = Path(raw_tmp)
            root = tmp / "train-validate"
            _materialise_expert_val_root(root, rows_per_file=2)
            prompts, per_source = brp._load_expert_validation_prompts(
                root, ["val_indist", "val_source", "val_topic"],
            )

        # 4 quadrants × 3 splits × 2 rows = 24 prompts
        self.assertEqual(len(prompts), 24)
        # source names exactly match the spec
        self.assertEqual(
            {row["source"] for row in prompts},
            {"expert_val_indist", "expert_val_source", "expert_val_topic"},
        )
        # 8 prompts per split (4 quadrants × 2 rows)
        self.assertEqual(per_source["expert_val_indist"], 8)
        self.assertEqual(per_source["expert_val_source"], 8)
        self.assertEqual(per_source["expert_val_topic"], 8)

        # canonical quadrant order honoured within each split: the very first
        # row should belong to left_lib / val_indist.
        self.assertEqual(prompts[0]["metadata"]["quadrant"], "left_lib")
        self.assertEqual(prompts[0]["metadata"]["split"], "val_indist")

        # metadata schema matches spec exactly
        for rec in prompts:
            md = rec["metadata"]
            self.assertEqual(
                set(md.keys()),
                {
                    "original_id",
                    "axis",
                    "source_file",
                    "quadrant",
                    "split",
                    "document_id",
                    "chunk_id",
                    "topic_label",
                    "original_source",
                },
            )
            self.assertIsNone(md["axis"])
            self.assertIn(md["quadrant"], CANONICAL_QUADRANTS)
            self.assertIn(md["split"], {"val_indist", "val_source", "val_topic"})
            self.assertIsNotNone(md["chunk_id"])
            self.assertIsNotNone(md["document_id"])
            self.assertEqual(md["original_source"], "allsides")
            # original_id prefers chunk_id when present
            self.assertEqual(md["original_id"], md["chunk_id"])

    def test_original_id_falls_back_to_document_id_then_local(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp = Path(raw_tmp)
            root = tmp / "train-validate"
            file_path = root / "left_lib" / "val_indist.jsonl"
            rows = [
                {  # has only document_id
                    "document_id": "doc_only_1",
                    "text": "doc-only fallback case.",
                    "source": "allsides",
                    "topic_label": "economy",
                },
                {  # has neither chunk_id nor document_id → deterministic local id
                    "text": "no-id fallback case.",
                    "topic_label": "economy",
                },
            ]
            _write_jsonl(file_path, rows)
            prompts, _ = brp._load_expert_validation_prompts(root, ["val_indist"])

        self.assertEqual(len(prompts), 2)
        self.assertEqual(prompts[0]["metadata"]["original_id"], "doc_only_1")
        self.assertIsNone(prompts[0]["metadata"]["chunk_id"])
        # local fallback id is deterministic from (quadrant, split, lineno)
        self.assertEqual(prompts[1]["metadata"]["original_id"], "left_lib/val_indist#2")
        self.assertIsNone(prompts[1]["metadata"]["original_source"])

    def test_missing_text_raises_value_error_naming_file_and_line(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp = Path(raw_tmp)
            file_path = tmp / "left_lib" / "val_indist.jsonl"
            _write_jsonl(file_path, [{"chunk_id": "c0", "text": "ok."}, {"chunk_id": "c1", "text": "  "}])
            with self.assertRaises(ValueError) as ctx:
                brp._build_expert_validation_prompts_for_file(
                    file_path=file_path, quadrant="left_lib", split="val_indist",
                )
        self.assertIn(":2", str(ctx.exception))
        self.assertIn("text", str(ctx.exception))

    def test_malformed_json_raises_value_error_naming_line(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp = Path(raw_tmp)
            file_path = tmp / "left_lib" / "val_indist.jsonl"
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(
                '{"chunk_id":"c0","text":"ok."}\nnot-json-here\n',
                encoding="utf-8",
            )
            with self.assertRaises(ValueError) as ctx:
                brp._build_expert_validation_prompts_for_file(
                    file_path=file_path, quadrant="left_lib", split="val_indist",
                )
        self.assertIn(":2", str(ctx.exception))


# === TESTS — CONFIG DEFAULTS / BACKWARDS COMPAT (GROUP C) ===

class ConfigDefaultsTests(unittest.TestCase):

    def test_minimal_prompt_set_block_loads_with_safe_defaults(self) -> None:
        # mimic _parse_prompt_set's contract: block without
        # include_expert_validation / expert_validation_splits should still
        # parse, defaulting to False / all three splits.
        cfg = rcc._parse_prompt_set({
            "include_method12": True,
            "include_method3":  False,
            "max_prompts": None,
            "seed": 42,
        })
        self.assertFalse(cfg.include_expert_validation)
        self.assertEqual(
            list(cfg.expert_validation_splits),
            list(rcc.DEFAULT_EXPERT_VALIDATION_SPLITS),
        )

    def test_explicit_include_expert_validation_respected(self) -> None:
        cfg = rcc._parse_prompt_set({
            "include_method12": True,
            "include_method3":  True,
            "include_expert_validation": True,
            "max_prompts": None,
            "seed": 42,
        })
        self.assertTrue(cfg.include_expert_validation)

    def test_explicit_expert_validation_splits_subset(self) -> None:
        cfg = rcc._parse_prompt_set({
            "include_method12": True,
            "include_method3":  False,
            "include_expert_validation": True,
            "expert_validation_splits": ["val_indist"],
            "max_prompts": None,
            "seed": 42,
        })
        self.assertEqual(list(cfg.expert_validation_splits), ["val_indist"])

    def test_invalid_split_raises(self) -> None:
        with self.assertRaises(ValueError):
            rcc._parse_prompt_set({
                "include_method12": True,
                "include_method3":  False,
                "include_expert_validation": True,
                "expert_validation_splits": ["val_unknown"],
                "max_prompts": None,
                "seed": 42,
            })


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


# === TESTS — END-TO-END run_build ===

class RunBuildIntegrationTests(unittest.TestCase):

    def _setup_full_synthetic_layout(
        self,
        tmp: Path,
        *,
        include_expert_validation: bool,
        rows_per_val_file: int = 1,
    ) -> tuple[Any, Path]:
        m12_path = tmp / "m12.jsonl"
        m3_path  = tmp / "m3.jsonl"
        _write_json(m12_path, {"statements": _method12_records()})
        _write_json(m3_path,  {"questions":  _method3_records()})
        val_root = tmp / "train-validate"
        if include_expert_validation:
            _materialise_expert_val_root(val_root, rows_per_file=rows_per_val_file)
        prompts_path = tmp / "out" / "prompts.jsonl"
        cfg = _SyntheticConfig(
            method12_path=m12_path,
            method3_path=m3_path,
            expert_validation_dir=val_root,
            prompts_path=prompts_path,
            include_method12=True,
            include_method3=True,
            include_expert_validation=include_expert_validation,
            max_prompts=None,
        )
        return cfg, prompts_path

    def test_run_build_method12_only_remains_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp = Path(raw_tmp)
            m12_path = tmp / "m12.jsonl"
            _write_json(m12_path, {"statements": _method12_records()})
            cfg = _SyntheticConfig(
                method12_path=m12_path,
                method3_path=tmp / "missing-m3.jsonl",
                expert_validation_dir=tmp / "missing-val",
                prompts_path=tmp / "out" / "prompts.jsonl",
                include_method12=True,
                include_method3=False,
                include_expert_validation=False,
                max_prompts=None,
            )
            out_path = brp.run_build(cfg)
            self.assertTrue(out_path.is_file())
            loaded = _read_jsonl(out_path)
        self.assertEqual(len(loaded), 3)
        self.assertEqual({r["source"] for r in loaded}, {"method12"})

    def test_run_build_with_expert_validation_includes_all_three_split_sources(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp = Path(raw_tmp)
            cfg, prompts_path = self._setup_full_synthetic_layout(
                tmp, include_expert_validation=True, rows_per_val_file=1,
            )
            brp.run_build(cfg)
            loaded = _read_jsonl(prompts_path)

        # 3 method12 + 2 method3 + (4 quadrants × 3 splits × 1 row) = 5 + 12 = 17
        self.assertEqual(len(loaded), 17)
        sources_in_order = [r["source"] for r in loaded]
        # method12 → method3 → expert_val_indist → expert_val_source → expert_val_topic
        self.assertEqual(sources_in_order[:3], ["method12"] * 3)
        self.assertEqual(sources_in_order[3:5], ["method3"] * 2)
        self.assertEqual(sources_in_order[5:9],  ["expert_val_indist"] * 4)
        self.assertEqual(sources_in_order[9:13], ["expert_val_source"] * 4)
        self.assertEqual(sources_in_order[13:17], ["expert_val_topic"] * 4)

        # ids contiguous from 1..17
        self.assertEqual(
            [r["example_id"] for r in loaded],
            [f"router_prompt_{i:06d}" for i in range(1, 18)],
        )

    def test_run_build_two_runs_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp = Path(raw_tmp)
            cfg, prompts_path = self._setup_full_synthetic_layout(
                tmp, include_expert_validation=True,
            )
            brp.run_build(cfg)
            first = prompts_path.read_text(encoding="utf-8")
            # second run with same inputs
            brp.run_build(cfg)
            second = prompts_path.read_text(encoding="utf-8")
        self.assertEqual(first, second)

    def test_run_build_max_prompts_caps_after_dedupe(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp = Path(raw_tmp)
            cfg, prompts_path = self._setup_full_synthetic_layout(
                tmp, include_expert_validation=True,
            )
            cfg.prompt_set.max_prompts = 6
            brp.run_build(cfg)
            loaded = _read_jsonl(prompts_path)
        self.assertEqual(len(loaded), 6)
        # source order preserved: 3 method12, then 2 method3, then 1 expert_val_indist
        self.assertEqual(
            [r["source"] for r in loaded],
            ["method12"] * 3 + ["method3"] * 2 + ["expert_val_indist"],
        )
        # ids contiguous
        self.assertEqual(
            [r["example_id"] for r in loaded],
            [f"router_prompt_{i:06d}" for i in range(1, 7)],
        )

    def test_run_build_dedupes_text_across_sources(self) -> None:
        # craft a method12 statement whose rendered prompt collides with
        # nothing else, then duplicate it inside method12 itself, then enable
        # expert val too — final dedup should remove the duplicate keeping
        # the first occurrence.
        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp = Path(raw_tmp)
            m12_records = _method12_records() + [
                {"id": "stmt_dup", "text": "All taxation is theft.", "axis": "economic"},
            ]
            m12_path = tmp / "m12.jsonl"
            _write_json(m12_path, {"statements": m12_records})
            m3_path  = tmp / "m3.jsonl"
            _write_json(m3_path, {"questions": []})
            val_root = tmp / "train-validate"
            _materialise_expert_val_root(val_root, rows_per_file=1)
            cfg = _SyntheticConfig(
                method12_path=m12_path,
                method3_path=m3_path,
                expert_validation_dir=val_root,
                prompts_path=tmp / "out" / "prompts.jsonl",
                include_method12=True,
                include_method3=False,  # empty list anyway
                include_expert_validation=True,
                max_prompts=None,
            )
            brp.run_build(cfg)
            loaded = _read_jsonl(tmp / "out" / "prompts.jsonl")

        # 4 method12 inputs, one duplicated → 3 method12 prompts kept
        method12_rows = [r for r in loaded if r["source"] == "method12"]
        self.assertEqual(len(method12_rows), 3)
        # duplicate's metadata never appears
        self.assertNotIn(
            "stmt_dup",
            {r["metadata"]["original_id"] for r in method12_rows},
        )
        # ids still contiguous across the merged set
        self.assertEqual(
            [r["example_id"] for r in loaded],
            [f"router_prompt_{i:06d}" for i in range(1, len(loaded) + 1)],
        )

    def test_run_build_method3_missing_file_raises(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp = Path(raw_tmp)
            m12_path = tmp / "m12.jsonl"
            _write_json(m12_path, {"statements": _method12_records()})
            cfg = _SyntheticConfig(
                method12_path=m12_path,
                method3_path=tmp / "no-such-m3.jsonl",
                expert_validation_dir=tmp / "irrelevant",
                prompts_path=tmp / "out" / "prompts.jsonl",
                include_method12=True,
                include_method3=True,
                include_expert_validation=False,
                max_prompts=None,
            )
            with self.assertRaises(FileNotFoundError):
                brp.run_build(cfg)

    def test_run_build_expert_validation_missing_root_raises(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp = Path(raw_tmp)
            m12_path = tmp / "m12.jsonl"
            _write_json(m12_path, {"statements": _method12_records()})
            cfg = _SyntheticConfig(
                method12_path=m12_path,
                method3_path=tmp / "m3.jsonl",
                expert_validation_dir=tmp / "no-such-val-root",
                prompts_path=tmp / "out" / "prompts.jsonl",
                include_method12=True,
                include_method3=False,
                include_expert_validation=True,
                max_prompts=None,
            )
            with self.assertRaises(FileNotFoundError):
                brp.run_build(cfg)


# === END-TO-END WRITE ===

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


# === SYNTHETIC CONFIG SHIM ===

class _SyntheticPromptSources:
    def __init__(
        self,
        method12_path: Path,
        method3_path: Path,
        expert_validation_dir: Path,
    ) -> None:
        self.method12_path = method12_path
        self.method3_path = method3_path
        self.expert_validation_dir = expert_validation_dir


class _SyntheticPaths:
    def __init__(
        self,
        method12_path: Path,
        method3_path: Path,
        expert_validation_dir: Path,
        prompts_path: Path,
    ) -> None:
        self.prompt_sources = _SyntheticPromptSources(
            method12_path, method3_path, expert_validation_dir,
        )
        self.prompts_path = prompts_path


class _SyntheticPromptSet:
    def __init__(
        self,
        include_method12: bool,
        include_method3: bool,
        include_expert_validation: bool,
        max_prompts: int | None,
    ) -> None:
        self.include_method12 = include_method12
        self.include_method3 = include_method3
        self.include_expert_validation = include_expert_validation
        self.expert_validation_splits = ["val_indist", "val_source", "val_topic"]
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
        expert_validation_dir: Path,
        prompts_path: Path,
        include_method12: bool,
        include_method3: bool,
        include_expert_validation: bool,
        max_prompts: int | None,
    ) -> None:
        self.paths = _SyntheticPaths(
            method12_path, method3_path, expert_validation_dir, prompts_path,
        )
        self.prompt_set = _SyntheticPromptSet(
            include_method12,
            include_method3,
            include_expert_validation,
            max_prompts,
        )


# === MAIN ===

def main() -> None:
    unittest.main()


if __name__ == "__main__":
    main()
