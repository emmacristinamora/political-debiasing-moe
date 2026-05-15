# tests/test_run_moce.py


# === IMPORTS ===

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml


# === MODULE LOADING ===

# src/10_run_moce.py begins with a digit, so it cannot be imported via normal
# "import" syntax. load it explicitly by absolute path with importlib.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_RUN_MOCE_PATH = _REPO_ROOT / "src" / "10_run_moce.py"
_REAL_CONFIG_PATH = _REPO_ROOT / "config" / "config.yaml"

_spec = importlib.util.spec_from_file_location("run_moce", _RUN_MOCE_PATH)
run_moce = importlib.util.module_from_spec(_spec)
sys.modules["run_moce"] = run_moce
_spec.loader.exec_module(run_moce)

CANONICAL = ("left_lib", "left_auth", "right_lib", "right_auth")


# === HELPERS ===

@contextlib.contextmanager
def _patched_argv(argv: list[str]):
    """Temporarily replace sys.argv so parse_args() reads from `argv`."""
    saved = sys.argv
    sys.argv = ["10_run_moce.py", *argv]
    try:
        yield
    finally:
        sys.argv = saved


def _parse_args_silently(argv: list[str]):
    """
    Run parse_args() with stderr suppressed.

    argparse prints usage messages and contract-violation errors to stderr;
    swallowing them keeps test output clean while preserving the SystemExit
    raise that callers assert on.
    """
    buf = io.StringIO()
    with _patched_argv(argv), contextlib.redirect_stderr(buf):
        return run_moce.parse_args()


def _minimal_inference_block() -> dict[str, Any]:
    """
    Build a moce_inference dict that satisfies every sub-key read by
    build_engine. Values are placeholders -- the tests use a fake moce
    module so no real engine is constructed.
    """
    return {
        "model": {
            "base_model": "fake/model",
            "dtype": "bfloat16",
            "device": "cpu",
        },
        "steering_vectors": {
            "economic_vector_path": "data/fake/economic.pt",
            "social_vector_path": "data/fake/social.pt",
        },
        "input_transformer": {
            "vector_method": "logistic_regression",
            "use_final_aggregated_vectors": True,
            "selected_layers": [8, 12, 16, 20, 24],
            "pooling_method": "mean",
            "use_centering": False,
            "neutral_reference_path": None,
            "calibration_input_dim": 4096,
        },
        "expert_checkpoints": {
            "left_lib_checkpoint": "data/fake/left_lib",
            "left_auth_checkpoint": "data/fake/left_auth",
            "right_lib_checkpoint": "data/fake/right_lib",
            "right_auth_checkpoint": "data/fake/right_auth",
        },
        "router": {
            "beta": 1.0,
            "temperature": 1.0,
            "fallback_to_uniform_if_centered": True,
            "center_threshold": 0.05,
        },
        "editor": {
            "max_edit_steps": 1,
            "correction_beta": 1.0,
            "initialization_mode": "router_policy",
            "use_recursive_editing": True,
            "initialize_from_router": True,
            "convergence_threshold": 1.0e-3,
            "keep_edit_trace": True,
            "stop_on_axis_proximity": True,
            "axis_proximity_threshold": 0.015,
        },
        "generation": {
            "max_new_tokens": 256,
            "temperature": 0.7,
            "do_sample": False,
            "top_p": 1.0,
        },
    }


class _RecordingConfig:
    """Generic *Config stand-in that records constructor kwargs as attributes."""

    def __init__(self, **kwargs: Any) -> None:
        for key, value in kwargs.items():
            setattr(self, key, value)


class _FakeRouter:
    """Fake Router that records load_calibration_checkpoint calls."""

    def __init__(self, config: Any) -> None:
        self.config = config
        self.checkpoint_calls: list[Path] = []

    def load_calibration_checkpoint(self, path: Path) -> None:
        self.checkpoint_calls.append(Path(path))


class _FakeEngine:
    """
    Fake MoCEEngine that mirrors the real constructor signature and exposes
    a router with an observable load_calibration_checkpoint.
    """

    def __init__(
        self,
        *,
        model: Any,
        tokenizer: Any,
        steering_config: Any,
        router_config: Any,
        expert_config: Any,
        editor_config: Any,
        generation_config: Any,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.steering_config = steering_config
        self.router_config = router_config
        self.expert_config = expert_config
        self.editor_config = editor_config
        self.generation_config = generation_config
        self.router = _FakeRouter(router_config)


def _make_fake_moce_module() -> SimpleNamespace:
    """
    Return a stand-in for the 09_moce_components module. Each *Config is the
    same _RecordingConfig class; build_engine identifies them by attribute
    access so distinct classes are not required.
    """
    return SimpleNamespace(
        SteeringVectorConfig=_RecordingConfig,
        RouterConfig=_RecordingConfig,
        ExpertConfig=_RecordingConfig,
        EditorConfig=_RecordingConfig,
        GenerationConfig=_RecordingConfig,
        MoCEEngine=_FakeEngine,
    )


def _fake_result(
    *,
    final_text: str = "decoded answer",
    prompt_text: str = "prompt?",
    heuristic_prior: dict[str, float] | None = None,
    calibrated_policy: dict[str, float] | None = None,
    final_alpha: dict[str, float] | None = None,
    num_steps_run: int = 1,
    stopped_early: bool = False,
    stop_reason: str | None = None,
) -> Any:
    """Build a fake MoCEResult-shaped object for serializer/formatter tests."""
    uniform = {k: 0.25 for k in CANONICAL}
    prior = heuristic_prior if heuristic_prior is not None else dict(uniform)
    policy = calibrated_policy if calibrated_policy is not None else dict(uniform)
    alpha = final_alpha if final_alpha is not None else dict(uniform)
    alignment = {k: 0.0 for k in CANONICAL}
    quadrant_scores = {"left_lib": 0.41, "left_auth": 0.18,
                       "right_lib": 0.27, "right_auth": 0.14}
    return SimpleNamespace(
        prompt_text=prompt_text,
        final_text=final_text,
        prompt_state=SimpleNamespace(
            bias_magnitude=0.34,
            economic_score=-0.12,
            social_score=0.27,
            quadrant_scores=quadrant_scores,
        ),
        router_state=SimpleNamespace(
            heuristic_prior=prior,
            calibrated_policy=policy,
        ),
        editor_result=SimpleNamespace(
            final_alpha=alpha,
            final_alignment=alignment,
            num_steps_run=num_steps_run,
            stopped_early=stopped_early,
            stop_reason=stop_reason,
            step_traces=[],
        ),
        metadata={
            "num_edit_steps": num_steps_run,
            "stopped_early": stopped_early,
        },
    )


# === TESTS: parse_args CLI contracts ===

class ParseArgsContractTests(unittest.TestCase):

    def test_help_exits_zero(self) -> None:
        with self.assertRaises(SystemExit) as ctx, \
                contextlib.redirect_stdout(io.StringIO()):
            _parse_args_silently(["--help"])
        self.assertEqual(ctx.exception.code, 0)

    def test_missing_config_exits(self) -> None:
        with self.assertRaises(SystemExit) as ctx:
            _parse_args_silently(["--prompt", "hi"])
        self.assertEqual(ctx.exception.code, 2)

    def test_missing_prompt_source_exits(self) -> None:
        with self.assertRaises(SystemExit) as ctx:
            _parse_args_silently(["--config", "config/config.yaml"])
        self.assertEqual(ctx.exception.code, 2)

    def test_prompt_and_prompts_file_mutually_exclusive(self) -> None:
        with self.assertRaises(SystemExit) as ctx:
            _parse_args_silently([
                "--config", "config/config.yaml",
                "--prompt", "hi",
                "--prompts-file", "p.jsonl",
            ])
        self.assertEqual(ctx.exception.code, 2)

    def test_calibrated_requires_checkpoint(self) -> None:
        with self.assertRaises(SystemExit) as ctx:
            _parse_args_silently([
                "--config", "config/config.yaml",
                "--prompt", "hi",
                "--calibrated",
            ])
        self.assertEqual(ctx.exception.code, 2)

    def test_checkpoint_without_calibrated_exits(self) -> None:
        with self.assertRaises(SystemExit) as ctx:
            _parse_args_silently([
                "--config", "config/config.yaml",
                "--prompt", "hi",
                "--router-checkpoint", "ckpt.pt",
            ])
        self.assertEqual(ctx.exception.code, 2)

    def test_minimal_valid_args_default_to_heuristic(self) -> None:
        args = _parse_args_silently([
            "--config", "config/config.yaml",
            "--prompt", "hi",
        ])
        self.assertFalse(args.calibrated)
        self.assertIsNone(args.router_checkpoint)
        self.assertIsNone(args.calibration_input_dim)
        self.assertIsNone(args.output_path)
        self.assertIsNone(args.device)
        self.assertEqual(args.prompt, "hi")
        self.assertIsNone(args.prompts_file)
        self.assertEqual(args.config, Path("config/config.yaml"))

    def test_calibrated_with_checkpoint_parses(self) -> None:
        args = _parse_args_silently([
            "--config", "config/config.yaml",
            "--prompt", "hi",
            "--calibrated",
            "--router-checkpoint", "/tmp/ckpt.pt",
            "--calibration-input-dim", "4096",
        ])
        self.assertTrue(args.calibrated)
        self.assertEqual(args.router_checkpoint, Path("/tmp/ckpt.pt"))
        self.assertEqual(args.calibration_input_dim, 4096)

    def test_router_hidden_dim_alias_populates_calibration_input_dim(self) -> None:
        args = _parse_args_silently([
            "--config", "config/config.yaml",
            "--prompt", "hi",
            "--calibrated",
            "--router-checkpoint", "/tmp/ckpt.pt",
            "--router-hidden-dim", "1024",
        ])
        self.assertEqual(args.calibration_input_dim, 1024)

    def test_prompts_file_mode_parses(self) -> None:
        args = _parse_args_silently([
            "--config", "config/config.yaml",
            "--prompts-file", "data/prompts.jsonl",
            "--output-path", "data/runs/out.jsonl",
            "--device", "cuda",
        ])
        self.assertIsNone(args.prompt)
        self.assertEqual(args.prompts_file, Path("data/prompts.jsonl"))
        self.assertEqual(args.output_path, Path("data/runs/out.jsonl"))
        self.assertEqual(args.device, "cuda")


# === TESTS: _load_moce_inference_block ===

class ConfigBlockLoaderTests(unittest.TestCase):

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _write_yaml(self, name: str, payload: Any) -> Path:
        path = self.tmp / name
        with path.open("w", encoding="utf-8") as fh:
            yaml.safe_dump(payload, fh)
        return path

    def test_missing_file_raises_filenotfound(self) -> None:
        missing = self.tmp / "nope.yaml"
        with self.assertRaises(FileNotFoundError):
            run_moce._load_moce_inference_block(missing)

    def test_non_mapping_top_level_raises(self) -> None:
        path = self._write_yaml("list.yaml", ["a", "b"])
        with self.assertRaisesRegex(ValueError, "top-level mapping"):
            run_moce._load_moce_inference_block(path)

    def test_missing_block_raises(self) -> None:
        path = self._write_yaml("nob.yaml", {"other": {}})
        with self.assertRaisesRegex(ValueError, "moce_inference"):
            run_moce._load_moce_inference_block(path)

    def test_block_not_mapping_raises(self) -> None:
        path = self._write_yaml("scalar.yaml", {"moce_inference": "yes"})
        with self.assertRaisesRegex(ValueError, "moce_inference"):
            run_moce._load_moce_inference_block(path)

    def test_missing_sub_key_raises_with_full_path(self) -> None:
        partial = _minimal_inference_block()
        del partial["steering_vectors"]
        path = self._write_yaml("partial.yaml", {"moce_inference": partial})
        with self.assertRaisesRegex(ValueError, "moce_inference.steering_vectors"):
            run_moce._load_moce_inference_block(path)

    def test_sub_key_not_mapping_raises(self) -> None:
        bad = _minimal_inference_block()
        bad["router"] = "not a mapping"  # type: ignore[assignment]
        path = self._write_yaml("badsub.yaml", {"moce_inference": bad})
        with self.assertRaisesRegex(ValueError, "moce_inference.router"):
            run_moce._load_moce_inference_block(path)

    def test_real_config_loads_with_seven_sub_keys(self) -> None:
        # the checked-in config must always be loadable; if this fails the
        # config has drifted away from the runner's contract
        block = run_moce._load_moce_inference_block(_REAL_CONFIG_PATH)
        self.assertEqual(
            sorted(block.keys()),
            sorted(run_moce.REQUIRED_INFERENCE_BLOCKS),
        )


# === TESTS: iter_prompts ===

class IterPromptsTests(unittest.TestCase):

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _write_jsonl(self, name: str, lines: list[str]) -> Path:
        path = self.tmp / name
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path

    def _args_with_prompt(self, text: str) -> SimpleNamespace:
        return SimpleNamespace(prompt=text, prompts_file=None)

    def _args_with_file(self, path: Path) -> SimpleNamespace:
        return SimpleNamespace(prompt=None, prompts_file=path)

    def test_single_prompt_mode_yields_one_pair(self) -> None:
        out = list(run_moce.iter_prompts(self._args_with_prompt("hello")))
        self.assertEqual(out, [(None, "hello")])

    def test_prompts_file_yields_id_and_text(self) -> None:
        path = self._write_jsonl("ok.jsonl", [
            json.dumps({"id": "q1", "prompt_text": "first"}),
            json.dumps({"id": "q2", "prompt_text": "second"}),
        ])
        out = list(run_moce.iter_prompts(self._args_with_file(path)))
        self.assertEqual(out, [("q1", "first"), ("q2", "second")])

    def test_id_optional(self) -> None:
        path = self._write_jsonl("noid.jsonl", [
            json.dumps({"prompt_text": "first"}),
        ])
        out = list(run_moce.iter_prompts(self._args_with_file(path)))
        self.assertEqual(out, [(None, "first")])

    def test_non_string_id_is_stringified(self) -> None:
        path = self._write_jsonl("intid.jsonl", [
            json.dumps({"id": 7, "prompt_text": "first"}),
        ])
        out = list(run_moce.iter_prompts(self._args_with_file(path)))
        self.assertEqual(out, [("7", "first")])

    def test_blank_lines_skipped(self) -> None:
        path = self._write_jsonl("blank.jsonl", [
            "",
            json.dumps({"prompt_text": "real"}),
            "   ",
        ])
        out = list(run_moce.iter_prompts(self._args_with_file(path)))
        self.assertEqual(out, [(None, "real")])

    def test_missing_file_raises_filenotfound(self) -> None:
        missing = self.tmp / "missing.jsonl"
        with self.assertRaises(FileNotFoundError):
            list(run_moce.iter_prompts(self._args_with_file(missing)))

    def test_invalid_json_raises_with_line_number(self) -> None:
        path = self._write_jsonl("bad.jsonl", [
            json.dumps({"prompt_text": "ok"}),
            "{not json",
        ])
        with self.assertRaisesRegex(ValueError, "line 2"):
            list(run_moce.iter_prompts(self._args_with_file(path)))

    def test_row_not_object_raises(self) -> None:
        path = self._write_jsonl("array.jsonl", [
            json.dumps(["just", "a", "list"]),
        ])
        with self.assertRaisesRegex(ValueError, "JSON object"):
            list(run_moce.iter_prompts(self._args_with_file(path)))

    def test_missing_prompt_text_raises(self) -> None:
        path = self._write_jsonl("noprompt.jsonl", [
            json.dumps({"id": "x"}),
        ])
        with self.assertRaisesRegex(ValueError, "prompt_text"):
            list(run_moce.iter_prompts(self._args_with_file(path)))

    def test_empty_prompt_text_raises(self) -> None:
        path = self._write_jsonl("empty.jsonl", [
            json.dumps({"prompt_text": "   "}),
        ])
        with self.assertRaisesRegex(ValueError, "non-empty"):
            list(run_moce.iter_prompts(self._args_with_file(path)))


# === TESTS: build_engine wiring ===

class BuildEngineTests(unittest.TestCase):

    def _heuristic_args(self) -> SimpleNamespace:
        return SimpleNamespace(
            calibrated=False,
            router_checkpoint=None,
            calibration_input_dim=None,
            temperature=None,
            top_p=None,
        )

    def _calibrated_args(self, ckpt: Path, dim: int | None = None) -> SimpleNamespace:
        return SimpleNamespace(
            calibrated=True,
            router_checkpoint=ckpt,
            calibration_input_dim=dim,
            temperature=None,
            top_p=None,
        )

    def test_heuristic_default_wires_use_calibrated_false(self) -> None:
        cfg = _minimal_inference_block()
        engine = run_moce.build_engine(
            inference_cfg=cfg,
            args=self._heuristic_args(),
            moce=_make_fake_moce_module(),
            model=object(),
            tokenizer=object(),
        )
        self.assertFalse(engine.router_config.use_calibrated_router)
        # heuristic mode must not touch the checkpoint loader
        self.assertEqual(engine.router.checkpoint_calls, [])

    def test_heuristic_uses_config_calibration_input_dim(self) -> None:
        cfg = _minimal_inference_block()
        cfg["input_transformer"]["calibration_input_dim"] = 4096
        engine = run_moce.build_engine(
            inference_cfg=cfg,
            args=self._heuristic_args(),
            moce=_make_fake_moce_module(),
            model=object(),
            tokenizer=object(),
        )
        # config value flows through even in heuristic mode (harmless)
        self.assertEqual(engine.router_config.calibration_input_dim, 4096)

    def test_calibrated_loads_checkpoint_with_correct_path(self) -> None:
        cfg = _minimal_inference_block()
        ckpt = Path("/tmp/fake_ckpt.pt")
        engine = run_moce.build_engine(
            inference_cfg=cfg,
            args=self._calibrated_args(ckpt),
            moce=_make_fake_moce_module(),
            model=object(),
            tokenizer=object(),
        )
        self.assertTrue(engine.router_config.use_calibrated_router)
        self.assertEqual(engine.router.checkpoint_calls, [ckpt])

    def test_cli_calibration_input_dim_overrides_config(self) -> None:
        cfg = _minimal_inference_block()
        cfg["input_transformer"]["calibration_input_dim"] = 4096
        engine = run_moce.build_engine(
            inference_cfg=cfg,
            args=self._calibrated_args(Path("/tmp/x.pt"), dim=512),
            moce=_make_fake_moce_module(),
            model=object(),
            tokenizer=object(),
        )
        self.assertEqual(engine.router_config.calibration_input_dim, 512)

    def test_config_values_flow_through_to_all_subconfigs(self) -> None:
        cfg = _minimal_inference_block()
        cfg["router"]["beta"] = 2.5
        cfg["router"]["temperature"] = 0.5
        cfg["editor"]["correction_beta"] = 0.75
        cfg["editor"]["max_edit_steps"] = 3
        cfg["generation"]["max_new_tokens"] = 128
        cfg["generation"]["do_sample"] = True
        engine = run_moce.build_engine(
            inference_cfg=cfg,
            args=self._heuristic_args(),
            moce=_make_fake_moce_module(),
            model=object(),
            tokenizer=object(),
        )
        self.assertEqual(engine.router_config.beta, 2.5)
        self.assertEqual(engine.router_config.temperature, 0.5)
        self.assertEqual(engine.editor_config.correction_beta, 0.75)
        self.assertEqual(engine.editor_config.max_edit_steps, 3)
        self.assertEqual(engine.generation_config.max_new_tokens, 128)
        self.assertTrue(engine.generation_config.do_sample)
        # steering + expert paths are converted to Path objects
        self.assertIsInstance(engine.steering_config.economic_vector_path, Path)
        self.assertIsInstance(engine.expert_config.left_lib_checkpoint, Path)

    def test_model_and_tokenizer_are_passed_through(self) -> None:
        model = object()
        tokenizer = object()
        engine = run_moce.build_engine(
            inference_cfg=_minimal_inference_block(),
            args=self._heuristic_args(),
            moce=_make_fake_moce_module(),
            model=model,
            tokenizer=tokenizer,
        )
        self.assertIs(engine.model, model)
        self.assertIs(engine.tokenizer, tokenizer)


# === TESTS: serialize_result ===

class SerializeResultTests(unittest.TestCase):

    def test_heuristic_mode_round_trips_as_json(self) -> None:
        args = SimpleNamespace(calibrated=False)
        row = run_moce.serialize_result("q1", args, _fake_result())
        encoded = json.dumps(row)
        decoded = json.loads(encoded)
        self.assertEqual(decoded["id"], "q1")
        self.assertEqual(decoded["router_mode"], "heuristic")
        self.assertEqual(decoded["final_text"], "decoded answer")
        self.assertEqual(decoded["num_edit_steps"], 1)
        self.assertFalse(decoded["stopped_early"])
        self.assertAlmostEqual(decoded["bias_magnitude"], 0.34)

    def test_calibrated_mode_records_router_mode(self) -> None:
        args = SimpleNamespace(calibrated=True)
        row = run_moce.serialize_result(None, args, _fake_result())
        self.assertIsNone(row["id"])
        self.assertEqual(row["router_mode"], "calibrated")

    def test_mapping_fields_have_canonical_keys(self) -> None:
        args = SimpleNamespace(calibrated=False)
        row = run_moce.serialize_result("q1", args, _fake_result())
        for key in ("heuristic_prior", "calibrated_policy", "final_alpha",
                    "final_alignment", "quadrant_scores"):
            self.assertEqual(sorted(row[key].keys()), sorted(CANONICAL))
            for v in row[key].values():
                self.assertIsInstance(v, float)


# === TESTS: format_stdout_summary ===

class FormatStdoutSummaryTests(unittest.TestCase):

    def test_with_id_prints_id_line(self) -> None:
        out = run_moce.format_stdout_summary("q1", _fake_result())
        self.assertIn("id: q1", out)
        self.assertIn("prompt:", out)
        self.assertIn("final_text:", out)
        self.assertIn("prior:", out)
        self.assertIn("final_alpha:", out)
        self.assertIn("edit_steps:", out)

    def test_without_id_uses_dash(self) -> None:
        out = run_moce.format_stdout_summary(None, _fake_result())
        self.assertIn("id: -", out)


# === MAIN ===

if __name__ == "__main__":
    unittest.main()
