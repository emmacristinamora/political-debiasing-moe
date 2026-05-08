# tests/test_train_router_calibration_pipeline.py


# === IMPORTS ===

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch


# === MODULE LOADING ===

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_DIR = _REPO_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import train_router_calibration_pipeline as trcp  # noqa: E402


# === FIXTURE HELPERS ===

def _make_cfg(
    tmp: Path,
    *,
    save_every_epoch: bool = False,
    calibration_input_dim_in_training: bool = False,
) -> SimpleNamespace:
    """Build a torch-free duck-typed RouterCalibrationConfig for tests."""
    paths = SimpleNamespace(
        output_dir=tmp,
        hidden_path=tmp / "hidden.pt",
        checkpoints_dir=tmp / "checkpoints",
        reports_dir=tmp / "reports",
    )
    training_kwargs: dict[str, Any] = dict(
        beta=1.0,
        temperature=1.0,
        learning_rate=1e-3,
        weight_decay=1e-4,
        batch_size=32,
        epochs=2,
        kl_weight=0.1,
        entropy_weight=0.01,
        seed=42,
        device="cpu",
    )
    if save_every_epoch:
        training_kwargs["save_every_epoch"] = True
    if calibration_input_dim_in_training:
        training_kwargs["calibration_input_dim"] = 64
    training = SimpleNamespace(**training_kwargs)
    input_transformer = SimpleNamespace(calibration_input_dim=128)
    return SimpleNamespace(
        paths=paths, training=training, input_transformer=input_transformer
    )


def _make_args(config_path: Path, **overrides: Any) -> SimpleNamespace:
    """Build a Namespace mirroring parse_args() with sensible defaults."""
    base: dict[str, Any] = dict(
        config=config_path,
        train_records_path=None,
        val_records_path=None,
        test_records_path=None,
        hidden_path=None,
        output_path=None,
        trainer_report_path=None,
        pipeline_report_path=None,
        device=None,
        dry_run=False,
        max_examples=None,
        skip_validation=False,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _touch(path: Path, content: str = "stub") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def _make_paths(tmp: Path) -> dict[str, Path]:
    return {
        "train_records_path": tmp / "train" / "records.jsonl",
        "val_records_path":   tmp / "val"   / "records.jsonl",
        "test_records_path":  tmp / "test"  / "records.jsonl",
        "hidden_path":        tmp / "hidden.pt",
        "router_checkpoint":  tmp / "checkpoints" / "calibrated_router.pt",
        "trainer_report":     tmp / "reports"     / "train_report.json",
        "pipeline_report":    tmp / "reports"     / "pipeline_train_report.json",
    }


def _make_hparams(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = dict(
        calibration_input_dim=128,
        beta=1.0,
        temperature=1.0,
        learning_rate=1e-3,
        weight_decay=1e-4,
        batch_size=32,
        epochs=2,
        kl_weight=0.1,
        entropy_weight=0.01,
        seed=42,
        device="cpu",
        save_every_epoch=False,
    )
    base.update(overrides)
    return base


# === TESTS — build_training_command ===

class BuildCommandTests(unittest.TestCase):

    def test_includes_required_args(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            paths = _make_paths(Path(raw_tmp))
            cmd = trcp.build_training_command(paths, _make_hparams())

        self.assertEqual(cmd[0], sys.executable)
        self.assertEqual(cmd[1], str(trcp.TRAINER_SCRIPT_RELATIVE))
        for flag in (
            "--records-path",
            "--hidden-path",
            "--output-path",
            "--report-path",
            "--calibration-input-dim",
            "--beta",
            "--temperature",
            "--learning-rate",
            "--weight-decay",
            "--batch-size",
            "--epochs",
            "--kl-weight",
            "--entropy-weight",
            "--seed",
            "--device",
        ):
            self.assertIn(flag, cmd, f"missing flag {flag} in command")
        self.assertNotIn("--save-every-epoch", cmd)
        self.assertNotIn("--max-examples", cmd)

    def test_max_examples_appended_when_passed(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            paths = _make_paths(Path(raw_tmp))
            cmd = trcp.build_training_command(paths, _make_hparams(), max_examples=7)
        self.assertIn("--max-examples", cmd)
        idx = cmd.index("--max-examples")
        self.assertEqual(cmd[idx + 1], "7")

    def test_save_every_epoch_appended_when_configured(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            paths = _make_paths(Path(raw_tmp))
            cmd = trcp.build_training_command(
                paths, _make_hparams(save_every_epoch=True)
            )
        self.assertIn("--save-every-epoch", cmd)

    def test_paths_are_stringified_in_command(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            paths = _make_paths(Path(raw_tmp))
            cmd = trcp.build_training_command(paths, _make_hparams())
        self.assertIn(str(paths["train_records_path"]), cmd)
        self.assertIn(str(paths["hidden_path"]), cmd)
        self.assertIn(str(paths["router_checkpoint"]), cmd)
        self.assertIn(str(paths["trainer_report"]), cmd)
        for entry in cmd:
            self.assertIsInstance(entry, str)


# === TESTS — tail_text ===

class TailTextTests(unittest.TestCase):

    def test_passes_through_short_text(self) -> None:
        self.assertEqual(trcp.tail_text("hello"), "hello")

    def test_truncates_to_last_n_chars(self) -> None:
        text = "abcdefghij" * 1000  # 10 000 chars
        out = trcp.tail_text(text, max_chars=4000)
        self.assertEqual(len(out), 4000)
        self.assertTrue(text.endswith(out))

    def test_none_passes_through(self) -> None:
        self.assertIsNone(trcp.tail_text(None))


# === TESTS — validation behavior ===

class ValidationTests(unittest.TestCase):

    def test_missing_train_records_raises(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp = Path(raw_tmp)
            paths = _make_paths(tmp)
            _touch(paths["hidden_path"])
            # train file deliberately absent
            with patch.object(trcp, "load_hidden_tensor_safe", return_value=None):
                with self.assertRaises(FileNotFoundError):
                    trcp.validate_training_inputs(paths, _make_hparams())

    def test_missing_val_test_only_warnings(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp = Path(raw_tmp)
            paths = _make_paths(tmp)
            _touch(paths["hidden_path"])
            _touch(paths["train_records_path"])
            # val/test deliberately absent
            with patch.object(trcp, "load_hidden_tensor_safe", return_value=None), \
                 patch.object(trcp, "load_records_jsonl", return_value=[{"x": 1}]), \
                 patch.object(trcp, "validate_router_dataset", return_value=None):
                result = trcp.validate_training_inputs(paths, _make_hparams())
        self.assertTrue(result["train"]["ok"])
        self.assertEqual(result["train"]["num_records"], 1)
        self.assertFalse(result["val"]["ok"])
        self.assertIsNotNone(result["val"]["warning"])
        self.assertFalse(result["test"]["ok"])
        self.assertIsNotNone(result["test"]["warning"])

    def test_missing_hidden_raises(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp = Path(raw_tmp)
            paths = _make_paths(tmp)
            _touch(paths["train_records_path"])
            # hidden file deliberately absent
            with self.assertRaises(FileNotFoundError):
                trcp.validate_training_inputs(paths, _make_hparams())


# === TESTS — run_pipeline ===

class RunPipelineTests(unittest.TestCase):

    def _setup_valid_dataset(self, tmp: Path) -> tuple[SimpleNamespace, SimpleNamespace]:
        cfg = _make_cfg(tmp)
        config_path = tmp / "config.yaml"
        config_path.write_text("dummy", encoding="utf-8")
        # materialize the inputs the validator needs to find on disk
        _touch(tmp / "hidden.pt")
        _touch(tmp / "train" / "records.jsonl")
        _touch(tmp / "val" / "records.jsonl")
        _touch(tmp / "test" / "records.jsonl")
        args = _make_args(
            config_path,
            pipeline_report_path=tmp / "pipeline_report.json",
            output_path=tmp / "checkpoint.pt",
            trainer_report_path=tmp / "trainer_report.json",
        )
        return cfg, args

    def test_dry_run_skips_subprocess(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp = Path(raw_tmp)
            cfg, args = self._setup_valid_dataset(tmp)
            args.dry_run = True
            with patch.object(trcp, "load_router_calibration_config", return_value=cfg), \
                 patch.object(trcp, "load_hidden_tensor_safe", return_value=None), \
                 patch.object(trcp, "load_records_jsonl", return_value=[{"x": 1}]), \
                 patch.object(trcp, "validate_router_dataset", return_value=None), \
                 patch.object(trcp.subprocess, "run") as run_mock:
                report = trcp.run_pipeline(args)
            run_mock.assert_not_called()
            self.assertFalse(report["execution"]["executed"])
            self.assertIsNone(report["execution"]["returncode"])
            # report still written before tempdir cleanup
            self.assertTrue((tmp / "pipeline_report.json").is_file())

    def test_non_dry_run_calls_subprocess_with_command(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp = Path(raw_tmp)
            cfg, args = self._setup_valid_dataset(tmp)
            completed = SimpleNamespace(returncode=0, stdout="ok", stderr="")
            with patch.object(trcp, "load_router_calibration_config", return_value=cfg), \
                 patch.object(trcp, "load_hidden_tensor_safe", return_value=None), \
                 patch.object(trcp, "load_records_jsonl", return_value=[{"x": 1}]), \
                 patch.object(trcp, "validate_router_dataset", return_value=None), \
                 patch.object(trcp.subprocess, "run", return_value=completed) as run_mock:
                report = trcp.run_pipeline(args)
        run_mock.assert_called_once()
        called_cmd = run_mock.call_args.args[0]
        self.assertEqual(called_cmd, report["command"])
        self.assertFalse(run_mock.call_args.kwargs.get("shell", False))
        self.assertTrue(run_mock.call_args.kwargs.get("capture_output"))
        self.assertTrue(report["execution"]["executed"])
        self.assertEqual(report["execution"]["returncode"], 0)

    def test_nonzero_returncode_raises_but_writes_report(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp = Path(raw_tmp)
            cfg, args = self._setup_valid_dataset(tmp)
            completed = SimpleNamespace(returncode=2, stdout="boom", stderr="trace")
            with patch.object(trcp, "load_router_calibration_config", return_value=cfg), \
                 patch.object(trcp, "load_hidden_tensor_safe", return_value=None), \
                 patch.object(trcp, "load_records_jsonl", return_value=[{"x": 1}]), \
                 patch.object(trcp, "validate_router_dataset", return_value=None), \
                 patch.object(trcp.subprocess, "run", return_value=completed):
                with self.assertRaises(RuntimeError):
                    trcp.run_pipeline(args)
            report_path = tmp / "pipeline_report.json"
            self.assertTrue(report_path.is_file())
            written = json.loads(report_path.read_text(encoding="utf-8"))
        self.assertEqual(written["execution"]["returncode"], 2)
        self.assertEqual(written["execution"]["stdout_tail"], "boom")
        self.assertEqual(written["execution"]["stderr_tail"], "trace")
        self.assertTrue(written["execution"]["executed"])

    def test_skip_validation_bypasses_validators(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp = Path(raw_tmp)
            cfg = _make_cfg(tmp)
            config_path = tmp / "config.yaml"
            config_path.write_text("dummy", encoding="utf-8")
            # NOTE: deliberately do NOT create train/hidden — skip-validation
            # must let us proceed past the missing artifacts
            args = _make_args(
                config_path,
                pipeline_report_path=tmp / "pipeline_report.json",
                skip_validation=True,
                dry_run=True,
            )
            load_records_mock = MagicMock(side_effect=AssertionError("must not be called"))
            validate_mock = MagicMock(side_effect=AssertionError("must not be called"))
            load_hidden_mock = MagicMock(side_effect=AssertionError("must not be called"))
            with patch.object(trcp, "load_router_calibration_config", return_value=cfg), \
                 patch.object(trcp, "load_hidden_tensor_safe", load_hidden_mock), \
                 patch.object(trcp, "load_records_jsonl", load_records_mock), \
                 patch.object(trcp, "validate_router_dataset", validate_mock):
                report = trcp.run_pipeline(args)
        load_hidden_mock.assert_not_called()
        load_records_mock.assert_not_called()
        validate_mock.assert_not_called()
        self.assertTrue(report["skip_validation"])
        self.assertIn(
            "validation skipped via --skip-validation",
            report["warnings"],
        )

    def test_pipeline_report_has_expected_keys(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp = Path(raw_tmp)
            cfg, args = self._setup_valid_dataset(tmp)
            args.dry_run = True
            with patch.object(trcp, "load_router_calibration_config", return_value=cfg), \
                 patch.object(trcp, "load_hidden_tensor_safe", return_value=None), \
                 patch.object(trcp, "load_records_jsonl", return_value=[{"x": 1}]), \
                 patch.object(trcp, "validate_router_dataset", return_value=None):
                report = trcp.run_pipeline(args)
        for key in (
            "config_path",
            "dry_run",
            "skip_validation",
            "input_paths",
            "output_paths",
            "hyperparameters",
            "validation",
            "command",
            "execution",
            "warnings",
        ):
            self.assertIn(key, report)
        for key in (
            "train_records_path",
            "val_records_path",
            "test_records_path",
            "hidden_path",
        ):
            self.assertIn(key, report["input_paths"])
        for key in ("router_checkpoint", "trainer_report", "pipeline_report"):
            self.assertIn(key, report["output_paths"])
        for key in ("executed", "returncode", "stdout_tail", "stderr_tail"):
            self.assertIn(key, report["execution"])
        self.assertEqual(report["validation"]["hidden_filename"], "hidden.pt")

    def test_stdout_stderr_tailed_to_max_length(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp = Path(raw_tmp)
            cfg, args = self._setup_valid_dataset(tmp)
            long_stdout = "a" * 10_000
            long_stderr = "b" * 9_000
            completed = SimpleNamespace(
                returncode=0, stdout=long_stdout, stderr=long_stderr
            )
            with patch.object(trcp, "load_router_calibration_config", return_value=cfg), \
                 patch.object(trcp, "load_hidden_tensor_safe", return_value=None), \
                 patch.object(trcp, "load_records_jsonl", return_value=[{"x": 1}]), \
                 patch.object(trcp, "validate_router_dataset", return_value=None), \
                 patch.object(trcp.subprocess, "run", return_value=completed):
                report = trcp.run_pipeline(args)
        self.assertEqual(len(report["execution"]["stdout_tail"]), trcp.TAIL_DEFAULT_CHARS)
        self.assertEqual(len(report["execution"]["stderr_tail"]), trcp.TAIL_DEFAULT_CHARS)
        self.assertTrue(long_stdout.endswith(report["execution"]["stdout_tail"]))
        self.assertTrue(long_stderr.endswith(report["execution"]["stderr_tail"]))

    def test_creates_output_directories(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp = Path(raw_tmp)
            cfg = _make_cfg(tmp)
            config_path = tmp / "config.yaml"
            config_path.write_text("dummy", encoding="utf-8")
            _touch(tmp / "hidden.pt")
            _touch(tmp / "train" / "records.jsonl")
            # nested output paths that don't exist yet
            args = _make_args(
                config_path,
                output_path=tmp / "out" / "ckpts" / "calibrated_router.pt",
                trainer_report_path=tmp / "out" / "rep" / "train_report.json",
                pipeline_report_path=tmp / "out" / "rep" / "pipeline_train_report.json",
                dry_run=True,
            )
            with patch.object(trcp, "load_router_calibration_config", return_value=cfg), \
                 patch.object(trcp, "load_hidden_tensor_safe", return_value=None), \
                 patch.object(trcp, "load_records_jsonl", return_value=[{"x": 1}]), \
                 patch.object(trcp, "validate_router_dataset", return_value=None):
                trcp.run_pipeline(args)
            self.assertTrue((tmp / "out" / "ckpts").is_dir())
            self.assertTrue((tmp / "out" / "rep").is_dir())
            self.assertTrue((tmp / "out" / "rep" / "pipeline_train_report.json").is_file())


# === TESTS — resolve_paths / resolve_training_hparams ===

class ResolvePathsAndHparamsTests(unittest.TestCase):

    def test_resolve_paths_uses_output_dir_layout(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp = Path(raw_tmp)
            cfg = _make_cfg(tmp)
            args = _make_args(tmp / "config.yaml")
            paths = trcp.resolve_paths(cfg, args)
        self.assertEqual(paths["train_records_path"], tmp / "train" / "records.jsonl")
        self.assertEqual(paths["val_records_path"],   tmp / "val"   / "records.jsonl")
        self.assertEqual(paths["test_records_path"],  tmp / "test"  / "records.jsonl")
        self.assertEqual(paths["hidden_path"], tmp / "hidden.pt")
        self.assertEqual(paths["router_checkpoint"], tmp / "checkpoints" / "calibrated_router.pt")
        self.assertEqual(paths["trainer_report"],    tmp / "reports"     / "train_report.json")
        self.assertEqual(paths["pipeline_report"],   tmp / "reports" / "pipeline_train_report.json")

    def test_resolve_paths_cli_overrides_win(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp = Path(raw_tmp)
            cfg = _make_cfg(tmp)
            override = tmp / "custom" / "train.jsonl"
            args = _make_args(tmp / "config.yaml", train_records_path=override)
            paths = trcp.resolve_paths(cfg, args)
        self.assertEqual(paths["train_records_path"], override)

    def test_resolve_hparams_prefers_training_calibration_dim(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp = Path(raw_tmp)
            cfg = _make_cfg(tmp, calibration_input_dim_in_training=True)
            args = _make_args(tmp / "config.yaml")
            hparams = trcp.resolve_training_hparams(cfg, args)
        self.assertEqual(hparams["calibration_input_dim"], 64)

    def test_resolve_hparams_falls_back_to_input_transformer(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp = Path(raw_tmp)
            cfg = _make_cfg(tmp)
            args = _make_args(tmp / "config.yaml")
            hparams = trcp.resolve_training_hparams(cfg, args)
        self.assertEqual(hparams["calibration_input_dim"], 128)

    def test_resolve_hparams_cli_device_overrides_config(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp = Path(raw_tmp)
            cfg = _make_cfg(tmp)
            args = _make_args(tmp / "config.yaml", device="cuda:0")
            hparams = trcp.resolve_training_hparams(cfg, args)
        self.assertEqual(hparams["device"], "cuda:0")


# === ENTRY POINT ===

if __name__ == "__main__":
    unittest.main()
