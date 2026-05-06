# tests/test_run_moce.py


# === IMPORTS ===

import contextlib
import importlib.util
import io
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

import torch


# === MODULE LOADING ===

# src/10_run_moce.py begins with a digit, so it cannot be imported via normal
# "import" syntax. load it explicitly by absolute path with importlib.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_RUN_MOCE_PATH = _REPO_ROOT / "src" / "10_run_moce.py"

_spec = importlib.util.spec_from_file_location("run_moce", _RUN_MOCE_PATH)
run_moce = importlib.util.module_from_spec(_spec)
sys.modules["run_moce"] = run_moce
_spec.loader.exec_module(run_moce)

# load the components module once via run_moce's own helper so we share the
# exact same Router class run_moce.build_router will instantiate
_components = run_moce._load_components_module()
CANONICAL_QUADRANT_ORDER = _components.CANONICAL_QUADRANT_ORDER


# === HELPERS ===

def _make_checkpoint(
    path: Path,
    hidden_dim: int,
    weight_value: float = 0.0,
    bias_value: float = 0.0,
) -> None:
    """
    Write a minimal valid calibrated-router checkpoint to `path`.

    The state_dict is built from a deterministic donor nn.Linear so the
    weight/bias shapes match Router.calibration_module exactly.
    """
    layer = torch.nn.Linear(hidden_dim, len(CANONICAL_QUADRANT_ORDER))
    with torch.no_grad():
        layer.weight.fill_(weight_value)
        layer.bias.fill_(bias_value)
    payload: dict[str, Any] = {
        "state_dict": layer.state_dict(),
        "calibration_input_dim": hidden_dim,
        # legacy alias kept so older runtimes can still load this checkpoint
        "router_hidden_dim": hidden_dim,
        "canonical_quadrant_order": list(CANONICAL_QUADRANT_ORDER),
        "beta": 1.0,
        "temperature": 1.0,
    }
    torch.save(payload, path)


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


# === TESTS ===

class HeuristicModeTests(unittest.TestCase):

    def test_default_args_construct_heuristic_router(self) -> None:
        args = _parse_args_silently([])
        self.assertFalse(args.calibrated)
        self.assertIsNone(args.router_checkpoint)
        # the default destination should be the new name
        self.assertTrue(hasattr(args, "calibration_input_dim"))

        router = run_moce.build_router(args, _components)
        self.assertIsNone(router.calibration_module)
        self.assertIsNone(router.calibration_input_dim)
        self.assertIsNone(router.calibration_checkpoint_metadata)


class CalibratedModeTests(unittest.TestCase):

    def setUp(self) -> None:
        self.hidden_dim = 4
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self._tmp.name)
        self.ckpt_path = self.tmp_path / "router.pt"

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_calibrated_with_valid_checkpoint_loads(self) -> None:
        _make_checkpoint(
            self.ckpt_path,
            hidden_dim=self.hidden_dim,
            weight_value=0.5,
            bias_value=0.25,
        )
        args = _parse_args_silently([
            "--calibrated",
            "--router-checkpoint", str(self.ckpt_path),
            "--calibration-input-dim", str(self.hidden_dim),
        ])
        self.assertTrue(args.calibrated)
        self.assertEqual(args.calibration_input_dim, self.hidden_dim)

        router = run_moce.build_router(args, _components)
        self.assertIsNotNone(router.calibration_module)
        self.assertEqual(router.calibration_input_dim, self.hidden_dim)

        meta = router.calibration_checkpoint_metadata
        self.assertIsNotNone(meta)
        self.assertEqual(meta["checkpoint_path"], str(self.ckpt_path))
        self.assertEqual(meta["calibration_input_dim"], self.hidden_dim)
        self.assertEqual(
            meta["canonical_quadrant_order"], list(CANONICAL_QUADRANT_ORDER)
        )

    def test_deprecated_router_hidden_dim_alias_still_works(self) -> None:
        # the legacy CLI flag must still populate calibration_input_dim so
        # existing scripts and docs do not break
        _make_checkpoint(self.ckpt_path, hidden_dim=self.hidden_dim)
        args = _parse_args_silently([
            "--calibrated",
            "--router-checkpoint", str(self.ckpt_path),
            "--router-hidden-dim", str(self.hidden_dim),
        ])
        self.assertEqual(args.calibration_input_dim, self.hidden_dim)
        router = run_moce.build_router(args, _components)
        self.assertEqual(router.calibration_input_dim, self.hidden_dim)


class CLIContractTests(unittest.TestCase):

    def setUp(self) -> None:
        self.hidden_dim = 4
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self._tmp.name)
        self.ckpt_path = self.tmp_path / "router.pt"

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_calibrated_without_checkpoint_exits(self) -> None:
        with self.assertRaises(SystemExit) as ctx:
            _parse_args_silently(["--calibrated"])
        self.assertEqual(ctx.exception.code, 2)

    def test_checkpoint_without_calibrated_exits(self) -> None:
        # write a real file so the failure is unambiguously about the CLI
        # contract, not about a missing path (which is checked later)
        _make_checkpoint(self.ckpt_path, hidden_dim=self.hidden_dim)
        with self.assertRaises(SystemExit) as ctx:
            _parse_args_silently([
                "--router-checkpoint", str(self.ckpt_path),
            ])
        self.assertEqual(ctx.exception.code, 2)


class CheckpointFailureTests(unittest.TestCase):

    def setUp(self) -> None:
        self.hidden_dim = 4
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self._tmp.name)
        self.ckpt_path = self.tmp_path / "router.pt"

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_calibration_input_dim_mismatch_raises_value_error(self) -> None:
        # checkpoint built with hidden_dim=8 but CLI says 4
        _make_checkpoint(self.ckpt_path, hidden_dim=8)
        args = _parse_args_silently([
            "--calibrated",
            "--router-checkpoint", str(self.ckpt_path),
            "--calibration-input-dim", str(self.hidden_dim),
        ])
        with self.assertRaisesRegex(ValueError, "calibration_input_dim"):
            run_moce.build_router(args, _components)

    def test_missing_checkpoint_file_raises(self) -> None:
        missing = self.tmp_path / "does_not_exist.pt"
        args = _parse_args_silently([
            "--calibrated",
            "--router-checkpoint", str(missing),
            "--calibration-input-dim", str(self.hidden_dim),
        ])
        with self.assertRaises(FileNotFoundError):
            run_moce.build_router(args, _components)


# === MAIN ===

if __name__ == "__main__":
    unittest.main()
