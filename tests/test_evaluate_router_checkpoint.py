# tests/test_evaluate_router_checkpoint.py


# === IMPORTS ===

from __future__ import annotations

import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any


try:
    import torch
    from torch import nn
    HAS_TORCH = True
except ImportError:
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    HAS_TORCH = False


# === MODULE LOADING ===

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_DIR = _REPO_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

if HAS_TORCH:
    import evaluate_router_checkpoint as erc  # noqa: E402
else:
    erc = None  # type: ignore[assignment]


CANONICAL = ("left_lib", "left_auth", "right_lib", "right_auth")


def load_tests(loader, standard_tests, pattern):  # noqa: ARG001
    """
    unittest load_tests hook — surfaces a single skipped placeholder when
    torch is missing instead of raising at module import time.
    """
    if not HAS_TORCH:
        suite = unittest.TestSuite()

        class _TorchUnavailable(unittest.TestCase):
            @unittest.skip("torch unavailable")
            def test_skipped(self) -> None:
                pass

        suite.addTests(loader.loadTestsFromTestCase(_TorchUnavailable))
        return suite
    return standard_tests


# === FIXTURES ===

def _quadrant_scores(values: tuple[float, float, float, float]) -> dict[str, float]:
    return dict(zip(CANONICAL, values))


def _policy(values: tuple[float, float, float, float]) -> dict[str, float]:
    total = sum(values)
    if total <= 0:
        raise ValueError("policy values must be positive")
    return {k: v / total for k, v in zip(CANONICAL, values)}


def _make_head(
    hidden_dim: int,
    *,
    weight: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
) -> nn.Linear:
    head = nn.Linear(hidden_dim, len(CANONICAL))
    with torch.no_grad():
        head.weight.zero_()
        head.bias.zero_()
        if weight is not None:
            head.weight.copy_(weight)
        if bias is not None:
            head.bias.copy_(bias)
    head.eval()
    return head


def _save_checkpoint(
    path: Path,
    *,
    hidden_dim: int = 4,
    use_legacy_key: bool = False,
    canonical_override: list[str] | None = None,
) -> None:
    head = _make_head(hidden_dim)
    payload: dict[str, Any] = {
        "state_dict": head.state_dict(),
        "canonical_quadrant_order": (
            canonical_override if canonical_override is not None else list(CANONICAL)
        ),
        "beta": 1.0,
        "temperature": 1.0,
        "kl_weight": 0.1,
        "entropy_weight": 0.01,
        "epochs": 2,
        "records_path": "data/router/train/records.jsonl",
        "hidden_path": "data/router/hidden.pt",
    }
    if use_legacy_key:
        payload["router_hidden_dim"] = hidden_dim
    else:
        payload["calibration_input_dim"] = hidden_dim
    torch.save(payload, path)


def _make_record(
    eid: str,
    row_index: int,
    target_values: tuple[float, float, float, float],
    quadrant_values: tuple[float, float, float, float],
    *,
    hidden_filename: str = "hidden.pt",
) -> dict[str, Any]:
    return {
        "example_id": eid,
        "prompt_text": f"prompt for {eid}",
        "quadrant_scores": _quadrant_scores(quadrant_values),
        "bias_magnitude": 0.5,
        "target_policy": _policy(target_values),
        "hidden_representation_ref": f"{hidden_filename}:{row_index}",
    }


# === TESTS — numeric helpers ===

class HeuristicPriorTests(unittest.TestCase):

    def test_heuristic_prior_matches_softmax_with_minus_beta_q(self) -> None:
        # quadrant_scores [0.0, 0.5, 1.0, -0.5], beta=1.0, T=1.0
        scores = _quadrant_scores((0.0, 0.5, 1.0, -0.5))
        prior = erc.heuristic_prior_from_scores(scores, beta=1.0, temperature=1.0)
        self.assertEqual(set(prior), set(CANONICAL))
        total = sum(prior.values())
        self.assertAlmostEqual(total, 1.0, places=6)
        # smaller score → higher prob (since logit is -beta*q)
        # q ordering ascending: right_auth=-0.5 < left_lib=0.0 < left_auth=0.5 < right_lib=1.0
        self.assertGreater(prior["right_auth"], prior["left_lib"])
        self.assertGreater(prior["left_lib"], prior["left_auth"])
        self.assertGreater(prior["left_auth"], prior["right_lib"])

    def test_heuristic_prior_uniform_when_scores_equal(self) -> None:
        scores = _quadrant_scores((0.3, 0.3, 0.3, 0.3))
        prior = erc.heuristic_prior_from_scores(scores, beta=2.0, temperature=1.0)
        for k in CANONICAL:
            self.assertAlmostEqual(prior[k], 0.25, places=6)


# === TESTS — calibrated policy ===

class CalibratedPolicyTests(unittest.TestCase):

    def test_zero_head_yields_calibrated_equal_heuristic(self) -> None:
        head = _make_head(4)  # zero weight + zero bias
        hidden_tensor = torch.randn(2, 4)
        rec = _make_record(
            "ex0",
            row_index=0,
            target_values=(0.4, 0.3, 0.2, 0.1),
            quadrant_values=(0.0, 0.5, 1.0, -0.5),
        )
        result = erc.evaluate_record(
            record=rec,
            hidden_tensor=hidden_tensor,
            head=head,
            beta=1.0,
            temperature=1.0,
            hidden_filename="hidden.pt",
        )
        for k in CANONICAL:
            self.assertAlmostEqual(
                result["calibrated_policy"][k],
                result["heuristic_prior"][k],
                places=5,
            )

    def test_nonzero_bias_shifts_calibrated_policy(self) -> None:
        # bias dominates: enormous push on left_lib makes it the top quadrant
        bias = torch.tensor([10.0, 0.0, 0.0, 0.0], dtype=torch.float32)
        head = _make_head(4, bias=bias)
        hidden_tensor = torch.zeros(1, 4)
        # heuristic alone would prefer right_auth (lowest score)
        rec = _make_record(
            "ex0",
            row_index=0,
            target_values=(0.25, 0.25, 0.25, 0.25),
            quadrant_values=(0.0, 0.5, 1.0, -0.5),
        )
        result = erc.evaluate_record(
            record=rec,
            hidden_tensor=hidden_tensor,
            head=head,
            beta=1.0,
            temperature=1.0,
            hidden_filename="hidden.pt",
        )
        self.assertEqual(result["metrics"]["top1_calibrated"], "left_lib")
        self.assertNotEqual(result["metrics"]["top1_heuristic"], "left_lib")


# === TESTS — KL / entropy ===

class KLEntropyTests(unittest.TestCase):

    def test_kl_zero_for_identical_distributions(self) -> None:
        p = _policy((0.4, 0.3, 0.2, 0.1))
        self.assertAlmostEqual(erc.kl_policy(p, p), 0.0, places=10)

    def test_kl_inf_when_q_is_zero_on_supported_index(self) -> None:
        p = _policy((0.5, 0.5, 1e-9, 1e-9))  # nearly all mass on first two
        q = {"left_lib": 0.0, "left_auth": 0.5, "right_lib": 0.25, "right_auth": 0.25}
        self.assertEqual(erc.kl_policy(p, q), float("inf"))

    def test_entropy_finite_and_positive(self) -> None:
        p = _policy((0.4, 0.3, 0.2, 0.1))
        h = erc.entropy(p)
        self.assertTrue(math.isfinite(h))
        self.assertGreater(h, 0.0)

    def test_entropy_uniform_is_log_n(self) -> None:
        p = {k: 0.25 for k in CANONICAL}
        self.assertAlmostEqual(erc.entropy(p), math.log(4), places=6)

    def test_l1_distance_zero_for_identical(self) -> None:
        p = _policy((0.4, 0.3, 0.2, 0.1))
        self.assertAlmostEqual(erc.l1_distance(p, p), 0.0, places=10)


# === TESTS — improvement_kl ===

class ImprovementKLTests(unittest.TestCase):

    def test_improvement_kl_positive_when_calibrated_closer_to_target(self) -> None:
        # construct a bias such that combined = log_heuristic + bias = log_target
        # exactly. log_softmax(log_target) == log_target since target is already
        # a normalized distribution, so calibrated_policy equals target and
        # KL(target || calibrated) drops to ~0 while KL(target || heuristic) > 0.
        quadrant_values = (0.0, 0.5, 1.0, -0.5)
        target_values = (0.7, 0.1, 0.1, 0.1)
        target = _policy(target_values)
        heuristic = erc.heuristic_prior_from_scores(
            _quadrant_scores(quadrant_values), beta=1.0, temperature=1.0,
        )
        bias = torch.tensor(
            [math.log(target[k]) - math.log(heuristic[k]) for k in CANONICAL],
            dtype=torch.float32,
        )
        head = _make_head(4, bias=bias)
        hidden_tensor = torch.zeros(1, 4)
        rec = _make_record(
            "ex0",
            row_index=0,
            target_values=target_values,
            quadrant_values=quadrant_values,
        )
        result = erc.evaluate_record(
            record=rec,
            hidden_tensor=hidden_tensor,
            head=head,
            beta=1.0,
            temperature=1.0,
            hidden_filename="hidden.pt",
        )
        self.assertGreater(result["metrics"]["improvement_kl"], 0.0)
        self.assertLess(
            result["metrics"]["kl_target_to_calibrated"],
            result["metrics"]["kl_target_to_heuristic"],
        )
        for k in CANONICAL:
            self.assertAlmostEqual(
                result["calibrated_policy"][k], target[k], places=5,
            )


# === TESTS — checkpoint loader ===

class CheckpointLoaderTests(unittest.TestCase):

    def test_loader_accepts_calibration_input_dim(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp = Path(raw_tmp)
            ckpt = tmp / "head.pt"
            _save_checkpoint(ckpt, hidden_dim=4)
            head, metadata = erc.load_checkpoint_head(ckpt, expected_dim=4)
        self.assertIsInstance(head, nn.Linear)
        self.assertEqual(head.in_features, 4)
        self.assertEqual(head.out_features, len(CANONICAL))
        self.assertNotIn("state_dict", metadata)
        self.assertEqual(metadata["calibration_input_dim"], 4)

    def test_loader_accepts_legacy_router_hidden_dim(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp = Path(raw_tmp)
            ckpt = tmp / "head.pt"
            _save_checkpoint(ckpt, hidden_dim=8, use_legacy_key=True)
            head, metadata = erc.load_checkpoint_head(ckpt, expected_dim=8)
        self.assertEqual(head.in_features, 8)
        self.assertEqual(metadata["router_hidden_dim"], 8)
        self.assertNotIn("calibration_input_dim", metadata)

    def test_loader_dim_mismatch_raises(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp = Path(raw_tmp)
            ckpt = tmp / "head.pt"
            _save_checkpoint(ckpt, hidden_dim=4)
            with self.assertRaises(ValueError):
                erc.load_checkpoint_head(ckpt, expected_dim=8)

    def test_loader_canonical_mismatch_raises(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp = Path(raw_tmp)
            ckpt = tmp / "head.pt"
            _save_checkpoint(
                ckpt,
                hidden_dim=4,
                canonical_override=["right_auth", "right_lib", "left_auth", "left_lib"],
            )
            with self.assertRaises(ValueError):
                erc.load_checkpoint_head(ckpt, expected_dim=4)

    def test_loader_missing_state_dict_raises(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp = Path(raw_tmp)
            ckpt = tmp / "head.pt"
            torch.save({"calibration_input_dim": 4}, ckpt)
            with self.assertRaises(ValueError):
                erc.load_checkpoint_head(ckpt, expected_dim=4)


# === TESTS — split evaluation + report writer ===

class EvaluateSplitTests(unittest.TestCase):

    def test_summary_has_required_keys(self) -> None:
        head = _make_head(4)  # zero weights → calibrated == heuristic
        hidden_tensor = torch.randn(3, 4)
        records = [
            _make_record("ex0", 0, (0.4, 0.3, 0.2, 0.1), (0.0, 0.5, 1.0, -0.5)),
            _make_record("ex1", 1, (0.3, 0.4, 0.2, 0.1), (0.1, 0.0, 0.5, -0.2)),
            _make_record("ex2", 2, (0.25, 0.25, 0.25, 0.25), (0.2, 0.2, 0.2, 0.2)),
        ]
        block = erc.evaluate_split(
            records=records,
            hidden_tensor=hidden_tensor,
            head=head,
            beta=1.0,
            temperature=1.0,
            hidden_filename="hidden.pt",
            num_report_examples=2,
        )
        self.assertEqual(block["num_records"], 3)
        self.assertEqual(len(block["examples"]), 2)
        for key in (
            "mean_kl_target_to_heuristic",
            "mean_kl_target_to_calibrated",
            "mean_improvement_kl",
            "median_improvement_kl",
            "frac_improved_kl",
            "mean_kl_calibrated_to_heuristic",
            "mean_entropy_heuristic",
            "mean_entropy_calibrated",
            "mean_entropy_target",
            "mean_l1_target_heuristic",
            "mean_l1_target_calibrated",
            "heuristic_top1_accuracy",
            "calibrated_top1_accuracy",
        ):
            self.assertIn(key, block["summary"])
        # zero-head identity sanity: calibrated equals heuristic on aggregate
        self.assertAlmostEqual(
            block["summary"]["mean_improvement_kl"], 0.0, places=5
        )
        self.assertAlmostEqual(
            block["summary"]["mean_kl_calibrated_to_heuristic"], 0.0, places=5
        )

    def test_max_examples_truncates_dataset(self) -> None:
        head = _make_head(4)
        hidden_tensor = torch.randn(5, 4)
        records = [
            _make_record(f"ex{i}", i, (0.4, 0.3, 0.2, 0.1), (0.0, 0.5, 1.0, -0.5))
            for i in range(5)
        ]
        block = erc.evaluate_split(
            records=records,
            hidden_tensor=hidden_tensor,
            head=head,
            beta=1.0,
            temperature=1.0,
            hidden_filename="hidden.pt",
            max_examples=2,
            num_report_examples=20,
        )
        self.assertEqual(block["num_records"], 2)
        self.assertEqual(len(block["examples"]), 2)


# === TESTS — report writer ===

class ReportWriterTests(unittest.TestCase):

    def test_writes_json_excluding_hidden_vectors(self) -> None:
        head = _make_head(4)
        hidden_tensor = torch.randn(2, 4)
        rec = _make_record("ex0", 0, (0.4, 0.3, 0.2, 0.1), (0.0, 0.5, 1.0, -0.5))
        per_record = erc.evaluate_record(
            record=rec,
            hidden_tensor=hidden_tensor,
            head=head,
            beta=1.0,
            temperature=1.0,
            hidden_filename="hidden.pt",
        )
        report = {
            "config_path": "config/config.yaml",
            "checkpoint_path": "ckpts/head.pt",
            "hidden_path": "hidden.pt",
            "splits": {
                "val": {
                    "records_path": "val/records.jsonl",
                    "num_records": 1,
                    "summary": erc.summarize_examples([per_record]),
                    "examples": [per_record],
                },
            },
            "checkpoint_metadata": {"beta": 1.0, "temperature": 1.0},
            "hyperparameters": {
                "beta": 1.0, "temperature": 1.0, "calibration_input_dim": 4,
            },
            "warnings": [],
        }
        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp = Path(raw_tmp)
            out = tmp / "nested" / "report.json"
            erc.write_report(report, out)
            self.assertTrue(out.is_file())
            roundtrip = json.loads(out.read_text(encoding="utf-8"))

        self.assertIn("splits", roundtrip)
        example = roundtrip["splits"]["val"]["examples"][0]
        # hidden vectors must not appear anywhere in the example block
        self.assertNotIn("hidden", example)
        self.assertNotIn("hidden_representation", example)
        self.assertNotIn("hidden_representation_ref", example)
        self.assertNotIn("hidden_vector", example)
        # exactly the expected example fields
        self.assertEqual(
            set(example.keys()),
            {"example_id", "prompt_text", "target_policy",
             "heuristic_prior", "calibrated_policy", "metrics"},
        )

    def test_json_safe_stringifies_unknown_objects(self) -> None:
        class Opaque:
            def __repr__(self) -> str:
                return "<opaque>"

        out = erc._json_safe({"x": Opaque(), "ys": [Opaque()], "n": 3})
        self.assertEqual(out, {"x": "<opaque>", "ys": ["<opaque>"], "n": 3})


# === ENTRY POINT ===

if __name__ == "__main__":
    unittest.main()
