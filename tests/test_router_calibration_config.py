# tests/test_router_calibration_config.py


# === IMPORTS ===

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

import yaml


# === MODULE LOADING ===

# add src/ to sys.path so router_calibration_config can be imported by name.
# router_calibration_config.py does not start with a digit, so it can be
# imported normally — unlike the 09_moce_components module imported elsewhere.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_DIR = _REPO_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from router_training import config as rcc  # noqa: E402


# === HELPERS ===

def _valid_block() -> dict[str, Any]:
    """
    Return a minimal but fully valid router_calibration block. Each test can
    deepcopy this and mutate one field to assert the corresponding validation.
    """
    return {
        "paths": {
            "prompt_sources": {
                "method12_path": "data/router/m12.jsonl",
                "method3_path": "data/router/m3.jsonl",
                "expert_validation_dir": "data/router/expert_validation",
            },
            "steering_vectors": {
                "economic_vector_path": "data/router/econ.pt",
                "social_vector_path": "data/router/soc.pt",
            },
            "expert_checkpoints": {
                "left_lib_checkpoint": "data/router/ckpt/left_lib",
                "left_auth_checkpoint": "data/router/ckpt/left_auth",
                "right_lib_checkpoint": "data/router/ckpt/right_lib",
                "right_auth_checkpoint": "data/router/ckpt/right_auth",
            },
            "output_dir": "data/router",
            "prompts_path": "data/router/prompts.jsonl",
            "features_path": "data/router/features.jsonl",
            "hidden_path": "data/router/hidden.pt",
            "records_path": "data/router/records.jsonl",
            "candidate_traces_path": "data/router/candidate_traces.jsonl",
            "splits_dir": "data/router/splits",
            "checkpoints_dir": "data/router/checkpoints",
            "reports_dir": "data/router/reports",
        },
        "model": {
            "base_model": "mistralai/Mistral-7B-v0.1",
            "dtype": "bfloat16",
            "device": "cuda",
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
        "prompt_set": {
            "include_method12": True,
            "include_method3": False,
            "include_expert_validation": False,
            "max_prompts": None,
            "seed": 42,
        },
        "candidate_policies": {
            "include_heuristic_prior": True,
            "include_uniform": True,
            "sharpen_temperatures": [0.5],
            "soften_temperatures": [2.0],
            "include_opposite_heavy": True,
            "include_adjacent_heavy": True,
            "dirichlet_samples": 16,
            "dirichlet_concentration": 64.0,
            "min_probability": 1.0e-6,
            "seed": 42,
        },
        "generation": {
            "max_new_tokens": 256,
            "temperature": 0.0,
            "do_sample": False,
            "top_p": 1.0,
        },
        "scoring": {
            "score_temperature": 0.2,
            "weights": {
                "bias_radius": 1.0,
                "quality": 0.5,
                "refusal": 0.5,
                "vagueness": 0.3,
                "kl_to_prior": 0.1,
            },
            "normalize_bias_radius": True,
            "baseline_bias_radius_path": None,
            "judge": {"enabled": False, "provider": None, "model": None},
        },
        "split": {
            "train_fraction": 0.8,
            "val_fraction": 0.1,
            "test_fraction": 0.1,
            "split_by": "source",
            "seed": 42,
        },
        "training": {
            "beta": 1.0,
            "temperature": 1.0,
            "learning_rate": 1.0e-3,
            "weight_decay": 1.0e-4,
            "batch_size": 32,
            "epochs": 20,
            "kl_weight": 0.1,
            "entropy_weight": 0.01,
            "seed": 42,
            "device": "cuda",
        },
    }


def _write_config(tmp: Path, block: dict[str, Any] | None) -> Path:
    """Write a config.yaml to tmp. If block is None, omit router_calibration."""
    cfg: dict[str, Any] = {"unrelated": {"x": 1}}
    if block is not None:
        cfg["router_calibration"] = block
    config_path = tmp / "config.yaml"
    with config_path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg, fh, sort_keys=False)
    return config_path


# === TESTS ===

class LoadRouterCalibrationConfigTests(unittest.TestCase):

    def test_valid_config_returns_resolved_dataclasses(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            cfg_path = _write_config(tmp_path, _valid_block())

            cfg = rcc.load_router_calibration_config(cfg_path)

        self.assertIsInstance(cfg, rcc.RouterCalibrationConfig)

        # paths are absolute and resolved against PROJECT_ROOT
        self.assertTrue(cfg.paths.output_dir.is_absolute())
        self.assertTrue(cfg.paths.prompts_path.is_absolute())
        self.assertTrue(cfg.paths.expert_checkpoints.left_lib_checkpoint.is_absolute())
        self.assertEqual(
            cfg.paths.expert_checkpoints.left_lib_checkpoint,
            (rcc.PROJECT_ROOT / "data/router/ckpt/left_lib").resolve(),
        )
        self.assertEqual(
            cfg.paths.steering_vectors.economic_vector_path,
            (rcc.PROJECT_ROOT / "data/router/econ.pt").resolve(),
        )

        # canonical quadrant order is preserved across the four checkpoint fields
        self.assertEqual(
            rcc.CANONICAL_QUADRANT_ORDER,
            ("left_lib", "left_auth", "right_lib", "right_auth"),
        )

        # representative scalar fields land on the right dataclass attribute
        self.assertEqual(cfg.input_transformer.calibration_input_dim, 4096)
        self.assertEqual(cfg.input_transformer.selected_layers, [8, 12, 16, 20, 24])
        self.assertEqual(cfg.model.dtype, "bfloat16")
        self.assertTrue(cfg.prompt_set.include_method12)
        self.assertFalse(cfg.prompt_set.include_method3)
        self.assertIsNone(cfg.input_transformer.neutral_reference_path)
        self.assertEqual(cfg.split.train_fraction, 0.8)
        self.assertEqual(cfg.training.epochs, 20)

    def test_missing_router_calibration_block_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg_path = _write_config(Path(tmp), None)
            with self.assertRaises(ValueError) as ctx:
                rcc.load_router_calibration_config(cfg_path)
        self.assertIn("router_calibration", str(ctx.exception))

    def test_missing_config_file_raises_filenotfound(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "does_not_exist.yaml"
            with self.assertRaises(FileNotFoundError):
                rcc.load_router_calibration_config(missing)

    def test_missing_required_subsection_raises(self) -> None:
        block = _valid_block()
        del block["scoring"]
        with tempfile.TemporaryDirectory() as tmp:
            cfg_path = _write_config(Path(tmp), block)
            with self.assertRaises(ValueError) as ctx:
                rcc.load_router_calibration_config(cfg_path)
        self.assertIn("scoring", str(ctx.exception))

    def test_missing_canonical_checkpoint_key_raises(self) -> None:
        block = _valid_block()
        del block["paths"]["expert_checkpoints"]["right_auth_checkpoint"]
        with tempfile.TemporaryDirectory() as tmp:
            cfg_path = _write_config(Path(tmp), block)
            with self.assertRaises(ValueError) as ctx:
                rcc.load_router_calibration_config(cfg_path)
        self.assertIn("right_auth_checkpoint", str(ctx.exception))

    def test_selected_layers_missing_20_raises(self) -> None:
        block = _valid_block()
        block["input_transformer"]["selected_layers"] = [8, 12, 16, 24]
        with tempfile.TemporaryDirectory() as tmp:
            cfg_path = _write_config(Path(tmp), block)
            with self.assertRaises(ValueError) as ctx:
                rcc.load_router_calibration_config(cfg_path)
        self.assertIn("20", str(ctx.exception))

    def test_non_mean_pooling_method_raises(self) -> None:
        block = _valid_block()
        block["input_transformer"]["pooling_method"] = "max"
        with tempfile.TemporaryDirectory() as tmp:
            cfg_path = _write_config(Path(tmp), block)
            with self.assertRaises(ValueError) as ctx:
                rcc.load_router_calibration_config(cfg_path)
        self.assertIn("pooling_method", str(ctx.exception))

    def test_invalid_vector_method_raises(self) -> None:
        block = _valid_block()
        block["input_transformer"]["vector_method"] = "random_projection"
        with tempfile.TemporaryDirectory() as tmp:
            cfg_path = _write_config(Path(tmp), block)
            with self.assertRaises(ValueError) as ctx:
                rcc.load_router_calibration_config(cfg_path)
        self.assertIn("vector_method", str(ctx.exception))

    def test_invalid_dtype_raises(self) -> None:
        block = _valid_block()
        block["model"]["dtype"] = "int8"
        with tempfile.TemporaryDirectory() as tmp:
            cfg_path = _write_config(Path(tmp), block)
            with self.assertRaises(ValueError) as ctx:
                rcc.load_router_calibration_config(cfg_path)
        self.assertIn("dtype", str(ctx.exception))

    def test_split_fractions_not_summing_to_one_raises(self) -> None:
        block = _valid_block()
        block["split"]["train_fraction"] = 0.7
        block["split"]["val_fraction"] = 0.1
        block["split"]["test_fraction"] = 0.1   # sums to 0.9
        with tempfile.TemporaryDirectory() as tmp:
            cfg_path = _write_config(Path(tmp), block)
            with self.assertRaises(ValueError) as ctx:
                rcc.load_router_calibration_config(cfg_path)
        self.assertIn("sum", str(ctx.exception).lower())

    def test_candidate_min_probability_zero_raises(self) -> None:
        block = _valid_block()
        block["candidate_policies"]["min_probability"] = 0.0
        with tempfile.TemporaryDirectory() as tmp:
            cfg_path = _write_config(Path(tmp), block)
            with self.assertRaises(ValueError) as ctx:
                rcc.load_router_calibration_config(cfg_path)
        self.assertIn("min_probability", str(ctx.exception))

    def test_candidate_min_probability_above_upper_bound_raises(self) -> None:
        block = _valid_block()
        block["candidate_policies"]["min_probability"] = 0.5
        with tempfile.TemporaryDirectory() as tmp:
            cfg_path = _write_config(Path(tmp), block)
            with self.assertRaises(ValueError) as ctx:
                rcc.load_router_calibration_config(cfg_path)
        self.assertIn("min_probability", str(ctx.exception))

    def test_score_temperature_zero_raises(self) -> None:
        block = _valid_block()
        block["scoring"]["score_temperature"] = 0.0
        with tempfile.TemporaryDirectory() as tmp:
            cfg_path = _write_config(Path(tmp), block)
            with self.assertRaises(ValueError) as ctx:
                rcc.load_router_calibration_config(cfg_path)
        self.assertIn("score_temperature", str(ctx.exception))

    def test_training_temperature_zero_raises(self) -> None:
        block = _valid_block()
        block["training"]["temperature"] = 0.0
        with tempfile.TemporaryDirectory() as tmp:
            cfg_path = _write_config(Path(tmp), block)
            with self.assertRaises(ValueError) as ctx:
                rcc.load_router_calibration_config(cfg_path)
        self.assertIn("temperature", str(ctx.exception))

    def test_non_bool_include_method12_raises(self) -> None:
        block = _valid_block()
        block["prompt_set"]["include_method12"] = "true"
        with tempfile.TemporaryDirectory() as tmp:
            cfg_path = _write_config(Path(tmp), block)
            with self.assertRaises(ValueError) as ctx:
                rcc.load_router_calibration_config(cfg_path)
        self.assertIn("include_method12", str(ctx.exception))

    def test_calibration_input_dim_zero_raises(self) -> None:
        block = _valid_block()
        block["input_transformer"]["calibration_input_dim"] = 0
        with tempfile.TemporaryDirectory() as tmp:
            cfg_path = _write_config(Path(tmp), block)
            with self.assertRaises(ValueError) as ctx:
                rcc.load_router_calibration_config(cfg_path)
        self.assertIn("calibration_input_dim", str(ctx.exception))


# === MAIN ===

def main() -> None:
    unittest.main()


if __name__ == "__main__":
    main()
