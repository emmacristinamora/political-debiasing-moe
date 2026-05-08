# tests/test_router_features.py


# === IMPORTS ===

from __future__ import annotations

import importlib.util
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

import torch


# === MODULE LOADING ===

# add src/ to sys.path so build_router_features and router_calibration_config
# can be imported by name. 09_moce_components.py uses a digit prefix and is
# loaded inside build_router_features via importlib already.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_DIR = _REPO_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import build_router_features as brf  # noqa: E402

# pull moce_components from inside the script so tests use the exact same
# class objects the code under test sees.
_COMPONENTS_PATH = _SRC_DIR / "09_moce_components.py"
_COMPONENTS_SPEC = importlib.util.spec_from_file_location(
    "moce_components_for_tests", _COMPONENTS_PATH,
)
moce_components = importlib.util.module_from_spec(_COMPONENTS_SPEC)
sys.modules["moce_components_for_tests"] = moce_components
assert _COMPONENTS_SPEC.loader is not None
_COMPONENTS_SPEC.loader.exec_module(moce_components)

PromptState = moce_components.PromptState
CANONICAL_QUADRANT_ORDER = moce_components.CANONICAL_QUADRANT_ORDER


# === FAKES ===

class _FakeOutput:
    pass


class _FakeModel:
    """
    Minimal stand-in for an HF causal LM. Returns a deterministic list of
    hidden_states whose layer-20 entry is configurable per instance, so each
    prompt yields a distinct unit-norm encoded vector.
    """

    def __init__(
        self,
        hidden_dim: int = 4,
        layer20_value: torch.Tensor | None = None,
        n_hidden_states: int = 22,
    ) -> None:
        self.hidden_dim = hidden_dim
        self._layer20_value = (
            layer20_value if layer20_value is not None
            else torch.tensor([1.0] + [0.0] * (hidden_dim - 1))
        )
        self._n_hidden_states = n_hidden_states

    def eval(self) -> None:
        pass

    def parameters(self):
        return iter([torch.zeros(1)])

    def __call__(self, *, input_ids, attention_mask, output_hidden_states):
        assert output_hidden_states is True
        seq_len = int(input_ids.shape[1])
        hidden_states: list[torch.Tensor] = [
            torch.full((1, seq_len, self.hidden_dim), float(i + 1))
            for i in range(self._n_hidden_states)
        ]
        if 20 < self._n_hidden_states:
            hidden_states[20] = (
                self._layer20_value.to(dtype=torch.float32)
                .unsqueeze(0).unsqueeze(0)
                .expand(1, seq_len, self.hidden_dim)
                .clone()
            )
        out = _FakeOutput()
        out.hidden_states = hidden_states
        return out


class _FakeTokenizer:
    """Returns a constant batch-1 token dict regardless of input text."""

    def __call__(self, text, *, return_tensors, truncation, max_length):
        n = max(1, min(len(text or " "), max_length))
        return {
            "input_ids":      torch.tensor([list(range(n))], dtype=torch.long),
            "attention_mask": torch.ones((1, n), dtype=torch.long),
        }


class _PerPromptModel:
    """
    Returns a layer-20 vector that depends on the input length, giving each
    prompt a unique encoded representation when paired with _FakeTokenizer
    (which produces token sequences whose length tracks the prompt length).
    Used to verify per-row uniqueness of hidden.pt rows.
    """

    def __init__(self, hidden_dim: int = 4, n_hidden_states: int = 22) -> None:
        self.hidden_dim = hidden_dim
        self._n_hidden_states = n_hidden_states

    def eval(self) -> None:
        pass

    def parameters(self):
        return iter([torch.zeros(1)])

    def __call__(self, *, input_ids, attention_mask, output_hidden_states):
        assert output_hidden_states is True
        seq_len = int(input_ids.shape[1])
        hidden_states: list[torch.Tensor] = [
            torch.full((1, seq_len, self.hidden_dim), float(i + 1))
            for i in range(self._n_hidden_states)
        ]
        # build a layer-20 vector keyed off seq_len so prompts of different
        # lengths produce visibly different encoded vectors after pooling.
        seed = float(seq_len)
        layer20 = torch.tensor(
            [seed, seed * 2.0] + [0.0] * (self.hidden_dim - 2),
        )
        hidden_states[20] = (
            layer20.unsqueeze(0).unsqueeze(0)
            .expand(1, seq_len, self.hidden_dim)
            .clone()
        )
        out = _FakeOutput()
        out.hidden_states = hidden_states
        return out


# === HELPERS ===

def _write_vector_artifact(
    path: Path,
    *,
    final_vector: torch.Tensor,
    layers=(8, 12, 16, 20, 24),
) -> None:
    """Write a synthetic stage-04 steering-vector .pt artifact."""
    final_vectors = {
        "logistic_regression": final_vector,
        "mean_difference": final_vector,
    }
    per_layer: dict[int, dict[str, dict[str, torch.Tensor]]] = {}
    for layer in layers:
        per_layer[layer] = {
            "logistic_regression": {"vector": final_vector},
            "mean_difference": {"vector": final_vector},
        }
    torch.save({"final_vectors": final_vectors, "per_layer": per_layer}, path)


def _build_synthetic_transformer(
    tmp: Path,
    *,
    hidden_dim: int = 4,
    layer20_value: torch.Tensor | None = None,
    model: Any | None = None,
) -> Any:
    """Build an InputTransformer from synthetic vectors + fake model/tokenizer."""
    econ = torch.tensor([1.0] + [0.0] * (hidden_dim - 1))
    soc  = torch.tensor([0.0, 1.0] + [0.0] * (hidden_dim - 2))
    econ_path = tmp / "economic_vectors.pt"
    soc_path  = tmp / "social_vectors.pt"
    _write_vector_artifact(econ_path, final_vector=econ)
    _write_vector_artifact(soc_path,  final_vector=soc)

    cfg = moce_components.SteeringVectorConfig(
        economic_vector_path=econ_path,
        social_vector_path=soc_path,
        vector_method="logistic_regression",
        use_final_aggregated_vectors=True,
        selected_layers=[8, 12, 16, 20, 24],
        pooling_method="mean",
        use_centering=False,
        neutral_reference_path=None,
    )
    fake_model = model if model is not None else _FakeModel(
        hidden_dim=hidden_dim, layer20_value=layer20_value,
    )
    return moce_components.InputTransformer(
        model=fake_model,
        tokenizer=_FakeTokenizer(),
        steering_config=cfg,
    )


def _make_prompt(example_id: str, text: str, source: str = "method12") -> dict:
    return {
        "example_id": example_id,
        "prompt_text": text,
        "source": source,
        "metadata": {"original_id": example_id, "axis": None, "source_file": "x.jsonl"},
    }


def _write_prompts_jsonl(path: Path, prompts: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for record in prompts:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")


# === TESTS — load_prompts ===

class LoadPromptsTests(unittest.TestCase):

    def test_happy_path_returns_records_in_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "prompts.jsonl"
            _write_prompts_jsonl(path, [
                _make_prompt("a", "first"),
                _make_prompt("b", "second"),
            ])
            out = brf.load_prompts(path)
            self.assertEqual([r["example_id"] for r in out], ["a", "b"])

    def test_missing_file_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                brf.load_prompts(Path(tmp) / "nope.jsonl")

    def test_empty_file_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "p.jsonl"
            path.write_text("", encoding="utf-8")
            with self.assertRaises(ValueError):
                brf.load_prompts(path)

    def test_malformed_json_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "p.jsonl"
            path.write_text("{not json}\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                brf.load_prompts(path)

    def test_missing_required_field_raises(self) -> None:
        bad = {"example_id": "x", "prompt_text": "y", "metadata": {}}  # source missing
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "p.jsonl"
            path.write_text(json.dumps(bad) + "\n", encoding="utf-8")
            with self.assertRaises(ValueError) as ctx:
                brf.load_prompts(path)
            self.assertIn("source", str(ctx.exception))

    def test_empty_prompt_text_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "p.jsonl"
            _write_prompts_jsonl(path, [_make_prompt("a", "   ")])
            with self.assertRaises(ValueError):
                brf.load_prompts(path)

    def test_duplicate_example_id_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "p.jsonl"
            _write_prompts_jsonl(path, [
                _make_prompt("dup", "one"),
                _make_prompt("dup", "two"),
            ])
            with self.assertRaises(ValueError) as ctx:
                brf.load_prompts(path)
            self.assertIn("dup", str(ctx.exception))


# === TESTS — validate_prompt_state ===

class ValidatePromptStateTests(unittest.TestCase):

    def _good_state(self, hidden: torch.Tensor | None = None) -> PromptState:
        h = hidden if hidden is not None else torch.tensor([1.0, 0.0, 0.0, 0.0])
        return PromptState(
            prompt_text="p",
            hidden_representation=h,
            economic_score=0.1,
            social_score=-0.2,
            quadrant_scores={k: 0.0 for k in CANONICAL_QUADRANT_ORDER},
            bias_magnitude=0.3,
            metadata={},
        )

    def test_happy_path_returns_float32_cpu(self) -> None:
        out = brf.validate_prompt_state(self._good_state(), where="x")
        self.assertEqual(out.dtype, torch.float32)
        self.assertEqual(out.device.type, "cpu")
        self.assertEqual(out.dim(), 1)

    def test_non_tensor_hidden_raises(self) -> None:
        bad = self._good_state()
        bad.hidden_representation = [1.0, 0.0, 0.0, 0.0]
        with self.assertRaises(ValueError):
            brf.validate_prompt_state(bad, where="x")

    def test_rank_2_hidden_raises(self) -> None:
        bad = self._good_state(hidden=torch.zeros((2, 4)))
        with self.assertRaises(ValueError):
            brf.validate_prompt_state(bad, where="x")

    def test_nan_hidden_raises(self) -> None:
        nan_vec = torch.tensor([1.0, float("nan"), 0.0, 0.0])
        with self.assertRaises(ValueError):
            brf.validate_prompt_state(self._good_state(hidden=nan_vec), where="x")

    def test_inf_hidden_raises(self) -> None:
        inf_vec = torch.tensor([float("inf"), 0.0, 0.0, 0.0])
        with self.assertRaises(ValueError):
            brf.validate_prompt_state(self._good_state(hidden=inf_vec), where="x")

    def test_wrong_quadrant_keys_raise(self) -> None:
        bad = self._good_state()
        bad.quadrant_scores = {"left_lib": 0.0, "left_auth": 0.0, "right_lib": 0.0}
        with self.assertRaises(ValueError):
            brf.validate_prompt_state(bad, where="x")

    def test_nan_quadrant_score_raises(self) -> None:
        bad = self._good_state()
        bad.quadrant_scores = {**bad.quadrant_scores, "left_lib": float("nan")}
        with self.assertRaises(ValueError):
            brf.validate_prompt_state(bad, where="x")

    def test_nan_bias_magnitude_raises(self) -> None:
        bad = self._good_state()
        bad.bias_magnitude = float("nan")
        with self.assertRaises(ValueError):
            brf.validate_prompt_state(bad, where="x")


# === TESTS — build_feature_records ===

class BuildFeatureRecordsTests(unittest.TestCase):

    def test_canonical_scores_and_hidden_refs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            transformer = _build_synthetic_transformer(Path(tmp))
            prompts = [
                _make_prompt("p1", "hello"),
                _make_prompt("p2", "world!"),
            ]
            records, hidden = brf.build_feature_records(
                prompts, transformer, hidden_filename="hidden.pt",
            )

        self.assertEqual(len(records), 2)
        for i, rec in enumerate(records):
            self.assertEqual(set(rec.keys()), {
                "example_id", "prompt_text", "source",
                "quadrant_scores", "bias_magnitude",
                "economic_score", "social_score",
                "hidden_representation_ref", "metadata",
            })
            # canonical order is preserved in the dict (Python preserves insertion)
            self.assertEqual(
                tuple(rec["quadrant_scores"].keys()), CANONICAL_QUADRANT_ORDER,
            )
            for v in rec["quadrant_scores"].values():
                self.assertTrue(math.isfinite(v))
            self.assertTrue(math.isfinite(rec["bias_magnitude"]))
            self.assertEqual(rec["hidden_representation_ref"], f"hidden.pt:{i}")
            md = rec["metadata"]
            self.assertEqual(md["feature_source"], "InputTransformer.transform")
            self.assertEqual(md["hidden_dtype"], "float32")
            self.assertIn("input_transformer", md)
            self.assertEqual(md["original_id"], rec["example_id"])

        self.assertEqual(hidden.dtype, torch.float32)
        self.assertEqual(hidden.shape[0], 2)
        self.assertEqual(hidden.dim(), 2)

    def test_hidden_rows_match_refs_per_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            transformer = _build_synthetic_transformer(
                Path(tmp), model=_PerPromptModel(),
            )
            prompts = [
                _make_prompt("a", "x"),
                _make_prompt("b", "longer prompt"),
                _make_prompt("c", "another distinct one"),
            ]
            records, hidden = brf.build_feature_records(
                prompts, transformer, hidden_filename="hidden.pt",
            )

        # each ref points to its own row; rows should be pairwise distinct
        for i, rec in enumerate(records):
            self.assertEqual(rec["hidden_representation_ref"], f"hidden.pt:{i}")
        for i in range(hidden.shape[0]):
            for j in range(i + 1, hidden.shape[0]):
                self.assertFalse(
                    torch.allclose(hidden[i], hidden[j], atol=1e-6),
                    msg=f"hidden rows {i} and {j} unexpectedly identical",
                )

    def test_empty_prompts_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            transformer = _build_synthetic_transformer(Path(tmp))
            with self.assertRaises(ValueError):
                brf.build_feature_records(
                    [], transformer, hidden_filename="hidden.pt",
                )

    def test_calibration_input_dim_mismatch_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            # synthetic transformer encodes at hidden_dim=4; passing 8 must trip
            # the post-stack guard with a message naming both dimensions.
            transformer = _build_synthetic_transformer(Path(tmp))
            prompts = [_make_prompt("p1", "alpha"), _make_prompt("p2", "beta")]
            with self.assertRaises(ValueError) as ctx:
                brf.build_feature_records(
                    prompts,
                    transformer,
                    hidden_filename="hidden.pt",
                    expected_hidden_dim=8,
                )
            msg = str(ctx.exception)
            self.assertIn("4", msg)
            self.assertIn("8", msg)
            self.assertIn("calibration_input_dim", msg)


# === TESTS — write_outputs ===

class WriteOutputsTests(unittest.TestCase):

    def test_writes_files_and_round_trips(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            transformer = _build_synthetic_transformer(Path(tmp))
            prompts = [_make_prompt("p1", "alpha"), _make_prompt("p2", "beta")]
            records, hidden = brf.build_feature_records(
                prompts, transformer, hidden_filename="hidden.pt",
            )
            features_path = Path(tmp) / "out" / "features.jsonl"
            hidden_path   = Path(tmp) / "out" / "hidden.pt"
            brf.write_outputs(records, hidden, features_path, hidden_path)

            self.assertTrue(features_path.is_file())
            self.assertTrue(hidden_path.is_file())

            loaded_records: list[dict] = []
            with features_path.open(encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        loaded_records.append(json.loads(line))
            self.assertEqual(loaded_records, records)

            loaded_hidden = torch.load(hidden_path, map_location="cpu")
            self.assertTrue(torch.is_tensor(loaded_hidden))
            self.assertEqual(loaded_hidden.dtype, torch.float32)
            self.assertEqual(tuple(loaded_hidden.shape), tuple(hidden.shape))
            self.assertTrue(torch.allclose(loaded_hidden, hidden))

            for i, rec in enumerate(loaded_records):
                self.assertEqual(rec["hidden_representation_ref"], f"hidden.pt:{i}")
                # the ref maps to the actual row in hidden.pt
                _, idx_str = rec["hidden_representation_ref"].split(":")
                self.assertTrue(torch.allclose(loaded_hidden[int(idx_str)], hidden[i]))

    def test_row_count_mismatch_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            features_path = Path(tmp) / "f.jsonl"
            hidden_path   = Path(tmp) / "h.pt"
            with self.assertRaises(ValueError):
                brf.write_outputs(
                    records=[{"x": 1}, {"x": 2}],
                    hidden_tensor=torch.zeros((1, 4), dtype=torch.float32),
                    features_path=features_path,
                    hidden_path=hidden_path,
                )

    def test_non_float32_hidden_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                brf.write_outputs(
                    records=[{"x": 1}],
                    hidden_tensor=torch.zeros((1, 4), dtype=torch.float64),
                    features_path=Path(tmp) / "f.jsonl",
                    hidden_path=Path(tmp) / "h.pt",
                )


# === TESTS — run_build (end-to-end via injected transformer) ===

class RunBuildEndToEndTests(unittest.TestCase):

    def _build_synthetic_config(
        self, tmp: Path, prompts: list[dict],
    ) -> tuple[Any, Path, Path]:
        prompts_path  = tmp / "prompts.jsonl"
        features_path = tmp / "features.jsonl"
        hidden_path   = tmp / "hidden.pt"
        _write_prompts_jsonl(prompts_path, prompts)

        cfg = _SyntheticConfig(
            prompts_path=prompts_path,
            features_path=features_path,
            hidden_path=hidden_path,
        )
        return cfg, features_path, hidden_path

    def test_end_to_end_writes_features_and_hidden(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            prompts = [_make_prompt("p1", "alpha"), _make_prompt("p2", "beta")]
            cfg, features_path, hidden_path = self._build_synthetic_config(tmp_path, prompts)
            transformer = _build_synthetic_transformer(
                tmp_path, model=_PerPromptModel(),
            )

            paths = brf.run_build(cfg, transformer=transformer)

            self.assertEqual(paths["features_path"], features_path)
            self.assertEqual(paths["hidden_path"], hidden_path)
            self.assertTrue(features_path.is_file())
            self.assertTrue(hidden_path.is_file())

            with features_path.open(encoding="utf-8") as fh:
                rows = [json.loads(line) for line in fh if line.strip()]
            self.assertEqual(len(rows), 2)
            self.assertEqual([r["example_id"] for r in rows], ["p1", "p2"])

            hidden = torch.load(hidden_path, map_location="cpu")
            self.assertEqual(hidden.dtype, torch.float32)
            self.assertEqual(hidden.shape[0], 2)

    def test_limit_truncates_deterministically(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            prompts = [
                _make_prompt("p1", "alpha"),
                _make_prompt("p2", "beta"),
                _make_prompt("p3", "gamma"),
            ]
            cfg, features_path, hidden_path = self._build_synthetic_config(tmp_path, prompts)
            transformer = _build_synthetic_transformer(
                tmp_path, model=_PerPromptModel(),
            )
            brf.run_build(cfg, transformer=transformer, limit=2)

            with features_path.open(encoding="utf-8") as fh:
                rows = [json.loads(line) for line in fh if line.strip()]
            self.assertEqual([r["example_id"] for r in rows], ["p1", "p2"])

            hidden = torch.load(hidden_path, map_location="cpu")
            self.assertEqual(hidden.shape[0], 2)

            # second run with same inputs should give byte-identical features
            features_first = features_path.read_bytes()
            hidden_first = hidden.clone()
            transformer2 = _build_synthetic_transformer(
                tmp_path, model=_PerPromptModel(),
            )
            brf.run_build(cfg, transformer=transformer2, limit=2)
            self.assertEqual(features_path.read_bytes(), features_first)
            hidden_second = torch.load(hidden_path, map_location="cpu")
            self.assertTrue(torch.allclose(hidden_first, hidden_second))


# === SYNTHETIC CONFIG SHIM ===

class _SyntheticSteeringPaths:
    def __init__(self, econ: Path, soc: Path) -> None:
        self.economic_vector_path = econ
        self.social_vector_path = soc


class _SyntheticPaths:
    def __init__(self, prompts_path: Path, features_path: Path, hidden_path: Path) -> None:
        self.prompts_path  = prompts_path
        self.features_path = features_path
        self.hidden_path   = hidden_path
        # placeholder steering paths; run_build only consults these when it
        # has to construct an InputTransformer itself, which the tests skip
        # by injecting `transformer=...`.
        self.steering_vectors = _SyntheticSteeringPaths(
            econ=prompts_path.with_name("__unused_econ.pt"),
            soc=prompts_path.with_name("__unused_soc.pt"),
        )


class _SyntheticInputTransformerCfg:
    vector_method = "logistic_regression"
    use_final_aggregated_vectors = True
    selected_layers = [8, 12, 16, 20, 24]
    pooling_method = "mean"
    use_centering = False
    neutral_reference_path = None
    # synthetic fakes use hidden_dim=4; matches the stacked hidden_tensor so
    # run_build's calibration_input_dim cross-check passes by default.
    calibration_input_dim = 4


class _SyntheticModelCfg:
    base_model = "synthetic/model"
    dtype = "float32"
    device = "cpu"


class _SyntheticConfig:
    """
    Duck-typed RouterCalibrationConfig stand-in. Exposes only the attributes
    that build_router_features.run_build reads when a transformer is injected,
    so tests don't need a real config.yaml or a real model.
    """
    def __init__(self, prompts_path: Path, features_path: Path, hidden_path: Path) -> None:
        self.paths = _SyntheticPaths(prompts_path, features_path, hidden_path)
        self.model = _SyntheticModelCfg()
        self.input_transformer = _SyntheticInputTransformerCfg()


# === MAIN ===

def main() -> None:
    unittest.main()


if __name__ == "__main__":
    main()
