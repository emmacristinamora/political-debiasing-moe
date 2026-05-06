# tests/test_input_transformer.py


# === IMPORTS ===

import importlib.util
import math
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

import torch


# === MODULE LOADING ===

# src/09_moce_components.py begins with a digit, so it cannot be imported via
# normal "import" syntax. load it explicitly by absolute path with importlib.
_COMPONENTS_PATH = Path(__file__).resolve().parents[1] / "src" / "09_moce_components.py"
_SPEC = importlib.util.spec_from_file_location("moce_components", _COMPONENTS_PATH)
moce_components = importlib.util.module_from_spec(_SPEC)
sys.modules["moce_components"] = moce_components
assert _SPEC.loader is not None
_SPEC.loader.exec_module(moce_components)

InputTransformer = moce_components.InputTransformer
SteeringVectorConfig = moce_components.SteeringVectorConfig
PromptState = moce_components.PromptState
CANONICAL_QUADRANT_ORDER = moce_components.CANONICAL_QUADRANT_ORDER
Router = moce_components.Router
RouterConfig = moce_components.RouterConfig


# === FAKES ===

class _FakeOutput:
    """Plain attribute carrier; mimics the .hidden_states surface only."""


class _FakeModel:
    """
    Minimal stand-in for an HF causal LM. Returns a deterministic list of
    hidden_states whose layer 20 / layer 24 contents can be controlled
    per-instance to verify encode_prompt picks layer 20.
    """

    def __init__(
        self,
        hidden_dim: int = 4,
        layer20_value: torch.Tensor | None = None,
        layer24_value: torch.Tensor | None = None,
        n_hidden_states: int = 22,
    ) -> None:
        self.hidden_dim = hidden_dim
        self._layer20_value = (
            layer20_value if layer20_value is not None
            else torch.tensor([1.0] + [0.0] * (hidden_dim - 1))
        )
        self._layer24_value = (
            layer24_value if layer24_value is not None
            else torch.tensor([0.0, 1.0] + [0.0] * (hidden_dim - 2))
        )
        self._n_hidden_states = n_hidden_states
        self.eval_called = False

    def eval(self) -> None:
        self.eval_called = True

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
        if 24 < self._n_hidden_states:
            hidden_states[24] = (
                self._layer24_value.to(dtype=torch.float32)
                .unsqueeze(0).unsqueeze(0)
                .expand(1, seq_len, self.hidden_dim)
                .clone()
            )
        out = _FakeOutput()
        out.hidden_states = hidden_states
        return out


class _FakeTokenizer:
    """Returns a constant batch-1 token dict; flags toggle malformed shapes."""

    def __init__(
        self,
        *,
        drop_input_ids: bool = False,
        drop_attention_mask: bool = False,
        force_zero_mask: bool = False,
    ) -> None:
        self.drop_input_ids = drop_input_ids
        self.drop_attention_mask = drop_attention_mask
        self.force_zero_mask = force_zero_mask

    def __call__(self, text, *, return_tensors, truncation, max_length):
        # produce a non-empty token sequence regardless of input
        n = max(1, min(len(text or " "), max_length))
        ids = list(range(n))
        attn_values = (
            torch.zeros((1, n), dtype=torch.long)
            if self.force_zero_mask
            else torch.ones((1, n), dtype=torch.long)
        )
        tokens: dict[str, torch.Tensor] = {
            "input_ids": torch.tensor([ids], dtype=torch.long),
            "attention_mask": attn_values,
        }
        if self.drop_input_ids:
            tokens.pop("input_ids", None)
        if self.drop_attention_mask:
            tokens.pop("attention_mask", None)
        return tokens


# === HELPERS: VECTOR ARTIFACTS ===

def _write_vector_artifact(
    path: Path,
    *,
    final_vector: torch.Tensor,
    layers=(8, 12, 16, 20, 24),
    per_layer_vectors: dict[int, torch.Tensor] | None = None,
    omit_layers: tuple[int, ...] = (),
) -> None:
    """Write a synthetic stage-04 steering-vector .pt artifact."""
    final_vectors = {
        "logistic_regression": final_vector,
        "mean_difference": final_vector,
    }
    per_layer: dict[int, dict[str, dict[str, torch.Tensor]]] = {}
    for layer in layers:
        if layer in omit_layers:
            continue
        layer_vector = (
            per_layer_vectors[layer]
            if per_layer_vectors and layer in per_layer_vectors
            else final_vector
        )
        per_layer[layer] = {
            "logistic_regression": {"vector": layer_vector},
            "mean_difference": {"vector": layer_vector},
        }
    torch.save({"final_vectors": final_vectors, "per_layer": per_layer}, path)


def _basis_econ(hidden_dim: int = 4) -> torch.Tensor:
    return torch.tensor([1.0] + [0.0] * (hidden_dim - 1))


def _basis_social(hidden_dim: int = 4) -> torch.Tensor:
    return torch.tensor([0.0, 1.0] + [0.0] * (hidden_dim - 2))


# === HELPERS: BUILD INPUTTRANSFORMER ===

def _build_transformer(
    tmp_dir: Path,
    *,
    hidden_dim: int = 4,
    use_centering: bool = False,
    selected_layers=(8, 12, 16, 20, 24),
    use_final_aggregated_vectors: bool = True,
    vector_method: str = "logistic_regression",
    econ_vector: torch.Tensor | None = None,
    social_vector: torch.Tensor | None = None,
    neutral_vector: torch.Tensor | None = None,
    layer20_value: torch.Tensor | None = None,
    layer24_value: torch.Tensor | None = None,
    n_hidden_states: int = 22,
    tokenizer: _FakeTokenizer | None = None,
    omit_layers_econ: tuple[int, ...] = (),
    omit_layers_social: tuple[int, ...] = (),
    model_hidden_dim: int | None = None,
) -> Any:
    econ = econ_vector if econ_vector is not None else _basis_econ(hidden_dim)
    soc = social_vector if social_vector is not None else _basis_social(hidden_dim)
    econ_path = tmp_dir / "economic_vectors.pt"
    soc_path = tmp_dir / "social_vectors.pt"
    _write_vector_artifact(
        econ_path,
        final_vector=econ,
        layers=selected_layers,
        omit_layers=omit_layers_econ,
    )
    _write_vector_artifact(
        soc_path,
        final_vector=soc,
        layers=selected_layers,
        omit_layers=omit_layers_social,
    )
    neutral_path: Path | None = None
    if neutral_vector is not None:
        neutral_path = tmp_dir / "neutral_reference.pt"
        torch.save({"vector": neutral_vector}, neutral_path)

    config = SteeringVectorConfig(
        economic_vector_path=econ_path,
        social_vector_path=soc_path,
        vector_method=vector_method,
        use_final_aggregated_vectors=use_final_aggregated_vectors,
        selected_layers=list(selected_layers),
        use_centering=use_centering,
        neutral_reference_path=neutral_path,
    )
    model = _FakeModel(
        hidden_dim=model_hidden_dim if model_hidden_dim is not None else hidden_dim,
        layer20_value=layer20_value,
        layer24_value=layer24_value,
        n_hidden_states=n_hidden_states,
    )
    tok = tokenizer if tokenizer is not None else _FakeTokenizer()
    return InputTransformer(model=model, tokenizer=tok, steering_config=config)


# === BASE TEST CLASS ===

class _BaseTempDirTest(unittest.TestCase):

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()


# === TESTS ===

class VectorLoadingTests(_BaseTempDirTest):

    def test_final_vectors_logistic_regression_normalized(self) -> None:
        # raw vectors carry non-unit norm; the loader normalizes them
        econ = torch.tensor([3.0, 0.0, 0.0, 0.0])
        soc = torch.tensor([0.0, 4.0, 0.0, 0.0])
        inst = _build_transformer(self.tmp, econ_vector=econ, social_vector=soc)
        self.assertAlmostEqual(
            float(torch.linalg.vector_norm(inst.economic_vector).item()), 1.0, places=6,
        )
        self.assertAlmostEqual(
            float(torch.linalg.vector_norm(inst.social_vector).item()), 1.0, places=6,
        )
        self.assertTrue(torch.allclose(
            inst.economic_vector, torch.tensor([1.0, 0.0, 0.0, 0.0]), atol=1e-6,
        ))
        self.assertTrue(torch.allclose(
            inst.social_vector, torch.tensor([0.0, 1.0, 0.0, 0.0]), atol=1e-6,
        ))
        self.assertEqual(inst.economic_vector.dtype, torch.float32)

    def test_quadrant_keys_canonical(self) -> None:
        inst = _build_transformer(self.tmp)
        self.assertEqual(set(inst.quadrant_vectors.keys()), set(CANONICAL_QUADRANT_ORDER))
        self.assertEqual(list(inst.quadrant_vectors.keys()), list(CANONICAL_QUADRANT_ORDER))

    def test_quadrant_formulas(self) -> None:
        inst = _build_transformer(self.tmp)
        sqrt2 = math.sqrt(2.0)
        expected = {
            "left_lib":   torch.tensor([-1 / sqrt2, -1 / sqrt2, 0.0, 0.0]),
            "left_auth":  torch.tensor([-1 / sqrt2,  1 / sqrt2, 0.0, 0.0]),
            "right_lib":  torch.tensor([ 1 / sqrt2, -1 / sqrt2, 0.0, 0.0]),
            "right_auth": torch.tensor([ 1 / sqrt2,  1 / sqrt2, 0.0, 0.0]),
        }
        for key in CANONICAL_QUADRANT_ORDER:
            self.assertTrue(
                torch.allclose(inst.quadrant_vectors[key], expected[key], atol=1e-6),
                f"{key}: got {inst.quadrant_vectors[key]}, expected {expected[key]}",
            )

    def test_missing_vector_file_raises(self) -> None:
        config = SteeringVectorConfig(
            economic_vector_path=self.tmp / "does_not_exist_econ.pt",
            social_vector_path=self.tmp / "does_not_exist_soc.pt",
        )
        with self.assertRaises(FileNotFoundError):
            InputTransformer(_FakeModel(hidden_dim=4), _FakeTokenizer(), config)

    def test_unsupported_vector_method_raises(self) -> None:
        econ_path = self.tmp / "economic_vectors.pt"
        soc_path = self.tmp / "social_vectors.pt"
        _write_vector_artifact(econ_path, final_vector=_basis_econ())
        _write_vector_artifact(soc_path, final_vector=_basis_social())
        config = SteeringVectorConfig(
            economic_vector_path=econ_path,
            social_vector_path=soc_path,
            vector_method="unsupported_method_name",
        )
        with self.assertRaisesRegex(ValueError, "vector_method"):
            InputTransformer(_FakeModel(hidden_dim=4), _FakeTokenizer(), config)

    def test_nan_vector_raises(self) -> None:
        bad = torch.tensor([float("nan"), 0.0, 0.0, 0.0])
        with self.assertRaisesRegex(ValueError, "NaN|inf"):
            _build_transformer(self.tmp, econ_vector=bad)

    def test_zero_vector_raises(self) -> None:
        zero = torch.zeros(4)
        with self.assertRaisesRegex(ValueError, "norm"):
            _build_transformer(self.tmp, econ_vector=zero)

    def test_econ_social_dim_mismatch_raises(self) -> None:
        econ_path = self.tmp / "economic_vectors.pt"
        soc_path = self.tmp / "social_vectors.pt"
        _write_vector_artifact(econ_path, final_vector=torch.tensor([1.0, 0.0, 0.0, 0.0]))
        _write_vector_artifact(soc_path, final_vector=torch.tensor([0.0, 1.0, 0.0]))
        config = SteeringVectorConfig(
            economic_vector_path=econ_path,
            social_vector_path=soc_path,
        )
        with self.assertRaisesRegex(ValueError, "shape"):
            InputTransformer(_FakeModel(hidden_dim=4), _FakeTokenizer(), config)


class PerLayerVectorLoadingTests(_BaseTempDirTest):

    def test_per_layer_aggregated_loads(self) -> None:
        # all per-layer vectors identical; the aggregate (post-normalize) is
        # the same vector again, with unit norm
        inst = _build_transformer(
            self.tmp,
            use_final_aggregated_vectors=False,
            selected_layers=(8, 12, 16, 20, 24),
        )
        self.assertAlmostEqual(
            float(torch.linalg.vector_norm(inst.economic_vector).item()), 1.0, places=6,
        )
        self.assertTrue(torch.allclose(
            inst.economic_vector, torch.tensor([1.0, 0.0, 0.0, 0.0]), atol=1e-6,
        ))

    def test_duplicate_selected_layers_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicate"):
            _build_transformer(
                self.tmp,
                use_final_aggregated_vectors=False,
                selected_layers=(8, 8, 16, 20, 24),
            )

    def test_missing_layer_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "missing"):
            _build_transformer(
                self.tmp,
                use_final_aggregated_vectors=False,
                selected_layers=(8, 12, 16, 20, 24),
                omit_layers_econ=(24,),
            )


class EncodePromptTests(_BaseTempDirTest):

    def test_encoded_is_unit_norm_rank1_float32(self) -> None:
        inst = _build_transformer(self.tmp)
        out = inst.encode_prompt("hello")
        self.assertIsInstance(out, torch.Tensor)
        self.assertEqual(out.dim(), 1)
        self.assertEqual(out.shape[0], 4)
        self.assertEqual(out.dtype, torch.float32)
        self.assertAlmostEqual(
            float(torch.linalg.vector_norm(out).item()), 1.0, places=6,
        )

    def test_uses_layer_20_not_layer_24(self) -> None:
        # layer 20 = econ basis, layer 24 = social basis. encode_prompt must
        # pull layer 20: result projects fully onto the econ axis.
        inst = _build_transformer(
            self.tmp,
            layer20_value=torch.tensor([1.0, 0.0, 0.0, 0.0]),
            layer24_value=torch.tensor([0.0, 1.0, 0.0, 0.0]),
        )
        encoded = inst.encode_prompt("hi")
        self.assertTrue(torch.allclose(
            encoded, torch.tensor([1.0, 0.0, 0.0, 0.0]), atol=1e-6,
        ))

    def test_eval_called_in_init(self) -> None:
        econ_path = self.tmp / "economic_vectors.pt"
        soc_path = self.tmp / "social_vectors.pt"
        _write_vector_artifact(econ_path, final_vector=_basis_econ())
        _write_vector_artifact(soc_path, final_vector=_basis_social())
        config = SteeringVectorConfig(
            economic_vector_path=econ_path, social_vector_path=soc_path,
        )
        model = _FakeModel(hidden_dim=4)
        InputTransformer(model, _FakeTokenizer(), config)
        self.assertTrue(model.eval_called)

    def test_empty_prompt_raises(self) -> None:
        inst = _build_transformer(self.tmp)
        for bad in ("", "   "):
            with self.assertRaisesRegex(ValueError, "non-empty"):
                inst.encode_prompt(bad)
        with self.assertRaisesRegex(ValueError, "non-empty"):
            inst.encode_prompt(None)  # type: ignore[arg-type]

    def test_tokenizer_missing_input_ids_raises(self) -> None:
        inst = _build_transformer(self.tmp, tokenizer=_FakeTokenizer(drop_input_ids=True))
        with self.assertRaisesRegex(ValueError, "input_ids"):
            inst.encode_prompt("hello")

    def test_tokenizer_missing_attention_mask_raises(self) -> None:
        inst = _build_transformer(
            self.tmp, tokenizer=_FakeTokenizer(drop_attention_mask=True),
        )
        with self.assertRaisesRegex(ValueError, "attention_mask"):
            inst.encode_prompt("hello")

    def test_zero_attention_mask_raises(self) -> None:
        inst = _build_transformer(
            self.tmp, tokenizer=_FakeTokenizer(force_zero_mask=True),
        )
        with self.assertRaisesRegex(ValueError, "non-padding"):
            inst.encode_prompt("hello")

    def test_short_hidden_states_raises(self) -> None:
        inst = _build_transformer(self.tmp, n_hidden_states=5)
        with self.assertRaisesRegex(ValueError, "out of range"):
            inst.encode_prompt("hello")

    def test_hidden_dim_mismatch_raises(self) -> None:
        # steering vectors are dim 4 but the model produces dim 8 at layer 20
        inst = _build_transformer(self.tmp, hidden_dim=4, model_hidden_dim=8)
        with self.assertRaisesRegex(ValueError, "hidden_dim"):
            inst.encode_prompt("hello")


class ScoringTests(_BaseTempDirTest):

    def test_compute_axis_scores_keys_and_values(self) -> None:
        inst = _build_transformer(self.tmp)
        h = torch.tensor([1.0, 0.0, 0.0, 0.0])
        scores = inst.compute_axis_scores(h)
        self.assertEqual(set(scores.keys()), {"economic_score", "social_score"})
        self.assertAlmostEqual(scores["economic_score"], 1.0, places=6)
        self.assertAlmostEqual(scores["social_score"], 0.0, places=6)
        self.assertIsInstance(scores["economic_score"], float)
        self.assertIsInstance(scores["social_score"], float)

    def test_compute_axis_scores_with_social_basis(self) -> None:
        inst = _build_transformer(self.tmp)
        h = torch.tensor([0.0, 1.0, 0.0, 0.0])
        scores = inst.compute_axis_scores(h)
        self.assertAlmostEqual(scores["economic_score"], 0.0, places=6)
        self.assertAlmostEqual(scores["social_score"], 1.0, places=6)

    def test_compute_axis_scores_rejects_bad_inputs(self) -> None:
        inst = _build_transformer(self.tmp)
        with self.assertRaisesRegex(ValueError, "torch.Tensor"):
            inst.compute_axis_scores([1.0, 0.0, 0.0, 0.0])  # type: ignore[arg-type]
        with self.assertRaisesRegex(ValueError, "rank-1"):
            inst.compute_axis_scores(torch.zeros((2, 4)))
        with self.assertRaisesRegex(ValueError, "length"):
            inst.compute_axis_scores(torch.zeros(7))
        with self.assertRaisesRegex(ValueError, "NaN|inf"):
            inst.compute_axis_scores(torch.tensor([float("nan"), 0.0, 0.0, 0.0]))

    def test_compute_quadrant_scores_canonical_order(self) -> None:
        inst = _build_transformer(self.tmp)
        out = inst.compute_quadrant_scores(torch.zeros(4))
        self.assertEqual(list(out.keys()), list(CANONICAL_QUADRANT_ORDER))
        for value in out.values():
            self.assertIsInstance(value, float)
            self.assertTrue(math.isfinite(value))

    def test_compute_quadrant_scores_values(self) -> None:
        # at h = e_0, scores = dot with each unit-norm quadrant vector =
        # the quadrant vector's first component (±1/sqrt(2))
        inst = _build_transformer(self.tmp)
        out = inst.compute_quadrant_scores(torch.tensor([1.0, 0.0, 0.0, 0.0]))
        sqrt2 = math.sqrt(2.0)
        self.assertAlmostEqual(out["left_lib"],   -1 / sqrt2, places=6)
        self.assertAlmostEqual(out["left_auth"],  -1 / sqrt2, places=6)
        self.assertAlmostEqual(out["right_lib"],   1 / sqrt2, places=6)
        self.assertAlmostEqual(out["right_auth"],  1 / sqrt2, places=6)

    def test_compute_bias_magnitude(self) -> None:
        inst = _build_transformer(self.tmp)
        self.assertAlmostEqual(inst.compute_bias_magnitude(3.0, 4.0), 5.0, places=12)
        self.assertEqual(inst.compute_bias_magnitude(0, 0), 0.0)

    def test_compute_bias_magnitude_rejects_bool(self) -> None:
        inst = _build_transformer(self.tmp)
        with self.assertRaisesRegex(ValueError, "bool"):
            inst.compute_bias_magnitude(True, 0.0)
        with self.assertRaisesRegex(ValueError, "bool"):
            inst.compute_bias_magnitude(0.0, False)

    def test_compute_bias_magnitude_rejects_nan_inf_str(self) -> None:
        inst = _build_transformer(self.tmp)
        with self.assertRaises(ValueError):
            inst.compute_bias_magnitude(float("nan"), 0.0)
        with self.assertRaises(ValueError):
            inst.compute_bias_magnitude(0.0, float("inf"))
        with self.assertRaises(ValueError):
            inst.compute_bias_magnitude("0.5", 0.0)  # type: ignore[arg-type]


class TransformContractTests(_BaseTempDirTest):

    def _build(self) -> Any:
        return _build_transformer(
            self.tmp,
            layer20_value=torch.tensor([1.0, 0.0, 0.0, 0.0]),
        )

    def test_returns_promptstate(self) -> None:
        inst = self._build()
        state = inst.transform("a prompt")
        self.assertIsInstance(state, PromptState)
        self.assertEqual(state.prompt_text, "a prompt")

    def test_hidden_representation_contract(self) -> None:
        inst = self._build()
        state = inst.transform("hi")
        h = state.hidden_representation
        self.assertIsInstance(h, torch.Tensor)
        self.assertEqual(h.dim(), 1)
        self.assertEqual(h.shape[0], 4)
        self.assertEqual(h.dtype, torch.float32)
        self.assertTrue(torch.isfinite(h).all().item())

    def test_axis_and_bias_finite_floats(self) -> None:
        inst = self._build()
        state = inst.transform("hi")
        for value in (state.economic_score, state.social_score, state.bias_magnitude):
            self.assertIsInstance(value, float)
            self.assertTrue(math.isfinite(value))
        self.assertGreaterEqual(state.bias_magnitude, 0.0)

    def test_quadrant_scores_keys_canonical_order(self) -> None:
        inst = self._build()
        state = inst.transform("hi")
        self.assertEqual(list(state.quadrant_scores.keys()), list(CANONICAL_QUADRANT_ORDER))

    def test_metadata_fields(self) -> None:
        inst = self._build()
        state = inst.transform("hi")
        self.assertEqual(state.metadata["encoding_layer"], 20)
        self.assertEqual(state.metadata["pooling_method"], "mean")
        self.assertEqual(state.metadata["vector_method"], "logistic_regression")
        self.assertTrue(state.metadata["use_final_aggregated_vectors"])
        self.assertFalse(state.metadata["use_centering"])

    def test_router_accepts_promptstate(self) -> None:
        inst = self._build()
        state = inst.transform("hi")
        # would raise if quadrant_scores keys / bias_magnitude were malformed
        Router(RouterConfig())._validate_prompt_state(state)


class CenteringTests(_BaseTempDirTest):

    def test_no_centering_returns_input_unchanged(self) -> None:
        inst = _build_transformer(self.tmp, use_centering=False)
        h = torch.tensor([0.0, 0.0, 1.0, 0.0])
        out = inst.maybe_center_representation(h)
        self.assertEqual(out.dtype, torch.float32)
        self.assertEqual(out.dim(), 1)
        self.assertEqual(out.shape[0], 4)
        self.assertTrue(torch.allclose(out, h))

    def test_centering_subtracts_and_renormalizes(self) -> None:
        # encoded prompt = e_0; neutral = 0.5 * e_0; centered = 0.5 * e_0;
        # after L2-normalize -> e_0 again with unit norm
        inst = _build_transformer(
            self.tmp,
            use_centering=True,
            neutral_vector=torch.tensor([0.5, 0.0, 0.0, 0.0]),
            layer20_value=torch.tensor([1.0, 0.0, 0.0, 0.0]),
        )
        encoded = inst.encode_prompt("hi")
        centered = inst.maybe_center_representation(encoded)
        self.assertAlmostEqual(
            float(torch.linalg.vector_norm(centered).item()), 1.0, places=6,
        )
        self.assertTrue(torch.allclose(
            centered, torch.tensor([1.0, 0.0, 0.0, 0.0]), atol=1e-6,
        ))

    def test_centering_without_reference_raises(self) -> None:
        # use_centering=True with neutral_reference_path=None must surface at
        # construction time (load_steering_vectors -> _maybe_load_neutral_reference)
        econ_path = self.tmp / "economic_vectors.pt"
        soc_path = self.tmp / "social_vectors.pt"
        _write_vector_artifact(econ_path, final_vector=_basis_econ())
        _write_vector_artifact(soc_path, final_vector=_basis_social())
        config = SteeringVectorConfig(
            economic_vector_path=econ_path,
            social_vector_path=soc_path,
            use_centering=True,
            neutral_reference_path=None,
        )
        with self.assertRaisesRegex(ValueError, "neutral_reference"):
            InputTransformer(_FakeModel(hidden_dim=4), _FakeTokenizer(), config)


# === MAIN ===

if __name__ == "__main__":
    unittest.main()
