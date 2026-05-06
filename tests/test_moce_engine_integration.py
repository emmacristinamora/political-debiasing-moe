# tests/test_moce_engine_integration.py


# === IMPORTS ===

import importlib.util
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

MoCEEngine = moce_components.MoCEEngine
SteeringVectorConfig = moce_components.SteeringVectorConfig
RouterConfig = moce_components.RouterConfig
EditorConfig = moce_components.EditorConfig
ExpertConfig = moce_components.ExpertConfig
GenerationConfig = moce_components.GenerationConfig


# === FAKES ===

class _FakeOutput:
    """Plain attribute carrier; mimics the .hidden_states surface only."""


class _FakeModel:
    """
    Returns 22 hidden_states tensors so layer index 20 is reachable.
    Layer 20 is filled with a fixed per-token vector so encode_prompt's
    mean pooling produces a deterministic representation.
    """

    def __init__(self, hidden_dim: int = 4) -> None:
        self.hidden_dim = hidden_dim
        self.eval_called = False

    def eval(self) -> None:
        self.eval_called = True

    def parameters(self):
        return iter([torch.zeros(1)])

    def __call__(self, *, input_ids, attention_mask, output_hidden_states):
        assert output_hidden_states is True
        seq_len = int(input_ids.shape[1])
        hidden_states = [
            torch.full((1, seq_len, self.hidden_dim), float(i + 1))
            for i in range(22)
        ]
        layer20 = torch.zeros((1, seq_len, self.hidden_dim))
        layer20[..., 0] = 1.0
        hidden_states[20] = layer20
        out = _FakeOutput()
        out.hidden_states = hidden_states
        return out


class _FakeTokenizer:
    def __call__(self, text, *, return_tensors, truncation, max_length):
        n = max(1, min(len(text or " "), max_length))
        ids = list(range(n))
        return {
            "input_ids": torch.tensor([ids], dtype=torch.long),
            "attention_mask": torch.ones((1, n), dtype=torch.long),
        }


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


# === TESTS ===

class MoCEEngineRuntimeBoundaryTests(unittest.TestCase):

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

        hidden_dim = 4
        econ_path = self.tmp / "economic_vectors.pt"
        soc_path = self.tmp / "social_vectors.pt"
        _write_vector_artifact(
            econ_path,
            final_vector=torch.tensor([1.0] + [0.0] * (hidden_dim - 1)),
        )
        _write_vector_artifact(
            soc_path,
            final_vector=torch.tensor([0.0, 1.0] + [0.0] * (hidden_dim - 2)),
        )

        self.steering_config = SteeringVectorConfig(
            economic_vector_path=econ_path,
            social_vector_path=soc_path,
            vector_method="logistic_regression",
            use_final_aggregated_vectors=True,
            selected_layers=[8, 12, 16, 20, 24],
            use_centering=False,
        )
        self.router_config = RouterConfig()
        self.editor_config = EditorConfig()
        self.expert_config = ExpertConfig(
            left_lib_checkpoint=Path("dummy_left_lib"),
            left_auth_checkpoint=Path("dummy_left_auth"),
            right_lib_checkpoint=Path("dummy_right_lib"),
            right_auth_checkpoint=Path("dummy_right_auth"),
        )
        self.generation_config = GenerationConfig()

        self.fake_model = _FakeModel(hidden_dim=hidden_dim)
        self.fake_tokenizer = _FakeTokenizer()

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _build_engine(self) -> Any:
        return MoCEEngine(
            model=self.fake_model,
            tokenizer=self.fake_tokenizer,
            steering_config=self.steering_config,
            router_config=self.router_config,
            expert_config=self.expert_config,
            editor_config=self.editor_config,
            generation_config=self.generation_config,
        )

    def test_engine_runs_until_expert_boundary(self) -> None:
        # ExpertManager.__init__ is also still NotImplementedError, so the
        # engine constructor itself raises before run() is reachable. Either
        # way the failure must be NotImplementedError pointing at the expert
        # boundary or the decode boundary -- never a silent ValueError.
        try:
            engine = self._build_engine()
        except NotImplementedError as exc:
            message = str(exc)
            # MoCEEngine.__init__ wires ExpertManager(...); ExpertManager
            # construction is the first NotImplementedError on this path.
            # Accept any of the documented unimplemented surfaces below.
            self.assertTrue(
                any(token in message for token in ("ExpertManager", "run_all_experts", "decode"))
                or message == "",
                f"unexpected NotImplementedError message: {message!r}",
            )
            return

        # If construction somehow succeeded (future state where ExpertManager
        # ships before this test is updated), engine.run must still fail at
        # ExpertManager.run_all_experts or the decode boundary.
        with self.assertRaises(NotImplementedError) as ctx:
            engine.run("test prompt")
        message = str(ctx.exception)
        self.assertTrue(
            any(token in message for token in ("ExpertManager", "run_all_experts", "decode"))
            or message == "",
            f"unexpected NotImplementedError message: {message!r}",
        )

    def test_router_route_is_reached_when_engine_runs(self) -> None:
        # Both ExpertManager.__init__ and Editor.__init__ are still stubs
        # raising NotImplementedError on this branch, so MoCEEngine.__init__
        # cannot complete unmodified. To exercise engine.run far enough to
        # confirm InputTransformer and Router were reached, swap both classes
        # for tiny no-op-construction stubs and make ExpertManager.run_all_experts
        # raise NotImplementedError at the expected boundary.
        original_expert_manager = moce_components.ExpertManager
        original_editor = moce_components.Editor
        original_route = moce_components.Router.route

        class _StubExpertManager:
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                pass

            def run_all_experts(self, prompt_text: str, prompt_state: Any) -> Any:
                raise NotImplementedError(
                    "stub ExpertManager.run_all_experts: expert boundary"
                )

        class _StubEditor:
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                pass

        moce_components.ExpertManager = _StubExpertManager
        moce_components.Editor = _StubEditor

        route_called = {"value": False}

        def _spy_route(self: Any, prompt_state: Any) -> Any:
            route_called["value"] = True
            return original_route(self, prompt_state)

        moce_components.Router.route = _spy_route

        try:
            engine = self._build_engine()
            with self.assertRaises(NotImplementedError) as ctx:
                engine.run("test prompt")
            self.assertTrue(route_called["value"], "Router.route was not reached")
            message = str(ctx.exception)
            self.assertTrue(
                any(
                    token in message
                    for token in ("ExpertManager", "run_all_experts", "expert", "decode")
                ),
                f"unexpected NotImplementedError message: {message!r}",
            )
        finally:
            moce_components.Router.route = original_route
            moce_components.ExpertManager = original_expert_manager
            moce_components.Editor = original_editor


# === MAIN ===

if __name__ == "__main__":
    unittest.main()
