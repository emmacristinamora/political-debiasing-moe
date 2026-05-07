# src/router_forced_policy_runner.py


# === IMPORTS ===

from __future__ import annotations

import copy
import importlib.util
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# router_calibration_config carries CANONICAL_QUADRANT_ORDER and does not
# require torch — safe to import at module load time.
_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from router_calibration_config import CANONICAL_QUADRANT_ORDER  # noqa: E402


# === LAZY LOAD: RouterState ===

# 09_moce_components.py imports torch at module top, which may be unavailable
# in lightweight test environments. We resolve the real RouterState class
# lazily so this module remains importable without torch — production runs
# import torch and get the real class (the real Editor isinstance-checks it);
# fake-engine tests get a duck-typed shim that the fake editor accepts.

_REAL_ROUTER_STATE_CLASS: Any | None = None
_router_state_load_attempted: bool = False


def _try_load_real_router_state_class() -> Any | None:
    """
    Attempt to load the real RouterState class from src/09_moce_components.py.
    Returns None if torch (or any other dependency of that module) is missing.
    The result is cached in module-level state.
    """
    global _REAL_ROUTER_STATE_CLASS, _router_state_load_attempted
    if _router_state_load_attempted:
        return _REAL_ROUTER_STATE_CLASS

    try:
        components_path = _SRC_DIR / "09_moce_components.py"
        spec = importlib.util.spec_from_file_location(
            "moce_components_for_forced_runner", components_path,
        )
        if spec is None or spec.loader is None:
            _REAL_ROUTER_STATE_CLASS = None
        else:
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            _REAL_ROUTER_STATE_CLASS = module.RouterState
    except Exception:
        # any import-time failure (typically torch missing) → use shim
        _REAL_ROUTER_STATE_CLASS = None
    finally:
        _router_state_load_attempted = True

    return _REAL_ROUTER_STATE_CLASS


@dataclass
class _ForcedRouterStateShim:
    """
    Shim used when the real RouterState cannot be imported. Mirrors the real
    RouterState's field set so duck-typed consumers (fake editors in tests)
    keep working. The real Editor explicitly isinstance-checks RouterState
    and would reject this shim — that path always uses the real class via
    _try_load_real_router_state_class().
    """
    heuristic_prior: dict[str, float]
    calibrated_policy: dict[str, float]
    diagnostics: dict[str, Any] = field(default_factory=dict)
    losses: dict[str, Any]   = field(default_factory=dict)


def _build_router_state(
    *,
    heuristic_prior: dict[str, float],
    calibrated_policy: dict[str, float],
    diagnostics: dict[str, Any],
    losses: dict[str, Any],
) -> Any:
    real_cls = _try_load_real_router_state_class()
    if real_cls is not None:
        return real_cls(
            heuristic_prior=heuristic_prior,
            calibrated_policy=calibrated_policy,
            diagnostics=diagnostics,
            losses=losses,
        )
    return _ForcedRouterStateShim(
        heuristic_prior=heuristic_prior,
        calibrated_policy=calibrated_policy,
        diagnostics=diagnostics,
        losses=losses,
    )


# === CONSTANTS ===

POLICY_SUM_TOLERANCE: float = 1e-6

# probed in priority order — first one that exists on engine wins
DECODE_FALLBACKS: tuple[str, ...] = (
    "decode_editor_result",
    "decode_final_text",
    "_decode_editor_result",
)


# === RESULT DATACLASS ===

@dataclass
class ForcedPolicyRunResult:
    """
    Complete result of a single forced-policy MoCE run.

    Carries both the final decoded text (the candidate-trace payload) and
    the live runtime objects (prompt_state, router_state, editor_result) so
    the caller can extract any extra fields it needs without re-running.
    Heavy fields (hidden tensors etc.) are deliberately not pulled out — see
    serialize_forced_policy_result for the JSON-safe trace shape.
    """
    example_id: str
    prompt_text: str
    forced_policy: dict[str, float]
    heuristic_prior: dict[str, float]
    final_text: str
    prompt_state: Any
    router_state: Any
    editor_result: Any | None
    metadata: dict[str, Any] = field(default_factory=dict)


# === VALIDATION HELPERS ===

def _is_finite_number(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return math.isfinite(float(value))


def _validate_policy(policy: Any, where: str) -> dict[str, float]:
    """
    Validate a policy dict and return a fresh copy in canonical key order.

    Required: keys exactly CANONICAL_QUADRANT_ORDER, every value a finite
    number (bool rejected), strictly positive, sum to 1 within
    POLICY_SUM_TOLERANCE.
    """
    if not isinstance(policy, dict):
        raise ValueError(
            f"{where}: policy must be a dict, got {type(policy).__name__}"
        )
    expected = set(CANONICAL_QUADRANT_ORDER)
    actual = set(policy.keys())
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(
            f"{where}: keys must equal canonical "
            f"{list(CANONICAL_QUADRANT_ORDER)}; missing={missing} extra={extra}"
        )

    out: dict[str, float] = {}
    total = 0.0
    for key in CANONICAL_QUADRANT_ORDER:
        v = policy[key]
        if not _is_finite_number(v):
            raise ValueError(
                f"{where}[{key!r}] = {v!r} must be a finite number"
            )
        v = float(v)
        if v <= 0:
            raise ValueError(
                f"{where}[{key!r}] = {v} must be strictly > 0"
            )
        out[key] = v
        total += v

    if abs(total - 1.0) > POLICY_SUM_TOLERANCE:
        raise ValueError(
            f"{where}: sum = {total} (must be 1 within {POLICY_SUM_TOLERANCE})"
        )
    return out


def _validate_engine_components(engine: Any) -> None:
    """Verify the engine exposes every component the runner consults."""
    if engine is None:
        raise ValueError("engine must not be None")

    for attr in ("input_transformer", "router", "expert_manager", "editor"):
        if not hasattr(engine, attr):
            raise ValueError(f"engine is missing required component: {attr!r}")

    if not callable(getattr(engine.input_transformer, "transform", None)):
        raise ValueError(
            "engine.input_transformer must expose a callable .transform(prompt_text)"
        )
    if not callable(getattr(engine.router, "build_heuristic_prior", None)):
        raise ValueError(
            "engine.router must expose a callable .build_heuristic_prior(prompt_state)"
        )
    if not callable(getattr(engine.expert_manager, "run_all_experts", None)):
        raise ValueError(
            "engine.expert_manager must expose a callable "
            ".run_all_experts(prompt_text, prompt_state)"
        )
    if not callable(getattr(engine.editor, "run_editing_loop", None)):
        raise ValueError(
            "engine.editor must expose a callable "
            ".run_editing_loop(prompt_text, prompt_state, router_state, expert_outputs)"
        )


def _resolve_decode_callable(engine: Any) -> tuple[Any, str]:
    """
    Walk DECODE_FALLBACKS and return the first (callable, attr_name) pair
    found on engine. Returning the attr name (not the bound method's
    __name__, which can be an inner helper like '_decode_impl' in tests)
    lets the runner report which decode boundary was actually consulted.
    """
    for name in DECODE_FALLBACKS:
        candidate = getattr(engine, name, None)
        if callable(candidate):
            return candidate, name
    raise ValueError(
        "engine is missing decode boundary; expected one of "
        f"{list(DECODE_FALLBACKS)}"
    )


# === RUNNER ===

class ForcedPolicyMoCERunner:
    """
    Run MoCE with a router policy injected from the outside.

    Replicates the engine's normal pipeline except for routing: the
    candidate policy is wired straight into RouterState.calibrated_policy
    so the Editor initializes its mixture from it (in router_policy mode).
    Router.route() is never called — the heuristic prior is either provided
    by the caller or computed via engine.router.build_heuristic_prior on
    the prompt_state. This keeps the normal MoCE runtime untouched: the
    runner is a sibling code path, never invoked by engine.run.

    Decode boundary resolution: at construction time the runner picks the
    first available method on engine from DECODE_FALLBACKS and caches it.
    Production engines typically expose one; the fallback list lets tests
    and mocks expose any of the three names without ambiguity.
    """

    def __init__(self, engine: Any) -> None:
        _validate_engine_components(engine)
        self.engine = engine
        self._decode_call, self._decode_name = _resolve_decode_callable(engine)

    def run(
        self,
        *,
        example_id: str,
        prompt_text: str,
        candidate_policy: dict[str, float],
        heuristic_prior: dict[str, float] | None = None,
    ) -> ForcedPolicyRunResult:
        # 1) string inputs — non-empty (and stripped check, since whitespace-only
        # is meaningless as an id or prompt)
        if not isinstance(example_id, str) or not example_id.strip():
            raise ValueError("example_id must be a non-empty string")
        if not isinstance(prompt_text, str) or not prompt_text.strip():
            raise ValueError("prompt_text must be a non-empty string")

        # 2) candidate policy — validated copy decouples downstream state from caller mutations
        forced_policy = _validate_policy(candidate_policy, "candidate_policy")

        # 3) prompt state via engine.input_transformer
        prompt_state = self.engine.input_transformer.transform(prompt_text)

        # 4) heuristic prior — caller-supplied wins; otherwise compute from prompt_state
        if heuristic_prior is None:
            computed = self.engine.router.build_heuristic_prior(prompt_state)
            prior = _validate_policy(
                computed, "engine.router.build_heuristic_prior output",
            )
        else:
            prior = _validate_policy(heuristic_prior, "heuristic_prior")

        # 5) construct RouterState (real class in production, shim in tests)
        diagnostics: dict[str, Any] = {
            "forced_policy": True,
            "heuristic_prior":   copy.deepcopy(prior),
            "calibrated_policy": copy.deepcopy(forced_policy),
        }
        router_state = _build_router_state(
            heuristic_prior=copy.deepcopy(prior),
            calibrated_policy=copy.deepcopy(forced_policy),
            diagnostics=diagnostics,
            losses={},
        )

        # 6) experts
        expert_outputs = self.engine.expert_manager.run_all_experts(
            prompt_text, prompt_state,
        )
        if not isinstance(expert_outputs, dict) or not expert_outputs:
            raise ValueError(
                "engine.expert_manager.run_all_experts must return a non-empty dict"
            )
        expected_keys = set(CANONICAL_QUADRANT_ORDER)
        actual_keys = set(expert_outputs.keys())
        if actual_keys != expected_keys:
            missing = sorted(expected_keys - actual_keys)
            extra = sorted(actual_keys - expected_keys)
            raise ValueError(
                "engine.expert_manager.run_all_experts must return keys exactly "
                f"{list(CANONICAL_QUADRANT_ORDER)}; missing={missing} extra={extra}"
            )

        # 7) editor
        editor_result = self.engine.editor.run_editing_loop(
            prompt_text, prompt_state, router_state, expert_outputs,
        )

        # 8) decode
        final_text = self._decode_call(
            prompt_text, prompt_state, router_state, expert_outputs, editor_result,
        )
        if not isinstance(final_text, str) or not final_text:
            raise ValueError(
                "decode boundary did not return a non-empty string final_text; "
                f"got {type(final_text).__name__} {final_text!r}"
            )

        metadata: dict[str, Any] = {
            "decode_callable":     self._decode_name,
            "router_state_class":  type(router_state).__name__,
        }

        return ForcedPolicyRunResult(
            example_id=example_id,
            prompt_text=prompt_text,
            forced_policy=copy.deepcopy(forced_policy),
            heuristic_prior=copy.deepcopy(prior),
            final_text=final_text,
            prompt_state=prompt_state,
            router_state=router_state,
            editor_result=editor_result,
            metadata=metadata,
        )


# === SERIALIZATION ===

def _safe_get_attr(obj: Any, name: str) -> Any:
    """Return getattr(obj, name) or None; never raise."""
    try:
        return getattr(obj, name, None)
    except Exception:
        return None


def serialize_forced_policy_result(
    result: ForcedPolicyRunResult,
) -> dict[str, Any]:
    """
    Produce a JSON-safe trace dict for downstream candidate-trace files.

    Excluded by design: prompt_state, router_state object, editor_result
    object, hidden tensors, anything else that would not survive json.dumps.
    Only lightweight diagnostic fields are surfaced under metadata:

    - router_diagnostics       (deep-copied dict)
    - editor_final_alpha       (deep-copied dict)
    - editor_final_alignment   (deep-copied dict)
    - editor_num_steps_run     (int)
    - editor_stopped_early     (bool)
    - editor_stop_reason       (str or omitted if None)
    """
    if not isinstance(result, ForcedPolicyRunResult):
        raise ValueError(
            f"result must be ForcedPolicyRunResult, got {type(result).__name__}"
        )

    metadata: dict[str, Any] = dict(result.metadata)

    diagnostics = _safe_get_attr(result.router_state, "diagnostics")
    if isinstance(diagnostics, dict):
        metadata["router_diagnostics"] = copy.deepcopy(diagnostics)

    er = result.editor_result
    for src_attr, out_key in (
        ("final_alpha",     "editor_final_alpha"),
        ("final_alignment", "editor_final_alignment"),
        ("num_steps_run",   "editor_num_steps_run"),
        ("stopped_early",   "editor_stopped_early"),
        ("stop_reason",     "editor_stop_reason"),
    ):
        value = _safe_get_attr(er, src_attr)
        if value is None:
            continue
        if isinstance(value, dict):
            metadata[out_key] = copy.deepcopy(value)
        else:
            metadata[out_key] = value

    return {
        "example_id":      result.example_id,
        "prompt_text":     result.prompt_text,
        "forced_policy":   copy.deepcopy(result.forced_policy),
        "heuristic_prior": copy.deepcopy(result.heuristic_prior),
        "final_text":      result.final_text,
        "metadata":        metadata,
    }
