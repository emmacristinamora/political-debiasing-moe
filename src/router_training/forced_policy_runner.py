# src/router_forced_policy_runner.py


# === IMPORTS ===

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import logging
import math
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


# router_training.config carries CANONICAL_QUADRANT_ORDER and does not
# require torch — safe to import at module load time.
_SRC_DIR = Path(__file__).resolve().parents[1]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from router_training.config import CANONICAL_QUADRANT_ORDER  # noqa: E402


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

# CLI driver constants
PROJECT_ROOT = Path(__file__).resolve().parents[2]

FALLBACK_FEATURES_PATH: Path = Path("data/router/features.jsonl")
FALLBACK_HIDDEN_PATH:   Path = Path("data/router/hidden.pt")
FALLBACK_OUTPUT_PATH:   Path = Path("data/router/candidate_traces.jsonl")
FALLBACK_REPORT_PATH:   Path = Path("data/router/reports/forced_policy_run_report.json")

REQUIRED_FEATURE_KEYS: tuple[str, ...] = (
    "example_id", "prompt_text", "quadrant_scores", "hidden_representation_ref",
)
LOG_PROGRESS_EVERY: int = 50

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO,
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


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


# === PIPELINE MAIN ===
# Step 3 of the router-calibration pipeline:
#   python src/router_training/forced_policy_runner.py --config config/config.yaml
#
# For each prompt in features.jsonl, builds ~28 candidate policies and generates
# one text trace per (prompt, policy) using the LoRA expert with the highest
# weight in that policy (argmax routing, v1). Output is candidate_traces.jsonl
# in the format consumed by scorer.py.
#
# Runtime on H200 (Mistral-7B, 256 tokens, greedy, batch=1):
#   ~1.5 s/generation → 28 candidates/prompt → ~42 s/prompt
#   full 33 850 prompts ≈ 394 h  (use --limit for a usable subset)
#   --limit 500  ≈  6 h   (recommended for initial router training)
#   --limit 100  ≈  1.2 h (smoke test)

if __name__ == "__main__":

    import argparse as _argparse
    import json as _json
    import logging as _logging
    import math as _math
    import sys as _sys
    from pathlib import Path as _Path

    import numpy as _np
    import torch as _torch
    from peft import PeftModel as _PeftModel
    from transformers import AutoModelForCausalLM as _AutoModelForCausalLM
    from transformers import AutoTokenizer as _AutoTokenizer

    _logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(message)s",
        level=_logging.INFO,
        datefmt="%H:%M:%S",
    )
    _log = _logging.getLogger("forced_policy_runner")

    _src_dir = _Path(__file__).resolve().parents[1]
    if str(_src_dir) not in _sys.path:
        _sys.path.insert(0, str(_src_dir))
    from router_training.config import load_router_calibration_config as _load_cfg  # noqa: E402

    # ── quadrant topology ─────────────────────────────────────────────────────
    _ORDER: tuple[str, ...] = ("left_lib", "left_auth", "right_lib", "right_auth")

    _OPPOSITE: dict[str, str] = {
        "left_lib":   "right_auth",
        "left_auth":  "right_lib",
        "right_lib":  "left_auth",
        "right_auth": "left_lib",
    }
    _ADJACENT: dict[str, tuple[str, str]] = {
        "left_lib":   ("left_auth",  "right_lib"),
        "left_auth":  ("left_lib",   "right_auth"),
        "right_lib":  ("left_lib",   "right_auth"),
        "right_auth": ("right_lib",  "left_auth"),
    }

    # ── policy helpers ────────────────────────────────────────────────────────

    def _softmax_list(logits: list[float]) -> list[float]:
        m = max(logits)
        exps = [_math.exp(x - m) for x in logits]
        s = sum(exps)
        return [e / s for e in exps]

    def _normalize_dict(d: dict) -> dict:
        s = sum(d.values())
        return {k: v / s for k, v in d.items()}

    def _clip_min(d: dict, min_p: float) -> dict:
        return _normalize_dict({k: max(v, min_p) for k, v in d.items()})

    def _heuristic_prior(quadrant_scores: dict) -> dict:
        """pi_0 = softmax(-q) — weights experts that oppose the prompt bias."""
        logits = [-quadrant_scores[k] for k in _ORDER]
        probs = _softmax_list(logits)
        return {k: p for k, p in zip(_ORDER, probs)}

    def _temperature_resample(prior: dict, temperature: float) -> dict:
        """Sharpen (T<1) or soften (T>1) a distribution via log rescaling."""
        log_probs = [_math.log(max(prior[k], 1e-12)) for k in _ORDER]
        probs = _softmax_list([lp / temperature for lp in log_probs])
        return {k: p for k, p in zip(_ORDER, probs)}

    def _opposite_heavy(focal_quadrant: str, heavy: float = 0.70) -> dict:
        """Put `heavy` weight on opposite(focal_quadrant), share the rest."""
        opp = _OPPOSITE[focal_quadrant]
        rest = (1.0 - heavy) / (len(_ORDER) - 1)
        return {k: (heavy if k == opp else rest) for k in _ORDER}

    def _adjacent_heavy(focal_quadrant: str, each_adj: float = 0.40) -> dict:
        """Put `each_adj` on each of the two adjacent quadrants, share remainder."""
        adj = _ADJACENT[focal_quadrant]
        other = (1.0 - 2 * each_adj) / 2
        return {k: (each_adj if k in adj else other) for k in _ORDER}

    def _build_candidate_policies(
        prior: dict,
        cfg_cand: Any,
        rng: "_np.random.Generator",
    ) -> list[dict]:
        policies: list[dict] = []
        min_p = float(cfg_cand.min_probability)

        if cfg_cand.include_heuristic_prior:
            policies.append(_clip_min(prior, min_p))

        if cfg_cand.include_uniform:
            policies.append({k: 1.0 / len(_ORDER) for k in _ORDER})

        for temp in cfg_cand.sharpen_temperatures:
            policies.append(_clip_min(_temperature_resample(prior, float(temp)), min_p))

        for temp in cfg_cand.soften_temperatures:
            policies.append(_clip_min(_temperature_resample(prior, float(temp)), min_p))

        if cfg_cand.include_opposite_heavy:
            for q in _ORDER:
                policies.append(_clip_min(_opposite_heavy(q), min_p))

        if cfg_cand.include_adjacent_heavy:
            for q in _ORDER:
                policies.append(_clip_min(_adjacent_heavy(q), min_p))

        samples = rng.dirichlet(
            [float(cfg_cand.dirichlet_concentration)] * len(_ORDER),
            size=cfg_cand.dirichlet_samples,
        )
        for row in samples:
            policies.append(_clip_min({k: float(v) for k, v in zip(_ORDER, row)}, min_p))

        return policies

    def _argmax_adapter(policy: dict) -> str:
        return max(policy, key=policy.__getitem__)

    # ── model loading ─────────────────────────────────────────────────────────

    def _load_model_and_adapters(
        base_model: str,
        checkpoints: dict,
        device: str,
        dtype: "_torch.dtype",
    ) -> tuple:
        _log.info("loading tokenizer: %s", base_model)
        tok = _AutoTokenizer.from_pretrained(base_model)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        tok.padding_side = "right"

        _log.info("loading base model  dtype=%s  device=%s", dtype, device)
        model = _AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=dtype)
        model = model.to(device)
        model.eval()

        names = list(checkpoints.keys())
        _log.info("loading adapter '%s' from %s", names[0], checkpoints[names[0]])
        model = _PeftModel.from_pretrained(
            model, str(checkpoints[names[0]]), adapter_name=names[0],
        )
        for name in names[1:]:
            _log.info("loading adapter '%s' from %s", name, checkpoints[name])
            model.load_adapter(str(checkpoints[name]), adapter_name=name)
        model.eval()
        _log.info("all %d adapters loaded", len(names))
        return model, tok

    # ── generation ────────────────────────────────────────────────────────────

    def _generate(
        model: Any,
        tok: Any,
        prompt_text: str,
        adapter_name: str,
        gen_cfg: Any,
        device: str,
    ) -> str:
        model.set_adapter(adapter_name)
        inputs = tok(
            prompt_text, return_tensors="pt", truncation=True, max_length=512,
        ).to(device)
        prompt_len = inputs["input_ids"].shape[1]
        gen_kwargs: dict = {
            "max_new_tokens": gen_cfg.max_new_tokens,
            "do_sample":      gen_cfg.do_sample,
            "pad_token_id":   tok.eos_token_id,
        }
        if gen_cfg.do_sample:
            gen_kwargs["temperature"] = gen_cfg.temperature
            gen_kwargs["top_p"]       = gen_cfg.top_p
        with _torch.inference_mode():
            out_ids = model.generate(**inputs, **gen_kwargs)
        return tok.decode(out_ids[0, prompt_len:], skip_special_tokens=True).strip()

    # ── argument parsing ──────────────────────────────────────────────────────

    def _parse_args() -> "_argparse.Namespace":
        p = _argparse.ArgumentParser(
            description=(
                "Step 3: run forced-policy MoCE per (prompt, candidate policy) "
                "and write candidate_traces.jsonl."
            )
        )
        p.add_argument("--config", type=_Path, required=True,
                       help="path to config.yaml")
        p.add_argument("--limit", type=int, default=None,
                       help="process at most this many prompts after selection (smoke test)")
        p.add_argument("--stratify", type=int, default=None,
                       help="take this many prompts per quadrant (recommended over --limit)")
        p.add_argument("--device", type=str, default=None,
                       help="override config device (cuda / cpu)")
        p.add_argument("--adapter-left-lib",   type=_Path, default=None,
                       help="override left_lib checkpoint path")
        p.add_argument("--adapter-left-auth",  type=_Path, default=None,
                       help="override left_auth checkpoint path")
        p.add_argument("--adapter-right-lib",  type=_Path, default=None,
                       help="override right_lib checkpoint path")
        p.add_argument("--adapter-right-auth", type=_Path, default=None,
                       help="override right_auth checkpoint path")
        return p.parse_args()

    # ── main ──────────────────────────────────────────────────────────────────

    def _main() -> None:
        args = _parse_args()
        cfg  = _load_cfg(args.config)

        device = args.device or cfg.model.device
        dtype  = (
            _torch.bfloat16
            if str(getattr(cfg.model, "dtype", "bfloat16")) == "bfloat16"
            else _torch.float32
        )

        ckpt = cfg.paths.expert_checkpoints
        checkpoints: dict[str, _Path] = {
            "left_lib":   args.adapter_left_lib   or _Path(ckpt.left_lib_checkpoint),
            "left_auth":  args.adapter_left_auth   or _Path(ckpt.left_auth_checkpoint),
            "right_lib":  args.adapter_right_lib   or _Path(ckpt.right_lib_checkpoint),
            "right_auth": args.adapter_right_auth  or _Path(ckpt.right_auth_checkpoint),
        }

        features_path = _Path(cfg.paths.features_path)
        output_path   = _Path(cfg.paths.candidate_traces_path)

        # ── pre-flight ───────────────────────────────────────────────────────
        if not features_path.is_file():
            _log.error("features.jsonl not found: %s — run features.py first", features_path)
            _sys.exit(1)
        for name, ckpt_path in checkpoints.items():
            if not (_Path(ckpt_path) / "adapter_config.json").is_file():
                _log.error(
                    "adapter '%s' missing adapter_config.json at %s", name, ckpt_path
                )
                _sys.exit(1)

        output_path.parent.mkdir(parents=True, exist_ok=True)

        # ── load features ────────────────────────────────────────────────────
        _log.info("loading features from %s", features_path)
        with features_path.open() as fh:
            features = [_json.loads(ln) for ln in fh if ln.strip()]
        _log.info("loaded %d prompt records total", len(features))

        if args.stratify is not None:
            # stratified sample: take up to N records per quadrant (by economic+social sign)
            from collections import defaultdict as _defaultdict
            import random as _random
            _random.seed(int(cfg.candidate_policies.seed))
            by_q: dict = _defaultdict(list)
            for r in features:
                e, s = r["economic_score"], r["social_score"]
                q = ("right" if e > 0 else "left") + "_" + ("auth" if s > 0 else "lib")
                by_q[q].append(r)
            selected = []
            for q, recs in sorted(by_q.items()):
                _random.shuffle(recs)
                chosen = recs[: args.stratify]
                _log.info("stratify: %s → %d / %d records", q, len(chosen), len(recs))
                selected.extend(chosen)
            _random.shuffle(selected)
            features = selected

        elif args.limit is not None:
            # plain limit without shuffle warns if ordering may be skewed
            _log.warning(
                "--limit without --stratify: features.jsonl is source-sorted, "
                "first %d rows may not cover all quadrants — consider --stratify instead",
                args.limit,
            )
            features = features[: args.limit]

        _log.info("selected %d prompt records for processing", len(features))

        # ── resume: skip already-written example_ids ─────────────────────────
        done_ids: set = set()
        if output_path.is_file():
            with output_path.open() as fh:
                for ln in fh:
                    ln = ln.strip()
                    if ln:
                        try:
                            done_ids.add(_json.loads(ln)["example_id"])
                        except Exception:
                            pass
            if done_ids:
                _log.info("resuming — %d unique example_ids already written", len(done_ids))

        todo = [f for f in features if f["example_id"] not in done_ids]
        _log.info("%d prompts to process (%d already done)", len(todo), len(done_ids))
        if not todo:
            _log.info("nothing to do")
            return

        # ── load model + all 4 adapters ──────────────────────────────────────
        model, tok = _load_model_and_adapters(base_model=cfg.model.base_model,
                                              checkpoints=checkpoints,
                                              device=device, dtype=dtype)

        # ── per-prompt candidate policy RNG (seeded, deterministic) ──────────
        rng = _np.random.default_rng(int(cfg.candidate_policies.seed))

        # ── main generation loop ─────────────────────────────────────────────
        n_traces = 0
        with output_path.open("a") as out_fh:
            for idx, feat in enumerate(todo):
                example_id  = feat["example_id"]
                prompt_text = feat["prompt_text"]
                q_scores    = feat["quadrant_scores"]

                prior    = _heuristic_prior(q_scores)
                policies = _build_candidate_policies(prior, cfg.candidate_policies, rng)

                for policy in policies:
                    adapter    = _argmax_adapter(policy)
                    final_text = _generate(model, tok, prompt_text, adapter,
                                           cfg.generation, device)
                    record = {
                        "example_id":      example_id,
                        "prompt_text":     prompt_text,
                        "forced_policy":   {k: round(policy[k], 8) for k in _ORDER},
                        "heuristic_prior": {k: round(prior[k],   8) for k in _ORDER},
                        "final_text":      final_text,
                        "metadata": {
                            "argmax_adapter": adapter,
                            "source":         feat.get("source", ""),
                        },
                    }
                    out_fh.write(_json.dumps(record, ensure_ascii=False) + "\n")
                    n_traces += 1

                if (idx + 1) % 50 == 0 or (idx + 1) == len(todo):
                    _log.info(
                        "progress  prompt %d/%d  |  %d traces written",
                        idx + 1, len(todo), n_traces,
                    )
                    out_fh.flush()

        _log.info("done — %d traces → %s", n_traces, output_path)

    _main()


# === CLI: ARGUMENT PARSING ===

def parse_args() -> argparse.Namespace:
    """
    Parse CLI args for the forced-policy candidate-trace collector.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Forced-policy MoCE runner: for each prompt in features.jsonl "
            "and each generated candidate policy, force the editor's alpha "
            "to that policy, decode under Resolution 1, and append a row "
            "to candidate_traces.jsonl."
        ),
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--features-path", type=Path, default=None)
    parser.add_argument("--hidden-path",   type=Path, default=None)
    parser.add_argument("--output-path",   type=Path, default=None)
    parser.add_argument("--report-path",   type=Path, default=None)
    parser.add_argument("--device",        type=str,  default=None)
    parser.add_argument("--max-examples",  type=int,  default=None)
    parser.add_argument("--start-index",   type=int,  default=None)
    parser.add_argument("--end-index",     type=int,  default=None)
    parser.add_argument("--dry-run",       action="store_true")
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Truncate the output file at startup instead of resuming.",
    )
    return parser.parse_args()


# === CLI: PATH RESOLUTION ===

def _resolve_path(
    *,
    cli_value: Path | None,
    cfg_value: Any,
    fallback: Path,
) -> Path:
    """CLI override > config field > project-relative fallback."""
    if cli_value is not None:
        return Path(cli_value)
    if cfg_value is not None:
        return Path(cfg_value)
    return PROJECT_ROOT / fallback


def resolve_paths(cfg: Any, args: argparse.Namespace) -> dict[str, Path]:
    """
    Resolve every path the CLI driver needs.

    Precedence per path: CLI override > router_calibration.paths field >
    project-relative fallback. The report path defaults to
    <reports_dir>/forced_policy_run_report.json when reports_dir is set
    on the config; otherwise FALLBACK_REPORT_PATH.
    """
    paths_cfg = getattr(cfg, "paths", None)

    cfg_features = getattr(paths_cfg, "features_path", None) if paths_cfg is not None else None
    cfg_hidden   = getattr(paths_cfg, "hidden_path",   None) if paths_cfg is not None else None
    cfg_output   = getattr(paths_cfg, "candidate_traces_path", None) if paths_cfg is not None else None
    cfg_reports_dir = getattr(paths_cfg, "reports_dir", None) if paths_cfg is not None else None

    if args.report_path is not None:
        report_path = Path(args.report_path)
    elif cfg_reports_dir is not None:
        report_path = Path(cfg_reports_dir) / "forced_policy_run_report.json"
    else:
        report_path = PROJECT_ROOT / FALLBACK_REPORT_PATH

    return {
        "features_path": _resolve_path(
            cli_value=args.features_path, cfg_value=cfg_features,
            fallback=FALLBACK_FEATURES_PATH,
        ),
        "hidden_path": _resolve_path(
            cli_value=args.hidden_path, cfg_value=cfg_hidden,
            fallback=FALLBACK_HIDDEN_PATH,
        ),
        "output_path": _resolve_path(
            cli_value=args.output_path, cfg_value=cfg_output,
            fallback=FALLBACK_OUTPUT_PATH,
        ),
        "report_path": Path(report_path),
    }


# === CLI: FEATURE LOADING ===

def load_feature_records(
    path: Path,
    *,
    start_index: int | None,
    end_index: int | None,
    max_examples: int | None,
) -> list[dict[str, Any]]:
    """
    Load and validate features.jsonl. Slicing is applied in this order:
    [start_index:end_index], then [:max_examples] on the slice. Each row
    is required to carry REQUIRED_FEATURE_KEYS — schema mismatches surface
    here, not deep inside the run loop.
    """
    if not path.is_file():
        raise FileNotFoundError(f"features file not found: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line_index, raw_line in enumerate(fh):
            line = raw_line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as err:
                raise ValueError(
                    f"features.jsonl line {line_index + 1}: invalid JSON ({err})"
                ) from err
            if not isinstance(row, dict):
                raise ValueError(
                    f"features.jsonl line {line_index + 1}: row must be a JSON "
                    f"object, got {type(row).__name__}"
                )
            for key in REQUIRED_FEATURE_KEYS:
                if key not in row:
                    raise ValueError(
                        f"features.jsonl line {line_index + 1}: missing key {key!r}"
                    )
            rows.append(row)

    if start_index is not None or end_index is not None:
        s = 0 if start_index is None else int(start_index)
        e = len(rows) if end_index is None else int(end_index)
        if s < 0 or e < 0 or s > e:
            raise ValueError(
                f"invalid slice: start_index={start_index} end_index={end_index}"
            )
        rows = rows[s:e]
    if max_examples is not None and max_examples >= 0:
        rows = rows[: int(max_examples)]
    return rows


# === CLI: HEURISTIC PRIOR ===

def compute_heuristic_prior(
    quadrant_scores: dict[str, float],
    *,
    beta: float,
    temperature: float,
) -> dict[str, float]:
    """
    softmax(-beta * q / T) over CANONICAL_QUADRANT_ORDER. Mirrors what
    Router.build_heuristic_prior does in the engine, but operates on a
    dict so the runner does not need a PromptState to compute it.
    """
    if not isinstance(quadrant_scores, dict):
        raise ValueError(
            f"quadrant_scores must be a dict, got {type(quadrant_scores).__name__}"
        )
    if set(quadrant_scores.keys()) != set(CANONICAL_QUADRANT_ORDER):
        raise ValueError(
            "quadrant_scores keys must equal canonical "
            f"{list(CANONICAL_QUADRANT_ORDER)}; got {sorted(quadrant_scores.keys())}"
        )
    if not _is_finite_number(beta):
        raise ValueError(f"beta must be a finite number; got {beta!r}")
    if not _is_finite_number(temperature) or float(temperature) == 0.0:
        raise ValueError(
            f"temperature must be a finite non-zero number; got {temperature!r}"
        )
    for k in CANONICAL_QUADRANT_ORDER:
        if not _is_finite_number(quadrant_scores[k]):
            raise ValueError(
                f"quadrant_scores[{k!r}]={quadrant_scores[k]!r} is not finite"
            )

    logits = [
        -float(beta) * float(quadrant_scores[k]) / float(temperature)
        for k in CANONICAL_QUADRANT_ORDER
    ]
    m = max(logits)
    exps = [math.exp(x - m) for x in logits]
    z = sum(exps)
    return {k: e / z for k, e in zip(CANONICAL_QUADRANT_ORDER, exps)}


# === CLI: DEVICE / DTYPE RESOLUTION ===

_DTYPE_NAMES: tuple[str, ...] = ("bfloat16", "float16", "float32")


def _resolve_device(config_device: str, override_device: str | None) -> str:
    """CLI override wins over config; final value is just stringified."""
    chosen = override_device if override_device is not None else config_device
    if not isinstance(chosen, str) or not chosen.strip():
        raise ValueError(
            f"device must be a non-empty string; got config_device={config_device!r} "
            f"override_device={override_device!r}"
        )
    return str(chosen)


def _resolve_dtype(dtype_name: str) -> Any:
    """Map a config string to a torch dtype. torch is imported locally."""
    if dtype_name not in _DTYPE_NAMES:
        raise ValueError(
            f"dtype must be one of {list(_DTYPE_NAMES)}; got {dtype_name!r}"
        )
    import torch  # noqa: PLC0415
    return {
        "bfloat16": torch.bfloat16,
        "float16":  torch.float16,
        "float32":  torch.float32,
    }[dtype_name]


# === CLI: MODEL + ENGINE CONSTRUCTION ===

def load_model_and_tokenizer(
    base_model_name: str,
    dtype: Any,
    device: str,
) -> tuple[Any, Any]:
    """
    Load the base causal LM and its tokenizer, alias pad→eos for Mistral,
    place the model on the requested device, and set eval mode. Mirrors
    src/router_training/features.py to keep prefill geometry consistent.

    transformers is imported locally so this module remains importable
    without it.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415

    log.info("loading tokenizer: %s", base_model_name)
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        # mistral has no pad token; alias to eos without resizing embeddings
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    log.info(
        "loading base model: %s  dtype=%s  device=%s",
        base_model_name, dtype, device,
    )
    try:
        model = AutoModelForCausalLM.from_pretrained(base_model_name, dtype=dtype)
    except TypeError:
        # older transformers versions only accept torch_dtype
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name, torch_dtype=dtype,
        )
    model = model.to(device)
    model.eval()
    return model, tokenizer


def build_moce_engine(cfg: Any, model: Any, tokenizer: Any) -> Any:
    """
    Construct a MoCEEngine wired for forced-policy candidate-trace collection.

    Engine-side config choices baked in here:
    - RouterConfig.use_calibrated_router=False (the runner forces the policy
      externally; the engine's router is bypassed by ForcedPolicyMoCERunner).
    - RouterConfig.beta / temperature = cfg.training.beta / temperature so
      compute_heuristic_prior and engine.router.build_heuristic_prior agree.
    - EditorConfig.correction_beta=0.0 so the editor preserves the forced
      candidate as alpha (no in-loop correction). EditorConfig.max_edit_steps
      stays at 1 (the dataclass minimum); the step is a pass-through under
      correction_beta=0.

    The engine module is loaded lazily via importlib so forced_policy_runner
    remains importable without torch.
    """
    components_path = _SRC_DIR / "09_moce_components.py"
    spec = importlib.util.spec_from_file_location(
        "moce_components_for_runner_cli", components_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to import {components_path}")
    moce = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(moce)

    sv_cfg = moce.SteeringVectorConfig(
        economic_vector_path=Path(cfg.paths.steering_vectors.economic_vector_path),
        social_vector_path=Path(cfg.paths.steering_vectors.social_vector_path),
        vector_method=cfg.input_transformer.vector_method,
        use_final_aggregated_vectors=cfg.input_transformer.use_final_aggregated_vectors,
        selected_layers=list(cfg.input_transformer.selected_layers),
        pooling_method=cfg.input_transformer.pooling_method,
        use_centering=cfg.input_transformer.use_centering,
        neutral_reference_path=cfg.input_transformer.neutral_reference_path,
    )

    router_cfg = moce.RouterConfig(
        use_calibrated_router=False,
        beta=float(cfg.training.beta),
        temperature=float(cfg.training.temperature),
        calibration_input_dim=int(
            getattr(cfg.input_transformer, "calibration_input_dim", 4096)
        ),
    )

    expert_cfg = moce.ExpertConfig(
        left_lib_checkpoint=Path(cfg.paths.expert_checkpoints.left_lib_checkpoint),
        left_auth_checkpoint=Path(cfg.paths.expert_checkpoints.left_auth_checkpoint),
        right_lib_checkpoint=Path(cfg.paths.expert_checkpoints.right_lib_checkpoint),
        right_auth_checkpoint=Path(cfg.paths.expert_checkpoints.right_auth_checkpoint),
    )

    editor_cfg = moce.EditorConfig(
        max_edit_steps=1,
        correction_beta=0.0,
        initialization_mode="router_policy",
    )

    gen_block = getattr(cfg, "generation", None)
    if gen_block is not None:
        generation_cfg = moce.GenerationConfig(
            max_new_tokens=int(getattr(gen_block, "max_new_tokens", 256)),
            temperature=float(getattr(gen_block, "temperature", 0.7)),
            do_sample=bool(getattr(gen_block, "do_sample", False)),
            top_p=float(getattr(gen_block, "top_p", 1.0)),
        )
    else:
        generation_cfg = moce.GenerationConfig()

    return moce.MoCEEngine(
        model=model,
        tokenizer=tokenizer,
        steering_config=sv_cfg,
        router_config=router_cfg,
        expert_config=expert_cfg,
        editor_config=editor_cfg,
        generation_config=generation_cfg,
    )


# === CLI: RESUME SCAN ===

def scan_completed_pairs(path: Path) -> set[tuple[str, int]]:
    """
    Read an existing candidate_traces.jsonl and return the set of
    (example_id, candidate_index) tuples it already contains. Returns an
    empty set when the file is absent.

    Rows missing example_id or metadata.candidate_index are skipped with a
    warning rather than treated as completed; a malformed historical row
    should not silently mark a pair as done.
    """
    if not path.is_file():
        return set()
    done: set[tuple[str, int]] = set()
    with path.open("r", encoding="utf-8") as fh:
        for line_index, raw_line in enumerate(fh):
            line = raw_line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                log.warning(
                    "scan_completed_pairs: skipping malformed JSON at line %d",
                    line_index + 1,
                )
                continue
            example_id = row.get("example_id")
            metadata = row.get("metadata") or {}
            candidate_index = metadata.get("candidate_index")
            if (
                not isinstance(example_id, str)
                or isinstance(candidate_index, bool)
                or not isinstance(candidate_index, int)
            ):
                log.warning(
                    "scan_completed_pairs: skipping row %d (missing example_id "
                    "or metadata.candidate_index)", line_index + 1,
                )
                continue
            done.add((example_id, int(candidate_index)))
    return done


# === CLI: RUN COLLECTION ===

_FAILED_PAIRS_REPORT_CAP: int = 1000


def _normalize_quadrant_scores(raw: Any, where: str) -> dict[str, float]:
    """Coerce a feature-row quadrant_scores value into a canonical-keyed dict."""
    if not isinstance(raw, dict):
        raise ValueError(
            f"{where}: quadrant_scores must be a dict, got {type(raw).__name__}"
        )
    if set(raw.keys()) != set(CANONICAL_QUADRANT_ORDER):
        raise ValueError(
            f"{where}: quadrant_scores keys must equal canonical "
            f"{list(CANONICAL_QUADRANT_ORDER)}; got {sorted(raw.keys())}"
        )
    out: dict[str, float] = {}
    for k in CANONICAL_QUADRANT_ORDER:
        v = raw[k]
        if not _is_finite_number(v):
            raise ValueError(f"{where}: quadrant_scores[{k!r}]={v!r} is not finite")
        out[k] = float(v)
    return out


def run_collection(args: argparse.Namespace) -> dict[str, Any]:
    """
    Drive the forced-policy collection pipeline end-to-end:
        load config -> resolve paths -> load features -> resume scan ->
        (dry-run early-exit) -> load model + build engine + runner ->
        per-prompt: compute prior, generate candidates, run+serialize+write,
        skipping completed pairs and tolerating per-pair failures ->
        write a JSON report at the end.

    Returns the report dict. Also persists it under report_path.
    """
    # lazy imports — keep the module importable without torch / yaml
    import random  # noqa: PLC0415

    from router_training.config import load_router_calibration_config  # noqa: E402, PLC0415
    from router_training.utils import (  # noqa: E402, PLC0415
        CandidatePolicyConfig,
        generate_candidate_policies,
    )

    started_at = datetime.now().isoformat(timespec="seconds")

    cfg = load_router_calibration_config(args.config)
    paths = resolve_paths(cfg, args)

    features_path: Path = paths["features_path"]
    output_path:   Path = paths["output_path"]
    report_path:   Path = paths["report_path"]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    feature_rows = load_feature_records(
        features_path,
        start_index=args.start_index,
        end_index=args.end_index,
        max_examples=args.max_examples,
    )
    log.info(
        "loaded %d feature rows from %s (start=%s end=%s max=%s)",
        len(feature_rows), features_path, args.start_index, args.end_index,
        args.max_examples,
    )

    # candidate-policy config — pulled straight from router_calibration.candidate_policies
    cp = cfg.candidate_policies
    sharpen = list(getattr(cp, "sharpen_temperatures", []) or [])
    soften  = list(getattr(cp, "soften_temperatures",  []) or [])
    candidate_cfg = CandidatePolicyConfig(
        num_dirichlet_samples=int(cp.dirichlet_samples),
        dirichlet_alpha=float(cp.dirichlet_concentration),
        include_uniform=bool(cp.include_uniform),
        include_sharpened=bool(sharpen),
        include_softened=bool(soften),
        include_opposite=bool(cp.include_opposite_heavy),
        include_adjacent=bool(cp.include_adjacent_heavy),
        min_probability=float(cp.min_probability),
    )
    candidate_seed = int(cp.seed)

    beta = float(cfg.training.beta)
    temperature = float(cfg.training.temperature)

    # resume scan + truncate-on-no-resume
    if args.no_resume and output_path.exists():
        log.info("--no-resume: truncating %s", output_path)
        output_path.write_text("", encoding="utf-8")
    completed_pairs = (
        set() if args.no_resume else scan_completed_pairs(output_path)
    )
    log.info("resume: %d completed (example_id, candidate_index) pairs", len(completed_pairs))

    if args.dry_run:
        log.info("--dry-run: skipping model load and per-prompt execution")
        return _build_collection_report(
            cfg=cfg, args=args, paths=paths, beta=beta, temperature=temperature,
            candidate_cfg=candidate_cfg, candidate_seed=candidate_seed,
            num_feature_rows=len(feature_rows), completed_count=len(completed_pairs),
            attempted=0, written=0, resumed=0, failed=[], started_at=started_at,
            finished_at=datetime.now().isoformat(timespec="seconds"),
            dry_run=True,
        )

    # build the engine — heavy lift starts here
    device = _resolve_device(str(getattr(cfg.model, "device", "cpu")), args.device)
    dtype = _resolve_dtype(str(getattr(cfg.model, "dtype", "float32")))
    base_model_name = str(cfg.model.base_model)

    model, tokenizer = load_model_and_tokenizer(base_model_name, dtype, device)
    engine = build_moce_engine(cfg, model, tokenizer)
    runner = ForcedPolicyMoCERunner(engine)

    attempted = 0
    written = 0
    resumed = 0
    failed: list[dict[str, Any]] = []
    failed_overflow = 0

    with output_path.open("a", encoding="utf-8") as out_fh:
        for prompt_index, feature in enumerate(feature_rows):
            example_id  = feature["example_id"]
            prompt_text = feature["prompt_text"]
            quadrant_scores = _normalize_quadrant_scores(
                feature["quadrant_scores"], where=f"example_id={example_id}",
            )

            heuristic_prior = compute_heuristic_prior(
                quadrant_scores, beta=beta, temperature=temperature,
            )

            rng = random.Random(candidate_seed + prompt_index)
            candidates = generate_candidate_policies(
                quadrant_scores=quadrant_scores,
                heuristic_prior=heuristic_prior,
                config=candidate_cfg,
                rng=rng,
            )

            for cand_index, candidate in enumerate(candidates):
                if (example_id, cand_index) in completed_pairs:
                    resumed += 1
                    continue
                attempted += 1
                try:
                    result = runner.run(
                        example_id=example_id,
                        prompt_text=prompt_text,
                        candidate_policy=candidate,
                        heuristic_prior=heuristic_prior,
                    )
                    trace = serialize_forced_policy_result(result)
                    trace_meta = trace.setdefault("metadata", {})
                    trace_meta["candidate_index"] = int(cand_index)
                    out_fh.write(json.dumps(trace, ensure_ascii=False) + "\n")
                    out_fh.flush()
                    written += 1
                except (ValueError, RuntimeError) as err:
                    if len(failed) < _FAILED_PAIRS_REPORT_CAP:
                        failed.append({
                            "example_id": example_id,
                            "candidate_index": int(cand_index),
                            "error_type": type(err).__name__,
                            "error_message": str(err),
                        })
                    else:
                        failed_overflow += 1
                    log.warning(
                        "pair (example_id=%s, candidate_index=%d) failed: %s: %s",
                        example_id, cand_index, type(err).__name__, err,
                    )

            n_done = prompt_index + 1
            if n_done % LOG_PROGRESS_EVERY == 0 or n_done == len(feature_rows):
                log.info(
                    "progress: %d/%d prompts | attempted=%d written=%d resumed=%d failed=%d",
                    n_done, len(feature_rows), attempted, written, resumed,
                    len(failed) + failed_overflow,
                )

    finished_at = datetime.now().isoformat(timespec="seconds")
    return _build_collection_report(
        cfg=cfg, args=args, paths=paths, beta=beta, temperature=temperature,
        candidate_cfg=candidate_cfg, candidate_seed=candidate_seed,
        num_feature_rows=len(feature_rows), completed_count=len(completed_pairs),
        attempted=attempted, written=written, resumed=resumed,
        failed=failed, started_at=started_at, finished_at=finished_at,
        dry_run=False, failed_overflow=failed_overflow,
    )


def _build_collection_report(
    *,
    cfg: Any,
    args: argparse.Namespace,
    paths: dict[str, Path],
    beta: float,
    temperature: float,
    candidate_cfg: Any,
    candidate_seed: int,
    num_feature_rows: int,
    completed_count: int,
    attempted: int,
    written: int,
    resumed: int,
    failed: list[dict[str, Any]],
    started_at: str,
    finished_at: str,
    dry_run: bool,
    failed_overflow: int = 0,
) -> dict[str, Any]:
    """Assemble + persist the run report. Returns the report dict."""
    report: dict[str, Any] = {
        "config_path":  str(args.config),
        "dry_run":      bool(dry_run),
        "started_at":   started_at,
        "finished_at":  finished_at,
        "input_paths": {
            "features_path": str(paths["features_path"]),
            "hidden_path":   str(paths["hidden_path"]),
        },
        "output_paths": {
            "candidate_traces": str(paths["output_path"]),
            "report":           str(paths["report_path"]),
        },
        "hyperparameters": {
            "beta":              beta,
            "temperature":       temperature,
            "candidate_seed":    candidate_seed,
            "candidate_config": {
                "num_dirichlet_samples": int(candidate_cfg.num_dirichlet_samples),
                "dirichlet_alpha":       float(candidate_cfg.dirichlet_alpha),
                "include_uniform":       bool(candidate_cfg.include_uniform),
                "include_sharpened":     bool(candidate_cfg.include_sharpened),
                "include_softened":      bool(candidate_cfg.include_softened),
                "include_opposite":      bool(candidate_cfg.include_opposite),
                "include_adjacent":      bool(candidate_cfg.include_adjacent),
                "min_probability":       float(candidate_cfg.min_probability),
            },
        },
        "totals": {
            "feature_rows_seen":    int(num_feature_rows),
            "previously_completed": int(completed_count),
            "pairs_attempted":      int(attempted),
            "pairs_written":        int(written),
            "pairs_resumed":        int(resumed),
            "pairs_failed":         int(len(failed) + failed_overflow),
        },
        "failed_pairs":           list(failed),
        "failed_pairs_overflow":  int(failed_overflow),
    }
    paths["report_path"].parent.mkdir(parents=True, exist_ok=True)
    with paths["report_path"].open("w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
    return report


# === CLI: ENTRYPOINT ===

def main() -> None:
    args = parse_args()
    run_collection(args)


if __name__ == "__main__":
    main()
