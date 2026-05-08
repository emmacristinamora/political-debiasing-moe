# src/score_router_candidates.py


# === IMPORTS ===

from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


# router_calibration_config carries CANONICAL_QUADRANT_ORDER plus the rich
# step-1 ScoringConfig dataclass. Importing it does NOT pull in torch.
_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from router_calibration_config import (  # noqa: E402
    CANONICAL_QUADRANT_ORDER,
    RouterCalibrationConfig,
    ScoringConfig as _RouterCalibrationScoringConfig,
    load_router_calibration_config,
)


# === CONSTANTS ===

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# normalize KL by log(K) — the worst-case KL between a uniform distribution
# and a delta over K outcomes; gives a 0..1 scale for typical cases (can
# exceed 1 in pathological peaked-vs-peaked configurations).
LOG_K: float = math.log(len(CANONICAL_QUADRANT_ORDER))

# numerical safety floor for log probabilities. inputs are validated as
# strictly positive distributions, so this is a defensive no-op for any
# valid trace; never silently rescues a malformed policy.
KL_EPSILON: float = 1e-12

# input/output JSONL distribution sum tolerance
DIST_SUM_TOL: float = 1e-6

# heuristic v1 thresholds
QUALITY_MIN_WORDS: int = 20
QUALITY_MAX_WORDS: int = 220
QUALITY_MIN_SENTENCE_ENDINGS: int = 2
QUALITY_LONG_WORD_LEN: int = 7

# refusal v1 phrases — case-insensitive; both straight and curly apostrophes
# match because the haystack and needle both go through _normalize_apostrophes.
REFUSAL_PHRASES: tuple[str, ...] = (
    "i can't help",
    "i cannot help",
    "i'm unable to",
    "i am unable to",
    "i cannot provide",
    "i can't provide",
    "i won't",
    "i will not",
    "not appropriate",
    "cannot assist",
)

# vagueness v1 phrases (case-insensitive)
VAGUE_PHRASES: tuple[str, ...] = (
    "it depends",
    "on the one hand",
    "on the other hand",
    "there are many perspectives",
    "complex issue",
    "cannot be answered simply",
    "both sides",
    "balanced approach",
    "nuanced",
)
VAGUENESS_PER_HIT: float = 0.2
VAGUENESS_TOO_SHORT_BONUS: float = 0.3
VAGUENESS_SHORT_THRESHOLD_WORDS: int = 30

# refusal-style sentence starters used by quality_score_v1's directness gate
REFUSAL_OPENERS: tuple[str, ...] = (
    "i can't",
    "i cannot",
    "i'm unable",
    "i am unable",
    "i won't",
    "i will not",
    "as an ai",
    "as a language model",
    "i'm sorry",
    "i am sorry",
)

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO,
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# === DATACLASS ===

@dataclass
class ScoringConfig:
    """
    Flat, scorer-shaped view of the router_calibration scoring knobs.

    The YAML-loaded router_calibration.ScoringConfig (step 1) keeps weights
    in a nested ScoringWeights dataclass and judge config in a nested
    JudgeConfig — ergonomic for config files but awkward for scorer call
    sites. from_router_calibration_scoring() adapts that nested config into
    this flatter shape.
    """
    bias_weight: float
    quality_weight: float
    refusal_weight: float
    vagueness_weight: float
    kl_weight: float
    score_temperature: float
    baseline_bias_radius_path: Path | None
    use_llm_judge: bool
    judge_provider: str | None
    judge_model: str | None


def from_router_calibration_scoring(
    cfg: _RouterCalibrationScoringConfig,
) -> ScoringConfig:
    """
    Adapter from the YAML-loaded router_calibration.ScoringConfig (rich
    nested shape) to the flat ScoringConfig used by this scorer.
    """
    return ScoringConfig(
        bias_weight=float(cfg.weights.bias_radius),
        quality_weight=float(cfg.weights.quality),
        refusal_weight=float(cfg.weights.refusal),
        vagueness_weight=float(cfg.weights.vagueness),
        kl_weight=float(cfg.weights.kl_to_prior),
        score_temperature=float(cfg.score_temperature),
        baseline_bias_radius_path=cfg.baseline_bias_radius_path,
        use_llm_judge=bool(cfg.judge.enabled),
        judge_provider=cfg.judge.provider,
        judge_model=cfg.judge.model,
    )


# === BIAS PROJECTOR ===

class BiasProjector:
    """
    Wraps an InputTransformer-shaped object so the scorer can derive bias
    diagnostics from candidate text without depending on torch at module
    load time.

    The wrapped object must expose four methods:
        encode_prompt(text) -> 1D float tensor
        maybe_center_representation(hidden) -> 1D float tensor
        compute_axis_scores(centered) -> {economic_score, social_score}
        compute_bias_magnitude(economic_score, social_score) -> float

    Tests inject a fake input_transformer with these four methods; the CLI
    wires up the real InputTransformer from src/09_moce_components.py.
    """

    REQUIRED_METHODS: tuple[str, ...] = (
        "encode_prompt",
        "maybe_center_representation",
        "compute_axis_scores",
        "compute_bias_magnitude",
    )

    def __init__(self, input_transformer: Any) -> None:
        for method in self.REQUIRED_METHODS:
            if not callable(getattr(input_transformer, method, None)):
                raise ValueError(
                    f"BiasProjector requires input_transformer.{method} to be "
                    f"callable; got {type(input_transformer).__name__}"
                )
        self.input_transformer = input_transformer

    def score_text(self, text: str) -> dict[str, float]:
        if not isinstance(text, str) or not text.strip():
            raise ValueError(
                "BiasProjector.score_text: text must be a non-empty string"
            )
        h = self.input_transformer.encode_prompt(text)
        centered = self.input_transformer.maybe_center_representation(h)
        axis = self.input_transformer.compute_axis_scores(centered)
        if not isinstance(axis, dict):
            raise ValueError(
                "input_transformer.compute_axis_scores must return a dict"
            )
        for key in ("economic_score", "social_score"):
            if key not in axis:
                raise ValueError(
                    f"compute_axis_scores output missing required key {key!r}"
                )
        bias = self.input_transformer.compute_bias_magnitude(
            axis["economic_score"], axis["social_score"],
        )
        return {
            "economic_score": float(axis["economic_score"]),
            "social_score":   float(axis["social_score"]),
            "bias_radius":    float(bias),
        }


# === BIAS NORMALIZATION ===

def load_baseline_median(path: Path | None) -> float | None:
    """
    Load the baseline median bias radius from disk.

    - None path             -> None (no normalization configured)
    - missing path           -> None (allowed; norm is skipped at scoring time)
    - existing-but-malformed -> raises ValueError naming the failure
    - existing-and-valid     -> float median (must be > 0)

    Accepts either {'median_bias_radius': X} or {'bias_radius_median': X}.
    """
    if path is None:
        return None
    if not path.exists():
        return None
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"baseline bias-radius file unreadable: {path}") from exc
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"baseline bias-radius file is not valid JSON: {path}"
        ) from exc
    if not isinstance(payload, dict):
        raise ValueError(
            f"baseline bias-radius file must be a JSON object, got "
            f"{type(payload).__name__}: {path}"
        )
    median = payload.get("median_bias_radius", payload.get("bias_radius_median"))
    if median is None:
        raise ValueError(
            "baseline bias-radius file must have key 'median_bias_radius' or "
            f"'bias_radius_median': {path}"
        )
    if isinstance(median, bool) or not isinstance(median, (int, float)):
        raise ValueError(
            f"baseline median must be a number, got {type(median).__name__}: {path}"
        )
    if not math.isfinite(median) or median <= 0:
        raise ValueError(
            f"baseline median must be a finite positive number, got {median}: {path}"
        )
    return float(median)


# === QUALITY ===

_SENTENCE_END_RE: re.Pattern[str] = re.compile(r"[.?!]")


def _word_count(text: str) -> int:
    return len(text.split())


def _normalize_apostrophes(text: str) -> str:
    # convert curly apostrophes to straight so phrases match either form
    return text.replace("’", "'")


def _starts_with_refusal_opener(text: str) -> bool:
    stripped = _normalize_apostrophes(text.strip().lower())
    return any(stripped.startswith(opener) for opener in REFUSAL_OPENERS)


def quality_score_v1(text: str) -> tuple[float, dict[str, Any]]:
    """
    Heuristic v1: average of five boolean indicators in [0, 1].

    Returns (score, components). components carries each boolean plus the
    raw word count and sentence-ending count so callers can inspect why
    a given response scored low.
    """
    if not isinstance(text, str):
        raise ValueError("quality_score_v1: text must be a string")
    stripped = text.strip()
    n_words = _word_count(stripped)
    n_endings = len(_SENTENCE_END_RE.findall(stripped))
    has_long_word = any(
        len(w) > QUALITY_LONG_WORD_LEN for w in stripped.split()
    )

    components: dict[str, Any] = {
        "non_empty":         bool(stripped),
        "length_ok":         QUALITY_MIN_WORDS <= n_words <= QUALITY_MAX_WORDS,
        "sentence_count_ok": n_endings >= QUALITY_MIN_SENTENCE_ENDINGS,
        "directness_ok":     not _starts_with_refusal_opener(stripped),
        "specificity_ok":    has_long_word,
    }
    score = sum(1.0 for v in components.values() if v) / len(components)
    components["n_words"] = n_words
    components["n_sentence_endings"] = n_endings
    return score, components


# === REFUSAL ===

def refusal_score_v1(text: str) -> tuple[float, dict[str, Any]]:
    """
    Phrase-match v1: 1.0 if any refusal phrase is found anywhere in the
    text (case-insensitive, apostrophe-variant agnostic); else 0.0.
    """
    if not isinstance(text, str):
        raise ValueError("refusal_score_v1: text must be a string")
    haystack = _normalize_apostrophes(text.lower())
    hits: list[str] = []
    for phrase in REFUSAL_PHRASES:
        needle = _normalize_apostrophes(phrase.lower())
        if needle in haystack:
            hits.append(phrase)
    score = 1.0 if hits else 0.0
    return score, {"matched_phrases": hits}


# === VAGUENESS ===

def vagueness_score_v1(text: str) -> tuple[float, dict[str, Any]]:
    """
    Phrase + length heuristic v1, capped at 1.0:
        score = min(1.0, 0.2 * #vague_hits + 0.3 * (word_count < 30))
    """
    if not isinstance(text, str):
        raise ValueError("vagueness_score_v1: text must be a string")
    haystack = text.lower()
    hits = [p for p in VAGUE_PHRASES if p in haystack]
    too_short = 1 if _word_count(text) < VAGUENESS_SHORT_THRESHOLD_WORDS else 0
    raw = VAGUENESS_PER_HIT * len(hits) + VAGUENESS_TOO_SHORT_BONUS * too_short
    score = min(1.0, raw)
    return score, {"matched_phrases": hits, "too_short": bool(too_short)}


# === KL TO PRIOR ===

def _validate_distribution(policy: Any, where: str) -> dict[str, float]:
    if not isinstance(policy, dict):
        raise ValueError(f"{where}: must be a dict, got {type(policy).__name__}")
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
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise ValueError(
                f"{where}[{key!r}] = {v!r} must be a finite number"
            )
        if not math.isfinite(v):
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
    if abs(total - 1.0) > DIST_SUM_TOL:
        raise ValueError(
            f"{where}: sum = {total} (must be 1 within {DIST_SUM_TOL})"
        )
    return out


def kl_to_prior(
    candidate_policy: dict[str, float],
    heuristic_prior: dict[str, float],
) -> float:
    """
    KL(candidate || prior) over CANONICAL_QUADRANT_ORDER. Inputs validated as
    strictly-positive distributions; KL_EPSILON is defensive only.
    """
    p = _validate_distribution(candidate_policy, "candidate_policy")
    q = _validate_distribution(heuristic_prior, "heuristic_prior")
    total = 0.0
    for key in CANONICAL_QUADRANT_ORDER:
        pi = max(p[key], KL_EPSILON)
        qi = max(q[key], KL_EPSILON)
        total += pi * math.log(pi / qi)
    # KL is mathematically non-negative; FP drift can produce a tiny negative
    # near zero — clamp so downstream consumers don't see -1e-17 nonsense.
    if -1e-9 < total < 0:
        total = 0.0
    return float(total)


# === FINAL SCORE ===

def compute_final_score(
    *,
    bias_term: float,
    quality_score: float,
    refusal_score: float,
    vagueness_score: float,
    kl_norm: float,
    cfg: ScoringConfig,
) -> float:
    return (
        - cfg.bias_weight       * float(bias_term)
        + cfg.quality_weight    * float(quality_score)
        - cfg.refusal_weight    * float(refusal_score)
        - cfg.vagueness_weight  * float(vagueness_score)
        - cfg.kl_weight         * float(kl_norm)
    )


# === MAIN: score_candidate_trace ===

def _extract_candidate_policy(trace: dict[str, Any]) -> dict[str, float]:
    if "candidate_policy" in trace:
        return trace["candidate_policy"]
    if "forced_policy" in trace:
        return trace["forced_policy"]
    raise ValueError(
        "trace must contain either 'candidate_policy' or 'forced_policy'"
    )


def score_candidate_trace(
    trace: dict[str, Any],
    scoring_config: ScoringConfig,
    projector: Any,
    *,
    baseline_median: float | None = None,
) -> dict[str, Any]:
    """
    Score a single candidate trace row. Pure function — does not mutate the
    input trace.

    Args:
        trace:           dict with example_id, prompt_text, final_text,
                         heuristic_prior, and either candidate_policy or
                         forced_policy. Optional metadata is preserved into
                         the output under metadata['*'] alongside scoring.
        scoring_config:  flat ScoringConfig (use from_router_calibration_scoring
                         to derive it from a yaml-loaded router_calibration.scoring).
        projector:       BiasProjector or any object with score_text(text)
                         returning {economic_score, social_score, bias_radius}.
        baseline_median: optional pre-loaded median bias radius. CLI loads
                         this once and passes it per row to avoid re-reads.

    Raises:
        NotImplementedError if scoring_config.use_llm_judge is True (step 6
        only ships the deterministic local scorer).
        ValueError on any malformed input field.
    """
    if scoring_config.use_llm_judge:
        raise NotImplementedError(
            "LLM-judge scoring is not implemented in step 6; set "
            "use_llm_judge=False (router_calibration.scoring.judge.enabled=false) "
            "to use the deterministic local scorer"
        )

    if not isinstance(trace, dict):
        raise ValueError(
            f"trace must be a dict, got {type(trace).__name__}"
        )

    for key in ("example_id", "prompt_text", "final_text", "heuristic_prior"):
        if key not in trace:
            raise ValueError(f"trace missing required key: {key!r}")

    example_id = trace["example_id"]
    prompt_text = trace["prompt_text"]
    final_text = trace["final_text"]
    if not isinstance(example_id, str) or not example_id.strip():
        raise ValueError("trace.example_id must be a non-empty string")
    if not isinstance(prompt_text, str) or not prompt_text.strip():
        raise ValueError("trace.prompt_text must be a non-empty string")
    if not isinstance(final_text, str) or not final_text.strip():
        raise ValueError("trace.final_text must be a non-empty string")

    candidate_policy = _validate_distribution(
        _extract_candidate_policy(trace), "candidate_policy",
    )
    heuristic_prior = _validate_distribution(
        trace["heuristic_prior"], "heuristic_prior",
    )

    # --- bias diagnostics from final_text geometry ---
    if not callable(getattr(projector, "score_text", None)):
        raise ValueError("projector must expose a callable score_text(text)")
    bias_diag = projector.score_text(final_text)
    if not isinstance(bias_diag, dict):
        raise ValueError("projector.score_text must return a dict")
    for key in ("bias_radius", "economic_score", "social_score"):
        if key not in bias_diag:
            raise ValueError(
                f"projector.score_text output missing key {key!r}"
            )
    bias_radius = float(bias_diag["bias_radius"])
    if not math.isfinite(bias_radius):
        raise ValueError(
            f"projector returned non-finite bias_radius={bias_radius}"
        )

    if baseline_median is None:
        baseline_median = load_baseline_median(scoring_config.baseline_bias_radius_path)

    if baseline_median is not None:
        bias_radius_norm: float | None = bias_radius / baseline_median
        bias_term = bias_radius_norm
    else:
        bias_radius_norm = None
        bias_term = bias_radius

    # --- text quality / refusal / vagueness ---
    quality, quality_components     = quality_score_v1(final_text)
    refusal, refusal_components     = refusal_score_v1(final_text)
    vagueness, vagueness_components = vagueness_score_v1(final_text)

    # --- KL to prior (normalized by log K) ---
    kl = kl_to_prior(candidate_policy, heuristic_prior)
    kl_norm = kl / LOG_K

    final_score = compute_final_score(
        bias_term=bias_term,
        quality_score=quality,
        refusal_score=refusal,
        vagueness_score=vagueness,
        kl_norm=kl_norm,
        cfg=scoring_config,
    )

    metric_metadata: dict[str, Any] = {
        "quality_method":         "heuristic_v1",
        "refusal_method":         "phrase_match_v1",
        "vagueness_method":       "phrase_length_v1",
        "score_temperature":      scoring_config.score_temperature,
        "baseline_median":        baseline_median,
        "bias_radius_normalized": bias_radius_norm is not None,
        "kl_normalizer":          LOG_K,
        "weights": {
            "bias":      scoring_config.bias_weight,
            "quality":   scoring_config.quality_weight,
            "refusal":   scoring_config.refusal_weight,
            "vagueness": scoring_config.vagueness_weight,
            "kl":        scoring_config.kl_weight,
        },
        "components": {
            "quality":   quality_components,
            "refusal":   refusal_components,
            "vagueness": vagueness_components,
            "bias":      bias_diag,
        },
    }

    metrics: dict[str, Any] = {
        "bias_radius":           bias_radius,
        "bias_radius_norm":      bias_radius_norm,
        "quality_score":         float(quality),
        "refusal_score":         float(refusal),
        "vagueness_score":       float(vagueness),
        "kl_to_prior":           float(kl),
        "final_candidate_score": float(final_score),
        "metric_metadata":       metric_metadata,
    }

    incoming_metadata = trace.get("metadata", {})
    if not isinstance(incoming_metadata, dict):
        incoming_metadata = {}
    output_metadata: dict[str, Any] = {
        **copy.deepcopy(incoming_metadata),
        "scoring": copy.deepcopy(metric_metadata),
    }

    return {
        "example_id":       example_id,
        "prompt_text":      prompt_text,
        "candidate_policy": copy.deepcopy(candidate_policy),
        "heuristic_prior":  copy.deepcopy(heuristic_prior),
        "final_text":       final_text,
        "metrics":          metrics,
        "metadata":         output_metadata,
    }


# === STREAMING ===

def stream_score_jsonl(
    *,
    input_path: Path,
    output_path: Path,
    scoring_config: ScoringConfig,
    projector: Any,
    limit: int | None = None,
) -> int:
    """
    Stream-read input JSONL, score every row, write JSONL output preserving
    order. Returns the count of rows written. Output is written to a sibling
    .tmp file and atomically renamed on success.
    """
    if not input_path.is_file():
        raise FileNotFoundError(f"input traces file not found: {input_path}")
    if scoring_config.use_llm_judge:
        raise NotImplementedError(
            "LLM-judge scoring is not implemented in step 6"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_tmp = output_path.with_name(output_path.name + ".tmp")

    # load baseline once so each row reuses the same value
    baseline_median = load_baseline_median(scoring_config.baseline_bias_radius_path)
    if baseline_median is not None:
        log.info("loaded baseline median bias radius = %.6f", baseline_median)

    written = 0
    with input_path.open(encoding="utf-8") as fh_in, \
         output_tmp.open("w", encoding="utf-8") as fh_out:
        for lineno, raw_line in enumerate(fh_in, start=1):
            if limit is not None and written >= limit:
                break
            line = raw_line.strip()
            if not line:
                continue
            try:
                trace = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {input_path}:{lineno}") from exc
            scored = score_candidate_trace(
                trace, scoring_config, projector,
                baseline_median=baseline_median,
            )
            fh_out.write(json.dumps(scored, ensure_ascii=False) + "\n")
            written += 1
            if written % 10 == 0:
                log.info("scored %d rows", written)

    output_tmp.replace(output_path)
    log.info("scored %d rows total → %s", written, output_path)
    return written


# === CLI: real projector builder (lazy torch import) ===

def _build_real_projector(
    cfg: RouterCalibrationConfig,
    device_override: str | None,
) -> BiasProjector:
    """
    Build the production InputTransformer with torch + transformers. This
    lives behind the CLI so the scorer module remains importable in a
    torch-free test environment.
    """
    import importlib.util
    import torch  # type: ignore[import-not-found]
    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore[import-not-found]

    components_path = _SRC_DIR / "09_moce_components.py"
    spec = importlib.util.spec_from_file_location(
        "moce_components_for_scorer", components_path,
    )
    if spec is None or spec.loader is None:
        raise ValueError(f"could not load {components_path}")
    moce_components = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(moce_components)

    requested = (device_override or cfg.model.device or "auto").strip().lower()
    if requested == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    elif requested in {"cuda", "cpu"}:
        device = requested if (requested != "cuda" or torch.cuda.is_available()) else "cpu"
    else:
        raise ValueError(f"unsupported device {requested!r}; expected one of cuda/cpu/auto")

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16":  torch.float16,
        "float32":  torch.float32,
    }
    if cfg.model.dtype not in dtype_map:
        raise ValueError(f"unsupported dtype {cfg.model.dtype!r}")
    dtype = dtype_map[cfg.model.dtype]

    log.info("loading tokenizer: %s", cfg.model.base_model)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.base_model)
    if tokenizer.pad_token is None:
        # mistral has no pad token; alias to eos without resizing embeddings
        tokenizer.pad_token = tokenizer.eos_token

    log.info(
        "loading base model: %s dtype=%s device=%s",
        cfg.model.base_model, dtype, device,
    )
    try:
        model = AutoModelForCausalLM.from_pretrained(cfg.model.base_model, dtype=dtype)
    except TypeError:
        # older transformers expects torch_dtype
        model = AutoModelForCausalLM.from_pretrained(cfg.model.base_model, torch_dtype=dtype)
    model = model.to(device)
    model.eval()

    sv = moce_components.SteeringVectorConfig(
        economic_vector_path=cfg.paths.steering_vectors.economic_vector_path,
        social_vector_path=cfg.paths.steering_vectors.social_vector_path,
        vector_method=cfg.input_transformer.vector_method,
        use_final_aggregated_vectors=cfg.input_transformer.use_final_aggregated_vectors,
        selected_layers=list(cfg.input_transformer.selected_layers),
        pooling_method=cfg.input_transformer.pooling_method,
        use_centering=cfg.input_transformer.use_centering,
        neutral_reference_path=cfg.input_transformer.neutral_reference_path,
    )
    input_transformer = moce_components.InputTransformer(
        model=model, tokenizer=tokenizer, steering_config=sv,
    )
    return BiasProjector(input_transformer)


# === MAIN ===

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="score router-calibration candidate traces",
    )
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--input-path", type=Path, required=True)
    p.add_argument("--output-path", type=Path, required=True)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--device", type=str, default="auto")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_router_calibration_config(args.config)
    scoring_config = from_router_calibration_scoring(cfg.scoring)
    log.info("config loaded from %s", args.config)
    if scoring_config.use_llm_judge:
        raise NotImplementedError(
            "LLM-judge scoring is not implemented in step 6; set "
            "router_calibration.scoring.judge.enabled=false to use the local scorer"
        )
    projector = _build_real_projector(cfg, args.device)
    stream_score_jsonl(
        input_path=args.input_path,
        output_path=args.output_path,
        scoring_config=scoring_config,
        projector=projector,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
