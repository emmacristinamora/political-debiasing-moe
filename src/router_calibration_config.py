# src/router_calibration_config.py


# === IMPORTS ===

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


# === CONSTANTS ===

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CANONICAL_QUADRANT_ORDER: tuple[str, ...] = (
    "left_lib",
    "left_auth",
    "right_lib",
    "right_auth",
)
CANONICAL_CHECKPOINT_KEYS: tuple[str, ...] = tuple(
    f"{q}_checkpoint" for q in CANONICAL_QUADRANT_ORDER
)

VALID_VECTOR_METHODS: frozenset[str] = frozenset({"logistic_regression", "mean_difference"})
VALID_DTYPES: frozenset[str] = frozenset({"bfloat16", "float16", "float32"})

REQUIRED_TOP_LEVEL_KEYS: tuple[str, ...] = (
    "paths",
    "model",
    "input_transformer",
    "prompt_set",
    "candidate_policies",
    "generation",
    "scoring",
    "split",
    "training",
)
REQUIRED_PATHS_KEYS: tuple[str, ...] = (
    "prompt_sources",
    "steering_vectors",
    "expert_checkpoints",
    "output_dir",
    "prompts_path",
    "features_path",
    "hidden_path",
    "records_path",
    "candidate_traces_path",
    "splits_dir",
    "checkpoints_dir",
    "reports_dir",
)
REQUIRED_PROMPT_SOURCES_KEYS: tuple[str, ...] = (
    "method12_path",
    "method3_path",
    "expert_validation_dir",
)
REQUIRED_STEERING_VECTOR_KEYS: tuple[str, ...] = (
    "economic_vector_path",
    "social_vector_path",
)

FRACTION_SUM_TOLERANCE: float = 1e-6
MIN_PROBABILITY_UPPER_BOUND: float = 0.25
REQUIRED_LAYER: int = 20


# === DATACLASSES ===

@dataclass
class PromptSourcesPaths:
    method12_path: Path
    method3_path: Path
    expert_validation_dir: Path


@dataclass
class SteeringVectorPaths:
    economic_vector_path: Path
    social_vector_path: Path


@dataclass
class ExpertCheckpointPaths:
    left_lib_checkpoint: Path
    left_auth_checkpoint: Path
    right_lib_checkpoint: Path
    right_auth_checkpoint: Path


@dataclass
class RouterPaths:
    prompt_sources: PromptSourcesPaths
    steering_vectors: SteeringVectorPaths
    expert_checkpoints: ExpertCheckpointPaths
    output_dir: Path
    prompts_path: Path
    features_path: Path
    hidden_path: Path
    records_path: Path
    candidate_traces_path: Path
    splits_dir: Path
    checkpoints_dir: Path
    reports_dir: Path


@dataclass
class RouterModelConfig:
    base_model: str
    dtype: str
    device: str


@dataclass
class InputTransformerConfig:
    vector_method: str
    use_final_aggregated_vectors: bool
    selected_layers: list[int]
    pooling_method: str
    use_centering: bool
    neutral_reference_path: Path | None
    calibration_input_dim: int


@dataclass
class PromptSetConfig:
    include_method12: bool
    include_method3: bool
    include_expert_validation: bool
    max_prompts: int | None
    seed: int


@dataclass
class CandidatePoliciesConfig:
    include_heuristic_prior: bool
    include_uniform: bool
    sharpen_temperatures: list[float]
    soften_temperatures: list[float]
    include_opposite_heavy: bool
    include_adjacent_heavy: bool
    dirichlet_samples: int
    dirichlet_concentration: float
    min_probability: float
    seed: int


@dataclass
class GenerationConfig:
    max_new_tokens: int
    temperature: float
    do_sample: bool
    top_p: float


@dataclass
class ScoringWeights:
    bias_radius: float
    quality: float
    refusal: float
    vagueness: float
    kl_to_prior: float


@dataclass
class JudgeConfig:
    enabled: bool
    provider: str | None
    model: str | None


@dataclass
class ScoringConfig:
    score_temperature: float
    weights: ScoringWeights
    normalize_bias_radius: bool
    baseline_bias_radius_path: Path | None
    judge: JudgeConfig


@dataclass
class SplitConfig:
    train_fraction: float
    val_fraction: float
    test_fraction: float
    split_by: str
    seed: int


@dataclass
class TrainingConfig:
    beta: float
    temperature: float
    learning_rate: float
    weight_decay: float
    batch_size: int
    epochs: int
    kl_weight: float
    entropy_weight: float
    seed: int
    device: str


@dataclass
class RouterCalibrationConfig:
    paths: RouterPaths
    model: RouterModelConfig
    input_transformer: InputTransformerConfig
    prompt_set: PromptSetConfig
    candidate_policies: CandidatePoliciesConfig
    generation: GenerationConfig
    scoring: ScoringConfig
    split: SplitConfig
    training: TrainingConfig


# === VALIDATION HELPERS ===

def _resolve_path(value: Any, field: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"router_calibration.{field} must be a non-empty string path, got {value!r}"
        )
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate
    return (PROJECT_ROOT / candidate).resolve()


def _resolve_optional_path(value: Any, field: str) -> Path | None:
    if value is None:
        return None
    return _resolve_path(value, field)


def _require_bool(value: Any, field: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(
            f"router_calibration.{field} must be a bool, got {type(value).__name__} ({value!r})"
        )
    return value


def _require_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(
            f"router_calibration.{field} must be an int, got {type(value).__name__} ({value!r})"
        )
    return value


def _require_positive_int(value: Any, field: str) -> int:
    n = _require_int(value, field)
    if n <= 0:
        raise ValueError(f"router_calibration.{field} must be a positive int, got {n}")
    return n


def _require_optional_positive_int(value: Any, field: str) -> int | None:
    if value is None:
        return None
    return _require_positive_int(value, field)


def _require_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(
            f"router_calibration.{field} must be a number, got {type(value).__name__} ({value!r})"
        )
    return float(value)


def _require_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"router_calibration.{field} must be a non-empty string, got {value!r}")
    return value


def _require_optional_string(value: Any, field: str) -> str | None:
    if value is None:
        return None
    return _require_string(value, field)


def _require_int_list(value: Any, field: str) -> list[int]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"router_calibration.{field} must be a non-empty list, got {value!r}")
    out: list[int] = []
    for i, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, int):
            raise ValueError(
                f"router_calibration.{field}[{i}] must be int, got {type(item).__name__} ({item!r})"
            )
        out.append(item)
    return out


def _require_number_list(value: Any, field: str) -> list[float]:
    if not isinstance(value, list):
        raise ValueError(f"router_calibration.{field} must be a list, got {type(value).__name__}")
    out: list[float] = []
    for i, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ValueError(
                f"router_calibration.{field}[{i}] must be a number, "
                f"got {type(item).__name__} ({item!r})"
            )
        out.append(float(item))
    return out


def _require_subsection(raw: dict, key: str, parent: str) -> dict:
    if key not in raw:
        raise ValueError(f"router_calibration.{parent} missing required subsection '{key}'")
    sub = raw[key]
    if not isinstance(sub, dict):
        raise ValueError(
            f"router_calibration.{parent}.{key} must be a mapping, got {type(sub).__name__}"
        )
    return sub


def _require_keys(raw: dict, keys: tuple[str, ...], parent: str) -> None:
    missing = [k for k in keys if k not in raw]
    if missing:
        raise ValueError(f"router_calibration.{parent} missing required keys: {missing}")


# === SECTION PARSERS ===

def _parse_paths(raw: Any) -> RouterPaths:
    if not isinstance(raw, dict):
        raise ValueError("router_calibration.paths must be a mapping")
    _require_keys(raw, REQUIRED_PATHS_KEYS, "paths")

    prompt_sources_raw = _require_subsection(raw, "prompt_sources", "paths")
    _require_keys(prompt_sources_raw, REQUIRED_PROMPT_SOURCES_KEYS, "paths.prompt_sources")
    prompt_sources = PromptSourcesPaths(
        method12_path=_resolve_path(
            prompt_sources_raw["method12_path"], "paths.prompt_sources.method12_path"
        ),
        method3_path=_resolve_path(
            prompt_sources_raw["method3_path"], "paths.prompt_sources.method3_path"
        ),
        expert_validation_dir=_resolve_path(
            prompt_sources_raw["expert_validation_dir"],
            "paths.prompt_sources.expert_validation_dir",
        ),
    )

    steering_raw = _require_subsection(raw, "steering_vectors", "paths")
    _require_keys(steering_raw, REQUIRED_STEERING_VECTOR_KEYS, "paths.steering_vectors")
    steering_vectors = SteeringVectorPaths(
        economic_vector_path=_resolve_path(
            steering_raw["economic_vector_path"], "paths.steering_vectors.economic_vector_path"
        ),
        social_vector_path=_resolve_path(
            steering_raw["social_vector_path"], "paths.steering_vectors.social_vector_path"
        ),
    )

    checkpoints_raw = _require_subsection(raw, "expert_checkpoints", "paths")
    missing_ckpt = [k for k in CANONICAL_CHECKPOINT_KEYS if k not in checkpoints_raw]
    if missing_ckpt:
        raise ValueError(
            "router_calibration.paths.expert_checkpoints missing canonical quadrant keys: "
            f"{missing_ckpt} (expected order: {list(CANONICAL_CHECKPOINT_KEYS)})"
        )
    expert_checkpoints = ExpertCheckpointPaths(
        left_lib_checkpoint=_resolve_path(
            checkpoints_raw["left_lib_checkpoint"], "paths.expert_checkpoints.left_lib_checkpoint"
        ),
        left_auth_checkpoint=_resolve_path(
            checkpoints_raw["left_auth_checkpoint"],
            "paths.expert_checkpoints.left_auth_checkpoint",
        ),
        right_lib_checkpoint=_resolve_path(
            checkpoints_raw["right_lib_checkpoint"],
            "paths.expert_checkpoints.right_lib_checkpoint",
        ),
        right_auth_checkpoint=_resolve_path(
            checkpoints_raw["right_auth_checkpoint"],
            "paths.expert_checkpoints.right_auth_checkpoint",
        ),
    )

    return RouterPaths(
        prompt_sources=prompt_sources,
        steering_vectors=steering_vectors,
        expert_checkpoints=expert_checkpoints,
        output_dir=_resolve_path(raw["output_dir"], "paths.output_dir"),
        prompts_path=_resolve_path(raw["prompts_path"], "paths.prompts_path"),
        features_path=_resolve_path(raw["features_path"], "paths.features_path"),
        hidden_path=_resolve_path(raw["hidden_path"], "paths.hidden_path"),
        records_path=_resolve_path(raw["records_path"], "paths.records_path"),
        candidate_traces_path=_resolve_path(
            raw["candidate_traces_path"], "paths.candidate_traces_path"
        ),
        splits_dir=_resolve_path(raw["splits_dir"], "paths.splits_dir"),
        checkpoints_dir=_resolve_path(raw["checkpoints_dir"], "paths.checkpoints_dir"),
        reports_dir=_resolve_path(raw["reports_dir"], "paths.reports_dir"),
    )


def _parse_model(raw: Any) -> RouterModelConfig:
    if not isinstance(raw, dict):
        raise ValueError("router_calibration.model must be a mapping")
    _require_keys(raw, ("base_model", "dtype", "device"), "model")
    dtype = _require_string(raw["dtype"], "model.dtype")
    if dtype not in VALID_DTYPES:
        raise ValueError(
            f"router_calibration.model.dtype must be one of {sorted(VALID_DTYPES)}, got {dtype!r}"
        )
    return RouterModelConfig(
        base_model=_require_string(raw["base_model"], "model.base_model"),
        dtype=dtype,
        device=_require_string(raw["device"], "model.device"),
    )


def _parse_input_transformer(raw: Any) -> InputTransformerConfig:
    if not isinstance(raw, dict):
        raise ValueError("router_calibration.input_transformer must be a mapping")
    _require_keys(
        raw,
        (
            "vector_method",
            "use_final_aggregated_vectors",
            "selected_layers",
            "pooling_method",
            "use_centering",
            "neutral_reference_path",
            "calibration_input_dim",
        ),
        "input_transformer",
    )

    vector_method = _require_string(raw["vector_method"], "input_transformer.vector_method")
    if vector_method not in VALID_VECTOR_METHODS:
        raise ValueError(
            "router_calibration.input_transformer.vector_method must be one of "
            f"{sorted(VALID_VECTOR_METHODS)}, got {vector_method!r}"
        )

    pooling_method = _require_string(raw["pooling_method"], "input_transformer.pooling_method")
    if pooling_method != "mean":
        raise ValueError(
            "router_calibration.input_transformer.pooling_method must be 'mean', "
            f"got {pooling_method!r}"
        )

    selected_layers = _require_int_list(
        raw["selected_layers"], "input_transformer.selected_layers"
    )
    if REQUIRED_LAYER not in selected_layers:
        raise ValueError(
            "router_calibration.input_transformer.selected_layers must include "
            f"layer {REQUIRED_LAYER}, got {selected_layers}"
        )

    calibration_input_dim = _require_positive_int(
        raw["calibration_input_dim"], "input_transformer.calibration_input_dim"
    )

    return InputTransformerConfig(
        vector_method=vector_method,
        use_final_aggregated_vectors=_require_bool(
            raw["use_final_aggregated_vectors"],
            "input_transformer.use_final_aggregated_vectors",
        ),
        selected_layers=selected_layers,
        pooling_method=pooling_method,
        use_centering=_require_bool(raw["use_centering"], "input_transformer.use_centering"),
        neutral_reference_path=_resolve_optional_path(
            raw["neutral_reference_path"], "input_transformer.neutral_reference_path"
        ),
        calibration_input_dim=calibration_input_dim,
    )


def _parse_prompt_set(raw: Any) -> PromptSetConfig:
    if not isinstance(raw, dict):
        raise ValueError("router_calibration.prompt_set must be a mapping")
    _require_keys(
        raw,
        (
            "include_method12",
            "include_method3",
            "include_expert_validation",
            "max_prompts",
            "seed",
        ),
        "prompt_set",
    )
    return PromptSetConfig(
        include_method12=_require_bool(raw["include_method12"], "prompt_set.include_method12"),
        include_method3=_require_bool(raw["include_method3"], "prompt_set.include_method3"),
        include_expert_validation=_require_bool(
            raw["include_expert_validation"], "prompt_set.include_expert_validation"
        ),
        max_prompts=_require_optional_positive_int(raw["max_prompts"], "prompt_set.max_prompts"),
        seed=_require_int(raw["seed"], "prompt_set.seed"),
    )


def _parse_candidate_policies(raw: Any) -> CandidatePoliciesConfig:
    if not isinstance(raw, dict):
        raise ValueError("router_calibration.candidate_policies must be a mapping")
    _require_keys(
        raw,
        (
            "include_heuristic_prior",
            "include_uniform",
            "sharpen_temperatures",
            "soften_temperatures",
            "include_opposite_heavy",
            "include_adjacent_heavy",
            "dirichlet_samples",
            "dirichlet_concentration",
            "min_probability",
            "seed",
        ),
        "candidate_policies",
    )

    min_probability = _require_number(
        raw["min_probability"], "candidate_policies.min_probability"
    )
    if not (0.0 < min_probability < MIN_PROBABILITY_UPPER_BOUND):
        raise ValueError(
            "router_calibration.candidate_policies.min_probability must be in "
            f"(0, {MIN_PROBABILITY_UPPER_BOUND}), got {min_probability}"
        )

    return CandidatePoliciesConfig(
        include_heuristic_prior=_require_bool(
            raw["include_heuristic_prior"], "candidate_policies.include_heuristic_prior"
        ),
        include_uniform=_require_bool(
            raw["include_uniform"], "candidate_policies.include_uniform"
        ),
        sharpen_temperatures=_require_number_list(
            raw["sharpen_temperatures"], "candidate_policies.sharpen_temperatures"
        ),
        soften_temperatures=_require_number_list(
            raw["soften_temperatures"], "candidate_policies.soften_temperatures"
        ),
        include_opposite_heavy=_require_bool(
            raw["include_opposite_heavy"], "candidate_policies.include_opposite_heavy"
        ),
        include_adjacent_heavy=_require_bool(
            raw["include_adjacent_heavy"], "candidate_policies.include_adjacent_heavy"
        ),
        dirichlet_samples=_require_positive_int(
            raw["dirichlet_samples"], "candidate_policies.dirichlet_samples"
        ),
        dirichlet_concentration=_require_number(
            raw["dirichlet_concentration"], "candidate_policies.dirichlet_concentration"
        ),
        min_probability=min_probability,
        seed=_require_int(raw["seed"], "candidate_policies.seed"),
    )


def _parse_generation(raw: Any) -> GenerationConfig:
    if not isinstance(raw, dict):
        raise ValueError("router_calibration.generation must be a mapping")
    _require_keys(
        raw, ("max_new_tokens", "temperature", "do_sample", "top_p"), "generation"
    )
    return GenerationConfig(
        max_new_tokens=_require_positive_int(
            raw["max_new_tokens"], "generation.max_new_tokens"
        ),
        temperature=_require_number(raw["temperature"], "generation.temperature"),
        do_sample=_require_bool(raw["do_sample"], "generation.do_sample"),
        top_p=_require_number(raw["top_p"], "generation.top_p"),
    )


def _parse_scoring(raw: Any) -> ScoringConfig:
    if not isinstance(raw, dict):
        raise ValueError("router_calibration.scoring must be a mapping")
    _require_keys(
        raw,
        (
            "score_temperature",
            "weights",
            "normalize_bias_radius",
            "baseline_bias_radius_path",
            "judge",
        ),
        "scoring",
    )

    score_temperature = _require_number(raw["score_temperature"], "scoring.score_temperature")
    if score_temperature <= 0:
        raise ValueError(
            f"router_calibration.scoring.score_temperature must be > 0, got {score_temperature}"
        )

    weights_raw = _require_subsection(raw, "weights", "scoring")
    _require_keys(
        weights_raw,
        ("bias_radius", "quality", "refusal", "vagueness", "kl_to_prior"),
        "scoring.weights",
    )
    weights = ScoringWeights(
        bias_radius=_require_number(weights_raw["bias_radius"], "scoring.weights.bias_radius"),
        quality=_require_number(weights_raw["quality"], "scoring.weights.quality"),
        refusal=_require_number(weights_raw["refusal"], "scoring.weights.refusal"),
        vagueness=_require_number(weights_raw["vagueness"], "scoring.weights.vagueness"),
        kl_to_prior=_require_number(weights_raw["kl_to_prior"], "scoring.weights.kl_to_prior"),
    )

    judge_raw = _require_subsection(raw, "judge", "scoring")
    _require_keys(judge_raw, ("enabled", "provider", "model"), "scoring.judge")
    judge = JudgeConfig(
        enabled=_require_bool(judge_raw["enabled"], "scoring.judge.enabled"),
        provider=_require_optional_string(judge_raw["provider"], "scoring.judge.provider"),
        model=_require_optional_string(judge_raw["model"], "scoring.judge.model"),
    )

    return ScoringConfig(
        score_temperature=score_temperature,
        weights=weights,
        normalize_bias_radius=_require_bool(
            raw["normalize_bias_radius"], "scoring.normalize_bias_radius"
        ),
        baseline_bias_radius_path=_resolve_optional_path(
            raw["baseline_bias_radius_path"], "scoring.baseline_bias_radius_path"
        ),
        judge=judge,
    )


def _parse_split(raw: Any) -> SplitConfig:
    if not isinstance(raw, dict):
        raise ValueError("router_calibration.split must be a mapping")
    _require_keys(
        raw,
        ("train_fraction", "val_fraction", "test_fraction", "split_by", "seed"),
        "split",
    )
    train_fraction = _require_number(raw["train_fraction"], "split.train_fraction")
    val_fraction   = _require_number(raw["val_fraction"], "split.val_fraction")
    test_fraction  = _require_number(raw["test_fraction"], "split.test_fraction")

    for name, value in (
        ("train_fraction", train_fraction),
        ("val_fraction", val_fraction),
        ("test_fraction", test_fraction),
    ):
        if value <= 0:
            raise ValueError(f"router_calibration.split.{name} must be > 0, got {value}")

    total = train_fraction + val_fraction + test_fraction
    if abs(total - 1.0) > FRACTION_SUM_TOLERANCE:
        raise ValueError(
            "router_calibration.split fractions must sum to 1 within "
            f"{FRACTION_SUM_TOLERANCE}, got sum={total}"
        )

    return SplitConfig(
        train_fraction=train_fraction,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
        split_by=_require_string(raw["split_by"], "split.split_by"),
        seed=_require_int(raw["seed"], "split.seed"),
    )


def _parse_training(raw: Any) -> TrainingConfig:
    if not isinstance(raw, dict):
        raise ValueError("router_calibration.training must be a mapping")
    _require_keys(
        raw,
        (
            "beta",
            "temperature",
            "learning_rate",
            "weight_decay",
            "batch_size",
            "epochs",
            "kl_weight",
            "entropy_weight",
            "seed",
            "device",
        ),
        "training",
    )

    temperature = _require_number(raw["temperature"], "training.temperature")
    if temperature == 0:
        raise ValueError("router_calibration.training.temperature must be != 0")

    return TrainingConfig(
        beta=_require_number(raw["beta"], "training.beta"),
        temperature=temperature,
        learning_rate=_require_number(raw["learning_rate"], "training.learning_rate"),
        weight_decay=_require_number(raw["weight_decay"], "training.weight_decay"),
        batch_size=_require_positive_int(raw["batch_size"], "training.batch_size"),
        epochs=_require_positive_int(raw["epochs"], "training.epochs"),
        kl_weight=_require_number(raw["kl_weight"], "training.kl_weight"),
        entropy_weight=_require_number(raw["entropy_weight"], "training.entropy_weight"),
        seed=_require_int(raw["seed"], "training.seed"),
        device=_require_string(raw["device"], "training.device"),
    )


# === PUBLIC API ===

def load_router_calibration_config(path: Path) -> RouterCalibrationConfig:
    """
    Load and validate the router_calibration block from a YAML config file.

    Args:
        path: path to config.yaml.

    Returns:
        Fully-validated RouterCalibrationConfig with relative paths resolved
        against PROJECT_ROOT.

    Logic:
        Reads the YAML, asserts the router_calibration block is present, then
        validates every subsection's required keys, types, and value ranges.
        Path fields are turned into absolute pathlib.Path objects but their
        existence is not checked — the calibration pipeline materialises them
        in later stages.
    """
    if not isinstance(path, Path):
        path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"config file not found: {path}")

    with path.open(encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    if not isinstance(raw, dict):
        raise ValueError(f"config root must be a mapping, got {type(raw).__name__}: {path}")

    if "router_calibration" not in raw:
        raise ValueError(f"config missing 'router_calibration' block: {path}")

    cfg = raw["router_calibration"]
    if not isinstance(cfg, dict):
        raise ValueError(
            f"router_calibration must be a mapping, got {type(cfg).__name__}: {path}"
        )

    missing = [k for k in REQUIRED_TOP_LEVEL_KEYS if k not in cfg]
    if missing:
        raise ValueError(f"router_calibration missing required subsections: {missing}")

    return RouterCalibrationConfig(
        paths=_parse_paths(cfg["paths"]),
        model=_parse_model(cfg["model"]),
        input_transformer=_parse_input_transformer(cfg["input_transformer"]),
        prompt_set=_parse_prompt_set(cfg["prompt_set"]),
        candidate_policies=_parse_candidate_policies(cfg["candidate_policies"]),
        generation=_parse_generation(cfg["generation"]),
        scoring=_parse_scoring(cfg["scoring"]),
        split=_parse_split(cfg["split"]),
        training=_parse_training(cfg["training"]),
    )
