# src/build_router_features.py


# === IMPORTS ===

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import math
import sys
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# === MODULE LOADING ===

# router_training.config lives in the same package. add src/ to sys.path so
# the package can be imported by name regardless of cwd.
_SRC_DIR = Path(__file__).resolve().parents[1]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from router_training.config import (  # noqa: E402
    RouterCalibrationConfig,
    load_router_calibration_config,
)

# 09_moce_components.py begins with a digit, so it cannot be imported via
# normal "import" syntax. load it explicitly by absolute path with importlib —
# same approach as tests/test_router.py and tests/test_input_transformer.py.
_COMPONENTS_PATH = _SRC_DIR / "09_moce_components.py"
_COMPONENTS_SPEC = importlib.util.spec_from_file_location(
    "moce_components", _COMPONENTS_PATH,
)
moce_components = importlib.util.module_from_spec(_COMPONENTS_SPEC)
sys.modules["moce_components"] = moce_components
assert _COMPONENTS_SPEC.loader is not None
_COMPONENTS_SPEC.loader.exec_module(moce_components)

CANONICAL_QUADRANT_ORDER: tuple[str, ...] = moce_components.CANONICAL_QUADRANT_ORDER
SteeringVectorConfig = moce_components.SteeringVectorConfig
InputTransformer = moce_components.InputTransformer
PromptState = moce_components.PromptState


# === CONSTANTS ===

PROJECT_ROOT = Path(__file__).resolve().parents[2]

REQUIRED_PROMPT_FIELDS: tuple[str, ...] = (
    "example_id",
    "prompt_text",
    "source",
    "metadata",
)

_DTYPE_MAP: dict[str, torch.dtype] = {
    "bfloat16": torch.bfloat16,
    "float16":  torch.float16,
    "float32":  torch.float32,
}

LOG_PROGRESS_EVERY: int = 10

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO,
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# === DATA LOADING ===

def load_prompts(path: Path) -> list[dict]:
    """
    Load and schema-check the prompts produced by build_router_prompt_set.

    Returns:
        list of prompt dicts in original order, each with example_id,
        prompt_text, source, metadata.

    Raises:
        FileNotFoundError if the file is missing.
        ValueError on empty file, malformed JSON, missing required fields,
        empty prompt_text, empty/duplicate example_id, or zero records.
    """
    if not path.is_file():
        raise FileNotFoundError(f"prompts file not found: {path}")

    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"prompts file is empty: {path}")

    records: list[dict] = []
    seen_ids: set[str] = set()
    for lineno, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON at {path}:{lineno}") from exc

        if not isinstance(record, dict):
            raise ValueError(f"prompt at {path}:{lineno} is not a JSON object")

        missing = [k for k in REQUIRED_PROMPT_FIELDS if k not in record]
        if missing:
            raise ValueError(
                f"prompt at {path}:{lineno} missing required fields: {missing}"
            )

        prompt_text = record["prompt_text"]
        if not isinstance(prompt_text, str) or not prompt_text.strip():
            raise ValueError(f"prompt at {path}:{lineno} has empty 'prompt_text'")

        example_id = record["example_id"]
        if not isinstance(example_id, str) or not example_id.strip():
            raise ValueError(f"prompt at {path}:{lineno} has empty 'example_id'")
        if example_id in seen_ids:
            raise ValueError(
                f"duplicate example_id at {path}:{lineno}: {example_id!r}"
            )
        seen_ids.add(example_id)

        metadata = record.get("metadata")
        if metadata is None:
            record["metadata"] = {}
        elif not isinstance(metadata, dict):
            raise ValueError(
                f"prompt at {path}:{lineno} 'metadata' must be a dict, "
                f"got {type(metadata).__name__}"
            )

        records.append(record)

    if not records:
        raise ValueError(f"prompts file produced no records: {path}")

    log.info("loaded %d prompts from %s", len(records), path)
    return records


# === MODEL / TRANSFORMER LOADING ===

def _resolve_device(config_device: str, override_device: str | None) -> str:
    """
    Pick the runtime device. CLI override wins over config; "auto" means
    cuda if available else cpu. Unknown values raise ValueError.
    """
    raw = (override_device if override_device is not None else config_device) or "auto"
    requested = raw.strip().lower()
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested not in {"cuda", "cpu"}:
        raise ValueError(
            f"unsupported device {requested!r}; expected one of cuda/cpu/auto"
        )
    if requested == "cuda" and not torch.cuda.is_available():
        log.warning("device=cuda requested but cuda is not available; using cpu")
        return "cpu"
    return requested


def _resolve_dtype(dtype_name: str) -> torch.dtype:
    if dtype_name not in _DTYPE_MAP:
        raise ValueError(
            f"unsupported dtype {dtype_name!r}; expected one of {sorted(_DTYPE_MAP)}"
        )
    return _DTYPE_MAP[dtype_name]


def load_model_and_tokenizer(
    base_model_name: str,
    dtype: torch.dtype,
    device: str,
) -> tuple[Any, Any]:
    """
    Load tokenizer and base causal LM, alias pad→eos for Mistral, and place
    the model on the requested device in eval mode.
    """
    log.info("loading tokenizer: %s", base_model_name)
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        # mistral has no pad token; alias to eos without resizing embeddings
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    log.info(
        "loading base model: %s  dtype=%s  device=%s", base_model_name, dtype, device,
    )
    try:
        model = AutoModelForCausalLM.from_pretrained(base_model_name, dtype=dtype)
    except TypeError:
        # older transformers versions only accept torch_dtype
        model = AutoModelForCausalLM.from_pretrained(base_model_name, torch_dtype=dtype)
    model = model.to(device)
    model.eval()
    return model, tokenizer


def build_input_transformer(
    model: Any,
    tokenizer: Any,
    cfg: RouterCalibrationConfig,
) -> InputTransformer:
    """Construct an InputTransformer from the router_calibration config."""
    sv = SteeringVectorConfig(
        economic_vector_path=cfg.paths.steering_vectors.economic_vector_path,
        social_vector_path=cfg.paths.steering_vectors.social_vector_path,
        vector_method=cfg.input_transformer.vector_method,
        use_final_aggregated_vectors=cfg.input_transformer.use_final_aggregated_vectors,
        selected_layers=list(cfg.input_transformer.selected_layers),
        pooling_method=cfg.input_transformer.pooling_method,
        use_centering=cfg.input_transformer.use_centering,
        neutral_reference_path=cfg.input_transformer.neutral_reference_path,
    )
    return InputTransformer(model=model, tokenizer=tokenizer, steering_config=sv)


# === VALIDATION ===

def _is_finite_number(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return math.isfinite(float(value))


def validate_prompt_state(state: PromptState, where: str) -> torch.Tensor:
    """
    Verify the InputTransformer output against the schema documented in
    src/09_moce_components.py and return the hidden representation as a 1D
    float32 CPU tensor ready to stack into the [N, hidden_dim] artifact.
    """
    hidden = state.hidden_representation
    if not isinstance(hidden, torch.Tensor):
        raise ValueError(
            f"{where}: hidden_representation must be torch.Tensor, "
            f"got {type(hidden).__name__}"
        )
    if hidden.dim() != 1:
        raise ValueError(
            f"{where}: hidden_representation must be rank 1, "
            f"got shape {tuple(hidden.shape)}"
        )
    if hidden.numel() == 0:
        raise ValueError(f"{where}: hidden_representation is empty (length 0)")
    if not torch.isfinite(hidden).all().item():
        raise ValueError(f"{where}: hidden_representation contains NaN or inf entries")

    qs = state.quadrant_scores
    if not isinstance(qs, dict):
        raise ValueError(
            f"{where}: quadrant_scores must be dict, got {type(qs).__name__}"
        )
    if set(qs.keys()) != set(CANONICAL_QUADRANT_ORDER):
        raise ValueError(
            f"{where}: quadrant_scores keys {sorted(qs.keys())} != "
            f"canonical {sorted(CANONICAL_QUADRANT_ORDER)}"
        )
    for key in CANONICAL_QUADRANT_ORDER:
        if not _is_finite_number(qs[key]):
            raise ValueError(
                f"{where}: quadrant_scores[{key!r}]={qs[key]!r} is not finite"
            )

    for name, value in (
        ("bias_magnitude", state.bias_magnitude),
        ("economic_score", state.economic_score),
        ("social_score", state.social_score),
    ):
        if not _is_finite_number(value):
            raise ValueError(f"{where}: {name}={value!r} is not finite")

    return hidden.detach().to(dtype=torch.float32, device="cpu")


# === FEATURE BUILDING ===

def build_feature_records(
    prompts: list[dict],
    transformer: InputTransformer,
    hidden_filename: str,
    *,
    expected_hidden_dim: int | None = None,
) -> tuple[list[dict], torch.Tensor]:
    """
    Run InputTransformer.transform over every prompt and assemble the
    features.jsonl rows + the [N, hidden_dim] hidden tensor.

    Args:
        prompts:             list of validated prompt dicts (load_prompts output).
        transformer:         pre-built InputTransformer (steering vectors loaded).
        hidden_filename:     basename of the hidden artifact to embed in
                             hidden_representation_ref strings (e.g. "hidden.pt").
        expected_hidden_dim: if provided, the stacked tensor's hidden_dim must
                             match this value. Set by run_build to
                             cfg.input_transformer.calibration_input_dim so a
                             feature/router-input geometry mismatch surfaces here
                             instead of at calibrated-router training time.

    Returns:
        (records, hidden_tensor) — records preserve prompt order; hidden_tensor
        is float32 with shape [len(records), hidden_dim].
    """
    if not prompts:
        raise ValueError("build_feature_records: prompts list is empty")
    if not isinstance(hidden_filename, str) or not hidden_filename.strip():
        raise ValueError("hidden_filename must be a non-empty string")

    records: list[dict] = []
    hiddens: list[torch.Tensor] = []
    expected_dim: int | None = None
    total = len(prompts)

    for row_index, prompt in enumerate(prompts):
        example_id  = prompt["example_id"]
        prompt_text = prompt["prompt_text"]
        source      = prompt["source"]
        in_metadata = prompt.get("metadata") or {}

        state: PromptState = transformer.transform(prompt_text)
        hidden_cpu = validate_prompt_state(state, where=f"example_id={example_id}")

        if expected_dim is None:
            expected_dim = int(hidden_cpu.shape[0])
        elif hidden_cpu.shape[0] != expected_dim:
            raise ValueError(
                f"example_id={example_id}: hidden_representation length "
                f"{hidden_cpu.shape[0]} differs from expected {expected_dim}"
            )

        hiddens.append(hidden_cpu)

        # serialise quadrant_scores in canonical order for stable JSONL output
        ordered_quadrant: dict[str, float] = {
            key: float(state.quadrant_scores[key]) for key in CANONICAL_QUADRANT_ORDER
        }

        records.append({
            "example_id": example_id,
            "prompt_text": prompt_text,
            "source": source,
            "quadrant_scores": ordered_quadrant,
            "bias_magnitude": float(state.bias_magnitude),
            "economic_score": float(state.economic_score),
            "social_score":   float(state.social_score),
            "hidden_representation_ref": f"{hidden_filename}:{row_index}",
            "metadata": {
                **in_metadata,
                "input_transformer": dict(state.metadata or {}),
                "feature_source": "InputTransformer.transform",
                "hidden_dtype": "float32",
            },
        })

        n_done = row_index + 1
        if n_done % LOG_PROGRESS_EVERY == 0 or n_done == total:
            log.info("processed %d/%d prompts", n_done, total)

    hidden_tensor = torch.stack(hiddens, dim=0).to(dtype=torch.float32).contiguous()
    if hidden_tensor.shape[0] != len(records):
        raise ValueError(
            f"hidden tensor row count {hidden_tensor.shape[0]} != "
            f"feature row count {len(records)}"
        )
    if expected_hidden_dim is not None and hidden_tensor.shape[1] != expected_hidden_dim:
        raise ValueError(
            f"hidden_tensor hidden_dim {hidden_tensor.shape[1]} does not match "
            f"calibration_input_dim {expected_hidden_dim}"
        )

    return records, hidden_tensor


# === IO ===

def write_outputs(
    records: list[dict],
    hidden_tensor: torch.Tensor,
    features_path: Path,
    hidden_path: Path,
) -> None:
    """
    Persist features and hidden tensor with crash-safe semantics: write to
    sibling .tmp paths first, then atomically rename. Avoids leaving a
    half-written features.jsonl next to a stale hidden.pt on failure.
    """
    if not isinstance(hidden_tensor, torch.Tensor):
        raise ValueError(
            f"hidden_tensor must be torch.Tensor; got {type(hidden_tensor).__name__}"
        )
    if hidden_tensor.dtype != torch.float32:
        raise ValueError(f"hidden_tensor must be float32; got {hidden_tensor.dtype}")
    if hidden_tensor.dim() != 2:
        raise ValueError(
            f"hidden_tensor must be 2D [N, hidden_dim]; "
            f"got shape {tuple(hidden_tensor.shape)}"
        )
    if hidden_tensor.shape[0] != len(records):
        raise ValueError(
            f"hidden_tensor row count {hidden_tensor.shape[0]} != "
            f"records count {len(records)}"
        )

    features_path.parent.mkdir(parents=True, exist_ok=True)
    hidden_path.parent.mkdir(parents=True, exist_ok=True)

    features_tmp = features_path.with_name(features_path.name + ".tmp")
    hidden_tmp   = hidden_path.with_name(hidden_path.name + ".tmp")

    with features_tmp.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    torch.save(hidden_tensor, hidden_tmp)

    features_tmp.replace(features_path)
    hidden_tmp.replace(hidden_path)

    log.info("wrote %d features to %s", len(records), features_path)
    log.info(
        "wrote hidden tensor shape=%s dtype=%s to %s",
        tuple(hidden_tensor.shape), hidden_tensor.dtype, hidden_path,
    )


# === ORCHESTRATION ===

def run_build(
    cfg: RouterCalibrationConfig,
    *,
    limit: int | None = None,
    device_override: str | None = None,
    transformer: InputTransformer | None = None,
) -> dict[str, Path]:
    """
    Top-level driver: read prompts, run the InputTransformer, write outputs.

    The optional `transformer` argument lets tests inject a pre-built
    InputTransformer (with fake model/tokenizer/synthetic vectors) so the
    end-to-end path can run without downloading a real HF model.
    """
    prompts = load_prompts(cfg.paths.prompts_path)
    log.info("input prompts: %s", cfg.paths.prompts_path)
    log.info("output features: %s", cfg.paths.features_path)
    log.info("output hidden:  %s", cfg.paths.hidden_path)

    if limit is not None:
        if limit <= 0:
            raise ValueError(f"--limit must be positive; got {limit}")
        prompts = prompts[:limit]
        log.info("truncated to first %d prompts via --limit", len(prompts))

    if transformer is None:
        device = _resolve_device(cfg.model.device, device_override)
        dtype = _resolve_dtype(cfg.model.dtype)
        log.info(
            "model=%s  dtype=%s  device=%s",
            cfg.model.base_model, dtype, device,
        )
        model, tokenizer = load_model_and_tokenizer(cfg.model.base_model, dtype, device)
        transformer = build_input_transformer(model, tokenizer, cfg)

    hidden_filename = cfg.paths.hidden_path.name
    records, hidden_tensor = build_feature_records(
        prompts,
        transformer,
        hidden_filename,
        expected_hidden_dim=cfg.input_transformer.calibration_input_dim,
    )

    write_outputs(records, hidden_tensor, cfg.paths.features_path, cfg.paths.hidden_path)
    log.info(
        "done — final hidden tensor shape=%s", tuple(hidden_tensor.shape),
    )
    return {
        "features_path": cfg.paths.features_path,
        "hidden_path": cfg.paths.hidden_path,
    }


# === MAIN ===

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "build router-calibration features (features.jsonl + hidden.pt) "
            "from data/router/prompts.jsonl"
        ),
    )
    p.add_argument(
        "--config",
        type=Path,
        required=True,
        help="path to config.yaml containing the router_calibration block",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="optional cap on number of prompts (smoke testing/debugging)",
    )
    p.add_argument(
        "--device",
        type=str,
        default="auto",
        help="device override: auto/cuda/cpu (default: auto)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_router_calibration_config(args.config)
    log.info("config loaded from %s", args.config)
    run_build(cfg, limit=args.limit, device_override=args.device)


if __name__ == "__main__":
    main()
